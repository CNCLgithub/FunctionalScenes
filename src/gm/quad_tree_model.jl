export QuadTreeModel

#################################################################################
# Model specification
#################################################################################

"""
Parameters for an instance of the `QuadTreeModel`.
"""
@with_kw struct QuadTreeModel

    #############################################################################
    # Room geometry
    #############################################################################
    #
    # Ground truth room
    gt::GridRoom
    # Same as `gt` but without obstacles
    template::GridRoom = clear_room(gt)

    entrance::Int64 = first(entrance(gt))
    exit::Int64 = first(exits(gt))

    #############################################################################
    # Tracker parameters
    #############################################################################
    #
    dims::Tuple{Int64, Int64} = steps(gt)
    # coarsest node is centered at [0,0]
    # and has a span of [1,1]
    center::SVector{2, Float64} = zeros(2)
    bounds::SVector{2, Float64} = ones(2)

    # maximum resolution of each tracker
    max_depth::Int64 = _max_depth(gt)
    # probablility of splitting node
    # TODO: currently unused (hard coded to 0.5)
    split_prob::Float64 = 0.5

    # coarsest node
    start_node::QTProdNode = QTProdNode(center, bounds, 1, max_depth, 1)

    #############################################################################
    # Multigranular empirical estimation
    #############################################################################
    #
    # number of draws from stochastic scene state
    instances::Int64 = 20

    #############################################################################
    # Graphics
    #############################################################################
    #
    img_size::Tuple{Int64, Int64} = (120, 180)
    device::PyObject = _load_device()
    spp::Int64 = 24
    # preload partial scene mesh
    scene_d::PyDict = _init_mitsuba_scene(gt, img_size)
    # minimum variance in prediction
    base_sigma::Float64 = 1.0
end

"""
Maximum depth of quad tree
"""
function _max_depth(r::GridRoom)
    @unpack bounds, steps = r
    # FIXME: allow for arbitrary room steps
    @assert all(ispow2.(steps))
    convert(Int64, minimum(log2.(steps)) + 1)
end

function load(::Type{QuadTreeModel}, path::String; kwargs...)
    QuadTreeModel(;read_json(path)..., kwargs...)
end


struct QTPath
    g::SimpleGraph
    dm::Matrix{Float64}
    edges::Vector{AbstractEdge}
end


struct QuadTreeState
    qt::QTAggNode
    gs::Matrix{Float64}
    img_mu::Array{Float64, 3}
    path::QTPath
    lv::Vector{QTAggNode}
end

function QuadTreeState(qt, gs, img_mu,  pg)
   QuadTreeState(qt, gs, img_mu, pg, leaf_vec(qt))
end

function QTPath(st::QTAggNode)
    g = SimpleGraph(1)
    dm = Matrix{Float64}(undef, 1, 1)
    dm[1] = weight(st) * area(st.node)
    edges = [Graphs.SimpleEdge(1,1)]
    QTPath(g, dm, edges)
end

function add_from_state_flip(params::QuadTreeModel,
                             occupied::AbstractVector{Bool})::GridRoom
    @unpack template = params
    d = vec(data(template))
    possible = @. occupied && !(d == wall_tile)
    add(template, Set{Int64}(findall(possible)))
end

"""
    project_qt(lv, dims)

Projects the quad tree to a nxn matrix

# Arguments
- `lv::Vector{QTAggNode}`: The leaves of a quad tree
- `dims`: Dimensions of the target grid
"""
function project_qt(lv::Vector{QTAggNode}, dims::Tuple{Int64, Int64})
    gs = Matrix{Float64}(undef, dims[1], dims[2])
    project_qt!(gs, lv)
    return gs
end

function project_qt!(gs::Matrix{Float64},
                     lv::Vector{QTAggNode})
    d = size(gs, 1)
    for x in lv
        idx = node_to_idx(x.node, d)
        # potentially broadcast coarse states
        gs[idx] .= weight(x)
    end
    return nothing
end


#################################################################################
# Graphics
#################################################################################

function node_to_mitsuba(n::QTAggNode, room_dims)
    @unpack center, dims = (n.node)
    pos = Vector{Float32}(undef, 3)
    pos[1:2] = center .* room_dims
    pos[3] = 0.75
    sdim = Vector{Float32}(undef, 3)
    sdim[1:2] = dims .* room_dims
    sdim[3] = 1.5
    m = @pycall fs_py.create_cube(pos, sdim, n.mu)::PyObject
end

function stats_from_qt(lv::Vector{QTAggNode},
                       p::QuadTreeModel)
    @unpack gt, img_size, spp, base_sigma = p
    n = length(lv)
    scene_d = _init_mitsuba_scene(gt, img_size)
    for i = 1:n
        scene_d["cube_$(i)"] = node_to_mitsuba(lv[i], gt.steps)
    end
    result = @pycall mi.render(mi.load_dict(scene_d), spp=spp)::PyObject
    # H x W x C
    mu = @pycall numpy.array(result)::Array{Float32, 3}
    sd = fill(p.base_sigma, size(mu))
    (mu, sd)
end


function tile_to_mitsuba(room::GridRoom, tile::Int64)
    r,c  = steps(room)
    center = idx_to_node_space(tile, r)
    pos = Vector{Float32}(undef, 3)
    pos[1:2] = center .* bounds(room)
    pos[3] = 0.75
    sdim = Vector{Float32}(undef, 3)
    delta = (bounds(room) ./ (r,c))
    sdim[1:2] = [delta[1], delta[2]]
    sdim[3] = 1.5
    m = @pycall fs_py.create_cube(pos, sdim, 1.0)::PyObject
end

function render_mitsuba(r::GridRoom, p::QuadTreeModel)
    @unpack img_size, spp, base_sigma = p
    (row,col) = steps(r)
    delta = bounds(r) ./ (row, col)
    obstacle_tiles = findall(vec(data(r)) .== obstacle_tile)
    no = length(obstacle_tiles)
    scene_d = _init_mitsuba_scene(r, img_size)
    for i = 1:no
        obs_idx = @inbounds obstacle_tiles[i]
        obs_pos = idx_to_node_space(obs_idx, row)
        scene_d["cube_$(i)"] = tile_to_mitsuba(r, obs_idx)
    end
    result = @pycall mi.render(mi.load_dict(scene_d), spp=spp)::PyObject
    # mi.util.write_bitmap("/spaths/tests/render_mitsuba.png", result)
    # H x W x C
    mu = @pycall numpy.array(result)::Array{Float32, 3}
end

#################################################################################
# Planning
#################################################################################

function a_star_heuristic(nodes::Vector{QTAggNode}, dest::Int64)
    _dest = nodes[dest]
    src -> dist(nodes[src].node, _dest.node)
end

function nav_graph(lv::Vector{QTAggNode})
    n = length(lv)
    adm = fill(false, (n, n))
    dsm = fill(Inf, (n, n))
    @inbounds for i = 1:(n-1), j = (i+1):n
        x = lv[i]
        y = lv[j]
        # only care when nodes are touching
        contact(x.node, y.node) || continue
        d = dist(x.node, y.node)
        #  work to traverse each node
        p = area(x.node) / (area(x.node) + area(y.node))
        work = d * (p * weight(x) + (1-p)*weight(y))
        adm[i, j] = adm[j, i] = true
        dsm[i, j] = dsm[j, i] = work
    end
    (adm, dsm, lv)
end

"""
    qt_a_star(st, d, ent, ext)

Applies `a_star` to the quad tree.

# Arguments
- `st::QTAggNode`: The root node of the QT
- `d::Int64`: The row dimensions of the room
- `ent::Int64`: The entrance tile
- `ext::Int64`: The exit tile

# Returns
A tuple, first element is `QTPath` and the second is a vector
 of the leave nodes in QT.
"""
function qt_a_star(lv::Vector{QTAggNode}, d::Int64, ent::Int64, ext::Int64)
    #REVIEW: Shouldn't this either be 1 or >4?
    length(lv) == 1 && return QTPath(first(lv))
    # adjacency, distance matrix, and leaves
    ad, dm = nav_graph(lv)
    # scale dist matrix by room size
    rmul!(dm, d)
    g = SimpleGraph(ad)
    # map entrance and exit in room to qt
    ent_p = idx_to_node_space(ent, d)
    a = findfirst(s -> contains(s, ent_p), lv)
    ext_p = idx_to_node_space(ext, d)
    b = findfirst(s -> contains(s, ext_p), lv)
    heuristic = a_star_heuristic(lv, b)
    # compute path and path grid
    path = a_star(g, a, b, dm, heuristic)
    QTPath(g, dm, path)
end

#################################################################################
# Inference utils
#################################################################################

"""
    ridx_to_leaf(st, idx, c)

Returns

# Arguments

- `st::QuadTreeState`
- `ridx::Int64`: Linear index in room space
- `c::Int64`: Column size of room
"""
function room_to_leaf(st::QuadTreeState, ridx::Int64, c::Int64)
    point = idx_to_node_space(ridx, c)
    # TODO: traverse through the QT should be faster
    l = findfirst(s -> contains(s, point), st.lv)
    st.lv[l]
end

const qt_model_all_downstream_selection = StaticSelection(select(:instances))
function all_downstream_selection(p::QuadTreeModel)
    s = select()
    for i = 1:p.instances
        push!(s, :instances => i)
    end
    return s
end

function _init_mitsuba_scene(room::GridRoom, res)
    variants = @pycall mi.variants()::PyObject
    if "cuda_ad_rgb" in variants
        @pycall mi.set_variant("cuda_ad_rgb")::PyObject
    else
        @pycall mi.set_variant("scalar_rgb")::PyObject
    end
    (r,c) = steps(room)
    delta = bounds(room) ./ (r,c)
    dim = [c, r, 10]
    # map exit tiles to y-shifts for `door` argument
    exit_tiles = exits(room)
    ne = length(exit_tiles)
    exit_ys = Matrix{Float64}(undef, 2, ne)
    @inbounds for i = 1:ne
        exit_pos_y = idx_to_node_space(exit_tiles[i], r)[2]
        exit_pos_y *= r
        exit_ys[1, i] = exit_pos_y + delta[2]
        exit_ys[2, i] = exit_pos_y
    end
    door_left = maximum(exit_ys)
    door_right = minimum(exit_ys)
    door = [door_left, -door_right]
    scene_d = @pycall fs_py.initialize_scene(dim, door, res)::PyDict
end

function create_obs(p::QuadTreeModel)
    @unpack gt = p
    constraints = Gen.choicemap()
    constraints[:viz] = render_mitsuba(gt, p)
    constraints
end
