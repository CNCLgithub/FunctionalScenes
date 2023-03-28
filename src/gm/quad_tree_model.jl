export QuadTreeModel,
    render_mitsuba

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
    # Planning / Navigation
    #############################################################################
    #
    # weight to balance cost of distance with obstacle occupancy
    dist_cost::Float64 = 1.0

    #############################################################################
    # Graphics
    #############################################################################
    #
    img_size::Tuple{Int64, Int64} = (128, 128)
    device::PyObject = _load_device()
    # samples per pixel
    spp::Int64 = 24
    # preload partial scene mesh
    scene::PyObject = _init_mitsuba_scene(gt, img_size)
    sparams::PyObject = @pycall mi.traverse(scene)::PyObject
    skey::String = "grid.interior_medium.sigma_t.data"
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

function QTPath(st::QTAggNode)
    g = SimpleGraph(1)
    dm = Matrix{Float64}(undef, 1, 1)
    dm[1] = weight(st) * area(st.node)
    edges = [Graphs.SimpleEdge(1,1)]
    QTPath(g, dm, edges)
end


struct QuadTreeState
    qt::QuadTree
    img_mu::Array{Float64, 3}
    img_sd::Array{Float64, 3}
    path::QTPath
end




#################################################################################
# Graphics
#################################################################################

function stats_from_qt(qt::QuadTree,
                       p::QuadTreeModel)
    @unpack projected = qt
    @unpack gt, dims, scene, sparams, skey, spp, base_sigma = p
    weights = Matrix{Float64}(projected)
    # turn off walls
    weights[data(gt) .== wall_tile] .= 0
    # need to transpose for mitsuba
    weights = Matrix{Float64}(weights')
    obs_ten = reshape(weights, (1, size(weights)..., 1))
    prev_val = sync_params!(sparams, skey, obs_ten)

    result = @pycall mi.render(scene, spp=spp)::PyObject
    # need to set gamma correction for proper numpy export
    result = @pycall mi.Bitmap(result).convert(srgb_gamma=true)::PyObject
    mu = @pycall numpy.array(result)::Array{Float32, 3}
    sd = fill(p.base_sigma, size(mu))

    # REVIEW: might not be necessary
    sync_params!(sparams, skey, prev_val)

    return (mu, sd)
end


function render_mitsuba(r::GridRoom, scene, params, key, spp)
    obs_mat = zeros(steps(r))
    obs_mat[data(r) .== obstacle_tile] .= 1.0
    # need to transport data matrix for mitsuba
    obs_mat = Matrix{Float64}(obs_mat')
    obs_ten = reshape(obs_mat, (1, size(obs_mat)..., 1))
    prev_val = sync_params!(params, key, obs_ten)

    result = @pycall mi.render(scene, spp=spp)::PyObject
    bitmap = @pycall mi.Bitmap(result).convert(srgb_gamma=true)::PyObject
    # H x W x C
    mu = @pycall numpy.array(bitmap)::Array{Float32, 3}
    # REVIEW: might not be necessary
    sync_params!(params, key, prev_val)
    return mu
end

#################################################################################
# Planning
#################################################################################

function a_star_heuristic(nodes::Vector{QTAggNode}, dest::QTAggNode,
                          scale::Float64)
    src -> dist(nodes[src].node, dest.node) * scale
end
function a_star_heuristic(nodes::Vector{QTAggNode}, dest::Int64, scale::Float64)
    _dest = nodes[dest]
    a_star_heuristic(nodes, _dest)
end

function nav_graph(lv::Vector{QTAggNode}, dist_weight::Float64)
    n = length(lv)
    adm = fill(false, (n, n))
    dsm = fill(Inf, (n, n))

    @inbounds for i = 1:(n-1), j = (i+1):n
        x = lv[i]
        y = lv[j]
        # only care when nodes are touching
        contact(x.node, y.node) || continue
        d = dist(x.node, y.node) * dist_weight
        #  work to traverse each node
        p = area(x.node) / (area(x.node) + area(y.node))
        work = d + (p * weight(x) + (1-p)*weight(y))
        adm[i, j] = adm[j, i] = true
        dsm[i, j] = dsm[j, i] = work
    end
    (adm, dsm)
end

"""
    qt_a_star(qt, d, ent, ext)

Applies `a_star` to the quad tree.

# Arguments
- `qt::QuadTree`: A quad tree to traverse over leaves
- `d::Int64`: The row dimensions of the room
- `ent::Int64`: The entrance tile
- `ext::Int64`: The exit tile

# Returns
A tuple, first element is `QTPath` and the second is a vector
 of the leave nodes in QT.
"""
function qt_a_star(qt::QuadTree, dw::Float64, ent::Int64, ext::Int64)
    @unpack root, leaves, mapping = qt
    length(leaves) == 1 && return QTPath(first(leaves))
    # adjacency, distance matrix
    ad, dm = nav_graph(leaves, dw)

    g = SimpleGraph(ad)

    # map entrance and exit in room to qt
    row_d::Int64 = max_leaves(qt.root.node)
    ent_point = idx_to_node_space(ent, row_d)
    ent_node = traverse_qt(root, ent_point)
    ent_idx = mapping[ent_node.node.tree_idx]

    ext_point = idx_to_node_space(ext, row_d)
    ext_node = traverse_qt(root, ext_point)
    ext_idx = mapping[ext_node.node.tree_idx]

    # L2 heuristic used in A*
    heuristic = a_star_heuristic(leaves, ext_node, dw)
    # compute path and path grid
    path = a_star(g, ent_idx, ext_idx, dm, heuristic)
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
    traverse_qt(st.qt, point)
end

#TODO: Remove
const qt_model_all_downstream_selection = StaticSelection(select(:instances))
function all_downstream_selection(p::QuadTreeModel)
    s = select()
    for i = 1:p.instances
        push!(s, :instances => i)
    end
    return s
end

function _init_mitsuba_scene(room::GridRoom, res)
    # variant should already be configured in project `__init__`
    # see src/FunctionalScenes.jl
    # mi.set_variant("cuda_ad_rgb")
    mi.set_variant("llvm_ad_rgb")
    (r,c) = steps(room)
    delta = bounds(room) ./ (r,c)
    dim = [c, r, 5]
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
    # empty room
    scene_d = @pycall fs_py.initialize_scene(dim, door, res)::PyDict
    # add volume grid
    scene_d["grid"] = @pycall fs_py.create_volume([r, c])::PyObject
    scene = @pycall mi.load_dict(scene_d)::PyObject
    return scene
end

function create_obs(p::QuadTreeModel)
    @unpack gt = p
    constraints = Gen.choicemap()
    constraints[:viz] = render_mitsuba(gt, p.scene, p.sparams,
                                       p.skey, p.spp)
    constraints
end

function sync_params!(params::PyObject, key, val)::PyObject
    prev = get(params, PyObject, key)
    set!(params, key, val)
    py"$params.update()"o
    return prev
end
