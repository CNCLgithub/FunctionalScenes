export QuadTreeModel, qt_a_star
export img_from_instance

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
    img_size::Tuple{Int64, Int64} = (256, 256)
    device::PyObject = _load_device()
    # configure pytorch3d render
    camera::PyObject = py"{'position': [-16.5, 0.0, -10.75]}"o
    graphics::PyObject = _init_graphics(img_size, device, camera)
    # preload partial scene mesh
    scene_mesh::PyObject = _init_scene_mesh(gt, device, graphics)
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
    instances::Vector{GridRoom}
    img_mu::Array{Float64, 3}
    path::QTPath
    lv::Vector{QTAggNode}
end

function QuadTreeState(qt, gs, instances, img_mu,  pg)
   QuadTreeState(qt, gs, instances, img_mu, pg, leaf_vec(qt))
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

"""
    instances_from_gen(params, instances)

Converts random samples of geometry from `obst_gen` into `GridRoom`.
"""
function instances_from_gen(params::QuadTreeModel, instances)
    n = length(instances)
    rs = Vector{GridRoom}(undef, n)
    @inbounds for i = 1:n
        rs[i] = add_from_state_flip(params, instances[i])
    end
    return rs
end

function stats_from_instances(inst::AbstractArray{<:Room},
                              p::QuadTreeModel)
    @unpack graphics, scene_mesh, dims, device = p
    n = length(inst)
    meshes = Vector{PyObject}(undef, n)
    vdim = maximum(dims) * 0.5
    @inbounds for i = 1:n
        meshes[i] = _init_scene_mesh(inst[i], device, graphics;
                                     obstacles=true)
    end
    # REVIEW: unwanted mutatation of scene mesh?
    # @inbounds for i = 1:n
    #     voxels = voxelize(inst[i], obstacle_tile)
    #     obs_mesh = @pycall fs_py.from_voxels(voxels, vdim, device;
    #                                          color="blue")::PyObject
    #     # TODO: shorten...
    #     meshes[i] = @pycall pytorch3d.structures.join_meshes_as_scene([scene_mesh,
    #                                               obs_mesh])::PyObject
    # end
    mu,sd = @pycall fs_py.batch_render_and_stats(meshes, graphics)::Tuple{PyArray, PyArray}
    # mu::Array{Float64, 3} = _mu
    # sd::Array{Float64, 3} = _sd
    sd .+= p.base_sigma
    # @show p.base_sigma
    # @show maximum(sd)
    (mu, sd .+ p.base_sigma)
end


function img_from_instance(i::Room,
                           p::QuadTreeModel)
    @unpack graphics, scene_mesh, dims, device, base_sigma = p
    voxels = voxelize(i, obstacle_tile)
    vdim = maximum(dims) * 0.5
    obs_mesh = @pycall fs_py.from_voxels(voxels, vdim, device; color="blue")::PyObject
    pyargs = py"[$scene_mesh, $obs_mesh]"o
    mesh = @pycall pytorch3d.structures.join_meshes_as_scene(pyargs)::PyObject
    # 1 x C x H x W
    img = @pycall fs_py.render_mesh_single(mesh, graphics)::Array{Float64, 4}
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

function _init_scene_mesh(r::GridRoom, device::PyObject, graphics::PyObject;
                          obstacles::Bool = false)
    n = obstacles ? 3 : 2
    meshes = Vector{PyObject}(undef, n)
    voxels = voxelize(r, floor_tile)
    vdim = maximum(size(voxels)) * 0.5
    meshes[1] = @pycall fs_py.from_voxels(voxels, vdim, device)::PyObject
    voxels = voxelize(r, wall_tile)
    meshes[2] = @pycall fs_py.from_voxels(voxels, vdim, device)::PyObject
    if obstacles
        voxels = voxelize(r, obstacle_tile)
        obs_mesh = @pycall fs_py.from_voxels(voxels, vdim, device;
                                            color="blue")::PyObject
        meshes[3] = obs_mesh
    end
    mesh = @pycall pytorch3d.structures.join_meshes_as_scene(meshes)::PyObject
end

function create_obs(p::QuadTreeModel)
    @unpack gt = p
    img = img_from_instance(gt, p)
    constraints = Gen.choicemap()
    constraints[:viz] = img[1, :, :, :]
    constraints
end
