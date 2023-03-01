export QuadTreeModel, qt_a_star

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
    instances::Int64 = 10

    #############################################################################
    # Graphics
    #############################################################################
    #
    img_size::Tuple{Int64, Int64} = (256, 256)
    device = _load_device()
    # configure pytorch3d render
    graphics = _init_graphics(template, img_size, device)
    # minimum variance in prediction
    base_sigma::Float64 = 0.1
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
    ws = Matrix{Float64}(undef, 1, 1)
    ws[1] = weight(st) * area(st.node)
    QTPath(SimpleGraph(ones(1,1)), ws, Int64[1, 1])
end

function add_from_state_flip(params::QuadTreeModel,
                             occupied::AbstractVector{Bool})::GridRoom
    @unpack template = params
    d = vec(data(template))
    possible = @. occupied && !(d == wall_tile)
    add(template, Set{Int64}(findall(possible)))
end

"""
    project_qt(qt, dims)

Projects the quad tree to a nxn matrix
"""
function project_qt(qt::QTAggNode, dims::Tuple{Int64, Int64})
    gs = Matrix{Float64}(undef, dims[1], dims[2])
    project_qt!(gs, qt)
    return gs
end

function project_qt(params::QuadTreeModel, qt::QTAggNode)
    project_qt(qt, params.dims)
end

function project_qt!(gs::Matrix{Float64},
                     st::QTAggNode)
    d = size(gs, 1)
    heads::Vector{QTAggNode} = [st]
    while !isempty(heads)
        head = pop!(heads)
        if isempty(head.children)
            idx = node_to_idx(head.node, d)
            # potentially broadcast coarse states
            gs[idx] .= weight(head)
        else
            append!(heads, head.children)
        end
    end
    return nothing
end


#################################################################################
# Graphics
#################################################################################

function instances_from_gen(params::QuadTreeModel, instances)
    rs = Vector{GridRoom}(undef, params.instances)
    @inbounds for i = 1:params.instances
        rs[i] = add_from_state_flip(params, instances[i])
    end
    return rs
end


function image_from_instances(instances, params)
    g = params.graphics
    if length(instances) > 1
        instances = [instances[1]]
    end
    instances = map(r -> translate(r, Int64[], cubes=true), instances)
    # TODO compute mean and std in python first
    batch = @pycall functional_scenes.render_scene_batch(instances, g)::PyObject
    Array{Float64, 4}(batch.cpu().numpy())
end



function graphics_from_instances(instances, params)
    g = params.graphics
    if length(instances) > 1
        instances = [instances[1]]
    end
    # println("printing instances")
    # foreach(viz_gt, instances)

    batch = image_from_instances(instances, params)
    mu = mean(batch, dims = 1)
    if length(instances) > 1
        sigma = std(batch, mean = mu, dims = 1)
        sigma .+= params.base_sigma
    else
        sigma = fill(params.base_sigma, size(mu))
    end
    (mu[1, :, :, :], sigma[1, :, :, :])
end


#################################################################################
# Planning
#################################################################################

function a_star_heuristic(dest::QTAggNode, d::Real)
    src -> dist(src.node, dest.node) * d
end

function nav_graph(st::QTAggNode)
    adm = fill(false, (st.leaves, st.leaves))
    dsm = fill(Inf, (st.leaves, st.leaves))
    lv = leaf_vec(st)
    # TODO: add @inbounds
    for i = 1:(st.leaves-1), j = (i+1):st.leaves
        x = lv[i]
        y = lv[j]
        d = dist(x.node, y.node)
        # only care when nodes are touching
        contact(x.node, y.node, d) || continue
        #  work to traverse each node
        work = area(x.node) * weight(x) + area(y.node) * weight(y)
        adm[i, j] = adm[j, i] = true
        dsm[i, j] = dsm[j, i] = work
    end
    (adj, dsm, lv)
end

function qt_a_star(st::QTAggNode, d::Int64, ent::Int64, ext::Int64)
    #REVIEW: Shouldn't this either be 1 or >4?
    st.leaves < 4 && return (QTPath(st), QTAggNode[st])
    # adjacency, distance matrix, and leaves
    ad, dm, lv = nav_graph(st)
    # scale dist matrix by room size
    rmul!(dm, d)
    g = SimpleGraph(ad)
    # map entrance and exit in room to qt
    ent_p = idx_to_node_space(ent, d)
    a = findfirst(s -> contains(s, ent_p), lv)
    ext_p = idx_to_node_space(ext, d)
    b = findfirst(s -> contains(s, ext_p), lv)
    # compute path and path grid
    path = a_star(g, a, b,
                distmx = dm,
                heurisitc = a_star_heuristic(b, d))
    (QTPath(g, ds, path), lv)
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

function create_obs(params::QuadTreeModel, r::GridRoom)
    mu, _ = graphics_from_instances([r], params)
    constraints = Gen.choicemap()
    constraints[:viz] = mu
    constraints
end
