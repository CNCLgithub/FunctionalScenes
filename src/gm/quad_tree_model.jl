export QuadTreeModel, qt_a_star

#################################################################################
# Model specification
#################################################################################

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
    start_node::QTNode = QTNode(center, bounds, 1, max_depth, 1)

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
    @assert all(ispow2.(steps))
    convert(Int64, minimum(log2.(steps)) + 1)
end

function load(::Type{QuadTreeModel}, path::String; kwargs...)
    QuadTreeModel(;read_json(path)..., kwargs...)
end

struct QuadTreeState
    qt::QTState
    gs::Matrix{Float64}
    instances::Vector{GridRoom}
    pg::Matrix{Bool}
    lv::Vector{QTState}
end

function QuadTreeState(qt, gs, instances, pg)
   QuadTreeState(qt, gs, instances, pg, leaf_vec(qt))
end


function add_from_state_flip(params::QuadTreeModel,
                             occupied::AbstractVector{Bool})::GridRoom
    @unpack template = params
    d = vec(data(template))
    possible = @. occupied && !(d == wall_tile)
    add(template, Set{Int64}(findall(possible)))
end

function consolidate_qt_states(params::QuadTreeModel, qt::QTState)::Matrix{Float64}
    @unpack dims = params
    gs = Matrix{Float64}(undef, dims[1], dims[2])
    consolidate_qt_states!(gs, qt)
    gs
end


function consolidate_qt_states!(gs::Matrix{Float64}, st::QTState)
    foreach(s -> consolidate_qt_states!(gs, s), st.children)
    # only update for terminal states
    !isempty(st.children) && return nothing
    idx = node_to_idx(st.node, size(gs, 1))
    # potentially broadcast coarse states
    gs[idx] .= weight(st)
    return nothing
end

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

function qt_a_star(st::QTState, d::Int64, ent::Int64, ext::Int64)
    st.leaves < 4 && return (Matrix{Bool}(trues(d,d)), QTState[st])
    # adjacency, distance matrix, and leaves
    adj, ds, lv = nav_graph(st)
    # display(sparse(adj))
    # display(ds)
    # scale dist matrix by room size
    rmul!(ds, d)
    g = SimpleGraph(adj)
    # map entrance and exit to qt
    ent_p = idx_to_node_space(ent, d)
    a = findfirst(s -> contains(s, ent_p), lv)
    ext_p = idx_to_node_space(ext, d)
    b = findfirst(s -> contains(s, ext_p), lv)
    # compute path and path grid
    ps = a_star(g, a, b, ds)
    pg = Matrix{Bool}(falses(d,d))
    # @show a
    # @show b
    for e in ps
        idxs = node_to_idx(lv[src(e)].node, d)
        pg[idxs] .= true
    end
    pg[node_to_idx(lv[b].node, d)] .= true
    (pg, lv)
end

function ridx_to_leaf(st::QuadTreeState, ridx::Int64)
    point = idx_to_node_space(ridx, d)
    l = findfirst(s -> contains(s, point), st.lv)
end

const qt_model_all_downstream_selection = StaticSelection(select(:instances))
all_downstream_selection(p::QuadTreeModel) = qt_model_all_downstream_selection

function create_obs(params::QuadTreeModel, r::GridRoom)
    mu, _ = graphics_from_instances([r], params)
    constraints = Gen.choicemap()
    constraints[:viz] = mu
    constraints
end
