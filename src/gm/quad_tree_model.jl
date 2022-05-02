export QuadTreeModel

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
    start_node::QTNode = QTNode(center, bounds, 1, max_depth)

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


@with_kw struct QuadTreeState
    qt::QTState
    gs::Matrix{Float64}
    instances::Vector{GridRoom}
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

function room_from_state_args(params::QuadTreeModel, gs::Matrix{Float64})
    @unpack instances = params
    gs = fill(gs, instances)
    ps = fill(params, instances)
    (gs, ps)
end

function graphics_from_instances(instances, params)
    g = params.graphics
    if length(instances) > 1
        instances = [instances[1]]
    end
    # println("printing instances")
    # foreach(viz_gt, instances)

    instances = map(r -> translate(r, Int64[], cubes=true), instances)
    batch = @pycall functional_scenes.render_scene_batch(instances, g)::PyObject
    features = Array{Float64, 4}(batch.cpu().numpy())
    # features = @pycall functional_scenes.nn_features.single_feature(params.model,
    #                                                                 "features.6",
    #                                                                 batch)::Array{Float64, 4}
    mu = mean(features, dims = 1)
    if length(instances) > 1
        sigma = std(features, mean = mu, dims = 1)
        sigma .+= params.base_sigma
    else
        sigma = fill(params.base_sigma, size(mu))
    end
    (mu[1, :, :, :], sigma[1, :, :, :])
end
