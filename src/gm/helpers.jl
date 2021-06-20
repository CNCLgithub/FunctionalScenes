using UnicodePlots
# using Images
using ImageInTerminal
using Statistics
using Parameters: @unpack

export ModelParams

@with_kw struct ModelParams

    gt::Room
    template::Room = template_from_room(gt)

    # thickness of walls
    offset::CartesianIndex{2} = CartesianIndex(2, 2)

    # tracker parameters
    dims::Tuple{Int64, Int64} = (6, 6) # size of each tracker
    inner_ref::CartesianIndices = CartesianIndices(dims)
    tracker_ref::CartesianIndices = _tracker_ref(template, dims, offset)
    n_trackers::Int64 = length(tracker_ref)
    linear_ref::LinearIndices = _linear_ref(template)
    # default spread for a given tracker
    default_tracker_p::Float64 = 1.0
    # spread for each tracker
    tracker_ps::Vector{Float64} = fill(default_tracker_p,
                                       length(tracker_ref))
    tracker_size::Int64 = prod(dims)

    base::Tuple{Int64, Int64} = (3, 3)
    factor::Int64 = 2
    levels::Int64 = _count_levels(dims, base, factor)

    # tracker prior
    level_weights::Vector{Float64} = fill(1.0/levels, levels)
    bounds::Tuple{Float64, Float64} = (0., 1.) # range of bernoulli weights

    # simulation
    instances::Int64 = 10

    # graphics
    feature_weights::String
    device = _load_device()
    img_size::Tuple{Int64, Int64} = (480, 720)
    graphics = _init_graphics(template, img_size, device)
    model = functional_scenes.init_alexnet(feature_weights, device)
    # minimum variance in prediction
    base_sigma::Float64 = 0.1
end

function load(::Type{ModelParams}, path::String; kwargs...)
    ModelParams(;read_json(path)..., kwargs...)
end

function _init_graphics(r, img_size, device)
    graphics = functional_scenes.SimpleGraphics(img_size, device)
    base_d = translate(r, false)
    graphics.set_from_scene(base_d)
    return graphics
end

function _tracker_ref(r::Room, dims::Tuple{Int64, Int64},
                      offset::CartesianIndex{2})
    offset = Tuple(offset) .* 2
    space = (steps(r) .- offset) ./ dims
    space = Int64.(space)
    CartesianIndices(space)
end

function _linear_ref(room)
    LinearIndices(steps(room))
end

function _count_levels(dims, base, factor)::Int64
    n = 1
    q = all(dims .> level_dims(base, factor, n))
    while q
        n += 1
        q = all(dims .> level_dims(base, factor, n))
    end
    return n
end


function level_dims(base::Tuple{T,T}, factor::T, lvl::T) where {T<:Int64}
    lvl == 1 ? (1, 1) : base .* (factor^(lvl - 2))
end
function level_dims(params::ModelParams, lvl::Int64)
    @assert lvl <= params.levels "level $(lvl) out of bounds $(params.levels)"
    @unpack base, factor = params
    level_dims(base, factor, lvl)
end

# TODO play with using `size(tracker_ref)`
function tracker_prior_args(params::ModelParams)
    @unpack tracker_ref, n_trackers = params
    _params = fill(params, n_trackers)
    tids = collect(Int64, 1:n_trackers)
    (_params, tids)
end

function level_prior(params::ModelParams, tid::Int64, lvl::Int64)
    @unpack base, tracker_ps = params
    d = level_dims(params, lvl)
    spread = tracker_ps[tid]
    bounds = (0., spread)
    fill(bounds, d)
end

function create_obs(params::ModelParams, r::Room)
    mu, _ = graphics_from_instances([r], params)
    constraints = Gen.choicemap()
    constraints[:viz] = mu
    constraints
end


function add_init_constraints!(cm::Gen.ChoiceMap, params::ModelParams, r::Room)
    @unpack n_trackers, instances = params
    state = occupancy_position(r)
    # first column in room that has furniture
    frontier = findfirst(vec(sum(state, dims = 1) .!= 0)) - 6
    state_ref = CartesianIndices(steps(r))
    constraints = Gen.choicemap()
    for t = 1 : n_trackers
        for idx in tracker_to_state(params, t)
            # tracker is out of view
            last(Tuple(state_ref[idx])) < frontier || continue
            cm[:trackers => t => :level] = 1
            cm[:trackers => t => :state] = zeros((1, 1))

            # for i = 1 : instances
            #     cm[:instances => i => :furniture => idx => :flip] = false
            # end
        end
    end
    return nothing
end

function consolidate_local_states(params::ModelParams, states)::Array{Float64, 3}
    @unpack dims, factor, tracker_size, n_trackers = params
    gs = Array{Float64}(undef, dims[1], dims[2], n_trackers)
    for (i, tracker) in enumerate(states)
        level, state = tracker
        # resize the tracker's state to span its slice in the scene
        gs[:, :, i] = refine_state(state, dims)
    end
    gs
end

function room_from_state_args(params::ModelParams, gs::Array{Float64, 3})
    @unpack instances = params
    gs = fill(gs, instances)
    ps = fill(params, instances)
    (gs, ps)
end

function add_from_state_flip(params::ModelParams, occupied)::Room
    @unpack dims, template, tracker_size, n_trackers = params
    occupied = reshape(occupied, (dims[1], dims[2], n_trackers))
    vs = findall(occupied)
    fs = state_to_room(params, vs)
    add(template, Set{Int64}(fs))
end

function tracker_to_state(params::ModelParams, tracker::Int64)
    start = (tracker - 1) * params.tracker_size + 1
    @>> (start:(start + params.tracker_size - 1)) collect(Int64) vec
end

"""
Converts cartesian coordinats in tracker state space (inner_ref, tracker_id) to
the linear indices of room coordinates (x, y)
"""
function state_to_room(params::ModelParams, vs::Vector{CartesianIndex{3}})
    @unpack dims, offset, tracker_ref, inner_ref, linear_ref = params
    result = Vector{Int64}(undef, length(vs))
    for (i, cind) in enumerate(vs)
        x, y, tracker = Tuple(cind)
        outer = tracker_ref[tracker]
        # the top left corner of the tracker
        outer = CartesianIndex((Tuple(outer) .- (1, 1)) .* dims)
        c = CartesianIndex{2}(x, y) + outer + offset
        # convert to linear space (graph vertex space)
        result[i] = linear_ref[c]
    end
    return result
end

function project_state_weights(params::ModelParams, state)
    @>> state begin
        CartesianIndices
        collect(CartesianIndex{3})
        vec
        state_to_room(params)
    end
end

function viz_global_state(trace::Gen.Trace)
    params = first(get_args(trace))
    grid = zeros(steps(params.template))
    state, _ = get_retval(trace)
    inds = project_state_weights(params, state)
    grid[inds] = state

    grid = reverse(grid, dims = 1)
    println(heatmap(grid, border = :none,
                    title = "inferred geometry",
                    colorbar_border = :none,
                    colormap = :inferno))
    return nothing
end

function viz_ddp_state(grid::Matrix{Float64})
    grid = reverse(grid, dims = 1)
    println(heatmap(grid, border = :none,
                    title = "ddp geometry",
                    colorbar_border = :none,
                    colormap = :inferno))
    return nothing
end

function viz_ocg(ocg; title = "occupancy grid")
    # ocg = mean(ocg)
    ocg = reverse(ocg, dims = 1)
    println(heatmap(ocg,
                    title = title,
                    border = :none,
                    colorbar_border = :none,
                    colormap = :inferno
                    ))
end

function viz_compute_weights(weights)
    viz_barplot(weights, "compute weights")
end

function viz_sensitivity(trace, weights)
    params = first(get_args(trace))
    # weights = exp.(weights)
    grid = reshape(weights, size(params.tracker_ref))
    display(grid)
    grid = reverse(grid, dims = 1)
    grid = repeat(grid, inner = (3,3))
    println(heatmap(grid,
                    title = "sensitivity",
                    border = :none,
                    # colorbar_border = :none,
                    colormap = :inferno
                    ))
end

function viz_barplot(weights, title)
    names = collect(keys(weights))
    vals = collect(values(weights))
    n = min(length(names), 5)
    inds = sortperm(vals, rev=true)[1:n]
    println(barplot(names[inds], vals[inds],
                    title = "top $(n) $(title)"))
end

function viz_render(trace::Gen.Trace)
    _, instances = get_retval(trace)
    params = first(get_args(trace))
    g = params.graphics
    translated = translate(first(instances), false, cubes = true)
    batch = @pycall functional_scenes.render_scene_batch([translated], g)::PyObject
    batch = Array{Float64, 4}(batch.cpu().numpy())
    display(colorview(RGB, batch[1, :, :, :]))
end


function viz_gt(gt::Room; kwargs...)
    grid = occupancy_position(gt)
    viz_ocg(grid; title = "gt state", kwargs...)
    display(gt)
end

# or pass average image to alexnet
function image_from_instances(instances, params)
    g = params.graphics
    if length(instances) > 1
        instances = [instances[1]]
    end
    instances = map(r -> translate(r, false, cubes=true), instances)
    batch = @pycall functional_scenes.render_scene_batch(instances, g)::PyObject
    batch = Array{Float64, 4}(batch.cpu().numpy())
end

function graphics_from_instances(instances, params)
    g = params.graphics
    if length(instances) > 1
        instances = [instances[1]]
    end
    instances = map(r -> translate(r, false, cubes=true), instances)
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


function all_selections_from_model(params::ModelParams)
    s = select()
    # associated rooms samples
    for tracker in 1:params.n_trackers
        idxs = tracker_to_state(params, tracker)
        for i = 1:params.instances, j in idxs
            push!(s, :instances => i => :furniture => j => :flip)
        end
    end
    StaticSelection(s)
end

function select_from_model(params::ModelParams, tracker::Int64)
    s = select()
    # associated rooms samples
    idxs = tracker_to_state(params, tracker)
    for i = 1:params.instances, j in idxs
        push!(s, :instances => i => :furniture => j => :flip)
    end
    StaticSelection(s)
end

function selections(params::ModelParams)
    n = length(params.tracker_ps)
    idxs = collect(Int64, 1:n)
    nodes = map(Symbol, idxs)
    ks = Tuple(nodes)
    vals = map(x -> select_from_model(params, x), idxs)
    vals = Tuple(vals)
    LittleDict(ks, vals)
end

function swap_tiles!(g::PathGraph, p::Tuple{Tile, Tile})
    x,y = p
    a = get_prop(g, y, :type)
    b = get_prop(g, x, :type)
    set_prop!(g, x, :type, a)
    set_prop!(g, y, :type, b)
    return nothing
end

function connected(g::PathGraph, v::Tile)::Set{Tile}
    s = @>> v bfs_tree(g) edges collect induced_subgraph(g) last Set
    isempty(s) ? Set([v]) : s
end


function template_from_room(r::Room)
    clear_room(r)
end


function batch_og(tr::Gen.Trace)
    @>> get_retval(tr) begin
        last
        collect(Room)
        map(x -> occupancy_grid(x, sigma = 0.0, decay = 0.0))
        mean
    end
end

function batch_compare_og(og_a, og_b)
    wsd(og_a, og_b)
end

function refine_state(params::ModelParams, state, lvl::Int64)
    @>> lvl begin
        level_dims(params)
        refine_state(state)
    end
end
function refine_state(mat, dims::Tuple{Int64, Int64})
    kdim = Int64.(dims ./ size(mat))
    repeat(mat, inner = kdim)
end
function refine_state(mat::Float64, dims::Tuple{Int64, Int64})
    fill(mat, dims)
end

function coarsen_state(params::ModelParams, state, lvl::Int64)
    @>> lvl begin
        level_dims(params)
        coarsen_state(state)
    end
end

function coarsen_state(mat::Array{T}, dims::Tuple{Int64, Int64}) where {T}
    m_dims = size(mat)
    ref = CartesianIndices(dims)
    lref = LinearIndices(ref)
    kdim = Int64.(m_dims ./ dims)
    # steps = Int64.(size(mat) ./ dims)
    k = prod(kdim)
    kc = 1.0 / k
    kern = CartesianIndices(kdim)
    lkern = LinearIndices(kern)

    mus = zeros(T, dims[1], dims[2])

    for outer in ref
        c = CartesianIndex((Tuple(outer) .- (1, 1)) .* kdim)
        for inner in kern
            idx = c + inner
            mus[outer] += mat[idx]
        end
        mus[outer] *= kc
    end
    mus
end

"""
Probability of refining or coarsening

lvl = 1   ? 1
      <N  ? 0.5,
      N   ? 0
"""
function refine_weight(params::ModelParams, lvl::Int64)
    @unpack levels = params
    if lvl == 1
        w = 1.0
    elseif lvl < levels
        w = 0.5
    else
        w = 0.0
    end
    return w
end

function clean_state(state::AbstractArray;
                     sigma = 1E-5)
    clamp.(state, sigma, 1.0 - sigma)
end

# # stable softmax
# function softmax(x)
#     x = x .- maximum(x)
#     return exp.(x) / sum(exp.(x))
# end
