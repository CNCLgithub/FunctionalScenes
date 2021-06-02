using UnicodePlots
using Images
using ImageInTerminal
using Statistics
using Parameters: @unpack

export ModelParams

@with_kw struct ModelParams

    gt::Room
    template::Room = template_from_room(gt)

    # thickness of walls
    thickness::Tuple{Int64, Int64} = (2,2)

    # tracker parameters
    dims::Tuple{Int64, Int64} = (6, 6) # size of each tracker
    offset::Tuple{Int64, Int64} = (0, 2) # offset in world space for each tracker
    inner_ref::CartesianIndices = CartesianIndices(dims)
    tracker_ref::CartesianIndices = _tracker_ref(template, dims, thickness, offset)
    linear_ref::LinearIndices = _linear_ref(template)
    default_tracker_p::Float64 = 0.5
    tracker_ps::Vector{Float64} = fill(default_tracker_p,
                                       length(tracker_ref))
    tracker_size::Int64 = prod(dims)

    base::Tuple{Int64, Int64} = (1, 1)
    factor::Int64 = 3
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

function _tracker_ref(r::Room, dims, thickness, offset)
    thickness = thickness .* 2
    space = (steps(r) .- thickness) ./ dims
    space = space .- offset
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
    base .* factor^(lvl - 1)
end
function level_dims(params::ModelParams, lvl::Int64)
    @unpack base, factor = params
    level_dims(base, factor, lvl)
end

function tracker_prior_args(params::ModelParams)
    fill(params, length(params.tracker_ref))
end

function level_prior(params::ModelParams, lvl::Int64)
    @unpack base, bounds, factor = params
    c = prod(level_dims(params, lvl))
    fill(bounds, c)
end


function create_obs(params::ModelParams, r::Room)
    mu, _ = graphics_from_instances([r], params)
    constraints = Gen.choicemap()
    constraints[:viz] = mu
    return constraints
end

function consolidate_local_states(params::ModelParams, states)::Matrix{Float64}
    @unpack dims, factor, tracker_size, tracker_ps = params

    gs = Matrix{Float64}(undef, tracker_size, length(tracker_ps))

    for (i, tracker) in enumerate(states)
        level, state = tracker
        # the dimensions of the tracker for a given level
        lds = level_dims(params, level)

        # resize the tracker's state to span its slice in the scene
        if prod(lds) == 1
            # 1x1 tracker, cannot use `Image.imresize`
            resized = fill(state[1], tracker_size)
        else
            resized = @> state begin
                # first convert from vec -> matrix
                reshape(lds)
                # resize from tracker space to scene space
                imresize(dims)
                vec
            end
        end
        gs[:, i] = resized
    end
    gs
end

function room_from_state_args(params::ModelParams, gs::Matrix{Float64})
    @unpack instances = params
    gs = fill(gs, instances)
    ps = fill(params, instances)
    (gs, ps)
end

# function map_tracker_weights(params::ModelParams, tid::Int64)
#     vs = @>> tid tracker_to_state(params) state_to_room(params)
#     ws = params.tile_weights[vs] |> vec
#     collect(Tuple{Float64, Float64}, zip(ws .- params.tile_window,
#                                          ws .+ params.tile_window))
# end

# """
# Maps the
# """
# function map_tracker_weights(params::ModelParams)
#     ids = 1:length(active)
#     @>> ids begin
#         map(i -> map_tracker_weights(params, i))
#     end
# end

function add_from_state_flip(params::ModelParams, occupied)::Room
    r = params.template
    vs = findall(occupied)
    fs = state_to_room(params, vs)
    add(r, Set{Int64}(fs))
end

function tracker_to_state(params::ModelParams, tracker::Int64)
    start = (tracker - 1) * params.tracker_size + 1
    @>> (start:(start + params.tracker_size - 1)) collect(Int64) vec
end

function state_to_room(params::ModelParams, vs::Vector{CartesianIndex{N}}) where {N}
    @>> vs LinearIndices collect(Int64) state_to_room(params)
end

function state_to_room(params::ModelParams, vs::Vector{Int64})
    result = Vector{Int64}(undef, length(vs))
    for (i,j) in enumerate(vs)
        tracker_idx = ceil(j / params.tracker_size) |> Int64
        inner_idx = j % params.tracker_size + 1 |> Int64

        outer = Tuple(params.tracker_ref[tracker_idx]) .- (1,1) .+ params.offset
        outer = Tuple(outer) .* params.dims
        cart = Tuple(params.inner_ref[inner_idx]) .+ outer .+ params.thickness
        cart = CartesianIndex(cart)
        result[i] = params.linear_ref[cart]
    end
    return result
end

function room_to_tracker(params::ModelParams, fs)
    fs = collect(Int64, fs)
    # ts = Vector{Int64}(undef, size(fs)...)
    ts = Int64[]
    full_coords = CartesianIndices(steps(params.template))
    tracker_linear = LinearIndices(size(params.tracker_ref))
    for (i,j) in enumerate(fs)
        full_cart = full_coords[j]
        xy = Tuple(full_cart) .- params.thickness
        xy = xy .- (params.dims .* params.offset)
        xy = Int64.(ceil.(xy ./ params.dims))
        if any(xy .> size(params.tracker_ref))
            continue
        end
        push!(ts, tracker_linear[xy...])
        # ts[i] = tracker_linear[xy...]
    end
    return ts
end

function project_state_weights(params::ModelParams, state)
    @show size(state)
    n = prod(size(state))
    state_to_room(params, collect(1:n))
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


function viz_ocg(ocg)
    ocg = mean(ocg)
    ocg = reverse(ocg, dims = 1)
    println(heatmap(ocg,
                    title = "occupancy grid",
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


function viz_gt(query)
    params = first(query.args)
    g = params.graphics
    display(params.gt)
    translated = translate(params.gt, false, cubes = true)
    batch = @pycall functional_scenes.render_scene_batch([translated], g)::PyObject
    batch = Array{Float64, 4}(batch.cpu().numpy())
    display(colorview(RGB, batch[1, :, :, :]))
end

# or pass average image to alexnet
function graphics_from_instances(instances, params)
    g = params.graphics
    if length(instances) > 1
        instances = [instances[1]]
    end
    instances = map(r -> translate(r, false, cubes=true), instances)
    batch = @pycall functional_scenes.render_scene_batch(instances, g)::PyObject
    features = @pycall functional_scenes.nn_features.single_feature(params.model,
                                                                    "features.6",
                                                                    batch)::Array{Float64, 4}
    mu = mean(features, dims = 1)
    if length(instances) > 1
        sigma = std(features, mean = mu, dims = 1)
        sigma .+= params.base_sigma
    else
        sigma = fill(params.base_sigma, size(mu))
    end
    (mu[1, :, :, :], sigma[1, :, :, :])
end

function select_from_model(params::ModelParams, tracker::Int64)
    s = select()
    # tracker state
    for i = 1:params.tracker_size
        push!(s, :trackers => tracker => :state )
    end
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
        map(x -> occupancy_grid(x, sigma = 0.7, decay = 0.0001))
        mean
    end
end

function batch_compare_og(og_a, og_b)
    cor(vec(mean(og_a)), vec(mean(og_b)))
    # map(wsd, og_a, og_b) |> sum
end
