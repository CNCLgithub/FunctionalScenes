using Parameters: @with_kw
using UnicodePlots
using Images
using ImageInTerminal
using Statistics

export ModelParams

@with_kw struct ModelParams

    gt::Room
    template::Room = template_from_room(gt)

    # thickness of walls
    thickness::Tuple{Int64, Int64} = (2,2)

    # tracker parameters
    dims::Tuple{Int64, Int64} = (6, 6)
    offset::Tuple{Int64, Int64} = (0, 2)
    inner_ref::CartesianIndices = CartesianIndices(dims)
    # tracker_verts::Matrix{Int64} = _tracker_verts(template)
    tracker_ref::CartesianIndices = _tracker_ref(template, dims, thickness, offset)
    linear_ref::LinearIndices = _linear_ref(template)
    default_tracker_p::Float64 = 0.5
    tracker_ps::Vector{Float64} = fill(default_tracker_p,
                                       length(tracker_ref))
    tracker_size::Int64 = prod(dims)

    # tracker prior
    active_bias::Float64 = 0.2
    inactive_mu::Float64 = 0.5
    tile_weights::Matrix{Float64} = biased_tile_prior(gt, active_bias)
    tile_window::Float64 = 0.05
    bounds::Vector{Float64} = [0.01, 0.02, 0.49, 0.51, 0.98, 0.99]
    active::Vector{Float64} = [0.8, 0.0, 0.0, 0.0, 0.2]
    inactive::Vector{Float64} = [0.0, 0.0, 1.0, 0.0, 0.0]

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

function _init_graphics(r, img_size, device)
    graphics = functional_scenes.SimpleGraphics(img_size, device)
    base_d = translate(r, false)
    graphics.set_from_scene(base_d)
    return graphics
end

function _load_device()
    if torch.cuda.is_available()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else
        device = torch.device("cpu")
    end
    return device
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

function load(::Type{ModelParams}, path::String; kwargs...)
    ModelParams(;read_json(path)..., kwargs...)
end

function biased_tile_prior(gt::Room, sigma::Float64)
    g = pathgraph(gt)
    # grid = zeros(steps(gt))
    grid = fill(0.2, steps(gt))
    vs = @>> g vertices Base.filter(v -> istype(g, v, :furniture))
    grid[vs] .= 0.8
    # vs = @>> g vertices Base.filter(v -> istype(g, v, :floor))
    # grid[vs] .= 0.1
    # gf = Kernel.gaussian(sigma)
    # grid = imfilter(grid, gf, "symmetric")
    vs = @>> g vertices Base.filter(v -> istype(g, v, :wall))
    grid[vs] .= 0.0

    # max_g = maximum(grid)
    # grid = grid ./ max_g .* 0.8
    _grid = reverse(grid, dims = 1)
    println(heatmap(_grid, border = :none,
                    title = "geometry prior",
                    colorbar_border = :none,
                    colormap = :inferno))
    return grid
end

function create_obs(params::ModelParams, r::Room)
    mu, _ = graphics_from_instances([r], params)
    constraints = Gen.choicemap()
    constraints[:viz] = mu
    return constraints
end


# TODO something better than vcat?
function consolidate_local_states(states)
    states = vcat(states...)
    return states
end

function active_tracker(params::ModelParams, tid::Int64)
    vs = @>> tid tracker_to_state(params) state_to_room(params)
    ws = params.tile_weights[vs] |> vec
    collect(Tuple{Float64, Float64}, zip(ws .- params.tile_window,
                                         ws .+ params.tile_window))
end


function passive_tracker(params::ModelParams, tid::Int64)
    ws = fill(params.inactive_mu, params.tracker_size)
    collect(Tuple{Float64, Float64}, zip(ws .- params.tile_window,
                                         ws .+ params.tile_window))
end

function define_trackers(params::ModelParams, active)
    f = (i,a) -> a ? active_tracker(params, i) : passive_tracker(params, i)
    # ws = @>> active map(f) map(w -> fill(w, params.tracker_size))
    ids = 1:length(active)
    ws = @>> active map(f, ids)
    return ws
end

function add_from_state_flip(params::ModelParams, occupied)::Room
    r = params.template
    vs = findall(occupied)
    fs = state_to_room(params, vs)
    add(r, Set{Int64}(fs))
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
    n = length(state)
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

function tracker_to_state(params::ModelParams, tracker::Int64)
    start = (tracker - 1) * params.tracker_size + 1
    @>> (start:(start + params.tracker_size - 1)) collect(Int64) vec
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
    push!(s, :active => tracker => :flip)
    # tracker state
    for i = 1:params.tracker_size
        push!(s, :trackers => tracker => :state => i => :sflip)
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

function softmax(x)
    x = x .- maximum(x)
    exs = exp.(x)
    sxs = sum(exs)
    n = length(x)
    isnan(sxs) || iszero(sxs) ? fill(1.0/n, n) : exs ./ sxs
end

"""
    read_json(path)
    opens the file at path, parses as JSON and returns a dictionary
"""
function read_json(path)
    open(path, "r") do f
        global data
        data = JSON.parse(f)
    end

    # converting strings to symbols
    sym_data = Dict()
    for (k, v) in data
        sym_data[Symbol(k)] = v
    end

    return sym_data
end
