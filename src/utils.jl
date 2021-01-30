using Parameters: @with_kw

export ModelParams

@with_kw struct ModelParams

    template::Room
    # thickness of walls
    thickness::Tuple{Int64, Int64} = (2,2)

    # tracker parameters
    dims::Tuple{Int64, Int64} = (6, 6)
    inner_ref::CartesianIndices = CartesianIndices(dims)
    # tracker_verts::Matrix{Int64} = _tracker_verts(template)
    tracker_ref::CartesianIndices = _tracker_ref(template, dims, thickness)
    linear_ref::LinearIndices = _linear_ref(template)
    default_tracker_p::Float64 = 0.5
    tracker_ps::Vector{Float64} = fill(default_tracker_p,
                                       length(tracker_ref))
    tracker_size::Int64 = prod(dims)
    bounds::Vector{Float64} = [0.01, 0.45, 0.65, 0.99]
    active::Vector{Float64} = [0.5, 0.0, 0.5]
    inactive::Vector{Float64} = [0.0, 1.0, 0.0]

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

function _tracker_ref(r::Room, dims, thickness)
    thickness = thickness .* 2
    space = (steps(r) .- thickness) ./ dims
    space = Int64.(space)
    CartesianIndices(space)
end

function _linear_ref(room)
    LinearIndices(steps(room))
end

function load(::Type{ModelParams}, path::String; kwargs...)
    ModelParams(;read_json(path)..., kwargs...)
end

function create_obs(params::ModelParams, r::Room)
    mu, _ = graphics_from_instances([r], params)
    display(size(mu))
    constraints = Gen.choicemap()
    constraints[:viz] = mu
    return constraints
end


# TODO something better than vcat?
function consolidate_local_states(states)
    states = vcat(states...)
    states = clamp.(states, 0.1, 0.9)
    display(states)
    return states
end

function define_trackers(params::ModelParams, active)
    f = x -> x ? (params.bounds, params.active) : (params.bounds, params.inactive)
    ws = @>> active map(f) map(w -> fill(w, params.tracker_size))
    display(ws)
    return ws
end

function add_from_state_flip(params::ModelParams, occupied)::Room
    r = params.template
    vs = findall(occupied)
    fs = Vector{Int64}(undef, length(vs))
    for (i,j) in enumerate(vs)
        tracker_idx = ceil(j / params.tracker_size) |> Int64
        inner_idx = j % params.tracker_size + 1 |> Int64

        outer = Tuple(params.tracker_ref[tracker_idx]) .- (1,1)
        outer = Tuple(outer) .* params.dims
        cart = Tuple(params.inner_ref[inner_idx]) .+ outer .+ params.thickness
        cart = CartesianIndex(cart)
        fs[i] = params.linear_ref[cart]
        # (or,oc) = Tuple(params.tracker_ref[tracker_idx])
        # (ir,ic) = Tuple(params.inner_ref[inner_idx])
        # fs[i] = (oc-1) * size(params.tracker_ref, 1) * params.tracker_size +
        #     (or - 1) * params.dims[1] +
        #     (ic - 1) * params.tracker_size +
        #     ir
    end
    add(r, Set{Int64}(fs))
end

function tracker_to_state(params::ModelParams, tracker::Int64)
    start = (tracker - 1) * params.tracker_size + 1
    @>> (start:(start + params.tracker_size - 1)) collect(Int64) vec
    # outer = Tuple(params.tracker_ref[tracker]) .- (1,1)
    # outer = Tuple(outer) .* params.dims
    # cart = params.inner_ref .+ CartesianIndex(outer)
    # @>> params.linear_ref[cart] collect(Int64) vec
end


# or pass average image to alexnet
function graphics_from_instances(instances, params)
    g = params.graphics
    instances = map(r -> translate(r, false, cubes=true), instances)
    batch = @pycall functional_scenes.render_scene_batch(instances, g)::PyObject
    features = @pycall functional_scenes.nn_features.single_feature(params.model,
                                                                    "features.6",
                                                                    batch)::Array{Float64, 4}
    mu = mean(features, dims = 1)
    if length(instances) > 1
        sigma = std(features, mean = mu, dims = 1) .+ params.base_sigma
    else
        sigma = fill(params.base_sigma, size(mu))
    end
    display(any(isnan.(mu)))
    display(any(isnan.(sigma)))
    (mu[1, :, :, :], sigma[1, :, :, :])
end

function select_from_model(params::ModelParams, tracker::Int64)
    s = select(:active => tracker => :flip)
    # # tracker state
    for i = 1:params.tracker_size
        push!(s, :trackers => tracker => :state => i => :bflip)
    end
    # # associated rooms samples
    # idxs = tracker_to_state(params, tracker)
    # display(idxs)
    # for i = 1:params.instances, j in idxs
    #     push!(s, :instances => i => :furniture => j => :flip)
    # end
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
        collect(Room)
        map(occupancy_grid)
        mean
    end
end

function softmax(x)
    x = x .- maximum(x)
    exs = exp.(x)
    sxs = sum(exs)
    n = length(x)
    iszero(sxs) ? fill(1.0/n, n) : exs ./ sxs
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
