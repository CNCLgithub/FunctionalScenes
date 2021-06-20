using CSV
using Gen
using Lazy
using JLD2
using Images, FileIO
using DataFrames
using LinearAlgebra:norm
using Statistics: mean, std
using Distributions
using UnicodePlots
using Parameters: @unpack

import FunctionalScenes: Room, furniture, shift_furniture, navigability, wsd,
    occupancy_grid, diffuse_og, safe_shortest_path, shift_tile, room_to_tracker,
    viz_ocg, model, select_from_model, batch_og, batch_compare_og, viz_global_state

function compare_pixels(a::String, b::String)
    x = load(a)
    y = load(b)
    norm(x .- y)
end

function move_type(d::Real)
    if d == -1
        :up
    elseif d == 1
        :down
    elseif d < -1
        :left
    elseif d > 1
        :right
    else
        error("Not a valid move")
    end
end

function relative_path(path)
    lp = length(path)
    p = Vector{Symbol}(undef, max(0, lp - 1))
    for i = 2:lp
        d = path[i] - path[i-1]
        p[i-1] = move_type(d)
    end
    return p
end


function lv_distance(a, b)
    size_x = length(a) + 1
    size_y = length(b) + 1
    (isempty(a) || isempty(b)) && (return max(size_x, size_y) - 1)
    matrix = zeros((size_x, size_y))
    matrix[:, 1] = 0:(size_x-1)
    matrix[1, :] = 0:(size_y-1)
    for x = 2:size_x, y = 2:size_y
        if a[x-1] == b[y-1]
            matrix[x,y] = min(
                matrix[x-1, y] + 1,
                matrix[x-1, y-1],
                matrix[x, y-1] + 1
            )
        else
            matrix[x,y] = min(
                matrix[x-1,y] + 1,
                matrix[x-1,y-1] + 1,
                matrix[x,y-1] + 1
            )
        end
    end
    return last(matrix)
end

function compare_og(a,b)
    og_a = occupancy_grid(a, decay = 0.0001, sigma = 0.7)
    og_b = occupancy_grid(b, decay = 0.0001, sigma = 0.7)
    viz_ocg(og_a)
    viz_ocg(og_b)
    r = cor(vec(mean(og_a)), vec(mean(og_b)))
    d = map(wsd, og_a, og_b) |> sum
    (r, d)
end


function room_sensitivity(history, params, fs)
    weights = hcat(history...)
    weights = sum(weights, dims = 2) |> vec
    top_trackers = sortperm(weights, rev = true)[1:3]
    tracker_ids = room_to_tracker(params, fs)
    (top_trackers, weights, tracker_ids)
    # mean(zs[tracker_ids])
end

function og_from_step(query, step)
    choices = step["estimates"][:trace]
    tr, _ = generate(model, query.args, choices)
    viz_global_state(tr)
    batch_og(tr)
end

function cross_geometry()
end

function cross_predict(query, choices_a, choices_b, trackers)
    params = first(query.args)

    from_a = choicemap()
    for t in trackers
        s = select_from_model(params, t)
        from_a = merge(from_a, get_selected(choices_a, s))
    end
    all_trackers = collect(1:prod(size(params.tracker_ref)))
    other_trackers = setdiff(all_trackers, trackers)
    from_b = choicemap()
    for t in other_trackers
        s = select_from_model(params, t)
        from_b = merge(from_b, get_selected(choices_b, s))
    end

    constraints = merge(from_a, from_b)

    constraints[:viz] = query.observations[:viz]
    _, ls = generate(model, query.args, constraints)
    return ls
end

function sample_steps(chain_p, k)
    chain = load(chain_p)
    n = length(chain)
    logscores = map(i -> chain["$(i)"]["log_score"], 13:n)
    ws = exp.(logscores .- logsumexp(logscores))
    println("$(n), $(k)")
    println(size(ws))
    step_ids = rand(Distributions.Categorical(ws), k)
    println(step_ids)
    steps = @>> step_ids map(i -> chain["$(i+12)"]) collect
    sens = sens_from_chain(chain, 13, n)
    (steps, sens)
end

function sens_from_chain(chain, m, n)
    @>> collect(m:n) begin
    	map(s -> chain["$(s)"]["aux_state"][:sensitivities])
        x -> hcat(x...)
        x -> sum(x, dims = 2)
        vec
    end
end

function load_map_chain(chain_p)
    chain = load(chain_p)
    k = length(chain) 

    # first 18 steps are deterministic
    current_step = chain["18"]
    n = length(chain)
    weight_history = []
    cycles = Vector{Int64}(undef, 18)
    for i = 19:n
        new_step = chain["$(i)"]
        sens = new_step["aux_state"][:sensitivities]
        push!(weight_history, sens)
        cycles = new_step["aux_state"][:cycles]
        current_step = current_step["log_score"] < new_step["log_score"] ? new_step : current_step
    end
    return current_step, weight_history, cycles
end

function viz_trace_history(history)
    weights = hcat(history...)
    nt = size(weights, 1)
    ymax = maximum(weights)
    plt = lineplot(weights[1, :], name = "1",
                   ylim = (0, ymax * 1.1))
    for i = 2:nt
        lineplot!(plt, weights[i, :], name = "$(i)")
    end
    println(plt)

    sums = sum(weights, dims = 2)
    names = ["$(i)" for i = 1:nt]
    plt = barplot(names, vec(sums))
    println(plt)
end

function lv_distance(a::Room, b::Room)
    es = exits(a)
    paths_a = @>> es map(e -> safe_shortest_path(a,e)) map(relative_path)
    paths_b = @>> es map(e -> safe_shortest_path(b,e)) map(relative_path)
    lvd = sum(map(lv_distance, paths_a, paths_b))
end

function extract_attention(chain::String, r::Room, f::Furniture)

    map_step, weights, cycles = load_map_chain(chain)

    query = query_from_params(r,
                              "/project/scripts/experiments/attention/gm.json";
                              instances = 20,
                              dims = (6,6),
                              img_size = (240, 360))

    trackers = Int64[]

    @unpack dims, n_trackers = params

    # find which trackers correspond to the furniture
    state_ref = CartesianIndices((dims..., n_trackers))
    for t = 1:params.n_trackers
        vs = state_to_room(params, vec(state_ref[:, :, t]))
        if !isempty(intersect(vs, f))
            push!(trackers, t)
        end
    end

    tracker_cycles = mean(cycles[trackers])
    total_cycles = sum(cycles)
    (tracker_cycles, total_cycles)
end


function main(exp::String, render::String)

    df = DataFrame(CSV.File("/scenes/$(exp).csv"))
    new_df = DataFrame(id = Int64[],
                       door = Int64[],
                       furniture = Int64[],
                       move = String[],
                       pixeld = Float64[], # image features
                       lvd = Float64[],
                       ogd = Float64[],
                       ogc = Float64[],  # ideal navigational affordances
                       # model based inferences and attention
                       base_cycles_furniture = Int64[],
                       base_cycles_total = Int64[],
                       move_cycles_furniture = Int64[],
                       move_cycles_total = Int64[],
                       )

    render_base = "/renders/$(exp)_$(render)"
    for r in eachrow(df)

        # calculate pixel distance
        base = "$(render_base)/$(r.id)_$(r.door).png"
        img = "$(render_base)/$(r.id)_$(r.door)_$(r.furniture)_$(r.move).png"
        pixeld = compare_pixels(base, img)

        # simulated model covariates

        base_p = "/scenes/$(exp)/$(r.id).jld2"

        base = load(base_p)["rs"][r.door]
        f = furniture(base)[r.furniture]
        move = Symbol(r.move)
        room = shift_furniture(base, r.furniture, move)
        shifted = @>> f collect map(v -> shift_tile(base, v, move)) collect Set

        lvd = lv_distance(base, room)
        ogr, ogd = compare_og(base, room)

        base_chain = "/experiments/$(exp)_attention/$(r.id)/1.jld2"
        move_chain = "/experiments/$(exp)_attention/$(r.id)_furniture_$(r.move)/1.jld2"
        # pred = compare_model_predictions(base, base_chain, move_chain,
        #                                  r.furniture, r.move)
        base_att = extract_attention(base_chain, r.furniture)
        move_att = extract_attention(move_chain, r.furniture)
        row = (r.id, r.door, r.furniture, r.move,
               pixeld, lvd, ogd, ogr,
               base_att..., move_att...)
        push!(new_df, row)
    end
    display(new_df)
    isdir("/experiments/$(exp)") || mkdir("/experiments/$(exp)")
    CSV.write("/experiments/$(exp)/covariates.csv", new_df)
end


main("1_exit_22x40_doors");
