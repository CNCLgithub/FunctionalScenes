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
    cor(vec(mean(og_a)), vec(mean(og_b)))
    # map(wsd, og_a, og_b) |> sum
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

    current_step = chain["13"]
    n = length(chain)
    weight_history = []
    for i = 14:n
        new_step = chain["$(i)"]
        sens = new_step["aux_state"][:sensitivities]
        push!(weight_history, sens)
        current_step = current_step["log_score"] < new_step["log_score"] ? new_step : current_step
    end
    return current_step, weight_history
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

function compare_rooms(base_p::String, base_chain, move_chain,
                       fid, move)
    @time base = load(base_p)["r"]

    f = furniture(base)[fid]
    move = Symbol(move)
    room = shift_furniture(base, fid, move)
    shifted = @>> f collect map(v -> shift_tile(base, v, move)) collect Set
 
    es = exits(base)
    paths_a = @>> es map(e -> safe_shortest_path(base,e)) map(relative_path)
    paths_b = @>> es map(e -> safe_shortest_path(room,e)) map(relative_path)
    lvd = sum(map(lv_distance, paths_a, paths_b))

    ogd = compare_og(base, room)
    
    og_gt = occupancy_grid(base) 
    viz_ocg(og_gt)
    
    # ab, ba = 0,0
    base_query = query_from_params(base,
                                   "/project/scripts/experiments/attention/gm.json";
                                   instances = 20,
                                   # offset = (0, 0),
                                   img_size = (240, 360),
                                   tile_window = 10.0, # must be high enough due to gt prior
                                   active_bias = 10.0, # must be high enough due to gt prior
                                   default_tracker_p = 1.0
                              )
    move_query = query_from_params(room,
                                   "/project/scripts/experiments/attention/gm.json";
                                   instances = 20,
                                   # offset = (0,0),
                                   img_size = (240, 360),
                                   tile_window = 10.0, # must be high enough due to gt prior
                                   active_bias = 10.0, # must be high enough due to gt prior
                                   default_tracker_p = 1.0
                              )
    params = first(move_query.args)

    base_steps, base_weights = sample_steps(base_chain, 1)
    move_steps, move_weights = sample_steps(move_chain, 1)

    bparams = first(base_query.args) 

    base_ids = room_to_tracker(first(base_query.args), f)
    move_ids = room_to_tracker(first(move_query.args), shifted)
    joined_ids = union(base_ids, move_ids)
    # joined_ids = collect(1:length(base_weights))    

    base_target_att = sum(base_weights[base_ids])
    move_target_att = sum(move_weights[move_ids])
   
    base_total_att = sum(base_weights)
    move_total_att = sum(move_weights)

    f = (a,b) -> cross_predict(move_query,
                       a["estimates"][:trace],
                       b["estimates"][:trace],
                       joined_ids) - b["log_score"]
    g = (a,b) -> cross_predict(base_query,
                       a["estimates"][:trace],
                       b["estimates"][:trace],
                       joined_ids) - b["log_score"]

    ab = @>> map(f, base_steps, move_steps) mean
    ba = @>> map(g, move_steps, base_steps) mean

    base_og = @>> base_steps map(s -> og_from_step(base_query, s)) mean
    move_og = @>> move_steps map(s -> og_from_step(move_query, s)) mean
    mogd = batch_compare_og(base_og, move_og)


    (lvd, ogd, ab, ba, mogd, base_target_att, move_target_att,
     base_total_att, move_total_att)
end


function main(exp::String)

    df = DataFrame(CSV.File("/scenes/$(exp).csv"))
    new_df = DataFrame(id = Int64[], furniture = Int64[],
                       move = String[], pixeld = Float64[],
                       lvd = Float64[], ogd = Float64[],
                       base_sense = Float64[],
                       move_sense = Float64[],
                       mogd = Float64[],
                       base_target_att = Float64[],
                       move_target_att = Float64[], 
                       base_total_att = Float64[],
                       move_total_att = Float64[])

    # df = df[df.id .<= 1, :]
    for r in eachrow(df)
        base = "/renders/$(exp)/$(r.id).png"
        img = "/renders/$(exp)/$(r.id)_$(r.furniture)_$(r.move).png"
	pixeld = compare_pixels(base, img)

        base = "/scenes/$(exp)/$(r.id).jld2"

        base_chain = "/experiments/$(exp)_attention_150/$(r.id)/1.jld2"
        move_chain = "/experiments/$(exp)_attention_150/$(r.id)_furniture_$(r.move)/1.jld2"
        # base_chain = "/experiments/$(exp)_attention/$(r.id)/1.jld2"
        # move_chain = "/experiments/$(exp)_attention/$(r.id)_furniture_$(r.move)/1.jld2"
        result = compare_rooms(base, base_chain, move_chain,
                                 r.furniture, r.move)

        push!(new_df, (r.id, r.furniture, r.move, pixeld, result...))
    end
    display(new_df)
    isdir("/experiments/$(exp)") || mkdir("/experiments/$(exp)")
    CSV.write("/experiments/$(exp)/covariates.csv", new_df)
end


main("2e_1p_30s_matchedc3");
