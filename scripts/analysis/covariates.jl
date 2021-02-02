using CSV
using Gen
using Lazy
using JLD2
using Images, FileIO
using DataFrames
using LinearAlgebra:norm
using Statistics: mean, std

import FunctionalScenes: Room, furniture, shift_furniture, navigability, wsd,
    occupancy_grid, diffuse_og, safe_shortest_path, shift_tile, room_to_tracker,
    viz_ocg, model, select_from_model

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
    println("og a")
    display(a)
    og_a = occupancy_grid(a, decay = 0.0001, sigma = 0.7)
    # viz_ocg(og_a)
    println("og b")
    display(b)
    og_b = occupancy_grid(b, decay = 0.0001, sigma = 0.7)
    # viz_ocg(og_b)
    # wsd(og_a, og_b)
    # norm(og_a .- og_b)
    map(wsd, og_a, og_b) |> sum
end


function room_sensitivity(chain, params, fs::Furniture)
    steps = length(keys(chain))
    n = "$(steps)"
    weights = chain["aux_state"][:sensitivities]
    weights = @>> weights values collect(Float64)
    top_trackers = sortperm(weights, rev = true)[1:5]
    tracker_ids = room_to_tracker(params, fs)
    (top_trackers, mean(weights[tracker_ids]))
    # mean(zs[tracker_ids])
end


function cross_predict(query, choices_a, choices_b, trackers)
    params = first(query.args)

    from_a = choicemap()
    # trackers = trackers[1:1]
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
    # open("/project/output/a.txt", "w") do io
    #     Base.show(io, "text/plain", choices_a)
    # end
    # open("/project/output/b.txt", "w") do io
    #     Base.show(io, "text/plain", choices_b)
    # end

    constraints = merge(from_a, from_b)

    # open("/project/output/c.txt", "w") do io
    #     Base.show(io, "text/plain", constraints)
    # end

    constraints[:viz] = query.observations[:viz]
    _, ls = generate(model, query.args, constraints)
    # println(ls)
    # @assert trackers != [5, 8, 11]
    return ls
end

function compare_rooms(base_p::String, base_chain, move_chain,
                       fid, move)
    @time base = load(base_p)["r"]

    f = furniture(base)[fid]
    move = Symbol(move)
    room = shift_furniture(base, fid, move)
    es = exits(base)
    paths_a = @>> es map(e -> safe_shortest_path(base,e)) map(relative_path)
    paths_b = @>> es map(e -> safe_shortest_path(room,e)) map(relative_path)
    lvd = sum(map(lv_distance, paths_a, paths_b))

    ogd = compare_og(base, room)
    display(ogd)

    ab, ba = 0,0
    # base_query = query_from_params(base,
    #                                "/project/scripts/experiments/attention/gm.json";
    #                                img_size = (240, 360),
    #                                tile_window = 2.0, # must be high enough due to gt prior
    #                                active_bias = 10.0, # must be high enough due to gt prior
    #                                base_sigma = 10.0
    #                           )

    # params = first(base_query.args)
    # base_chain = load(base_chain, "50")
    # top_base, base_weights = room_sensitivity(base_chain, params,f)


    # move_chain = load(move_chain, "50")
    # shifted = @>> f collect map(v -> shift_tile(base, v, move)) collect Set
    # top_move, move_weights = room_sensitivity(move_chain, params,
    #                               shifted)

    # move_query = query_from_params(room,
    #                                "/project/scripts/experiments/attention/gm.json";
    #                                img_size = (240, 360),
    #                                tile_window = 2.0, # must be high enough due to gt prior
    #                                active_bias = 10.0, # must be high enough due to gt prior
    #                                base_sigma = 10.0
    #                           )
    # ab = cross_predict(base_query,
    #                    base_chain["estimates"][:trace],
    #                    move_chain["estimates"][:trace],
    #                    top_base)

    # ba = cross_predict(move_query,
    #                    move_chain["estimates"][:trace],
    #                    base_chain["estimates"][:trace],
    #                    top_move)

    # # @assert !isinf(ab)
    # # @assert !isinf(ba)
    # display((ab, ba))
    (lvd, ogd, ab, ba)
    # (lvd, ogd, base_weights, move_weights)
end


function main(exp::String)

    df = DataFrame(CSV.File("/scenes/$(exp).csv"))
    new_df = DataFrame(id = Int64[], furniture = Int64[],
                       move = String[], pixeld = Float64[],
                       lvd = Float64[], ogd = Float64[],
                       base_sense = Float64[],
                       move_sense = Float64[])

    for r in eachrow(df)
        base = "/renders/$(exp)/$(r.id).png"
        img = "/renders/$(exp)/$(r.id)_$(r.furniture)_$(r.move).png"
        pixeld = compare_pixels(base, img)
        # pixeld = 0

        base = "/scenes/$(exp)/$(r.id).jld2"

        base_chain = "/experiments/$(exp)_attention/$(r.id)/1.jld2"
        move_chain = "/experiments/$(exp)_attention/$(r.id)_furniture_$(r.move)/1.jld2"
        result = compare_rooms(base, base_chain, move_chain,
                                 r.furniture, r.move)

        push!(new_df, (r.id, r.furniture, r.move, pixeld, result...))
    end
    isdir("/experiments/$(exp)") || mkdir("/experiments/$(exp)")
    CSV.write("/experiments/$(exp)/covariates.csv", new_df)
end


# main("pilot");
# main("1exit");
main("2e_1p_30s_matchedc3");
