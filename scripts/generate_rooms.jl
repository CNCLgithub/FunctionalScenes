using CSV
using Lazy
using JLD2
using Statistics
using FunctionalScenes

using DataFrames

import Random:shuffle

import FunctionalScenes: expand, furniture, valid_moves,
    shift_furniture, move_map, labelled_categorical,
    translate, k_shortest_paths, entrance, exits

import FunctionalScenes: torch, functional_scenes

import Gen:categorical

using Profile
using StatProfilerHTML


features = Dict(
    "features.6" => "c3",
)

if torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else
    device = torch.device("cpu")
end


model = functional_scenes.init_alexnet("/datasets/alexnet_places365.pth.tar", device)
graphics = functional_scenes.SimpleGraphics((480, 720), device)


function feat_pred(a_img, x)
    b_d = translate(x, false)
    b_img = functional_scenes.render_scene_pil(b_d, graphics)
    feats = functional_scenes.nn_features.compare_features(model, features, a_img, b_img)
    feats["c3"] < 0.97
end

function change_in_path(x)
    any(iszero.(x.d)) && any((!iszero).(x.d))
end


function digest(df::DataFrame, base)
    # base_d = translate(base, false)
    # graphics.set_from_scene(base_d)
    # base_img = functional_scenes.render_scene_pil(base_d, graphics)
    @>> DataFrames.groupby(df, :furniture) begin
        # at least one valid furniture with two directions
        filter(g -> nrow(g) >= 2)
        # pick the first two
        map(g -> g[1:2, :])
        filter(change_in_path)
        # whether the change leads to a large c3 distance
        # filter(g -> all(
        #                 map(x -> feat_pred(base_img, x),
        #                 g.room)))
        x -> isempty(x) ? x : labelled_categorical(x)
        DataFrame
    end
end

function search(r::Room, move_set)
    # avoid furniture close to camera
    fs = furniture(r)[4:end]
    data = DataFrame()
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        moves = intersect(moves, move_set)
        for m in moves
            shifted = shift_furniture(r,f,m)
            d = mean(compare(r, shifted))
            append!(data, DataFrame(furniture = i+3,
                                    move = m,
                                    d = d,
                    room = shifted))
        end
    end
    isempty(data) && return  DataFrame()
    result = @> data sort([:furniture, :d]) digest(r)
    isempty(result) && return  DataFrame()
    select(result, [:furniture, :move, :d])
end

function build(r::Room;
               k::Int64 = 8, factor::Int64 = 1,
               pct_open::Float64 = 0.4,
               moves::Vector{Symbol} = move_map)
    weights = zeros(steps(r))
    # ensures that there is no furniture near the observer
    start_x = Int(last(steps(r)) * pct_open)
    stop_x = last(steps(r)) - 4 # nor blocking the exit
    start_y = 2
    stop_y = first(steps(r)) - 1
    weights[start_y:stop_y, start_x:stop_x] .= 1.0


    #entrance = entrance(r)
    #exits = exits(r)
    paths = k_shortest_paths(r,10,entrance(r)[1],exits(r)[1])
    probs = fill(1.0 / 10, 10)
    index = categorical(probs)
    path = paths[index] # hint you can use Gen.categorical
    weights[path] .= 0.0
    new_r = last(furniture_chain(k, r, weights))
    new_r = FunctionalScenes.expand(new_r, factor)
    dist = search(new_r, moves)
    (new_r, dist)
end

function saver(id::Int64, r::Room, out::String)
    @save "$(out)/$(id).jld2" r
end


function create(room_dims::Tuple{Int64, Int64},
                entrance::Int64,
                doors::Vector{Int64};
                n::Int64 = 8)
    inds = LinearIndices(room_dims)
    seeds = Vector{Room}(undef, n * length(doors))
    df = DataFrame()
    for (idx, door) in enumerate(doors)
	    r = Room(room_dims, room_dims, [entrance], 
                     [inds[door, room_dims[2]]])
	    display(r)
	    i = 1
	    max_move = Int64(ceil(n / 4.0))
	    @show max_move
	    move_counts = Dict{Symbol, Int64}(zip(move_map, zeros(4)))
	    while i <= n
		seed, _df = build(r, factor = 2,
				  moves = collect(Symbol, keys(move_counts)))
		if !isempty(_df)
		    move = filter(:d => d -> d > 0, _df)
		    move = first(move[!, :move])
		    if move_counts[move] >= max_move
			println("limit reached for $(move)")
			length(move_counts) > 2 && pop!(move_counts, move)
			continue
		    end
		    move_counts[move] += 1
		    @show move_counts
                    id = (idx-1)*n + i
		    seeds[id] = seed
		    _df[!, :id] .= id
		    _df[!, :door] .= idx
		    append!(df, _df)
		    i += 1
		    println("$(i)/$(n)")
		end

	    end

    end
    return seeds, df
end


function main()
    name = "1_exit_22x40"
    room_dims = (11,20)
    entrance = 6
    doors = [3, 5, 7, 9]
    n = 8
    seeds, df = create(room_dims, entrance, doors; n = n)
    out = "/scenes/$(name)"
    CSV.write("$(out).csv", df)
    isdir(out) || mkdir(out)
    @>> seeds enumerate foreach(x -> saver(x..., out))
    return seeds, df
end

main();
