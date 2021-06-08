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

function predicate(x)
    any(iszero.(x.d)) && any((!iszero).(x.d))
end


function digest(df::DataFrame, base)
    base_d = translate(base, false)
    graphics.set_from_scene(base_d)
    base_img = functional_scenes.render_scene_pil(base_d, graphics)
    @>> DataFrames.groupby(df, :furniture) begin
        filter(g -> nrow(g) >= 2)
        map(g -> g[1:2, :])
        filter(predicate)
        filter(g -> all(
                        map(x -> feat_pred(base_img, x), 
                        g.room)))
        x -> isempty(x) ? x : labelled_categorical(x)
        DataFrame
    end
end

function search(r::Room)
    # avoid furniture close to camera
    fs = furniture(r)[4:end]
    data = DataFrame()
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        # moves = intersect(moves, [:up, :down])
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

# TODO: fill in the blanks 
function build(r::Room; k = 12, factor = 1)
    weights = zeros(steps(r))
    # ensures that there is no furniture near the observer
    start_x = Int(last(steps(r)) * 0.4)
    stop_x = last(steps(r)) - 2
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
    dist = search(new_r)
    (new_r, dist)
end

function saver(id::Int64, r::Room, out::String)
    @save "$(out)/$(id).jld2" r
end


function create(base::Room; n::Int64 = 15)
    seeds = Vector{Room}(undef, n)
    df = DataFrame()
    i = 1
    max_move = Int64(ceil(n / 4.0))
    move_counts = Dict{Symbol, Int64}(zip([:up, :down, :left, :right],
                                          zeros(4)))
    while i <= n
        @time seed, _df = build(base, factor = 2)
        if !isempty(_df)
            move = filter(:d => d -> d > 0, _df)
            @show move
            move = first(move[:move])
            if move_counts[move] >= max_move 
                continue
            end
            move_counts[move] += 1
            seeds[i] = seed
            _df[!, :id] .= i
            append!(df, _df)
            i += 1
            println("$(i)/$(n)")
        end
    end
    return seeds, df
end


function main()
    name = "pytorch_rep"
    name = "2e_1p_30s_matchedc3"
    n = 30
    room_dims = (11,20)
    entrance = [6]
    exits = [215]
    #exits = [213,217]
    r = Room(room_dims, room_dims, entrance, exits)
    display(r)
    @time seeds, df = create(r, n = n)
    out = "/scenes/$(name)"
    CSV.write("$(out).csv", df)
    isdir(out) || mkdir(out)
    @>> seeds enumerate foreach(x -> saver(x..., out))
    return seeds, df
end

main();
