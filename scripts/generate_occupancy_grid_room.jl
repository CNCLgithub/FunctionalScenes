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
    result = @> data sort([:furniture, :d]) 
    isempty(result) && return  DataFrame()
    select(result, [:furniture, :move, :d])
end

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
    while i <= n
        @time seed, _df = build(base, factor = 2)
        if !isempty(_df)
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
    #name = "pytorch_rep"
    #name = "2e_1p_30s_matchedc3"
    name = "occupancy_grid_data_driven"
    n = 800
    room_dims = (11,20)
    entrance = [6]
    exits = [215]
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
