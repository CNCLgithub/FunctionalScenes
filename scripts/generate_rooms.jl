using CSV
using Lazy
using JLD2
using Statistics
using LinearAlgebra: norm
using FunctionalScenes

using DataFrames

import Random:shuffle

import FunctionalScenes: expand, furniture, valid_moves,
    shift_furniture, move_map, labelled_categorical,
    translate, k_shortest_paths, entrance, exits, wsd

import Gen:categorical

using Profile
using StatProfilerHTML


function change_in_path(x)
    minimum(x.d) == 0. && maximum(x.d) > 10
    # any(x.d .< 0.02) && any(x.d .> 0.05)
end


function digest(df::DataFrame, base)
    @>> DataFrames.groupby(df, :furniture) begin
        # at least one valid furniture with two directions
        filter(g -> nrow(g) >= 2)
        # pick the first two directions for a given furniture
        map(g -> g[1:2, :])
        filter(change_in_path)
        x -> isempty(x) ? x : labelled_categorical(x)
        DataFrame
    end
end

function search(r::Room, move_set)
    # avoid furniture close to camera
    fs = furniture(r)[4:end]
    data = DataFrame()
    base_og = occupancy_grid(r; sigma = 0., decay = 0.)
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        moves = intersect(moves, move_set)
        for m in moves
            shifted = shift_furniture(r,f,m)
            shifted_og = occupancy_grid(shifted; sigma = 0., decay = 0.)
            # d = wsd(base_og, shifted_og)
            d = norm(base_og - shifted_og)
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
               k::Int64 = 6, factor::Int64 = 1,
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
    moves = [:up, :down]
    # max move per door
    max_move = Int64(ceil(n / length(moves)))
    for (idx, door) in enumerate(doors)
        r = Room(room_dims, room_dims, [entrance],
                     [inds[door, room_dims[2]]])
        println("starting room")
        display(r)
        i = 1
        move_counts = Dict{Symbol, Int64}(zip(moves, zeros(length(moves))))
        while i <= n
            (seed, _df) = build(r, factor = 2,
                                moves = collect(Symbol, keys(move_counts)))
            if !isempty(_df)
                move = filter(:d => d -> d > 0, _df)
                move = first(move[!, :move])
                if move_counts[move] >= max_move
                    println("limit reached for $(move)")
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

                # visualize the accepted room pair
                println("seed")
                display((seed, FunctionalScenes.all_shortest_paths(seed)))
                og_a = occupancy_grid(seed, sigma = 0.)
                for row in eachrow(_df)
                    shifted = shift_furniture(seed,
                                            row.furniture,
                                            row.move)
                    @show row
                    display((shifted, FunctionalScenes.all_shortest_paths(shifted)))
                    # og_b = occupancy_grid(shifted, sigma = 0.)
                    # FunctionalScenes.viz_ocg(og_a - og_b)
                end
            end

        end

    end
    return seeds, df
end


function main()
    name = "1_exit_22x40_ud"
    room_dims = (11,20)
    entrance = 6
    doors = [3, 5, 7, 9]
    n = 8
    seeds, df = create(room_dims, entrance, doors; n = n)
    out = "/scenes/$(name)"
    CSV.write("$(out).csv", df)
    isdir(out) || mkdir(out)

    # save base rooms
    @>> seeds enumerate foreach(x -> saver(x..., out))



    return seeds, df
end

main();
