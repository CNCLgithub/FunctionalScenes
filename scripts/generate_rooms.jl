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


function examine_moves(r::Room, move_set::Vector{Symbol})
    # avoid furniture close to camera
    fs = furniture(r)
    data = DataFrame()
    base_og = occupancy_grid(r; sigma = 0., decay = 0.)
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        moves = intersect(moves, move_set)
        for m in moves
            # only consider strongly connected moves
            connected = strongly_connected(r, f, m)
            isempty(connected) && continue

            shifted = shift_furniture(r,f,m)
            shifted_og = occupancy_grid(shifted; sigma = 0., decay = 0.)
            d = norm(base_og - shifted_og)
            append!(data,
                    DataFrame(furniture = i,
                              move = m,
                              d = d,
                              room = shifted))
        end
    end
    data
end

function change_in_path(x)
    min_i = argmin(x[:, :d])
    c1 = x[min_i, :d] == 0.0 && x[min_i, :door] == 1

    max_i = argmax(x[:, :d])
    c2 = x[max_i, :d] > 10.0 && x[max_i, :door] == 2

    c1 && c2
end

function digest(df::DataFrame)::DataFrame
    df = @>> DataFrames.groupby(df, [:furniture, :move]) begin
        # look find change for one but not another door
        filter(change_in_path)
        # get the largest and smallest (0) change
        map(g -> sort(g, :d)[[1,end], :])
    end
    isempty(df) && return DataFrame()

    df = vcat(df...)
end

function build(rooms::Vector{Room};
               k::Int64 = 16, factor::Int64 = 1,
               pct_open::Float64 = 0.5,
               moves::Vector{Symbol} = move_map)
    # assuming all rooms have the same entrance and dimensions
    r = first(rooms)
    r = expand(r, factor)
    weights = zeros(steps(r))
    # ensures that there is no furniture near the observer
    start_x = Int64(last(steps(r)) * pct_open)
    stop_x = last(steps(r)) - 4 # nor blocking the exit
    start_y = 2
    stop_y = first(steps(r)) - 1
    weights[start_y:stop_y, start_x:stop_x] .= 1.0

    new_rooms = Vector{Room}(undef, length(rooms))
    new_r = last(furniture_chain(k, r, weights))
    results = DataFrame()
    for (i, ri) in enumerate(rooms)
        new_r = i > 1 ? add(new_r, expand(ri, factor)) : new_r
        # expanded = expand(new_r, factor)
        # _df = examine_moves(expanded, moves)
        _df = examine_moves(new_r, moves)
        isempty(_df) && continue
        _df[!, :door] .= i
        append!(results, _df)
        # new_rooms[i] = expanded
        new_rooms[i] = new_r
    end
    results = isempty(results) ? results : digest(results)
    (new_rooms, results)
end


function create(room_dims::Tuple{Int64, Int64},
                entrance::Int64,
                doors::Vector{Int64};
                n::Int64 = 8)
    df = DataFrame()
    moves = [:down, :up]
    max_count = Int64(ceil(n / length(moves)))

    println("starting room")
    r = Room(room_dims, room_dims, [entrance], Int64[])
    display(r)

    templates = @>> doors begin
        map(d -> Room(room_dims, room_dims, [entrance], [d]))
        collect(Room)
    end
    seeds = Matrix{Vector{Room}}(undef, max_count, length(moves))
    ids = LinearIndices(seeds)
    for (idx, move) in enumerate(moves)
        i = 1
        while i <= max_count
            (seed, _df) = build(templates, factor = 2, moves = [move])
            isempty(_df) && continue

            grps = DataFrames.groupby(_df, [:furniture, :move])
            selected_i = @>> grps begin
                gs -> combine(gs, :d => maximum)
                gs -> sortperm(gs, :d_maximum)
                last
            end
            selected = @>> grps[selected_i] DataFrame
            seeds[i, idx] = seed
            selected[!, :id] .= ids[i, idx]
            selected[!, :flip] .= i % 2
            @show selected
            append!(df, selected)
            i += 1
            println("$(i)/$(n)")

            # visualize the accepted room pair
            # println("seed")
            # display((seed, FunctionalScenes.all_shortest_paths(seed)))
            # og_a = occupancy_grid(seed, sigma = 0.)
            # for row in eachrow(_df)
            #     shifted = shift_furniture(seed,
            #                             row.furniture,
            #                             row.move)
            #     @show row
            #     display((shifted, FunctionalScenes.all_shortest_paths(shifted)))
            #     # og_b = occupancy_grid(shifted, sigma = 0.)
            #     # FunctionalScenes.viz_ocg(og_a - og_b)
            # end
        end
    end
    @show df
    return seeds, df
end

# function saver(id::Int64, r::Room, out::String)
function saver(id::Int64, rs::Vector{Room}, out::String)
    @save "$(out)/$(id).jld2" rs
end


function main()
    name = "1_exit_22x40_doors"
    room_dims = (11,20)
    entrance = 6
    inds = LinearIndices(room_dims)
    doors = [3, 9]
    doors = @>> doors map(d -> inds[d, room_dims[2]]) collect(Int64)
    n = 32
    seeds, df = create(room_dims, entrance, doors; n = n)
    out = "/scenes/$(name)"
    CSV.write("$(out).csv", df)
    isdir(out) || mkdir(out)

    # save base rooms
    @>> seeds enumerate foreach(x -> saver(x..., out))

    return seeds, df
end

main();
