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



function build(rooms::Vector{Room};
               k::Int64 = 16, factor::Int64 = 1,
               pct_open::Float64 = 0.5)
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
end


function create(room_dims::Tuple{Int64, Int64},
                entrance::Int64,
                doors::Vector{Int64};
                n::Int64 = 8)
    df = DataFrame()

    r = Room(room_dims, room_dims, [entrance], Int64[])
    display(r)

    templates = @>> doors begin
        map(d -> Room(room_dims, room_dims, [entrance], [d]))
        collect(Room)
    end
    i = 0
    while i < n
        x = build(templates, factor = 2, moves = [move])
    end
    return seeds, df
end

# function saver(id::Int64, r::Room, out::String)
function saver(id::Int64, rs::Vector{Room}, out::String)
    @save "$(out)/$(id).jld2" rs
end


function main()
    name = "train_ddp_1_exit_22x40_doors"
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
