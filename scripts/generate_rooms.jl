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
    fs = furniture(r) # [4:end]
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
            append!(data,
                    DataFrame(furniture = i,
                              move = m,
                              d = d,
                              room = shifted))
        end
    end
    data
    # result = @> data sort([:furniture, :d]) digest(r)
    # isempty(result) && return  DataFrame()
    # select(result, [:furniture, :move, :d])
end

function change_in_path(x)
    nrow(x) >= 2 && minimum(x.d) == 0. && maximum(x.d) > 10
    # any(x.d .< 0.02) && any(x.d .> 0.05)
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
               k::Int64 = 6, factor::Int64 = 1,
               pct_open::Float64 = 0.4,
               moves::Vector{Symbol} = move_map)
    # assuming all rooms have the same entrance and dimensions
    r = first(rooms)
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
        new_r = i > 1 ? add(new_r, ri) : new_r
        expanded = expand(new_r, factor)
        _df = examine_moves(expanded, moves)
        _df[!, :door] .= i
        append!(results, _df)
        new_rooms[i] = expanded
    end
    results = digest(results)
    @show results
    (new_rooms, results)
end


function create(room_dims::Tuple{Int64, Int64},
                entrance::Int64,
                doors::Vector{Int64};
                n::Int64 = 8)
    df = DataFrame()
    moves = [:up, :down]
    # max move per door
    max_counts = n # Int64(ceil(n / length(doors)))
    r = Room(room_dims, room_dims, [entrance], Int64[])
    println("starting room")
    display(r)
    i = 1
    chng_counts = zeros(length(doors))
    noch_counts = zeros(length(doors))

    templates = @>> doors begin
        map(d -> Room(room_dims, room_dims, [entrance], [d]))
        collect(Room)
    end
    seeds = Vector{Vector{Room}}(undef, n)
    while i <= n
        (seed, _df) = build(templates, factor = 2, moves = moves)
        @show chng_counts
        @show noch_counts
        isempty(_df) && continue

        grps = DataFrames.groupby(_df, [:furniture, :move])
        selected_i = @>> grps begin
            filter(g -> (noch_counts[first(g[:, :door])] < max_counts &&
                         chng_counts[last(g[:, :door])] < max_counts))
        end
        isempty(selected_i) && continue
        selected_i = @>> selected_i begin
            grps -> combine(grps, :d => maximum)
            grps -> sortperm(grps, :d_maximum)
            last
        end
        selected = @>> grps[selected_i] DataFrame
        noch_door, chng_door = selected[!, :door]
        chng_counts[chng_door] += 1
        noch_counts[noch_door] += 1
        seeds[i] = seed
        selected[!, :id] .= i
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
    doors = [4, 8]
    doors = @>> doors map(d -> inds[d, room_dims[2]]) collect(Int64)
    n = 64
    seeds, df = create(room_dims, entrance, doors; n = n)
    out = "/scenes/$(name)"
    CSV.write("$(out).csv", df)
    isdir(out) || mkdir(out)

    # save base rooms
    @>> seeds enumerate foreach(x -> saver(x..., out))



    return seeds, df
end

main();
