using CSV
using JSON
using Dates
using DataFrames
using StaticArrays
using SparseArrays
using LinearAlgebra
using Gen
using Graphs
using FunctionalScenes
using FunctionalCollections

function room_to_template(r::GridRoom;
                          gap::Int64 = 2)
    d = data(r)
    occupied = d.== obstacle_tile
    y, x = size(d) # room dimensions (e.g., 16x16)
    # hyper tile dimensions (8x8)
    r, c = Int64(y / 2), Int64(x / 2)
    result = Matrix{Int64}(undef, r, c)
    for ir = 1:r, ic = 1:c
        row_start = (ir - 1) * 2 + 1
        row_end = ir * 2
        col_start = (ic - 1) * 2 + 1
        col_end = ic * 2
        cell = occupied[row_start:row_end, col_start:col_end]
        result[ir, ic] = htmap[cell]
    end
    # gap_rng = 2:(r-gap-1)
    # result[gap_rng, 2:(end-1)] .= 0
    return sparse(result)
end

function empty_room(steps, bounds, entrance, exits)
    r = GridRoom(steps, bounds, entrance, exits)
    m = data(r)
    m[minimum(entrance) - 1 : maximum(entrance) + 1] .= floor_tile
    GridRoom(r, m)
end

function dpath(r::GridRoom)
    dpath(pathgraph(r), entrance(r), exits(r))
end
function dpath(g::SimpleGraph, ents, exts)
    state = dijkstra_shortest_paths(g, ents)
    paths = enumerate_paths(state, exts)
    reduce(union, paths)
end

# borrowed from https://github.com/BorisTheBrave/chiseled-random-paths
function chisel_path(r::GridRoom, w::Float64,
                     visited = Matrix{Bool}(data(r) .!= floor_tile))
    visited[entrance(r)] .= true
    visited[exits(r)] .= true
    g = init_pathgraph(r.data)
    path_tiles = fill(false, size(visited))
    weights = zeros(size(visited))
    normed_weights = zeros(length(visited))

    @inbounds for i = eachindex(weights)
        weights[i] = visited[i] ? -Inf : 0.0
    end

    # shortest paths
    paths = dpath(g, entrance(r), exits(r))
    @inbounds for i = paths
        path_tiles[i] = true
        weights[i] = w
    end

    while !all(visited)
        # pick a random tile
        softmax!(normed_weights, weights)
        tile = categorical(normed_weights)
        # block tile
        ns = collect(neighbors(g, tile))
        for n = ns
            rem_edge!(g, tile, n)
        end

        if path_tiles[tile]
            # new paths
            new_paths = dpath(g, entrance(r), exits(r))
            if isempty(new_paths)
                # blocked all paths - undo
                for n = ns
                    add_edge!(g, tile, n)
                end
            else
                # remove old path
                @inbounds for i = paths
                    path_tiles[i] = false
                    weights[i] = 0.0
                end

                # accept new path
                @inbounds for i = new_paths
                    path_tiles[i] = true
                    weights[i] = w
                end
                paths = new_paths
            end
        end

        # mark tile as visited
        visited[tile] = true
        weights[tile] = -Inf
    end
    path_tiles
end

function gen_obstacle_weights(template::GridRoom,
                              path;
                              pct_open::Float64 = 0.15,
                              side_buffer::Int64 = 1)
    dims = steps(template)
    # prevent furniture generated in either:
    # -1 out of sight
    # -2 blocking entrance exit
    # -3 hard to detect spaces next to walls
    weights = Matrix{Bool}(zeros(dims))
    # ensures that there is no furniture near the observer
    start_x = 5 # Int64(ceil(last(dims) * pct_open))
    stop_x = last(dims) - 4 # nor blocking the exit
    # buffer along sides
    start_y = side_buffer + 1
    stop_y = first(dims) - side_buffer
    weights[start_y:stop_y, start_x:stop_x] .= 1.0
    weights[path] .= 0.0
    # vectorize for `furniture_gm`
    PersistentVector(vec(weights))
end

function neighbor_count(r::GridRoom)
    neighbor_count(pathgraph(r))
end
function neighbor_count(g::SimpleGraph)
    m = zeros(16,16)
    @inbounds for i = 1:length(m)
        m[i] = length(neighbors(g, i))
    end
    return m
end

function sample_pair(left_room::GridRoom,
                     right_room::GridRoom,
                     chisel_temp::Float64 = -1.0,
                     fix_steps::Int64 = 10,
                     extra_pieces::Int64 = 10,
                     piece_size::Int64 = 4)

    # sample some obstacles in the middle section of the roo
    oweights = gen_obstacle_weights(right_room, Int64[])
    right_room = furniture_gm(right_room, oweights,
                              extra_pieces, piece_size)
    left_room  = add(left_room, right_room)
    # viz_room(right_room)

    # sample a random path for right room
    path_tiles = chisel_path(right_room, chisel_temp)
    right_path = findall(vec(path_tiles))
    # add some obstacles to coerce this path
    fixed = fix_shortest_path(right_room, right_path, fix_steps)
    right_room = add(right_room, fixed)
    right_path = dpath(right_room)

    # add obstacles to left room
    left_room = add(left_room, fixed)
    # left path accounting for right path
    visited = Matrix{Bool}(data(left_room) .!= floor_tile)
    visited[right_path] .= true
    path_tiles = chisel_path(left_room, chisel_temp, visited)
    left_path = findall(vec(path_tiles))
    fixed = fix_shortest_path(left_room, left_path, fix_steps)
    left_room = add(left_room, fixed)
    left_path = dpath(left_room)

    # add left chisels to right door
    right_room = add(right_room, fixed)

    # add more obstacles, avoiding both paths
    # oweights = gen_obstacle_weights(right_room, union(left_path,
    #                                                   right_path))
    # right_room = furniture_gm(right_room, oweights,
    #                           extra_pieces, piece_size)
    # left_room = add(left_room, right_room)

    # evaulate the paths one last time
    right_path = dpath(right_room)
    # left_path = dpath(left_room)


    (left_room, left_path, right_room, right_path)
end

function eval_pair(left_door::GridRoom,
                   left_path::Vector{Int64},
                   right_door::GridRoom,
                   right_path::Vector{Int64}
                   )

    tile = 0
    # Not a valid sample
    (isempty(left_path) || isempty(right_path)) && return tile

    # Where to place obstacle that blocks the right path
    nl = length(right_path)
    # avoid blocking entrance or exit
    trng = 3:(nl-5)
    g = pathgraph(right_door)
    gt = deepcopy(g)
    ens = entrance(right_door)
    ext = exits(right_door)
    left_path = Int64[]
    # look for first tile that blocks the path
    @inbounds for i = trng
        tid = right_path[i]
        # block tile
        ns = collect(neighbors(g, tid))
        for n = ns
            rem_edge!(gt, tid, n)
        end
        # does it block path ?
        new_path = dpath(gt, ens, ext)
        if isempty(new_path)
            tile = tid
            left_temp = add(left_door, Set{Int64}(tile))
            left_path = dpath(pathgraph(left_temp),
                              entrance(left_temp),
                              exits(left_temp))
            break
        else
            # reset block
            for n = ns
                add_edge!(gt, tid, n)
            end
        end
    end

    tile = isempty(left_path) ? 0 : tile

    return tile
end



function main()
    name = "path_block_$(Dates.today())"
    dataset_out = "/spaths/datasets/$(name)"
    isdir(dataset_out) || mkdir(dataset_out)

    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)


    # Parameters
    room_steps = (16, 16)
    room_bounds = (32., 32.)
    entrance = [8, 9]
    door_rows = [5, 12]
    inds = LinearIndices(room_steps)
    doors = inds[door_rows, room_steps[2]]

    # number of trials
    n = 6

    # empty room with doors
    left_cond = empty_room(room_steps, room_bounds, entrance, [doors[1]])
    right_cond = empty_room(room_steps, room_bounds, entrance, [doors[2]])

    # will store summary of generated rooms here
    df = DataFrame(scene = Int64[],
                   flipx = Bool[],
                   tidx = Int64[])

    i = 1 # scene id
    c = 0 # number of attempts;
    while i <= n && c < 100 * n
        # generate a room pair
        (left, lpath, right, rpath) = sample_pair(left_cond, right_cond)
        tile = eval_pair(left, lpath, right, rpath)
        # no valid pair generated, try again or finish
        c += 1
        tile == 0 && continue

        println("accepted pair!")
        viz_room(right, rpath)
        viz_room(left, lpath)
        viz_room(add(right, Set{Int64}(tile)))

        # save
        toflip = (i-1) % 2
        push!(df, [i, toflip, tile])
        open("$(scenes_out)/$(i)_1.json", "w") do f
            write(f, left |> json)
        end
        open("$(scenes_out)/$(i)_2.json", "w") do f
            write(f, right |> json)
        end

        print("scene $(i)/$(n)\r")
        i += 1
    end
    @show df
    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)
    return nothing
end

main();
