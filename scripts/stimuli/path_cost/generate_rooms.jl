using CSV
using Gen
using JSON
using DataFrames
using Statistics
using FunctionalScenes
using LinearAlgebra: norm
using FunctionalCollections

# using Profile
# using StatProfilerHTML

function valid_pair(pair::Tuple, m::Move;
                    min_pc_thresh::Float64 = 5.0,
                    max_pc_thresh::Float64 = 0.32,
                    max_pc_diff_thresh::Float64 = 20.0,
                    min_pd_diff_thresh::Float64 = 3.0
                    )::Bool

    @show pair[4:end]
    (d1, d2, f, pc1, pc2, pd1, pd2) = pair
    mu_pc = 0.5 * (pc1 + pc2)
    f > 0 &&
        mu_pc > min_pc_thresh &&
        mu_pc < max_pc_thresh &&
        abs(pc1 - pc2) < max_pc_diff_thresh &&
        abs(pd1 - pd2) > min_pd_diff_thresh
end

function gen_obstacle_weights(template::GridRoom;
                              pct_open::Float64 = 0.3,
                              side_buffer::Int64 = 1)
    dims = steps(template)
    # prevent furniture generated in either:
    # -1 out of sight
    # -2 blocking entrance exit
    # -3 hard to detect spaces next to walls
    weights = Matrix{Bool}(zeros(dims))
    # ensures that there is no furniture near the observer
    start_x = Int64(ceil(last(dims) * pct_open))
    stop_x = last(dims) - 2 # nor blocking the exit
    # buffer along sides
    start_y = side_buffer + 1
    stop_y = first(dims) - side_buffer
    weights[start_y:stop_y, start_x:stop_x] .= 1.0
    # vectorize for `furniture_gm`
    PersistentVector(vec(weights))
end

function average_tile_position(r::GridRoom, tiles)
    ny, _ = steps(r)
    pos = zeros(2)
    for t in tiles
        pos[1] += ceil(t / ny) / ny - 0.5
        pos[2] += (t % ny) / ny - 0.5
    end
    pos .*= 1/length(tiles)
    return pos
end

function select_obstacle(r::GridRoom, m::Move;
                         xbuffer::Float64 = 0.0,
                         ybuffer::Float64 = 0.3)
    fs = furniture(r)
    nf = length(fs)
    raw_weights = zeros(nf)
    @inbounds for i = 1:nf
        f = fs[i]
        valid_move(r, f, m) || continue
        avg_pos = average_tile_position(r, f)
        raw_weights[i] = 1.0 * (
            (avg_pos[1] > xbuffer) &&
            (abs(avg_pos[2]) < ybuffer))
    end
    srw = sum(raw_weights)
    srw == 0 && return (fs, 0)
    ws = raw_weights ./ srw
    fs, categorical(ws)
end

function create_pair(door_conditions::Vector{GridRoom},
                     move::Move,
                     obstacle_weights::PersistentVector,
                     path_params::NoisyPath;
                     samples::Int64 = 100,
                     max_f::Int64 = 8,
                     max_size::Int64 = 4,
                     factor::Int64 = 2)
    # generate furniture once and then apply to
    # each door condition
    template = door_conditions[1]
    with_furn =
        furniture_gm(template, obstacle_weights, max_f, max_size)

    # add the furniture and expand from 16x16 -> 32x32
    d1 = expand(with_furn, factor) # door1 already added
    d2 = expand(add(with_furn, door_conditions[2]), factor)

    # pick a random obstacle to move
    f, fi = select_obstacle(d1, move)
    # failed to select
    fi == 0 && return (d1, d2, fi, 0, 0, 0, 0)
    # measure the differences in path complexity between the two doors
    pc_1, p1 = noisy_path(d1, path_params; samples = samples)
    pd_1 = FunctionalScenes.min_distance_to_path(f[fi], p1)
    pc_2, p2 = noisy_path(d2, path_params; samples = samples)
    pd_2 = FunctionalScenes.min_distance_to_path(f[fi], p2)

    (d1, d2, fi, pc_1, pc_2, pd_1, pd_2)
end

function update_df!(df::DataFrame, id::Int64, move::Move,
                    flipx::Bool, pair::Tuple)
    d1, d2, furn_selected, pc_1, pc_2, pd_1, pd_2 = pair
    push!(df, (id, flipx, 1, furn_selected, move, pc_1, pd_1))
    push!(df, (id, flipx, 2, furn_selected, move, pc_2, pd_2))
    return nothing
end

function save_pair(out::String, id::Int64, pair::Tuple)
    d1, d2, _... = pair
    display(d1)
    open("$(out)/$(id)_1.json", "w") do f
        _d = d1 |> json
        write(f, _d)
    end
    open("$(out)/$(id)_2.json", "w") do f
        _d = d2 |> json
        write(f, _d)
    end
    return nothing
end

function main()
    # dataset name
    name = "pathcost_6.0"

    dataset_out = "/spaths/datasets/$(name)"
    isdir(dataset_out) || mkdir(dataset_out)
    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)

    # Parameters
    room_dims = (16, 16)
    entrance = [8, 9]
    door_rows = [5, 12]
    inds = LinearIndices(room_dims)
    doors = inds[door_rows, room_dims[2]]

    # will only consider these moves
    moves = [down_move, up_move]

    # Path planning parameters
    # - 05/30/23 based on 5.2.0 results
    path_params = NoisyPath(
        obstacle_cost = 3.2,
        floor_cost = 0.01,
        kernel_width = 7,
        kernel_beta = 2.0,
        kernel_sigma = 5.0,
        wall_cost = 32.0,
    )

    # Acceptance criteria, see `valid_pair`
    criteria = Dict(
        :min_pc_thresh => 76.0,
        :max_pc_thresh => 86.0,
        :max_pc_diff_thresh => 7.5,
        :min_pd_diff_thresh => 1.5,
    )

    # number of trials
    n = 20

    # number of trials per move condition
    max_per_move = Int64(ceil(n / length(moves)))

    # empty rooms with doors
    t1 = GridRoom(room_dims, room_dims, entrance, [doors[1]])
    t2 = GridRoom(room_dims, room_dims, entrance, [doors[2]])
    templates = [t1, t2]

    obstacle_weights = gen_obstacle_weights(templates[1])

    # will store summary of generated rooms here
    df = DataFrame(scene = Int64[],
                   flipx = Bool[],
                   door = Int64[],
                   furniture = Int64[],
                   move = Symbol[],
                   path_cost = Float64[],
                   path_dist = Float64[])
    for (idx, move) in enumerate(moves)
        i = 1
        while i <= max_per_move
            # generate a room pair
            pair = create_pair(templates, move, obstacle_weights,
                               path_params; samples = 250)

            # no valid pair generated, try again or finish
            !valid_pair(pair, move; criteria...) && continue

            # organize and save
            id = (idx - 1) * max_per_move + i
            pd1, pd2 = pair[end-1:end]
            # balance most complex door along L,R
            # pc1 < pc2 ; odd   ; result
            # T         ; T     ; T
            # T         ; F     ; F
            # F         ; T     ; F
            # F         ; F     ; T
            flipx = !xor(pd1 < pd2, Bool((i-1) % 2))
            update_df!(df, id, move, flipx, pair)

            # save scenes as json
            save_pair(scenes_out, id, pair)

            print("move $(idx): $(i)/$(max_per_move)\r")
            i += 1
        end
        println()
    end
    @show df
    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)
    return nothing
end

main();
