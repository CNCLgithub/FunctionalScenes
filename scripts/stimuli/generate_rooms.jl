using CSV
using JSON
using DataFrames
using SparseArrays
using FunctionalScenes
using FunctionalScenes: path_cost
using WaveFunctionCollapse

function prop_weights()
    n = length(HT2D_vec)
    ws = zeros(n, n)
    for (i, hti) = enumerate(HT2D_vec)
        ni = count(hti)
        for (j,  htj) = enumerate(HT2D_vec)
            nj = count(htj)
            ws[i, j] = ((hti[3] == htj[1]) + (hti[4] == htj[2])) *
                    ((1+nj)/(1+ni))
        end
    end

    ws .+= 0.1 # adds noise to prop rules

    # ws = gen_prop_rules()
    ws[1, :] .+= 0.75 # bump empty hyper tile
    ws[2:end, 1] .+= 0.25 #
    ws[end, 1] += 0.25 #
    return ws
end

function room_to_template(r::GridRoom;
                          gap::Int64 = 2)
    d = data(r)
    walls = d .== wall_tile
    y, x = size(d)
    r, c = Int64(y / 2), Int64(x / 2)
    result = Matrix{Int64}(undef, r, c)
    for ir = 1:r, ic = 1:c
        row_start = (ir - 1) * 2 + 1
        row_end = ir * 2
        col_start = (ic - 1) * 2 + 1
        col_end = ic * 2
        cell = walls[row_start:row_end, col_start:col_end]
        result[ir, ic] = WaveFunctionCollapse._ht2_hash_map[cell]
    end
    gap_rng = (gap+1):(r-gap)
    result[gap_rng, gap_rng] .= 0
    return sparse(result)
end

function empty_room(steps, bounds, entrance, exits)
    r = GridRoom(steps, bounds, entrance, exits)
    m = data(r)
    m[minimum(entrance) - 1 : maximum(entrance) + 1] .= floor_tile
    GridRoom(r, m)
end

function sample_obstacles(template::AbstractMatrix{Int64},
                          pr::Matrix{Float64})
    ws = WaveState(template, pr)
    collapse!(ws, pr)
    # empty out original walls
    ws.wave[template .!= 0] .= 1
    expanded = WaveFunctionCollapse.expand(ws)
    Set{Int64}(findall(vec(expanded)))
end

function sample_pair(left_room::GridRoom,
                     right_room::GridRoom,
                     template::AbstractMatrix{Int64},
                     pws::Matrix{Float64})

    obstacles = sample_obstacles(template, pws)

    # generate furniture once and then apply to
    # each door condition
    left = add(left_room, obstacles)
    right = add(right_room, obstacles)

    (left, right)

end

function analyze_path(room::GridRoom, params::PathProcedure)
    path, _, _, dm = path_procedure(room, params)
    path_cost(path, dm)
end

function eval_pair(left_door::GridRoom, right_door::GridRoom,
                   move::Move,
                   path_params::PathProcedure)

    # initial paths
    initial_left_pc = analyze_path(left_door, path_params)
    initial_right_pc = analyze_path(right_door, path_params)

    # jitter each furniture
    # consider as candidate if:
    #   1. move does not change left path
    #   2. move changes right path
    fs = furniture(left_door)
    candidates = falses(length(fs))
    for (fi, f) in enumerate(fs)
        length(f) > 3 && continue
        # shifted_left = shift_furniture(left_door, f, move)
        shifted_left = remove(left_door, f)
        shifted_left_pc = analyze_path(shifted_left, path_params)
        abs(initial_left_pc - shifted_left_pc) > 0.1 && continue
        # shifted_right = shift_furniture(right_door, f, move)
        shifted_right = remove(right_door, f)
        shifted_right_pc = analyze_path(shifted_right, path_params)
        (initial_right_pc - shifted_right_pc) < 2.5 && continue
        candidates[fi] = true
    end

    # pick a random candidate
    any(candidates) ? rand(findall(candidates)) : 0
end



function main()
    name = "09_18_2023"
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

    # path_params = AStarPath(obstacle_cost=1.0)
    path_params = NoisyPath(;obstacle_cost=2.0,
                            kernel_sigma = 0.5,
                            kernel_width = 3)

    # number of trials
    n = 10

    # will only consider these moves
    moves = [down_move, up_move]
    n_moves = length(moves)

    # number of trials per move condition
    max_count = Int64(ceil(n / n_moves))

    # empty room with doors
    left_cond = empty_room(room_steps, room_bounds, entrance, [doors[1]])
    right_cond = empty_room(room_steps, room_bounds, entrance, [doors[2]])
    template = room_to_template(left_cond)
    pws = prop_weights()

    display(pws)
    display(template)

    # will store summary of generated rooms here
    df = DataFrame(scene = Int64[],
                   flipx = Bool[],
                   furniture = Int64[],
                   move = Symbol[])

    for (idx, move) in enumerate(moves)
        i = 1
        while i <= max_count
            # generate a room pair
            (left, right) = sample_pair(left_cond, right_cond, template, pws)
            fi = eval_pair(left, right, move, path_params)
            # no valid pair generated, try again or finish
            fi == 0 && continue

            # valid pair found, organize and store
            id = (idx - 1) * max_count + i
            toflip = (i-1) % 2
            push!(df, [id, toflip, fi, move])

            right_path, _... = path_procedure(right, path_params)
            viz_room(right, right_path)
            # shifted_right = shift_furniture(right, fi, move)
            shifted_right = remove(right, furniture(right)[fi])
            shifted_right_path, _... = path_procedure(shifted_right, path_params)
            viz_room(shifted_right, shifted_right_path)
            left_path, _... = path_procedure(left, path_params)
            viz_room(left, left_path)
            # shifted_left = shift_furniture(left, fi, move)
            shifted_left = remove(left, furniture(left)[fi])
            shifted_left_path, _... = path_procedure(shifted_left, path_params)
            viz_room(shifted_left, shifted_left_path)
            # save scenes as json
            open("$(scenes_out)/$(id)_1.json", "w") do f
                write(f, left |> json)
            end
            open("$(scenes_out)/$(id)_2.json", "w") do f
                write(f, right |> json)
            end

            print("move $(idx): $(i)/$(max_count)\r")
            i += 1
        end
    end
    @show df
    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)

    return nothing
end

main();
