using CSV
using JSON
using DataFrames
using SparseArrays
using FunctionalScenes
using FunctionalScenes: path_cost
using WaveFunctionCollapse

function prop_weights()
    n = length(HT2D_vec)
    # column `i` in `ws` defines the weights over
    # h-tiles (the rows) associated with `i`.
    ws = zeros(n, n)
    # Want mostly empty space
    ws[1, :] .= 25.0 #
    # Add "islands" proportional to size
    # ws[2:end, 1] .= 20
    for (i, hti) = enumerate(HT2D_vec)
        ws[i, 1] += 10.0 * count(hti)
    end
    # Grow islands with smaller parts
    for (i, hti) = enumerate(HT2D_vec)
        ni = count(hti)
        # ni < 2 && continue
        for (j,  htj) = enumerate(HT2D_vec)
            nj = count(htj)
            (nj == 0 || nj >= ni) && continue
            ws[j, i] += 6 * (4 - nj) *
                ((hti[3] == htj[1]) || (hti[4] == htj[2]))
        end
    end
    return ws
end

function room_to_template(r::GridRoom;
                          gap::Int64 = 1)
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
    gap_rng = 2:(r-gap-1)
    result[gap_rng, 2:(end-1)] .= 0
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
    # REVIEW: why rot?
    Set{Int64}(findall(vec(rotr90(expanded))))
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
    path, _... = path_procedure(room, params)
    d = FunctionalScenes.obstacle_diffusion(room, path, 0.5, 5)
end

function eval_pair(left_door::GridRoom, right_door::GridRoom,
                   path_params::PathProcedure)

    # initial paths
    diff_left = analyze_path(left_door, path_params)
    diff_right = analyze_path(right_door, path_params)

    fs = furniture(left_door)
    candidates = Vector{Bool}(undef, length(fs))
    @inbounds for (fi, f) in enumerate(fs)
        candidates[fi] = diff_left[fi] < 0.1 &&
            diff_right[fi] > 3.0
    end
    length(fs) > 3  && any(candidates) ? rand(findall(candidates)) : 0
end



function main()
    name = "diffusion_n_block"
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
    n = 15

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
                   fidx = Int64[])

    i = 1
    c = 0
    while i <= n && c < 100 * n
        # generate a room pair
        (left, right) = sample_pair(left_cond, right_cond, template, pws)
        fi = eval_pair(left, right,  path_params)
        # no valid pair generated, try again or finish
        c += 1
        fi == 0 && continue

        # valid pair found, organize and store
        toflip = (i-1) % 2
        push!(df, [i, toflip, fi])

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
