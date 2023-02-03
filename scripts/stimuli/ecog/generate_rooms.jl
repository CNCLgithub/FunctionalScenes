using CSV
using Lazy
using JSON
using DataFrames
using Statistics
using FunctionalScenes
using LinearAlgebra: norm
using FunctionalCollections

using Random
Random.seed!(1235)
# using Profile
# using StatProfilerHTML


function build(template::GridRoom, p;
               temp::Float64 = 1.,
               max_f::Int64 = 5,
               max_size::Int64 = 3,
               factor::Int64 = 2,
               pct_open::Float64 = 0.15,
               side_buffer::Int64 = 2,
               quant::Float64 = 0.5)

    # p = recurse_path_gm(template, temp)
    # template = fix_shortest_path(template, p)
    # # viz_room(template, p)
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
    weights[start_y:stop_y, start_x:stop_x] .= true

    # prevent obstacles on path
    weights[p] .= false

    vmap = PersistentVector(vec(weights))

    obstacles = length(FunctionalScenes.get_tiles(template, obstacle_tile))

    # generate furniture once and then apply to
    # each door condition
    # adjusted_f = ceil(Int64, max_f - obstacles * quant)
    # with_furn = furniture_gm(template, vmap, adjusted_f, max_size)

    # uncomment lines below if using `main()`
    adjusted_f = abs.(ceil(Int64, max_f - obstacles * 0.5))
    with_furn = furniture_gm(template, vmap, adjusted_f, max_size)


    # with_furn = template
    # viz_room(with_furn, p)
    with_furn = expand(with_furn, factor)
    # perform path analysis
    # results = examine_furniture(with_furn)
    (with_furn, adjusted_f)
end

# essentially the same function as main, just wrapped in 3 loops to set params
function set_params()
    quant_ls = collect(range(0.25, stop=0.75, step=0.25))
    temp_ls = collect(range(1.0, stop=5.0, step=0.5))
    max_f_ls = collect(range(7, stop=12, step=1))

    for t in eachindex(temp_ls)
        for m in eachindex(max_f_ls)
            for q in eachindex(quant_ls)
                name = "max-f=$(max_f_ls[m])_temp=$(temp_ls[t])_quant=$(quant_ls[q])"
                dataset_out = "/spaths/datasets/$(name)"
                isdir(dataset_out) || mkdir(dataset_out)

                scenes_out = "$(dataset_out)/scenes"
                isdir(scenes_out) || mkdir(scenes_out)

                # Parameters
                room_dims = (16, 16)
                entrance = [8, 9]
                door_rows = [8]
                inds = LinearIndices(room_dims)
                doors = inds[door_rows, room_dims[2]]

                groups = 10
                group_size = 10
                n = groups * group_size
                template = GridRoom(room_dims, room_dims, entrance, doors)

                # will store summary of generated rooms here
                df = DataFrame(id = Int64[],
                                temp = Float64[],
                                density = Float64[],
                                adjusted_f = Float64[],
                                quant = Float64[],
                                obstacles = Int64[],
                                path_len = Int64[])
                # generate rooms
                # temps = LinRange(0.1, t_max, groups)

                for i = 1:groups
                    # sample a random path in an empty room (the template)
                    # with a temperature parameter determining complextiy
                    p = recurse_path_gm(template, temp_ls[t])
                    # solve for the placement of obstacles such that path
                    # `p` is the shortest path in the room
                    ri = fix_shortest_path(template, p)

                    for j = 1:group_size
                        # adds the obstacle to the room object `ri`
                        (r, adjusted_f) = build(ri, p; max_f = max_f_ls[m], quant=quant_ls[q])
                        id = (i - 1) * group_size + j

                        obstacles = length(FunctionalScenes.get_tiles(r, obstacle_tile))
                        path_len = length(p)

                        _df = DataFrame(id = id,
                                        temp = temp_ls[t],
                                        density = max_f_ls[m],
                                        adjusted_f = adjusted_f,
                                        quant = quant_ls[q],
                                        obstacles = obstacles,
                                        path_len = path_len)
                        # invalid room generated, try again or finish
                        # isempty(_df) && continue

                        # valid pair found, organize and store
                        append!(df, _df)

                        # save scenes as json
                        open("$(scenes_out)/$(id).json", "w") do f
                            _d = r |> json
                            write(f, _d)
                        end
                    end
                end
                @show df
                # saving summary / manifest
                CSV.write("$(scenes_out).csv", df)
            end
        end
    end
    return nothing
end

function main()
    name = "ecog_pilot"
    dataset_out = "/spaths/datasets/$(name)"
    isdir(dataset_out) || mkdir(dataset_out)

    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)

    navigation = true

    # Parameters
    room_dims = (16, 16)
    entrance = [8, 9]
    door_rows = [8]
    inds = LinearIndices(room_dims)
    doors = inds[door_rows, room_dims[2]]
    # unused for now - Mario 11/08/2022
    # t_max = 1.0
    # f_factor = 2

    groups = 10
    group_size = 10
    n = groups * group_size
    template = GridRoom(room_dims, room_dims, entrance, doors)

    # will store summary of generated rooms here
    df = DataFrame(id = Int64[],
                    temp = Float64[],
                    density = Float64[],
                    adjusted_f = Float64[],
                    obstacles = Int64[],
                    path_len = Int64[])
    # generate rooms
    # temps = LinRange(0.1, t_max, groups)
    temp = 4.0
    max_f = 10 # limit of additional obstacles to add
    for i = 1:groups
        # sample a random path in an empty room (the template)
        # with a temperature parameter determining complextiy
        p = recurse_path_gm(template, temp)

        # solve for the placement of obstacles such that path
        # `p` is the shortest path in the room
        ri = fix_shortest_path(template, p)

        for j = 1:group_size
            # adds the obstacle to the room object `ri`
            #
            (r, adjusted_f) = build(ri, p; max_f = max_f)
            id = (i - 1) * group_size + j

            obstacles = length(FunctionalScenes.get_tiles(r, obstacle_tile))
            path_len = length(p)

            _df = DataFrame(id = id,
                            temp = temp,
                            density = max_f,
                            adjusted_f = adjusted_f,
                            obstacles = obstacles,
                            path_len = path_len)
            # invalid room generated, try again or finish
            # isempty(_df) && continue

            # valid pair found, organize and store
            append!(df, _df)

            # save scenes as json
            open("$(scenes_out)/$(id).json", "w") do f
                # FIXME: save p to JSON
                _d = JSON.lower(r)
                _d["paths"] = p

                _d = r |> json
 
                write(f, _d)
            end
        end
    end
    @show df
    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)
    return nothing
end

main();
# set_params();