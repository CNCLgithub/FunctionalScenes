using Gen
using CSV
using JSON
using PyCall
using FileIO
using ArgParse
using DataFrames
using FunctionalScenes
using Random

np = pyimport("numpy")

name = "pathcost_3.0"
function main()

    src = "/spaths/datasets/$(name)"
    df = DataFrame(CSV.File("$(src)/scenes.csv"))
    samples = 20

    n = nrow(df) * 2

    out = "/spaths/datasets/$(name)_path"
    isdir(out) || mkdir(out)

    # for c = LinRange(0.5, 4.5, 8), k = [5,7,9]
    # for c = LinRange(1.0, 5.0, 4), fc = LinRange(0.25, 1.0, 4), k = [5]
    # for c = LinRange(0.5, 2.5, 4), fc = LinRange(0.1, 1.0, 4), k = [7]
    for c = [10.00], fc = [2.00], k = [7]
    # for c = [1.64], k = [7]
        params = NoisyPath(
            obstacle_cost = c,
            floor_cost = fc,
            kernel_width = k,
            kernel_sigma = 5.0,
            wall_cost = 2*c,
        )
        paths = Array{Float64}(undef, (30, 2, 2, 32, 32))
        results = DataFrame(scene = Int64[],
                            door = Int64[],
                            is_shifted = Bool[],
                            path_cost = Float64[],
                            path_dist = Float64[],
                            obstacle_cost = Float64[],
                            floor_cost = Float64[],
                            kernel_width = Int64[])
        metric_out = "$(out)/$(c)_$(fc)_$(k)_path_metrics.csv"
        @show metric_out
        isfile(metric_out) && continue
        for row in eachrow(df)

            @show (row.scene)
            # row.scene == 4 && break
            base_p = "/spaths/datasets/$(name)/scenes/$(row.scene)_$(row.door).json"
            local base_s
            open(base_p, "r") do f
                base_s = JSON.parse(f)
            end
            base = from_json(GridRoom, base_s)
            to_shift = furniture(base)[row.furniture]

            # base_cost, base_path = noisy_path(base, params;
            #                                   samples=samples)
            base_cost, base_path = noisy_shortest_paths(base, params;
                                                        k=samples)
            viz_room(base, base_path)

            base_dist = FunctionalScenes.distance_to_path(to_shift, base_path)
            paths[row.scene, row.door, 1, :, :] = base_path
            push!(results, (row.scene, row.door, false, base_cost, base_dist, c, fc, k))

            shifted = shift_furniture(base,
                                    to_shift,
                                    Symbol(row.move))
            # shift_cost, shift_path  = noisy_path(shifted, params;
            #                                      samples=samples)
            shift_cost, shift_path = noisy_shortest_paths(shifted, params;
                                                          k=samples)
            viz_room(shifted, shift_path)

            paths[row.scene, row.door, 2, :, :] = shift_path
            shift_dist = FunctionalScenes.distance_to_path(to_shift, shift_path)
            push!(results, (row.scene, row.door, true, shift_cost, shift_dist, c, fc, k))
        end

        display(results)

        np.save("$(out)/$(c)_$(fc)_$(k)_noisy_paths.npy", paths)
        CSV.write(metric_out, results)
    end
end

main();
