using Gen
using CSV
using JSON
using PyCall
using FileIO
using Images
using ArgParse
using DataFrames
using FunctionalScenes
using Random

np = pyimport("numpy")

function save_path_render(out::String, scene::Int, door::Int,
                          isshifted::Bool,
                          r::GridRoom, path::Matrix{Float64})
    m = FunctionalScenes.draw_room(r, path)
    m = imresize(m, (128,128))
    shifted = isshifted ? "shifted" : "unshifted"
    save("$(out)/$(scene)_$(door)_$(shifted).png", m)
    return nothing
end

name = "ccn_2023_exp"
# name = "pathcost_4.0"
function main()

    src = "/spaths/datasets/$(name)"
    df = DataFrame(CSV.File("$(src)/scenes.csv"))
    samples = 100

    n = nrow(df) * 2

    out = "/spaths/datasets/$(name)/path_analysis"
    isdir(out) || mkdir(out)

    fc = 1.0
    for c = LinRange(0.5, 20, 20), k = [5,7,9]
    for c = [5.0], fc = [1.0]
        params = AStarPath(
            obstacle_cost = c,
            floor_cost = fc,
        )
        paths = Array{Float64}(undef, (30, 2, 2, 32, 32))
        results = DataFrame(scene = Int64[],
                            door = Int64[],
                            is_shifted = Bool[],
                            obstacle_size = Int64[],
                            path_cost = Float64[],
                            path_dist = Float64[],
                            obstacle_cost = Float64[],
                            floor_cost = Float64[])
        render_out = "$(out)/$(c)_$(fc)_renders"
        isdir(render_out) || mkdir(render_out)
        metric_out = "$(out)/$(c)_$(fc)_path_metrics.csv"
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
            obs_size = length(to_shift)

            base_cost, base_path = astar_path(base, params;
                                              samples=samples)
            save_path_render(render_out, row.scene, row.door, false, base, base_path)

            base_dist = FunctionalScenes.distance_to_path(base,to_shift, base_path)
            paths[row.scene, row.door, 1, :, :] = base_path
            push!(results, (row.scene, row.door, false, obs_size,
                            base_cost, base_dist, c, fc))


            shifted = shift_furniture(base,
                                    to_shift,
                                    Symbol(row.move))
            shift_cost, shift_path  = astar_path(shifted, params;
                                                 samples=samples)
            save_path_render(render_out, row.scene, row.door, true, shifted, shift_path)

            paths[row.scene, row.door, 2, :, :] = shift_path
            shift_dist = FunctionalScenes.distance_to_path(shifted, to_shift, shift_path)
            push!(results, (row.scene, row.door, true, obs_size,
                            shift_cost, shift_dist, c, fc))
        end

        display(results)

        np.save("$(out)/$(c)_$(fc)_noisy_paths.npy", paths)
        CSV.write(metric_out, results)
    end
end

main();
