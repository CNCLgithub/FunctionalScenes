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
                          r::GridRoom, path)
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

    n = nrow(df) * 2

    out = "/spaths/datasets/$(name)/path_analysis"
    isdir(out) || mkdir(out)


    fc = 1.0
    # for c = LinRange(0.1, 5.0, 5), k = [5,7,9]
    for c = [32.0], k = [5]
        # params = AStarPath(
        #     obstacle_cost = c,
        #     floor_cost = fc,
        # )
        params = NoisyPath(
            obstacle_cost = c,
            floor_cost = fc,
            kernel_sigma = 3.0,
            kernel_width = k,
        )
        analysis_params = Dict(
            :kernel => k,
            :p => 0.35,
            :n => 7
        )
        param_data = Dict{Symbol, Any}(
            :obstacle_cost => c,
            :floor_cost => fc,
            :kernel_width => k)
        # paths = Array{Float64}(undef, (30, 2, 2, 32, 32))
        results = DataFrame(scene = Int64[],
                            door = Int64[],
                            is_shifted = Bool[],
                            obstacle_size = Int64[],
                            density = Float64[],
                            diffusion_ct = Float64[],
                            diffusion_ct_max = Float64[],
                            diffusion_ct_alt = Float64[],
                            diffusion_prop = Float64[],
                            diffusion_tot = Float64[],
                            path_dist = Float64[],
                            path_length = Int64[],
                            obstacle_cost = Float64[],
                            floor_cost = Float64[],
                            kernel_width = Int64[])
        render_out = "$(out)/$(c)_$(fc)_$(k)_renders"
        isdir(render_out) || mkdir(render_out)
        metric_out = "$(out)/$(c)_$(fc)_$(k)_path_metrics.csv"
        @show metric_out
        # isfile(metric_out) && continue
        for row in eachrow(df)

            base_p = "/spaths/datasets/$(name)/scenes/$(row.scene)_$(row.door).json"
            local base_s
            open(base_p, "r") do f
                base_s = JSON.parse(f)
            end
            base = from_json(GridRoom, base_s)
            to_shift = furniture(base)[row.furniture]
            obs_size = length(to_shift)

            scene_data = Dict{Symbol, Any}(
                :scene => row.scene,
                :door => row.door,
                :obstacle_size => obs_size)

            base_path, base_result = path_analysis(base, params, to_shift;
                                                   analysis_params...)

            save_path_render(render_out, row.scene, row.door, false, base,
                             base_path)

            # paths[row.scene, row.door, 1, :, :] = base_path

            trial_data = Dict{Symbol, Any}(:is_shifted => false)
            merge!(trial_data, base_result, scene_data, param_data)
            push!(results, trial_data)


            # shifted = remove(base, to_shift)
            shifted = shift_furniture(base,
                                    to_shift,
                                    Symbol(row.move))
            shifted_path, shifted_result = path_analysis(shifted, params, to_shift;
                                                         analysis_params...)

            save_path_render(render_out, row.scene, row.door, true, shifted,
                             shifted_path)

            # paths[row.scene, row.door, 2, :, :] = shift_path
            trial_data = Dict{Symbol, Any}(:is_shifted => true)
            merge!(trial_data, shifted_result, scene_data, param_data)
            push!(results, trial_data)
        end

        display(results)

        # np.save("$(out)/$(c)_$(fc)_$(k)_noisy_paths.npy", paths)
        CSV.write(metric_out, results)
    end
end

main();
