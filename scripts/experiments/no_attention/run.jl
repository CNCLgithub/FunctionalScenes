using CSV
using Gen
using JSON
using JLD2
using Images: imresize
using FileIO
using ArgParse
using DataFrames
using LinearAlgebra: norm
using FunctionalScenes
using FunctionalScenes: shift_furniture, generate_qt_from_ddp

dataset = "ccn_2023_exp"
scale = 4

function vae_geo(room::GridRoom, ddp_params)
    model_params = FunctionalScenes.load(QuadTreeModel, "$(@__DIR__)/gm.json"; gt = room)
    gt_img = render_mitsuba(room, model_params.scene, model_params.sparams,
                            model_params.skey, model_params.spp)
    tracker_cm = generate_qt_from_ddp(ddp_params, gt_img, model_params)
    trace,_ = Gen.generate(qt_model, (model_params,), tracker_cm)
    st = get_retval(trace)
    geo = imresize(st.qt.projected, (scale, scale))
end

function main()

    ddp_params = DataDrivenState(;config_path = "/project/scripts/nn/configs/og_decoder.yaml",
                                 var = 0.175)
    exp_path = "/spaths/experiments/$(dataset)_no_attention"

    try
        isdir(exp_path) || mkpath(exp_path)
    catch e
        println("could not make dir $(out_path)")
    end

    src = "/spaths/datasets/$(dataset)"
    df = DataFrame(CSV.File("$(src)/scenes.csv"))

    results = DataFrame(scene = Int64[],
                        max_diff_geo = Float64[],
                        mag_diff_geo = Float64[],
                        tot_diff_geo = Float64[])
    geo_comp = zeros((scale, scale, 2, 30))
    for r in eachrow(df)
        base_p = "/spaths/datasets/$(dataset)/scenes/$(r.id)_$(r.door).json"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)
        base_geo = vae_geo(base, ddp_params)
       
        shifted = shift_furniture(base,
                                  furniture(base)[r.furniture],
                                  Symbol(r.move))
        shifted_geo = vae_geo(shifted, ddp_params)

        geo_comp[:, :, r.door, r.id] = base_geo - shifted_geo
    end
    for i = 1:30
        # diff_att = norm(base_att - shift_att)
        diff = geo_comp[:, :, 1, i] - geo_comp[:, :, 2, i]
        tot_diff_geo = norm(diff)
        mag_diff_geo = sum(diff)
        # max_diff_geo = maximum(abs.(geo_comp[:, :, 1, i])) - maximum(abs.(geo_comp[:, :, 2, i]))
        max_diff_geo = diff[findmax(abs.(diff))[2]]
        push!(results, (scene = i, max_diff_geo = max_diff_geo, mag_diff_geo=mag_diff_geo,
                        tot_diff_geo=tot_diff_geo))
    end
    CSV.write("$(exp_path)/chain_summary.csv", results)

    return nothing
end

main();
