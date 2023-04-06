using CSV
using JLD2
using JSON
using FileIO
using PyCall
using DataFrames
# using FunctionalScenes
using Statistics
using LinearAlgebra
using Images: imresize


np = pyimport("numpy")

# assuming scenes are 32x32
dataset = "ccn_2023_exp"
burnin = 50
chains = 5
steps = 200

function aggregate_chains(path::String, chains::Int64, steps)
    att = zeros((chains, steps, 32, 32))
    geo = zeros((chains, steps, 32, 32))
    pmat = zeros(Int64, (chains, steps, 32, 32))
    gran = zeros(Int64, (chains, steps, 32, 32))
    img = zeros((chains, steps, 128, 128, 3))
    for c = 1:chains
        c_path = "$(path)/$(c).jld2"
        jldopen(c_path, "r") do file
        for s = 1:steps
            # data = load(c_path, "$s")
            data = file["$s"]
            att[c, s, :, :] = data[:attention][:sensitivities]
            geo[c, s, :, :] = data[:projected]
            pmat[c, s, :, :] = data[:path]
            gran[c, s, :, :] = data[:granularity]
            img[c, s, :, :, :] = data[:img_mu]
        end
        end
    end
    # np.savez("$(path)_aggregated.npz",
    #                   att = att,
    #                   geo=geo,
    #                   pmat=pmat,
    #                   gran=gran,
    #                   img=img)

    att_mu = reshape(mean(att[:, burnin:end, :, :], dims = (1, 2)), 32, 32)
    att_mu = imresize(att_mu, (4,4))
    geo_mu = reshape(mean(geo[:, burnin:end, :, :], dims = (1, 2)), 32, 32)
    geo_mu = imresize(geo_mu, (4,4))
    geo_init = reshape(mean(geo[:, 1:2, :, :], dims = (1, 2)), 32, 32)
    geo_init = imresize(geo_init, (4,4))
    GC.gc(true)
    return att_mu, geo_mu - geo_init
end

function main()
    exp_path = "/spaths/experiments/$(dataset)"
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    results = DataFrame(scene = Int64[],
                        diff_geo = Float64[])
    geo_comp = zeros((4, 4, 2, 30))
    for r in eachrow(df)
        base_path = "$(exp_path)/$(r.id)_$(r.door)"
        @show base_path
        base_att, base_geo = aggregate_chains(base_path, chains, steps)

        shift_path = "$(exp_path)/$(r.id)_$(r.door)_furniture_$(r.move)"
        @show shift_path
        shift_att, shift_geo = aggregate_chains(shift_path, chains, steps)

        geo_comp[:, :, r.door, r.id] = base_geo - shift_geo
    end
    for i = 1:30
        # diff_att = norm(base_att - shift_att)
        # diff_geo = norm(geo_comp[:, :, 1, i] - geo_comp[:, :, 2, i])
        diff_geo = maximum(abs.(geo_comp[:, :, 1, i])) - maximum(abs.(geo_comp[:, :, 2, i]))
        push!(results, (scene = i, diff_geo = diff_geo))
    end
    CSV.write("$(exp_path)/chain_summary_max_diff_$(burnin).csv", results)
    return nothing
end

main();
