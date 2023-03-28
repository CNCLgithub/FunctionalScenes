using CSV
using JLD2
using JSON
using FileIO
using PyCall
using DataFrames
# using FunctionalScenes
using Statistics
using LinearAlgebra
# using FunctionalScenes:


np = pyimport("numpy")

# assuming scenes are 32x32
dataset = "ccn_2023_exp"
burnin = 50

function aggregate_chains(path::String, chains::Int64, steps)
    att = zeros((chains, steps, 32, 32))
    geo = zeros((chains, steps, 32, 32))
    pmat = zeros(Int64, (chains, steps, 32, 32))
    gran = zeros(Int64, (chains, steps, 32, 32))
    img = zeros((chains, steps, 128, 128, 3))
    for c = 1:chains
        c_path = "$(path)/$(c).jld2"
        isfile(c_path) || continue
        for s = 1:steps
            local data
            data = load(c_path, "$s")
            att[c, s, :, :] = data[:attention][:sensitivities]
            geo[c, s, :, :] = data[:projected]
            pmat[c, s, :, :] = data[:path]
            gran[c, s, :, :] = data[:granularity]
            img[c, s, :, :, :] = data[:img_mu]
        end
    end
    @pycall (np.savez("$(path)/aggregated.npz",
                      att = att,
                      geo=geo,
                      pmat=pmat,
                      gran=gran,
                      img=img))::PyObject

    att_mu = reshape(mean(att[:, burnin:end, :, :], dims = (1, 2)), 32, 32)
    geo_mu = reshape(mean(geo[:, burnin:end, :, :], dims = (1, 2)), 32, 32)
    return att_mu, geo_mu
end

# function add_metrics!(results, r, att, gs)
#     base_p = "/spaths/datasets/$(dataset)/scenes/$(r.id)_$(r.door).json"
#     local base_s
#     open(base_p, "r") do f
#         base_s = JSON.parse(f)
#     end
#     base = from_json(GridRoom, base_s)
#     idxs = furniture(base)[r.furniture]

#     raw_att = sum(att[idxs])
#     tot_att = sum(att)
#     prop_att = raw_att / tot_att
#     # geo_p = sum(gs[idxs]) / length(idxs)
#     geo_p = sum(gs[idxs]) / length(idxs)
#     push!(results, (scene = r.id,
#                     door = r.door,
#                     furniture = r.furniture,
#                     move = r.move,
#                     raw_att = raw_att,
#                     prop_att = prop_att,
#                     tot_att = tot_att,
#                     geo_p = geo_p
#                     ))
# end

function main()
    exp_path = "/spaths/experiments/$(dataset)"
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    results = DataFrame(scene = Int64[],
                        att_diff = Float64[],
                        geo_diff = Float64[])
    for scene = 1:30
        base_path = "$(exp_path)/$(scene)_1"
        @show base_path
        att1, gs1 = aggregate_chains(base_path, 5, 100)
        base_path = "$(exp_path)/$(scene)_2"
        @show base_path
        att2, gs2 = aggregate_chains(base_path, 5, 100)
        l2_att = norm(att1 - att2)
        l2_geo = norm(gs1 - gs2)
        push!(results, (scene = scene, att_diff = l2_att,
                        geo_diff = l2_geo))

    end
    CSV.write("$(exp_path)/chain_summary.csv", results)
    return nothing
end

main();
