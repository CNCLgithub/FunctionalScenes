using Images
using Statistics
using CSV
using JLD2
using FileIO
using DataFrames
# using FunctionalScenes
using LinearAlgebra


dataset = "ccn_2023_exp"

function _img(data)

end

function aggregate_chains(path::String, chains::Int64)
    att_d = "$(path)/attention"
    isdir(att_d) || mkdir(att_d)
    pg_d = "$(path)/path"
    isdir(att_d) || mkdir(att_d)
    gs_d = "$(path)/global_state"
    isdir(gs_d) || mkdir(gs_d)
    img_d = "$(path)/img_mu"
    isdir(img_d) || mkdir(img_d)

    avg_sens = zeros(32, 32)
    avg_pg = zeros(32, 32)
    steps = 50
    for i = 1:steps
        sens = zeros(32, 32)
        gs = zeros(32, 32)
        pg = zeros(32, 32)
        img_mu = zeros(128, 128, 3)
        for c = 1:chains
            c_path = "$(path)/$(c).jld2"
            isfile(c_path) || return nothing
            local data
            data = load(c_path, "$i")
            sens += data[:attention][:sensitivities]
            gs += data[:projected]
            pg += data[:path]
            img_mu += permutedims(data[:img_mu], (2, 3, 1))
        end

        rmul!(sens, 1.0 / chains)
        rmul!(gs, 1.0 / chains)
        rmul!(pg, 1.0 / chains)
        rmul!(img_mu, 1.0 / chains)
        avg_sens += sens
        avg_pg += pg

        path_hm = Plots.heatmap(reverse(pg, dims = 1),
                               c = cgrad([:black, :white]))
                               # c = :thermal)
        Plots.savefig(path_hm, "$(pg_d)/$(i).svg")

        att_hm = Plots.heatmap(reverse(sens, dims = 1),
                               c = :gist_heat)
                               # c = :thermal)
                               # c = cgrad([:black, :orange]))
        Plots.savefig(att_hm, "$(att_d)/$(i).svg")
        gs_hm = Plots.heatmap(reverse(gs, dims = 1),
                                c = cgrad(["#b3acb0ff", "#393a8bff"]))
        Plots.savefig(gs_hm, "$(gs_d)/$(i).svg")
        # if i == steps
        #     gs_hm = Plots.heatmap(reverse(gs, dims = 1),
        #                           c = cgrad(["#b3acb0ff", "#393a8bff"]))
        #     Plots.savefig(gs_hm, "$(gs_d)/$(i).svg")
        # end
        FileIO.save("$(img_d)/$(i).png", img_mu)
    end

    rmul!(avg_sens, 1.0 / steps)
    rmul!(avg_pg, 1.0 / steps)
    path_hm = Plots.heatmap(reverse(avg_pg, dims = 1),
                            c = cgrad([:black, :white]))
    Plots.savefig(path_hm, "$(pg_d).svg")
    att_hm = Plots.heatmap(reverse(avg_sens, dims = 1),
                           c = cgrad([:black, :orange]))
    Plots.savefig(att_hm, "$(att_d).svg")
end
function main()
    exp_path = "/spaths/experiments/$(dataset)"
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    df = df[df.id .== 7, :]
    for r in eachrow(df)
        base_path = "$(exp_path)/$(r.id)_$(r.door)"
        isdir(base_path) || continue
        @show base_path
        aggregate_chains(base_path, 1)
        # shift_path = "$(base_path)_furniture_$(r.move)/1.jld2"
        # isdir(shift_path) || continue
        # @show shift_path
        # aggregate_chains(shift_path, 2)
    end
    return nothing
end

main();
