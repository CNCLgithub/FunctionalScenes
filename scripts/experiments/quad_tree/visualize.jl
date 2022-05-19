using CSV
using JLD2
using FileIO
using DataFrames
using FunctionalScenes

using Plots
using Images

dataset = "vss_pilot_11f_32x32_restricted"

function viz_chain(path::String)
    att_d = "$(dirname(path))/attention"
    isdir(att_d) || mkdir(att_d)
    gs_d = "$(dirname(path))/global_state"
    isdir(gs_d) || mkdir(gs_d)
    img_d = "$(dirname(path))/img_mu"
    isdir(img_d) || mkdir(img_d)
    last_idx = load(path, "current_idx")
    @show last_idx
    # current_chain = load(path, "current_chain")
    # FunctionalScenes.viz_gt(current_chain.state)
    local data
    for i = 1:100
        data = load(path, "$i")
        # att_hm = Gadfly.spy(data[:attention][:sensitivities])
        # att_hm |> PNG("$(att_d)/$(i).png", 4inch, 4inch)
        att = reverse(data[:attention][:sensitivities],
                      dims = 1)
        att_hm = Plots.heatmap(att)
        Plots.savefig(att_hm, "$(att_d)/$(i).png")
        gs_hm = Plots.heatmap(reverse(data[:global_state],
                                      dims = 1))
        Plots.savefig(gs_hm, "$(gs_d)/$(i).png")
        # gs_hm = Gadfly.spy(data[:global_state])
        # gs_hm |> PNG("$(gs_d)/$(i).png", 4inch, 4inch)
        FileIO.save("$(img_d)/$(i).png",
                    permutedims(data[:img_mu], (2, 3, 1)))
    end
end

function main()
    exp_path = "/spaths/experiments/$(dataset)"
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    for r in eachrow(df)
        base_path = "$(exp_path)/$(r.id)_$(r.door)"
        chain_path = "$(base_path)/1.jld2"
        @show chain_path
        isfile(chain_path) || continue
        viz_chain(chain_path)
        chain_path = "$(base_path)_$(r.furniture)_$(r.move)/1.jld2"
        isfile(chain_path) || continue
        viz_chain(chain_path)
    end
    return nothing
end

main();
