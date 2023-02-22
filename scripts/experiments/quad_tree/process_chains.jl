using Images
using CSV
using JLD2
using FileIO
using DataFrames
using FunctionalScenes
using LinearAlgebra


dataset = "vss_pilot_11f_32x32_restricted"

function aggregate_chains(path::String, chains::Int64, steps)
    avg_sens = zeros(32, 32)
    avg_geo = zeros(32, 32)
    for i = 1:steps
        for c = 1:chains
            c_path = "$(path)/$(c).jld2"
            isfile(c_path) || return nothing
            local data
            data = load(c_path, "$i")
            # att_hm = Gadfly.spy(data[:attention][:sensitivities])
            # att_hm |> PNG("$(att_d)/$(i).png", 4inch, 4inch)
            avg_sens += data[:attention][:sensitivities]
            if i == steps
                avg_gep += data[:global_state]
            end
        end
    end

    rmul!(avg_sens, 1.0 / (steps * chains))
    rmul!(avg_geo, 1.0 / chains)
    (avg_sens, avg_geo)
end

function add_metrics!(results, r, att, gs)
    base_p = "/spaths/datasets/$(dataset)/scenes/$(r.id)_$(r.door).json"
    local base_s
    open(base_p, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
    idxs = furniture(base)[r.furniture]
    raw_att = sum(att[idxs])
    tot_att = sum(att)
    prop_att = raw_att / tot_att
    geo_p = sum(gs[idxs])
    push!(results, (scene = r.id,
                    door = r.door,
                    furniture = r.furniture,
                    move = r.move,
                    raw_att = raw_att,
                    prop_att = prop_att,
                    tot_att = tot_att,
                    geo_p = geo_p
                    ))
end

function main()
    exp_path = "/spaths/experiments/$(dataset)"
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    results = DataFrame(scene = Int64[],
                        door = Int64[],
                        furniture = Int64[],
                        move = String[],
                        raw_att = Float64[],
                        prop_att = Float64[],
                        tot_att = Float64[],
                        geo_p = Float64[])
    for r in eachrow(df)
        base_path = "$(exp_path)/$(r.id)_$(r.door)"
        isdir(base_path) || continue
        @show base_path
        att, gs = aggregate_chains(base_path, 3, 100)
        add_metrics!(results, att, gs)
    end
    CSV.write("$(exp_path)/chain_summary.csv", results)
    return nothing
end

main();
