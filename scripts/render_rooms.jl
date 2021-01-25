using CSV
using Lazy
using JLD2
using FileIO
using FunctionalScenes

import FunctionalScenes: shift_furniture

using DataFrames

function render_base(bases::Vector{Int64}, name::String)
    out = "/renders/$(name)"
    isdir(out) || mkdir(out)
    for id in bases
        base_p = "/scenes/$(name)/$(id).jld2"
        base = load(base_p)["r"]
        p = "$(out)/$(id)"
        display(base)
        render(base, p, mode = "full", threads = 4)
    end
end

function render_stims(df::DataFrame, name::String)
    out = "/renders/$(name)"
    isdir(out) || mkdir(out)
    for r in eachrow(df)
        base_p = "/scenes/$(name)/$(r.id).jld2"
        base = load(base_p)["r"]
        p = "$(out)/$(r.id)_$(r.furniture)_$(r.move)"
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               Symbol(r.move))
        render(room, p, mode = "full", threads = 4)
    end
end

function main()
    name = "2e_1p_30s_matchedc3"
    src = "/scenes/$(name)"
    df = DataFrame(CSV.File("$(src).csv"))
    seeds = unique(df.id)
    render_base(seeds, name)
    render_stims(df, name)
    return nothing
end

main();
