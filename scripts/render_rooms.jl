using CSV
using Lazy
using JLD2
using FileIO
using FunctionalScenes

import FunctionalScenes: shift_furniture, functional_scenes, translate,
    _init_graphics, _load_device

using DataFrames

device = _load_device()

function render_base(bases::Vector{Int64}, name::String)
    out = "/renders/$(name)"
    isdir(out) || mkdir(out)
    for id in bases
        base_p = "/scenes/$(name)/$(id).jld2"
        base = load(base_p)["r"]
        p = "$(out)/$(id)"
        display(base)
        # render(base, p, mode = "none", threads = 4, navigation = true)
        render(base, p, mode = "full",threads = 4,  navigation = false)
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
        # render(room, p, mode = "none", threads = 4, navigation = true)
        render(room, p, mode = "full",threads = 4,  navigation = false)
    end
end


function render_torch(bases::Vector{Int64}, name::String)
    out = "/renders/$(name)"
    isdir(out) || mkdir(out)
    for id in bases
        base_p = "/scenes/$(name)/$(id).jld2"
        base = load(base_p)["r"]
        display(base)
        p = "$(out)/$(id).png"
        graphics = _init_graphics(base, (480, 720), device)
        d = translate(base, false; cubes = true)
        img = functional_scenes.render_scene_pil(d, graphics)
        img.save(p)
    end
end

function render_torch_stims(df::DataFrame, name::String)
    out = "/renders/$(name)"
    isdir(out) || mkdir(out)
    for r in eachrow(df)
        base_p = "/scenes/$(name)/$(r.id).jld2"
        base = load(base_p)["r"]
        p = "$(out)/$(r.id)_$(r.furniture)_$(r.move).png"
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               Symbol(r.move))
        graphics = _init_graphics(base, (480, 720), device)
        d = translate(room, false; cubes = true)
        img = functional_scenes.render_scene_pil(d, graphics)
        img.save(p)
    end
end
function main()
    #name = "pytorch_rep"
    name = "2e_1p_30s_matchedc3_cycles"
    src = "/scenes/$(name)"
    df = DataFrame(CSV.File("$(src).csv"))
    seeds = unique(df.id)
    #render_torch(seeds, name)
    #render_torch_stims(df, name)

    #seeds = [1]
    #df = df[df.id .== 1, :]
    render_base(seeds, name)
    render_stims(df, name)
    return nothing
end

main();
