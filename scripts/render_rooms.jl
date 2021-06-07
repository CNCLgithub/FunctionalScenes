using CSV
using Lazy
using JLD2
using FileIO
using ArgParse
using FunctionalScenes

import FunctionalScenes: shift_furniture, functional_scenes, translate,
    _init_graphics, _load_device

using DataFrames

device = _load_device()

function render_base(bases::Vector{Int64}, name::String;
                     spheres = false)
    out = spheres ? "spheres" : "cubes"
    out = "/renders/$(name)_cycles_$(out)"
    isdir(out) || mkdir(out)
    for id in bases
        base_p = "/scenes/$(name)/$(id).jld2"
        base = load(base_p)["r"]
        p = "$(out)/$(id)"
        display(base)
        render(base, p, mode = "full", threads = 8, navigation = false,
               spheres = spheres)
        # render(base, p, mode = "full", threads = 8, navigation = false,
        #        spheres = true)
        # render(base, p, mode = "full",threads = 8,  navigation = false)
    end
end

function render_stims(df::DataFrame, name::String;
                     spheres = false)
    out = spheres ? "spheres" : "cubes"
    out = "/renders/$(name)_cycles_$(out)"
    isdir(out) || mkdir(out)
    for r in eachrow(df)
        base_p = "/scenes/$(name)/$(r.id).jld2"
        base = load(base_p)["r"]
        p = "$(out)/$(r.id)_$(r.furniture)_$(r.move)"
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               Symbol(r.move))
        render(room, p, mode = "full", threads = 8, navigation = false,
               spheres = spheres)
        # render(room, p, mode = "full", threads = 8, navigation = false,
        #        spheres = true)
        # render(room, p, mode = "full",threads = 4,  navigation = false)
    end
end


function render_torch(bases::Vector{Int64}, name::String)
    out = "/renders/$(name)_torch3d"
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
    out = "/renders/$(name)_torch3d"
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
    args = parse_commandline()
    name = "1_exit_22x40"
    src = "/scenes/$(name)"
    df = DataFrame(CSV.File("$(src).csv"))
    seeds = unique(df.id)
    # render_torch(seeds, name)
    # render_torch_stims(df, name)

    @show args
    seeds = [args["scene"]]
    df = df[df.id .== args["scene"], :]
    render_base(seeds, name,
                spheres = true
                )
    render_stims(df, name,
                 spheres = true,
                 )
    render_base(seeds, name,
                spheres = false
                )
    render_stims(df, name,
                 spheres = false,
                 )
    return nothing
end



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "scene"
        help = "Which scene to run"
        arg_type = Int64
        required = true
    end

    return parse_args(s)
end



main();
