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

cycles_args = Dict(
    # :mode => "full"
    :mode => "none"
)

# function render_base(bases::Vector{Int64}, name::String;
#                      spheres = false)
#     out = spheres ? "spheres" : "cubes"
#     out = "/renders/$(name)_cycles_$(out)"
#     isdir(out) || mkdir(out)
#     for id in bases
#         base_p = "/scenes/$(name)/$(id).jld2"
#         base = load(base_p)["r"]
#         p = "$(out)/$(id)"
#         display(base)
#         render(base, p;
#                cycles_args...,
#                spheres = spheres)
#     end
# end

function render_stims(df::DataFrame, name::String;
                     spheres = false)
    out = spheres ? "spheres" : "cubes"
    out = "/renders/$(name)_cycles_$(out)"
    isdir(out) || mkdir(out)
    for r in eachrow(df)
        base_p = "/scenes/$(name)/$(r.id).jld2"
        base = load(base_p)["rs"][r.door]
        p = "$(out)/$(r.id)_$(r.door)"
        render(base, p;
               cycles_args...,
               spheres = spheres)
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               Symbol(r.move))
        p = "$(out)/$(r.id)_$(r.door)_$(r.furniture)_$(r.move)"
        render(room, p;
               cycles_args...,
               spheres = spheres)
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
    # args = parse_commandline()
    # args = Dict("scene" => 0)
    args = Dict("scene" => 1)

    name = "1_exit_22x40_doors"
    src = "/scenes/$(name)"
    df = DataFrame(CSV.File("$(src).csv"))
    seeds = unique(df.id)
    # render_torch(seeds, name)
    # render_torch_stims(df, name)
    if args["scene"] == 0
        seeds = unique(df.id)
    else
        seeds = [args["scene"]]
        df = df[df.id .== args["scene"], :]
    end

    # render_base(seeds, name,
    #             spheres = true
    #             )
    # render_stims(df, name,
    #              spheres = true,
    #              )
    # render_base(seeds, name,
    #             spheres = false
    #             )
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
