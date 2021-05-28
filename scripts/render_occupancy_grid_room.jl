using CSV
using Lazy
using JLD2
using FileIO
using FunctionalScenes
using JSON

import FunctionalScenes: shift_furniture, functional_scenes, translate,
    _init_graphics, _load_device,occupancy_grid

using DataFrames

device = _load_device()

function render_base(bases::Vector{Int64}, name::String)
    out = "/renders/$(name)_new"
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
    out = "/renders/$(name)_new"
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

function save_occupancy_grid(og::Matrix{Float64}, out::String, og_name::String)
    isdir(out) || mkpath(out)
    og_out = joinpath(out, "$(og_name).json")
    og = og |> json
    open(og_out, "w") do f
        write(f, og)
    end
end

function render_torch(bases::Vector{Int64}, name::String)
    for id in bases
        base_p = "/scenes/$(name)/$(id).jld2"
        base = load(base_p)["r"]
        display(base)

	out = "/datasets/$(name)/$(id)"
	isdir(out) || mkdir(out)

        p = "$(out)/render.png"
        graphics = _init_graphics(base, (480, 720), device)
        og = occupancy_grid(base, decay = 0.0, sigma = 1.0)[1]
	#display(og)
	og_name = "og"

	og_json = save_occupancy_grid(og,out,og_name)
	d = translate(base, false; cubes = true)
        img = functional_scenes.render_scene_pil(d, graphics)
        img.save(p)
    end
end

function render_torch_stims(df::DataFrame, name::String)
    #out = "/renders/$(name)"
    #isdir(out) || mkdir(out)
    for r in eachrow(df)
        base_p = "/scenes/$(name)/$(r.id).jld2"
        base = load(base_p)["r"]
     
	out = "/datasets/$(name)/$(r.id)_$(r.furniture)_$(r.move)"
    	isdir(out) || mkdir(out)

	p = "$(out)/render.png"
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               Symbol(r.move))
        graphics = _init_graphics(base, (480, 720), device)
        og = occupancy_grid(room, decay = 0.0, sigma = 1.0)[1]
        og_name = "og"

        og_json = save_occupancy_grid(og,out,og_name)
	d = translate(room, false; cubes = true)
        img = functional_scenes.render_scene_pil(d, graphics)
        img.save(p)
    end
end
function main()
    #name = "pytorch_rep"
    name = "occupancy_grid_data_driven"
    src = "/scenes/$(name)"
    df = DataFrame(CSV.File("$(src).csv"))
    seeds = unique(df.id)
    render_torch(seeds, name)
    render_torch_stims(df, name)

    # seeds = [1]
    # df = df[df.id .== 1, :]
    # render_base(seeds, name)
    # render_stims(df, name)
    return nothing
end

main();
