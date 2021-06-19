using CSV
using Lazy
using JLD2
using FileIO
using FunctionalScenes
using JSON

import FunctionalScenes: shift_furniture, functional_scenes, translate,
    _init_graphics, _load_device,occupancy_position

using DataFrames

device = _load_device()

function save_occupancy_grid(og::Matrix{Float64}, out::String)
    og = og |> json
    open(out, "w") do f
        write(f, og)
    end
end

function render_torch(rooms::Vector{Room}, out::String)
    for (i, r) in enumerate(rooms)
        path = "$(out)/$(i)"
        @show path
        isdir(path) || mkdir(path)

        # render image
        p = "$(path)/render.png"
        graphics = _init_graphics(r, (480, 720), device)
        d = translate(r, false; cubes = true)
        img = functional_scenes.render_scene_pil(d, graphics)
        img.save(p)

        # save og
        og_path = "$(path)/og.json"
        og = occupancy_position(r)
        og_json = save_occupancy_grid(og, og_path)
    end
    return nothing
end

function main()
    ##name = "train_ddp_1_exit_22x40_doors"
    name = "test_ddp_1_exit_22x40_doors"
    src = "/scenes/$(name)/rooms.jld2"
    rooms = load(src)["rs"]


    out = "/datasets/$(name)"
    isdir(out) || mkdir(out)
    render_torch(rooms, out)
    return nothing
end

main();
