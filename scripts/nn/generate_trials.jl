using PyCall
using JSON
using Images
using Lazy: @>>
using FunctionalScenes
using FunctionalScenes: _init_mitsuba_scene,
    mi
using FunctionalCollections: PersistentVector

mi.set_variant("cuda_ad_rgb")
IMG_RES = (128, 128)
SPP = 24
KEY = "grid.interior_medium.sigma_t.data"

function occupancy_position(r::GridRoom)::Matrix{Float64}
    grid = zeros(steps(r))
    grid[FunctionalScenes.data(r) .== obstacle_tile] .= 1.0
    grid
end

function build(r::GridRoom;
               max_f::Int64 = 11,
               max_size::Int64 = 5,
               pct_open::Float64 = 0.3,
               side_buffer::Int64 = 0,
               factor = 2)

    dims = steps(r)
    # prevent furniture generated in either:
    # -1 out of sight
    # -2 blocking entrance exit
    # -3 hard to detect spaces next to walls
    weights = Matrix{Bool}(zeros(dims))
    # ensures that there is no furniture near the observer
    start_x = Int64(ceil(last(dims) * pct_open))
    stop_x = last(dims) - 2 # nor blocking the exit
    # buffer along sides
    start_y = side_buffer + 1
    stop_y = first(dims) - side_buffer
    weights[start_y:stop_y, start_x:stop_x] .= 1.0
    vmap = PersistentVector(vec(weights))

    # generate furniture
    with_furn = furniture_gm(r, vmap, max_f, max_size)
    expand(with_furn, factor)
end

function save_trial(dpath::String, i::Int64, r::GridRoom,
                    img, og)
    out = "$(dpath)/$(i)"
    isdir(out) || mkdir(out)

    open("$(out)/room.json", "w") do f
        rj = r |> json
        write(f, rj)
    end
    open("$(out)/scene.json", "w") do f
        r2 = translate(r, Int64[]; cubes = false)
        r2j = r2 |> json
        write(f, r2j)
    end
    cimg = map(clamp01nan, img)
    save_img_array(cimg, "$(out)/render.png")
    # occupancy grid saved as grayscale image
    save("$(out)/og.png", og)
    return nothing
end

function main()
    # Parameters
    name = "ccn_2023_ddp_train_11f_32x32"
    n = 5000
    # name = "ccn_2023_ddp_test_11f_32x32"
    # n = 16
    room_dims = (16, 16)
    entrance = [8, 9]
    door_rows = [5, 12]
    inds = LinearIndices(room_dims)
    doors = inds[door_rows, room_dims[2]]

    # empty rooms with doors
    templates = Vector{GridRoom}(undef, length(doors))
    # initialize mitsuba scenes
    mi_scenes = Vector{PyObject}(undef, length(templates))
    mi_params = Vector{PyObject}(undef, length(templates))
    for i = 1:length(templates)
        r = GridRoom(room_dims, room_dims, entrance, [doors[i]])
        templates[i] = r
        r = expand(r, 2)
        scene = _init_mitsuba_scene(r, IMG_RES)
        mi_scenes[i] = scene
        mi_params[i] = @pycall mi.traverse(scene)::PyObject
    end

    # will store summary of generated rooms here
    m = Dict(
        :n => n,
        :templates => templates,
        :og_shape => (32, 32),
        :img_res => IMG_RES
    )
    out = "/spaths/datasets/$(name)"
    isdir(out) || mkdir(out)

    for i = 1:n
        idx = ceil(Int64, i / n)
        t = templates[idx]
        r = build(t)
        # select mitsuba scene
        ms = mi_scenes[idx]
        mp = mi_params[idx]
        @time r_img = render_mitsuba(r, ms, mp, KEY, SPP)
        r_og = occupancy_position(r)
        save_trial(out, i, r, r_img, r_og)
    end
    
    m[:img_mu] = zeros(3)
    m[:img_sd] = ones(3)

    open("$(out)_manifest.json", "w") do f
        write(f, m |> json)
    end
    return nothing
end

main();
