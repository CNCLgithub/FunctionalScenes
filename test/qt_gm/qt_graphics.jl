using PyCall
using JSON
using FunctionalScenes
using FunctionalScenes: render_mitsuba,
    _init_mitsuba_scene,
    mi
using Images:colorview, RGB

dataset = "ccn_2023_exp"

function load_base_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
end

function mytest()

    scene_id = 7
    door = 1
    base_path = "/spaths/datasets/$(dataset)/scenes"
    base_p = joinpath(base_path, "$(scene_id)_$(door).json")
    room = load_base_scene(base_p)
    display(room)

    scene = _init_mitsuba_scene(room, (512, 512))
    params = @pycall mi.traverse(scene)::PyObject
    key = "grid.interior_medium.sigma_t.data"
    @time mu = render_mitsuba(room, scene, params, key, 24)
    # # mu : 1 x C x H x W
    # @time mu = img_from_instance(r, params)
    img = colorview(RGB, permutedims(mu, (3, 1, 2)))
    display(img)
    save_img_array(mu, "/spaths/tests/mitsuba_scene_$(scene_id).png")

    return nothing
end

mytest();
