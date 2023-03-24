using Gen
using JSON
using FileIO
using FunctionalScenes
using FunctionalScenes: generate_qt_from_ddp,
    DataDrivenState,
    create_obs,
    display_mat,
    render_mitsuba

dataset = "ccn_2023_exp"

function load_base_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
end
function mytest()

    scene = 4
    door = 1
    base_path = "/spaths/datasets/$(dataset)/scenes"
    base_p = joinpath(base_path, "$(scene)_$(door).json")
    room = load_base_scene(base_p)
    display(room)
    model_params = QuadTreeModel(;gt = room, base_sigma = 0.2)


    ddp_path = "/project/scripts/nn/configs/og_decoder.yaml"
    ddp_params = DataDrivenState(;config_path = ddp_path,
                                 var = 0.15)
    gt_img = render_mitsuba(room, model_params.scene, model_params.sparams,
                            model_params.skey, model_params.spp)
    tracker_cm = generate_qt_from_ddp(ddp_params, gt_img, model_params)

    cm = create_obs(model_params)
    set_submap!(cm, :trackers,
                get_submap(tracker_cm, :trackers))

    tr, ls = generate(qt_model, (model_params,), cm)

    @show ls

    trace_st = get_retval(tr)
    println("Inferred state")
    display_mat(trace_st.qt.projected)


    return nothing
end

mytest();
