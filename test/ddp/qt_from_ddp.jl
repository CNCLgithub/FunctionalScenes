using Gen
using JSON
using FileIO
using FunctionalScenes
using FunctionalScenes: generate_qt_from_ddp,
    DataDrivenState,
    create_obs,
    display_mat

dataset = "ccn_2023_exp"

function load_base_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
end
function mytest()

    scene = 1
    door = 1
    base_path = "/spaths/datasets/$(dataset)/scenes"
    base_p = joinpath(base_path, "$(scene)_$(door).json")
    room = load_base_scene(base_p)

    model_params = QuadTreeModel(;gt = room, instances = 10, base_sigma = 0.2)


    ddp_path = "/project/scripts/nn/configs/og_decoder.yaml"
    ddp_params = DataDrivenState(;config_path = ddp_path,
                                 var = 0.13)
    gt_img = img_from_instance(room, model_params)
    ddp_args = (ddp_params, gt_img, model_params)

    tracker_cm = generate_qt_from_ddp(ddp_args...)
    display(tracker_cm)

    cm = create_obs(model_params)
    set_submap!(cm, :trackers,
                get_submap(tracker_cm, :trackers))

    tr, ls = generate(qt_model, (model_params,), cm)

    @show ls

    trace_st = get_retval(tr)
    println("Inferred state")
    display_mat(trace_st.gs)
    # set_submap!(cm, :trackers,
    #             get_submap(tracker_cm, :trackers))


    return nothing
end

mytest();
