using Gen
using FunctionalScenes
using Images: colorview, RGB, channelview
using FileIO: save
using Graphs
using JSON
# using Profile
# using BenchmarkTools
# using StatProfilerHTML

using FunctionalScenes: generate_qt_from_ddp,
    DataDrivenState,
    create_obs,
    display_mat,
    render_mitsuba,
    node_to_idx


using Random
Random.seed!(1234)

dataset = "ccn_2023_exp"

function load_base_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
end

function ex_path(trace)
    st = get_retval(trace)
    n = size(st.qt.projected, 1)
    leaves = st.qt.leaves
    m = fill(false, (n,n))
    for e in st.path.edges
        src_node = leaves[src(e)].node
        idx = node_to_idx(src_node, n)
        m[idx] .= true
        dst_node = leaves[dst(e)].node
        idx = node_to_idx(src_node, n)
        m[idx] .= true
    end
    Matrix{Float64}(m)
end

function mytest()

    scene = 22
    door = 1
    base_path = "/spaths/datasets/$(dataset)/scenes"
    base_p = joinpath(base_path, "$(scene)_$(door).json")
    room = load_base_scene(base_p)
    display(room)
    model_params = QuadTreeModel(;gt = room, base_sigma = 0.1,
                                 obs_cost = 1E5
                                 )


    ddp_path = "/project/scripts/nn/configs/og_decoder.yaml"
    ddp_params = DataDrivenState(;config_path = ddp_path,
                                 var = 0.175)
    gt_img = render_mitsuba(room, model_params.scene, model_params.sparams,
                            model_params.skey, model_params.spp)
    tracker_cm = generate_qt_from_ddp(ddp_params, gt_img, model_params)

    cm = create_obs(model_params)
    set_submap!(cm, :trackers,
                get_submap(tracker_cm, :trackers))

    tr, ls = generate(qt_model, (model_params,), cm)

    display_mat(get_retval(tr).qt.projected)
    display_mat(ex_path(tr))

    return nothing
end

mytest();
