using Gen
using JSON
using FunctionalScenes
# using ProfileView


function mytest()
    room_dims = (16, 16)
    entrance = [8,9]
    exits = [16*16 - 8]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    r = expand(r, 2)

    # base_p = "/spaths/datasets/vss_pilot_11f_32x32_restricted/scenes/1_1.json"
    # local base_s
    # open(base_p, "r") do f
    #     base_s = JSON.parse(f)
    # end
    # r = from_json(GridRoom, base_s)

    params = QuadTreeModel(;gt = r)



    println(params.device)
    local trace
    for _ = 1:10
        trace, ll = generate(qt_model, (params,))
    end
    st = get_retval(trace)
    # @time generate(qt_model, (params,))
    viz_render(trace)
    viz_gt(trace)
    viz_room(r)
    viz_room(st.instances[1])
    return nothing
end

mytest();
