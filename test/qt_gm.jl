using Gen
using FunctionalScenes
using FunctionalScenes: expand, cubify, translate
# using ProfileView


function mytest()
    room_dims = (32,32)
    entrance = [3]
    exits = [32*32 - 16]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    params = QuadTreeModel(;gt = r)

    println(params.device)

    trace, ll = generate(qt_model, (params,))
    @time generate(qt_model, (params,))
    return nothing
end

mytest();
