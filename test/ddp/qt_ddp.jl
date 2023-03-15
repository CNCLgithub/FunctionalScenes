using Gen
using FunctionalScenes
using FunctionalScenes: graphics_from_instances
using Images:colorview, RGB



function mytest()

    room_dims = (16, 16)
    entrance = [8,9]
    exits = [16*16 - 8]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    r = add(r, Set(16 * 8 + 8))
    display(r)
    r = expand(r, 2)

    params = QuadTreeModel(;gt = r)

    @time (mu, _) = graphics_from_instances(r, params)
    img = colorview(RGB, mu)
    display(img)

    return nothing
end

mytest();
