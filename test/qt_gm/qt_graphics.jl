using FunctionalScenes
using FunctionalScenes: img_from_instances,
    create_obs
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

    # mu : 1 x C x H x W
    @time mu = img_from_instance(r, params)
    img = colorview(RGB, mu[1])
    display(img)

    return nothing
end

mytest();
