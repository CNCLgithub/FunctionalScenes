using PyCall
using FunctionalScenes
using FunctionalScenes: render_mitsuba
using Images:colorview, RGB


function mytest()

    room_dims = (16, 16)
    entrance = [8,9]
    exits = [16*15 + 4]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    r = add(r, Set(16 * 8 + 8))
    r = expand(r, 2)
    display(r)

    params = QuadTreeModel(;gt = r)
    mu = render_mitsuba(r, params)
    # # mu : 1 x C x H x W
    # @time mu = img_from_instance(r, params)
    img = colorview(RGB, permutedims(mu, (3, 1, 2)))
    display(img)

    return nothing
end

mytest();
