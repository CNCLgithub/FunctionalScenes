using Gen
using BenchmarkTools
using FunctionalScenes
using FunctionalScenes: prod_grow_path, agg_grow_path

function mytest()
    room_dims = (16, 16)
    entrance = [8,9]
    exits = [16*15 + 5]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    r = expand(r, 2)
    for _ = 1:5
        p = recurse_path_gm(r, 0.2)
        viz_room(r, p)
    end
    @btime recurse_path_gm($r, 0.2)
    # display(p)
    return nothing
end

mytest();
