using Gen
using Random
using BenchmarkTools
using FunctionalScenes
using FunctionalScenes: prod_grow_path, agg_grow_path

Random.seed!(1234)

function mytest()
    room_dims = (16, 24)
    entrance = [8,9]
    exits = [room_dims[1]*(room_dims[2] - 1) + 5]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    # r = expand(r, 2)
    temp = 2.0
    local p
    for _ = 1:5
        p = recurse_path_gm(r, temp)
        # viz_room(r, p)
        new_r = fix_shortest_path(r, p)
        viz_room(new_r, p)
        viz_room(new_r,
                occupancy_grid(new_r;
                                decay = 0.,
                                sigma = 0.))
    end
    # @btime recurse_path_gm($r, 0.2)
    # display(p)
    return nothing
end

mytest();
