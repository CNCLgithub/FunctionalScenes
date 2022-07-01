using Gen
using JSON
using Random
using BenchmarkTools
using FunctionalScenes

Random.seed!(1235)

cycles_args = Dict(
    # :mode => "full",
    :mode => "none",
    :navigation => false
)
function mytest()
    room_dims = (16, 24)
    entrance = [8,9]
    exits = [room_dims[1]*(room_dims[2] - 1) + 5]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    # r = expand(r, 2)
    temp = 1.0
    out = "/spaths/datasets/path_dgp"
    isdir(out) || mkdir(out)
    for i = 1:5
        p = recurse_path_gm(r, temp)
        # viz_room(r, p)
        new_r = fix_shortest_path(r, p)
        new_r = expand(new_r, 2)
        # viz_room(new_r)
        # viz_room(r, p)
        # viz_room(r,
        #         occupancy_grid(new_r;
        #                         decay = 0.,
        #                         sigma = 0.))
        # save scenes as json
        open("$(out)/$(i).json", "w") do f
            _d = new_r |> json
            write(f, _d)
        end
        rout = "$(out)/$(i)"
        render(new_r, rout;
               cycles_args...)
    end
    return nothing
end

mytest();
