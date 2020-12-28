using Revise


using Lazy
using Statistics
using FunctionalScenes

import FunctionalScenes: expand, furniture, valid_moves, shift_furniture, move_map
# template_rooms = [

# ];


function search(r::Room)
    fs = furniture(r)
    moved = Dict()
    dist = Dict()
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        for m in moves
            shifted = shift_furniture(r,f,m)
            moved[i => m] = shifted
            dist[i => m] = mean(compare(r, shifted))
        end
    end
    return (moved, dist)
end

function build(r::Room; steps = 5, factor = 1)
    new_r = last(furniture_chain(steps, r))
    new_r = FunctionalScenes.expand(new_r, factor)
    moved, dist = search(new_r)
    (new_r, moved, dist)
end

r = Room((11,20), (11,20), [6], [213, 217])

rooms = furniture_chain(10, r);

r2 = last(rooms);
f = first(furniture(r2))
shift_furniture(r2, f, :right)


@time build(r2)

# using Profile
# Profile.init(10000000, 0.01)
# using StatProfilerHTML



# furniture(r2);

# @profilehtml furniture(r2);
# @profilehtml furniture(r2);
# @profilehtml s = search(r2);
# @profilehtml s = search(r2);
