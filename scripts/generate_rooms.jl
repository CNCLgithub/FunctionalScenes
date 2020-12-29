using Revise


using Lazy
using JLD2
using Statistics
using FunctionalScenes

import DataFrames: DataFrame

import FunctionalScenes: expand, furniture, valid_moves, shift_furniture, move_map

function search(r::Room)
    fs = furniture(r)
    dist = Dict()
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        for m in moves
            shifted = shift_furniture(r,f,m)
            dist[i => m] = mean(compare(r, shifted))
        end
    end
    return dist
end

function build(r::Room; steps = 5, factor = 1)
    new_r = last(furniture_chain(steps, r))
    new_r = FunctionalScenes.expand(new_r, factor)
    dist = @time search(new_r)
    (new_r, dist)
end

function saver(id::Int64, r::Room, out::String)
    @save "$(out)/$(id).jld2" r
end

function bar(xy::Tuple)::Vector{Tuple}
    id, distd = xy
    [(id, k,v) for (k,v) in distd]
end

function foo(base::Room; n::Int64 = 100)
    steps = @>> begin
        repeatedly(() -> build(base, factor = 2))
        filter(x -> @>> x last values map(!iszero) any)
        take(n)
    end
    seeds, dists = zip(steps...)
    out = "/scenes/pilot"
    isdir(out) || mkdir(out)
    @>> seeds enumerate foreach(x -> saver(x..., out))
    df = DataFrame(id = Int64[], move = Pair[], d = Float64[])
    something = @>> dists enumerate map(bar)
    foreach(x -> push!(df, x), vcat(something...))
    return df
end

r = Room((11,20), (11,20), [6], [213, 217])
rooms = furniture_chain(10, r);
r2 = last(rooms);
# f = first(furniture(r2))
# shift_furniture(r2, f, :right)


@time df = foo(r2)

# using Profile
# Profile.init(10000000, 0.01)
# using StatProfilerHTML



# furniture(r2);

# @profilehtml furniture(r2);
# @profilehtml furniture(r2);
# @profilehtml s = search(r2);
# @profilehtml s = search(r2);
