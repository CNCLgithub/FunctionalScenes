using CSV
using Lazy
using JLD2
using Statistics
using FunctionalScenes

using DataFrames

import FunctionalScenes: expand, furniture, valid_moves, shift_furniture, move_map

function search(r::Room)
    fs = furniture(r)
    data = []
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        for m in moves
            shifted = shift_furniture(r,f,m)
            d = mean(compare(r, shifted))
            push!(data, DataFrame(furniture = i,
                                  move = m,
                                  d = d))
        end
    end
    vcat(data...)
end

function build(r::Room; steps = 10, factor = 1)
    new_r = last(furniture_chain(steps, r))
    new_r = FunctionalScenes.expand(new_r, factor)
    dist = @time search(new_r)
    (new_r, dist)
end

function saver(id::Int64, r::Room, out::String)
    @save "$(out)/$(id).jld2" r
end

function predicate(df::DataFrame)
    @> df begin
        DataFrames.groupby(:furniture)
        @>> filter(r -> (!isempty)(r) && any(iszero.(r.d)) && any((!iszero).(r.d)))
        isempty; !
    end
end

function foo(base::Room; n::Int64 = 10)
    steps = @>> begin
        repeatedly(() -> build(base, factor = 2))
        filter(x -> @>> x last predicate)
        take(n)
    end
    seeds, dfs = zip(steps...)
    out = "/scenes/pilot"
    isdir(out) || mkdir(out)
    @>> seeds enumerate foreach(x -> saver(x..., out))
    for (i,d) in enumerate(dfs)
        d[!, :id] .= i
    end
    df = vcat(dfs...)
    return collect(seeds), df
end

sort_search(df, rev) = @> df begin
    sort([:id, :furniture, :d], rev = rev)
    DataFrames.groupby([:id, :furniture])
    @>> filter(r -> (!isempty)(r) && any(iszero.(r.d)) && any((!iszero).(r.d)))
    @>> map(first)
    DataFrame
end


function render_base(bases::Vector{Room})
    out = "/renders/pilot"
    isdir(out) || mkdir(out)
    for (id,r) in enumerate(bases)
        p = "$(out)/$(id)"
        render(r, p, mode = "full")
    end
end

function render_stims(bases::Vector{Room}, df::DataFrame)
    out = "/renders/pilot"
    isdir(out) || mkdir(out)
    for r in eachrow(xy)
        base = bases[r.id]
        p = "$(out)/$(r.id)_$(r.furniture)_$(r.move)"
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               r.move)
        render(room, p, mode = "full")
    end
end

r = Room((11,20), (11,20), [6], [213, 217])
rooms = furniture_chain(10, r);
r2 = last(rooms);
# f = first(furniture(r2))
# shift_furniture(r2, f, :right)


@time seeds, df = foo(r)
x = sort_search(df, true)
y = sort_search(df, false)
xy = vcat(x,y)

CSV.write("/scenes/pilot.csv", xy)
render_base(seeds)
render_stims(seeds, xy)
