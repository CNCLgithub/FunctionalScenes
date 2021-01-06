using CSV
using Lazy
using JLD2
using Statistics
using FunctionalScenes

using DataFrames

import FunctionalScenes: expand, furniture, valid_moves,
    shift_furniture, move_map, labelled_categorical


function predicate(x)
    any((!iszero).(x.d))
    # any(iszero.(x.d)) && any((!iszero).(x.d))
end

function digest(df::DataFrame)
    @>> DataFrames.groupby(df, :furniture) begin
        filter(g -> nrow(g) >= 2)
        map(g -> g[1:2, :])
        filter(predicate)
        x -> isempty(x) ? x : labelled_categorical(x)
        DataFrame
    end
end

function search(r::Room)
    # avoid furniture close to camera
    fs = furniture(r)
    data = DataFrame()
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = intersect(move_map[moves], [:up, :down])
        for m in moves
            shifted = shift_furniture(r,f,m)
            d = mean(compare(r, shifted))
            append!(data, DataFrame(furniture = i,
                                    move = m,
                                    d = d))
        end
    end
    isempty(data) ? DataFrame() : @> data sort([:furniture, :d]) digest
end

function build(r::Room; k = 7, factor = 1)
    strt = Int(last(steps(r)) / 2.0)
    weights = zeros(steps(r))
    weights[:, strt:end] .= 1.0
    new_r = last(furniture_chain(k, r, weights))
    new_r = FunctionalScenes.expand(new_r, factor)
    dist = search(new_r)
    (new_r, dist)
end

function saver(id::Int64, r::Room, out::String)
    @save "$(out)/$(id).jld2" r
end


function create(base::Room; n::Int64 = 10)
    seeds = Vector{Room}(undef, n)
    df = DataFrame()
    i = 1
    while i <= n
        seed, _df = build(base, factor = 2)
        if !isempty(_df)
            seeds[i] = seed
            _df[!, :id] .= i
            append!(df, _df)
            i += 1
        end
    end
    return collect(seeds), df
end

function render_base(bases::Vector{Room})
    out = "/renders/1exit"
    isdir(out) || mkdir(out)
    for (id,r) in enumerate(bases)
        p = "$(out)/$(id)"
        render(r, p, mode = "full")
    end
end

function render_stims(bases::Vector{Room}, df::DataFrame)
    out = "/renders/1exit"
    isdir(out) || mkdir(out)
    for r in eachrow(df)
        base = bases[r.id]
        p = "$(out)/$(r.id)_$(r.furniture)_$(r.move)"
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               r.move)
        render(room, p, mode = "full")
    end
end

function main()
    room_dims = (8,16)
    entrance = [4]
    seeds = Room[]
    df = DataFrame()
    exits = collect(((8*15+1):8*16))
    for ex = 2:7
        r = Room(room_dims, room_dims, entrance, exits[ex:ex])
        display(r)
        @time _seeds, _df = create(r, n = 5)
        append!(seeds, _seeds)
        _df[!, :id] = _df.id .+ (ex - 2)*5
        append!(df, _df)
    end
    CSV.write("/scenes/1exit.csv", df)
    out = "/scenes/1exit"
    isdir(out) || mkdir(out)
    @>> seeds enumerate foreach(x -> saver(x..., out))
    render_base(seeds)
    render_stims(seeds, df)
    return seeds, df
end

main();
