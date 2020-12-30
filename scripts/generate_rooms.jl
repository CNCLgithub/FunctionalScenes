using CSV
using Lazy
using JLD2
using Statistics
using FunctionalScenes

using DataFrames

import FunctionalScenes: expand, furniture, valid_moves,
    shift_furniture, move_map, labelled_categorical


function predicate(x)
    any(iszero.(x.d)) && any((!iszero).(x.d))
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
    fs = furniture(r)[3:end]
    data = []
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        for m in moves
            shifted = shift_furniture(r,f,m)
            d = mean(compare(r, shifted))
            push!(data, DataFrame(furniture = i+2,
                                  move = m,
                                  d = d))
        end
    end
    @> vcat(data...) sort([:furniture, :d]) digest
end

function build(r::Room; steps = 10, factor = 1)
    new_r = last(furniture_chain(steps, r))
    new_r = FunctionalScenes.expand(new_r, factor)
    dist = search(new_r)
    (new_r, dist)
end

function saver(id::Int64, r::Room, out::String)
    @save "$(out)/$(id).jld2" r
end


function create(base::Room; n::Int64 = 10)
    steps = @>> begin
        repeatedly(() -> build(base, factor = 2))
        filter(x -> @>> x last isempty !)
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
    r = Room((11,20), (11,20), [6], [213, 217])
    @time seeds, df = create(r, n = 15)
    CSV.write("/scenes/pilot.csv", df)
    render_base(seeds)
    render_stims(seeds, df)
    return r, df
end


r, xy = main();
