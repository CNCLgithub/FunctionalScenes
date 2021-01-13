using CSV
using Lazy
using JLD2
using Statistics
using FunctionalScenes

using DataFrames

import Random:shuffle

import FunctionalScenes: expand, furniture, valid_moves,
    shift_furniture, move_map, labelled_categorical


# Use if we don't get a strong effect with 1pair
# function predicate(x)
#     all(iszero.(x.d)) || all((!iszero).(x.d))
# end
# function digest(df::DataFrame)
#     nochng = @>> DataFrames.groupby(df, :furniture) begin
#         filter(x -> all(iszero.(x.d)) && nrow(x) >= 2)
#         x-> isempty(x) ? DataFrame() : labelled_categorical(x)
#         DataFrame
#     end

#     somechng = @>> DataFrames.groupby(df, :furniture) begin
#         filter(x -> !any(iszero.(x.d)) && nrow(x) >= 2)
#         x-> isempty(x) ? DataFrame() : labelled_categorical(x)
#         DataFrame
#     end
#     isempty(nochng) || isempty(somechng) return DataFrame()
#     rand(Bool) ? somechng : nochng
# end
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
    data = DataFrame()
    for (i,f) in enumerate(fs)
        moves = collect(Bool, valid_moves(r, f))
        moves = move_map[moves]
        # moves = intersect(move_map[moves], [:up, :down])
        for m in moves
            shifted = shift_furniture(r,f,m)
            d = mean(compare(r, shifted))
            append!(data, DataFrame(furniture = i+2,
                                    move = m,
                                    d = d))
        end
    end
    # isempty(data) ? DataFrame() : @> data sort([:furniture, :d])
    isempty(data) ? DataFrame() : @> data sort([:furniture, :d]) digest
end

function build(r::Room; k = 10, factor = 1)
    weights = ones(steps(r))
    # strt = Int(last(steps(r)) * 0.4)
    # weights[:, strt:end] .= 1.0
    new_r = last(furniture_chain(k, r, weights))
    new_r = FunctionalScenes.expand(new_r, factor)
    dist = search(new_r)
    (new_r, dist)
end

function saver(id::Int64, r::Room, out::String)
    @save "$(out)/$(id).jld2" r
end


function create(base::Room; n::Int64 = 15)
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
    return seeds, df
end

function render_base(bases::Vector{Room}, name::String)
    out = "/renders/$(name)"
    isdir(out) || mkdir(out)
    for (id,r) in enumerate(bases)
        p = "$(out)/$(id)"
        display(r)
        render(r, p, mode = "full")
    end
end

function render_stims(bases::Vector{Room}, df::DataFrame, name::String)
    out = "/renders/$(name)"
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
    name = "2e_1p_30s"
    n = 30
    room_dims = (11,20)
    entrance = [6]
    exits = [213, 217]
    r = Room(room_dims, room_dims, entrance, exits)
    display(r)
    @time seeds, df = create(r, n = n)
    out = "/scenes/$(name)"
    CSV.write("$(out).csv", df)
    isdir(out) || mkdir(out)
    @>> seeds enumerate foreach(x -> saver(x..., out))
    render_base(seeds, name)
    render_stims(seeds, df, name)
    return seeds, df
end

main();
