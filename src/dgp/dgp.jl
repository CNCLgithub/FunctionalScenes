export DGP, GrowState, valid_spaces, valid_move, valid_moves
abstract type DGP end

struct GrowState <: DGP
    head::Int64
    vm::PersistentVector{Bool}
    g::PathGraph
end

# function GrowState(head::Int64, vmap::BitMatrix,
#                    g::PathGraph)
#     GrowState(head, P)

function GrowState(ns, ni::Int64, st::GrowState)::GrowState
    @unpack vm, g = st
    # done growing
    ni == 0 && return st
    # update head
    new_head = ns[ni]
    new_vm = assoc(vm, new_head, false)
    GrowState(new_head, new_vm, g)
end

function neighboring_candidates(st::GrowState)::Vector{Int64}
    @unpack head, vm, g = st
    ns = neighbors(g, head)
    ns[vm[ns]]
end

function valid_spaces(r::Room)::PersistentVector{Bool} end

function valid_spaces(r::Room, vm::PersistentVector{Bool})
    PersistentVector(valid_spaces(r) .& vm)
end

function valid_spaces(r::GridRoom)
    g = pathgraph(r)
    d = data(r)
    valid_map = vec(d) .== floor_tile
    valid_map[entrance(r)] .= false
    valid_map[exits(r)] .= false

    # cannot block entrances
    for v in entrance(r)
        ns = neighbors(g, v)
        valid_map[ns] .= false
    end
    # cannot block exits
    for v in exits(r)
        ns = neighbors(g, v)
        valid_map[ns] .= false
    end
    PersistentVector(valid_map)
end

# having to deal with type instability
function merge_growth(head::Int64, children::Set{Int64})
    union(children, head)
end
function merge_growth(head::Int64, children::Vector{Set{Int64}})
    isempty(children) ? Set(head) : union(first(children), head)
end

function is_floor(r::GridRoom, t::Int64)::Bool
    g = pathgraph(r)
    d = data(r)
    has_vertex(g, t) && d[t] == floor_tile
end

function valid_move(r::Room, f::Furniture, m::Move)::Bool
    g = pathgraph(r)
    mf = move(r, f, m)
    @>> setdiff(mf, f) begin
        collect(Int64)
        map(v -> is_floor(r, v))
        all
    end
end

function valid_moves(r::Room, f::Furniture)::BitVector
    g = pathgraph(r)
    vs = vertices(g)
    moves = Vector{Bool}(undef, 4)
    @inbounds for i = 1:4
        moves[i] = valid_move(r, f, move_map[i])
    end
    BitVector(moves)
end

function strongly_connected(r::Room, f::Furniture, move::Symbol)

    f_inds = collect(Int64, f)
    shifted = @>> f_inds begin
        map(v -> shift_tile(r, v, move))
        collect(Tile)
    end

    h, _ = steps(r)
    g = grid(steps(r))
    gs = gdistances(g, f_inds)
    shifted_gs = gdistances(g, shifted)

    fs = furniture(r)
    other_inds = map(x -> collect(Int64, x), fs)

    connected = Int64[]
    gaps = BitVector(zeros(h))
    for i = 1:length(fs)

        # skip rest if same as f
        fs[i] == f && continue

        # fill in gaps
        gaps[(other_inds[i] .- 1) .% h .+ 1] .= 1

        touching1 = any(d -> d == 1, gs[other_inds[i]])
        touching2 = any(d -> d == 1, shifted_gs[other_inds[i]])

        # not touching
        (!touching1 && !touching2)  && continue

        # weakly connected
        xor(touching1, touching2) && return Int64[]

        # strongly connected
        push!(connected, i)

    end

    # check to see if `f` is in the front
    !all(gaps[(f_inds .- 1) .% h .+ 1]) && return Int64[]

    # location after move
    gaps[(shifted .- 1) .% h .+ 1] .= 1

    # gap was made with shift, reject
    !all(gaps[(f_inds .- 1) .% h .+ 1]) && return Int64[]

    return connected
end

# @gen functions
include("gen.jl")
