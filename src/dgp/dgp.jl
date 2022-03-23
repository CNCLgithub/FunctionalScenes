
abstract type DGP end

struct GrowState <: DGP
    head::Int64
    vm::PersistentVector{Bool}
    g::PathGraph
end

function GrowState(ns, ni::Int64, st::GrowState)::GrowState
    @unpack vm, g = st
    # done growing
    ni == 0 && return st
    # update head
    new_vm = assoc(vm, ns[ni], false)
    GrowState(ni, new_vm, g)
end

function neighboring_candidates(st::GrowState)::Int64
    @unpack head, vm, g = st
    ns = neighbors(g, vm)
    # xs = ns[findall(vm[ns])]
    @>> ns begin
        getindex(vm) # neighbors free tiles?
        findall      # local indices
        getindex(ns) # global indices
    end
end

function valid_spaces(r::Room) end

function valid_spaces(r::GridRoom)
    g = pathgraph(r)
    d = data(r)
    valid_map = vec(d) .== floor_tile
    valid_map[entrance(r)] .= false
    valid_map[exits(r)] .= false

    # cannot block entrances
    for v in entrace(r)
        ns = neighbors(g, v)
        valid_map[ns] .= false
    end
    # cannot block exits
    for v in exits(r)
        ns = neighbors(g, v)
        valid_map[ns] .= false
    end

    findall(valid_map)
end


const move_map = [up_move, down_move, left_move, right_move]

function valid_move(g::PathGraph, f::Furniture, m)
    @>> setdiff(m, f) begin
        map(v -> has_vertex(g, v) && isfloor(g, v))
        all
    end
end

function valid_moves(r::Room, f::Furniture)
    g = pathgraph(r)
    vs = vertices(g)
    rows, cols = steps(r)
    moves = Matrix{Tile}(undef, length(f), 4)
    moves[:, 1] = f .- 1
    moves[:, 2] = f .+ 1
    moves[:, 3] = f .- rows
    moves[:, 4] = f .+ rows
    @>> moves eachcol lazymap(m -> valid_move(g,f,m)) collect(Float64)
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
