export DGP, GrowState, valid_spaces, valid_move, valid_moves,
    strongly_connected

abstract type DGP end

@with_kw struct FurnishState <: DGP
    template::GridRoom
    vm::PersistentVector{Bool}
    f::Set{Int64}
    count::Int64
    max_size::Int64 = 5
    max_count::Int64 = 10
end

function FurnishState(st::FurnishState, f::Set{Int64})
    _vm = collect(Bool, st.vm)
    _vm[f] .= false
    purge_around!(_vm, st.template, f)
    FurnishState(st.template,
                 PersistentVector(_vm),
                 f,
                 st.count + 1,
                 st.max_size,
                 st.max_count)
end


@with_kw struct GrowState <: DGP
    head::Int64
    vm::PersistentVector{Bool}
    g::PathGraph
    current_depth::Int64
    max_depth::Int64 = 5
end

# function GrowState(head::Int64, vmap::BitMatrix,
#                    g::PathGraph)
#     GrowState(head, P)

function GrowState(ns, ni::Int64, st::GrowState)::GrowState
    @unpack vm, g, current_depth = st
    # done growing
    ni == 0 && return st
    # update head
    new_head = ns[ni]
    new_vm = assoc(vm, new_head, false)
    GrowState(new_head, new_vm, g, current_depth + 1,
              st.max_depth)
end

function neighboring_candidates(st::GrowState)::Vector{Int64}
    @unpack head, vm, g, current_depth, max_depth= st
    # reached max depth. terminate
    current_depth == max_depth && return Int64[]
    ns = neighbors(g, head)
    ns[vm[ns]]
end

function valid_spaces(r::Room)::PersistentVector{Bool} end

function valid_spaces(r::Room, vm::PersistentVector{Bool})
    PersistentVector{Bool}(valid_spaces(r) .& vm)
end

function valid_spaces(r::GridRoom)
    g = pathgraph(r)
    d = data(r)
    vec_d = vec(d)
    valid_map = Vector{Bool}(vec_d .== floor_tile)
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

    # want to prevent "stitching" new pieces together
    # fvs = Set{Int64}(findall(vec_d .== obstacle_tile))
    fvs = findall(vec_d .== obstacle_tile)
    ds = gdistances(g, fvs)
    valid_map[ds .<= 1] .= false
    # purge_around!(valid_map, r, fvs)

    PersistentVector(valid_map)
end

# having to deal with type instability
function merge_prod(st::GrowState, children::Set{Int64})
    @unpack head = st
    union(children, head)
end
function merge_prod(st::GrowState, children::Vector{Set{Int64}})
    @unpack head = st
    isempty(children) ? Set(head) : union(first(children), head)
end

function merge_prod(st::FurnishState, children::Set{Int64})
    @unpack f = st
    # @show f
    union(children, f)
    # children
end
function merge_prod(st::FurnishState, children::Vector{Set{Int64}})
    @unpack f = st
    # @show f
    isempty(children) ? f : union(first(children), f)
    # isempty(children) ? f : first(children)
end


function is_floor(r::GridRoom, t::Int64)::Bool
    g = pathgraph(r)
    d = data(r)
    has_vertex(g, t) && d[t] == floor_tile
end

function valid_move(r::Room, fid::Int64, m::Move)::Bool
    valid_move(r, furniture(r)[fid], m)
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

function purge_around!(vm::Vector{Bool}, r::GridRoom, f::Furniture)
    # want to prevent "stitching" new pieces together
    # viz_room(r)
    for m in move_map
        k = move(r, f, m)
        vm[k] .= false
    end
    return nothing
end

# checks whether moving furnition
# -1 changes contact with other furnitures
# -2 creates a "visual gap" when looking straight down the room
function strongly_connected(r::GridRoom, f::Furniture, m::Move)

    f_inds = collect(Int64, f)
    shifted = move(r, f, m)

    h, _ = steps(r)
    g = grid(steps(r))
    gs = gdistances(g, f_inds)
    shifted_gs = gdistances(g, collect(shifted))

    fs = furniture(r)
    other_inds = map(x -> collect(Int64, x), fs)

    connected = Int64[]
    # represents the horizon of blocks
    # ie. what spaces are occupied between
    # the left towards the right wall
    gaps = Vector{Bool}(zeros(h))
    for i = 1:length(fs)

        # skip rest if same as f
        fs[i] == f && continue

        # fill in gaps
        gaps[(other_inds[i] .- 1) .% h .+ 1] .= true

        # is fs[i] touching original f?
        touching1 = any(d -> d == 1, gs[other_inds[i]])
        # is fs[i] touching shifted f?
        touching2 = any(d -> d == 1, shifted_gs[other_inds[i]])

        # not touching
        (!touching1 && !touching2)  && continue

        # weakly connected, shifting changes contact
        xor(touching1, touching2) && return Int64[]

        # strongly connected
        push!(connected, i)

    end

    # make sure f is behind something
    !all(gaps[(f_inds .- 1) .% h .+ 1]) && return Int64[]

    # location after move
    gaps[(shifted .- 1) .% h .+ 1] .= true

    # gap was made with shift, reject
    !all(gaps[(f_inds .- 1) .% h .+ 1]) && return Int64[]

    return connected
end

# @gen functions
include("gen.jl")
include("path_based/path_based.jl")
