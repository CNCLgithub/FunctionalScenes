export PathState, recurse_path_gm, fix_shortest_path

using DataStructures: Stack

struct PathState
    dims::Tuple{Int64, Int64}
    # graph of the room so far
    g::PathGraph
    # geodisic distances
    gs::Vector{Float64}
    # current tile
    head::Int64
    # temperature for next head
    temp::Function
    # step count
    step::Int64
end

function PathState(r::GridRoom; temp = x ->  1.0 / x)
    dims = steps(r)
    ent = first(entrance(r))
    ext = first(exits(r))
    @show ext
    g = pathgraph(r)
    dm = noisy_distm(r, 0.1)
    ds = dijkstra_shortest_paths(g, entrance(r), dm).dists
    # ds = 1.0 ./ gdistances(g, ext)
    PathState(dims, g, ds, ext, temp, 0)
end

function PathState(new_head::Int64, st::PathState)
    PathState(st.dims, st.g, st.gs, new_head, st.temp, st.step + 1)
end

function head_weights(st::PathState)
    @unpack g, gs, head, temp, step = st
    ns = neighbors(g, head)
    d = gs[head]
    ds = d .- gs[ns]
    t = temp(step)
    ws = softmax(ds; t = t)
    @show step
    @show t
    @show d
    @show ds
    @show ws
    ns, ws
end

function is_adjacent(t1::Int64, t2::Int64, col_dim::Int64)::Bool
    dist = abs(t2 - t1)
    dist == 1 || dist == col_dim
end

function rem_redundant_path(path::Pair, k::Int64, col_dim::Int64)::Pair
    cnt = 0
    is_redundant = false
    p = path

    while true
        # remove adjacent subsections
        is_redundant = (cnt > 0 && is_adjacent(p.first, k, col_dim))
        (is_redundant || p.second == 0) && break

        p = p.second
        cnt += 1
    end

    (is_redundant ? rem_redundant_path(p, k, col_dim) : path)
end

function merge_grow_path(st::PathState, children)::Pair
    @unpack head, dims = st
    # no children, then we are at the end of the path
    isempty(children) && return head => 0
    # otherwise will have 1 child
    path_so_far::Pair = first(children)
    head => rem_redundant_path(path_so_far, head, dims[2])
end

function parse_recurse_path(x::Pair)
    p = Int64[]
    while x.second != 0
        push!(p, x.first)
        x = x.second
    end
    push!(p, x.first)
    return p
end

function find_invalid_path(omat, g, x, gs, pmat)
    y::Int64 = 0
    @inbounds for n in collect(neighbors(g, x))
        omat[n] && continue        # already blocked
        pmat[n] && continue        # part of the path
        gs[n] == 0 && continue     # end of path
        # block if shorter
        # if gs[x] >= gs[n]

        if gs[x] >= gs[n]
            y = n
            break
        end
    end
    return y
end

const MoveDeque = Vector{Tuple{Int64, Int64, Int64}}

function noisy_distm(r::GridRoom, w::Float64)
    g = pathgraph(r)
    d = data(r) .== floor_tile
    nrow = size(d, 1)
    n = length(d)
    # m = Matrix{Float64}(undef, n, n)
    m = fill(Inf, (n, n))

    @inbounds for i = 1:n, j = 1:n
        vd = abs(i - j)
        (vd == 1 || vd == nrow) || continue
        # case which di is (free tile, free tile)
            # m[i,j] should be 0.1
        # case which di is (free_tile, obstacle) or any permutation
            # m[i,j] should be 0.9
        m[i,j] = (d[i] && d[j]) ? w : 1 - w
    end
    return m
end


function _update_fix_weights!(weights::Array{Float64},
                              path::Vector{Int64},
                              g::SimpleGraph,
                              gds::Array,
                              mpd::Int64)
    # weights to add obstacle
    @inbounds for i = eachindex(weights)
        weights[i] = gds[i] <= mpd ? mpd - gds[i] : -Inf
    end
    # prioritize tiles neighboring the path
    @inbounds for (i, x) in enumerate(path)
        for n = neighbors(g, x)
            # `n` is closer to exit
            if gds[n] <= gds[x]
                weights[n] = mpd - gds[n] + 2
            end
        end
    end
    weights[path] .= -Inf
    weights[1:32] .= -Inf
end

function fix_shortest_path(r::GridRoom, p::Vector{Int64},
                           max_steps = 20)
    # no path to fix
    isempty(p) && return r
    # info of current room
    g = deepcopy(pathgraph(r))

    # reference distances
    gds = gdistances(g, exits(r))
    max_path_d = length(p)

    blocked = Set{Int64}()
    # weights to add obstacle
    weights = fill(-Inf, length(gds))
    _update_fix_weights!(weights, p, g, gds, max_path_d)

    c = 0
    while c < max_steps && any(!isinf, weights)
        to_block = categorical(softmax(weights))
        weights[to_block] = -Inf
        push!(blocked, to_block)
        # remove edges
        for n = neighbors(g, to_block)
            rem_edge!(g, to_block, n)
            rem_edge!(g, n, to_block)
        end
        # update distances and weights
        gds = gdistances(g, exits(r))
        _update_fix_weights!(weights, p, g, gds, max_path_d)
        c += 1
    end
    blocked
end

include("gen.jl")
include("path_cost.jl")
