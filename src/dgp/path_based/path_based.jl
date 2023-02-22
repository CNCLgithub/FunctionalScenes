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
    temp::Float64
    # step count
    step::Int64
end

function PathState(r::GridRoom; temp::Float64 = 1.0)
    dims = steps(r)
    ent = first(entrance(r))
    ext = first(exits(r))
    g = pathgraph(r)
    dm = noisy_distm(r, 0.1)
    ds = 1.0 ./ dijkstra_shortest_paths(g, ext, dm).dists

    PathState(dims, g, ds, ent, temp, 0)
end

function PathState(new_head::Int64, st::PathState)
    PathState(st.dims, st.g, st.gs, new_head, st.temp, st.step + 1)
end

function head_weights(st::PathState, ns)
    @unpack gs, temp, step = st
    ds = gs[ns]
    ws = softmax(ds; t = temp / step)
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
    d = data(r)
    n = length(d)
    m = Matrix{Float64}(undef, n, n)

    @inbounds for i = 1:n, j = 1:n
        # case which di is (free tile, free tile)
            # m[i,j] should be 0.1
        # case which di is (free_tile, obstacle) or any permutation
            # m[i,j] should be 0.9
        m[i,j] = (d[i] == d[j]) ? w : 1 - w
    end
    return m
end

function fix_shortest_path(r::GridRoom, p::Vector{Int64})::GridRoom
    # no path to fix
    isempty(p) && return r
    # info of current room
    ent = first(entrance(r))
    ext = first(exits(r))
    g = deepcopy(pathgraph(r))
    # an empty room for reference
    ref_g = @>> r clear_room pathgraph
    # distances from entrance
    dm = noisy_distm(r, 0.1)
    ds = dijkstra_shortest_paths(g, ext, dm, allpaths=true, trackvertices=true)

    # matrices representing obstacles and paths
    omat = Matrix{Bool}(data(r) .== obstacle_tile)
    pmat = Matrix{Bool}(falses(steps(r)))
    pmat[p] .= true

    saturated = Vector{Bool}(falses(length(p)))
    s = MoveDeque()
    np = length(p)
    i = 1
    @inbounds while i <= np
        # @show i => saturated[i]
        if saturated[i]
            i += 1
            continue
        end
        x = p[i]
        y::Int64 = find_invalid_path(omat, g, x, ds.dists, pmat)

        if y == 0
            saturated[i] = true
            i += 1
        else
            # not able to fix further up the path
            if i == 1 || saturated[i-1]
                # reset saturation
                saturated[:] .= false
                # clear later placements
                # in case they are no longer needed
                j,v,o = isempty(s) ? (i, x, y) : last(s)
                # @show (i, x, y)
                # @show s
                while j > i && !isempty(s)
                    # @show (j, v, o)
                    pop!(s)
                    for n = collect(neighbors(ref_g, o))
                        # only reconnect if open
                        omat[n] || add_edge!(g, Edge(o, n))
                    end
                    omat[o] = false
                    (j, v, o) = isempty(s) ? (i,x,y) : last(s)
                end
                
                # add block move
                omat[y] = true
                for ny in collect(neighbors(g, y))
                    rem_edge!(g, Edge(y, ny))
                end
                push!(s, (i, x, y))
                # synchronize distances
                dm = noisy_distm(r, 0.1)
                ds = dijkstra_shortest_paths(g, ext, dm, allpaths=true, trackvertices=true)
            end
            # restart
            i = 1
        end
    end
    new_r::GridRoom = @>> omat vec findall Set add(r)
end

include("gen.jl")