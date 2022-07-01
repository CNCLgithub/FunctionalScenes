export PathState, recurse_path_gm, fix_shortest_path

using DataStructures: Stack

struct PathState
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
    ent = first(entrance(r))
    ext = first(exits(r))
    g = pathgraph(r)
    gs = 1.0 ./ gdistances(g, ent)
    PathState(g, gs, ext, temp, 0)
end

function PathState(new_head::Int64, st::PathState)
    PathState(st.g, st.gs, new_head, st.temp, st.step + 1)
end

function head_weights(st::PathState, ns)
    @unpack gs, temp, step = st
    ds = gs[ns]
    ws = softmax(ds; t = temp / (0.1 * step))
end

function find_step(path::Pair{Int64, Int64}, k::Int64)::Pair
    path
end

function find_step(path::Pair, k::Int64)::Pair
    # found tile           | keep searching
    path.first == k ? path : find_step(path.second, k)
end

function merge_grow_path(st::PathState, children)::Pair
    @unpack head = st
    # no children, then we are at the end of the path
    isempty(children) && return head => 0
    # otherwise will have 1 child
    path::Pair = first(children)
    # does the head tile close a loop in path?
    step::Pair = find_step(path, head)
    # remove the loop | add to path
    step.first == head ? step : (head => path)
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
        if gs[x] >= gs[n]
            y = n
            break
        end
    end
    return y
end

const MoveDeque = Vector{Tuple{Int64, Int64, Int64}}

function fix_shortest_path(r::GridRoom, p::Vector{Int64})::GridRoom
    isempty(p) && return r
    ent = first(entrance(r))
    ext = first(exits(r))
    g = deepcopy(pathgraph(r))
    ref_g = @>> r clear_room pathgraph
    gs = gdistances(g, ent)
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
        y::Int64 = find_invalid_path(omat, g, x, gs, pmat)
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
                gs = gdistances(g, ent)
            end
            # restart
            i = 1
        end
    end
    new_r::GridRoom = @>> omat vec findall Set add(r)
end

include("gen.jl")
