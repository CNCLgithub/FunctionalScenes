export PathState, recurse_path_gm

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
    gs = 1.0 ./ gdistances(g, ext)
    PathState(g, gs, ent, temp, 0)
end

function PathState(new_head::Int64, st::PathState)
    PathState(st.g, st.gs, new_head, st.temp, st.step + 1)
end

function head_weights(st::PathState, ns)
    @unpack gs, temp, step = st
    ds = gs[ns]
    ws = softmax(ds; t = temp / step)
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

include("gen.jl")
