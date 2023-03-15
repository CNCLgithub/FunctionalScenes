export rw_move

@gen function qt_node_random_walk(t::Gen.Trace, i::Int64)
    addr = :trackers => (i, Val(:aggregation)) => :mu
    mu::Float64 = t[addr]
    low::Float64 = max(0., mu - 0.05)
    high::Float64 = min(1., mu + 0.05)
    # @show (low, high)
    {addr} ~ uniform(low, high)
end

function rw_move(t::Gen.Trace, i::Int64)
    (new_trace, w1) = apply_random_walk(t, qt_node_random_walk, (i,))
    # downstream = downstream_selection(no_change, t, i)
    # used to update dowstream hcoices
    # (new_trace, w2) = regenerate(new_trace, downstream)
    # (new_trace, w1 + w2)
end

function rw_move(::NoChange, t::Gen.Trace, i::Int64)
    rw_move(t, i)
end

function rw_move(::Split, tr::Gen.Trace, node::Int64)
    nt = tr
    result = 0.0
    for i = 1:4
        nt, w = rw_move(nt, Gen.get_child(node, i, 4))
        result += w
    end
    (nt, result)
end
function rw_move(::Merge, tr::Gen.Trace, node::Int64)
    rw_move(tr, Gen.get_parent(node, 4))
end
