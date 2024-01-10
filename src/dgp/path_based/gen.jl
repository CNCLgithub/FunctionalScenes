@gen static function prod_grow_path(st::PathState)
    ns, ws = head_weights(st)
    c = @trace(categorical(ws), :new_head)
    new_head  = ns[c]
    new_state::PathState = PathState(new_head, st)
    d = st.gs[new_head]
    result = Production(new_state, fill(new_state, !isinf(d)))
    return result
end

@gen static function agg_grow_path(st::PathState,
                                   children)
    result = merge_grow_path(st, children)
    return result
end

const recurse_path = Recurse(prod_grow_path,
            agg_grow_path,
            1,               # max 1 child
            PathState,       # U (passed from production to its children)
            PathState,       # V (passed from production to aggregation)
            Pair)            # W (passed from aggregation to its parents)



@gen static function recurse_path_gm(r::GridRoom,
                                     temp::Function)
    x = PathState(r; temp = temp)
    y = @trace(recurse_path(x, 1), :steps)
    path::Vector{Int64} = parse_recurse_path(y)
    return path
end
