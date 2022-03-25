export furnish, furniture_gm

@gen static function prod_grow_furniture(state::GrowState)
    ns = neighboring_candidates(state)
    # assuming uniform weights
    ws = safe_uniform_weights(ns)
    c = @trace(categorical(ws), :new_head)
    # ni == 1: stop
    # n > 1: head = n - 1
    new_state::GrowState = GrowState(ns, c - 1, state)
    result = Production(new_state, fill(new_state, c != 1))
    return result
end

@gen static function agg_grow_furniture(st::GrowState,
                                 children)
                                          # children::Vector{Set{Int64}})
    result = merge_prod(st, children)
    return result
end

function grow_rec(max_size::Int64)
    Recurse(prod_grow_furniture,
            agg_grow_furniture,
            max_size,
            GrowState,
            GrowState,
            Set{Int64})
end

const fixed_grow = grow_rec(10);

"""
Randomly samples a new piece of furniture
"""
@gen static function furnish(r::GridRoom,
                             vmap::PersistentVector{Bool},
                             max_depth::Int64)

    # first sample a vertex to add furniture to
    # - baking in a prior about number of immediate neighbors
    g = pathgraph(r)
    # annoying type instability...
    # passed = PersistentVector(vmap .& valid_spaces(r))
    passed = valid_spaces(r, vmap)
    # if !any(passed)
    #     return Furniture()
    # end
    np = sum(passed)
    ws = (1.0 / np) * passed
    head = @trace(categorical(ws), :head)
    v = assoc(passed, head, false)
    gs = GrowState(head, v, g, 1, max_depth)
    tree = @trace(fixed_grow(gs, 1), :tree)
    result = union(tree, head)
    return result
end



@gen static function prod_furnish(st::FurnishState)
    sampled_f = @trace(furnish(st.template, st.vm, st.max_size),
                       :furniture)
    new_state = FurnishState(st, sampled_f)
    c = new_state.max_count > new_state.count
    result = Production(new_state, fill(new_state, c))
    return result
end


@gen static function agg_furnish(st::FurnishState, children)
    result = merge_prod(st, children)
    return result
end


function furnish_rec(max_count::Int64)
    Recurse(prod_furnish,
            agg_furnish,
            max_count,
            FurnishState,
            FurnishState,
            Set{Int64})
end

const fixed_furnish = furnish_rec(20);

"""
Adds a randomly generated piece of furniture
"""
@gen (static) function furniture_gm(r::GridRoom, v::PersistentVector{Bool},
                                    max_count::Int64, max_size::Int64)
    fs = FurnishState(r, v, Set{Int64}(), 0, max_size, max_count)
    f = @trace(fixed_furnish(fs, 1), :collection)
    result::GridRoom = add(r, f)
    return result
end

# furniture_chain = Gen.Unfold(furniture_step)

# """
# Move a piece of furniture
# """
# @gen function reorganize(r::Room)
#     # pick a random furniture block, this will prefer larger pieces
#     g = pathgraph(r)
#     vs = @>> g vertices filter(v -> istype(g, v, :furniture))
#     n = length(vs)
#     ps = fill(1.0/n, n)
#     vi = @trace(categorical(ps), :block)
#     v = vs[vi]
#     f = connected(g, v)
#     # find the valid moves and pick one at random
#     # each move will be one unit
#     moves = valid_moves(r, f)

#     inds = CartesianIndices(steps(r))
#     move_probs = moves ./ sum(moves)
#     move_id = @trace(categorical(move_probs), :move)
#     move = move_map[move_id]
#     new_r = shift_furniture(r, f, move)
# end
