export furnish, furniture_step, furniture_chain, reorganize

@gen static function prod_grow_furniture(state::GrowState)
    @unpack head, vm, g = state
    ns = neighboring_candidates(vm, g, head)
    # assuming uniform weights
    ws = safe_uniform_weights(ns)
    c = @trace(categorical(ws), :new_head)
    # ni == 1: stop
    # n > 1: head = n - 1
    new_state::GrowState = GrowState(ns, ni - 1, state)
    result = Production(new_state, fill(new_state, ni != 1))
    return result
end

@gen function agg_grow_furniture(st::GrowState, children::Vector{GrowState})
    # no children, at max depth
    if length(children) == 0
        result = st
    # simply carry the deepest state to the top
    else
        result = first(children)
    end
end

function grow_furniture_gm(max_depth::Int64)
    Recurse(prod_grow_furniture,
            agg_grow_furniture,
            max_depth, Vector{GrowState}, GrowState, GrowState)
end

"""
Randomly samples a new piece of furniture
"""
@gen function furnish(r::GridRoom, vmap::PersistentVector{Bool},
                      max_depth::Int64)

    # first sample a vertex to add furniture to
    # - baking in a prior about number of immediate neighbors
    g = pathgraph(r)
    candidates = valid_spaces(r)
    passed = vmap[candidates]
    if !any(passed)
        return Furniture()
    end
    np = sum(passed)
    ws = (1.0 / np) * vmap
    vi = @trace(categorical(ws), :head)
    head = candidates[vi]

    gs = GrowState(head, vmap, g)

    result = @trace(grow_furniture_gm(max_depth)(gs), :tree)
    return result
end

# """
# Adds a randomly generated piece of furniture
# """
# @gen (static) function furniture_step(t::Int, r::Room, weights::Matrix{Float64})
#     f = @trace(furnish(r, weights), :furniture)
#     new_r = add(r, f)
#     return new_r
# end

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
