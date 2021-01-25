
# @gen (static) function occupancy_flip(p::Float64)
#     f = @trace(bernoulli(p), :flip)
#     return f
# end

# occupancy_map = Gen.Map(occupancy_flip)

# """
# Take a room and place a random set of objects in an xy plane
# """
# @gen (static) function furniture_prior(r::Room, rate::Float64)
#     floor_tiles = valid_spaces(r)
#     ps = fill(rate, length(floor_tiles))
#     occupied = @trace(occupancy_map(ps),  :occupancy)
#     result = add_from_grid(r, floor_tiles, occupied)
#     return result
# end

# @gen (static) function model(params::ModelParams)
#     #prior
#     room = @trace(furniture_prior(params.template, params.rate),
#                   :prior)
#     c3_means = c3_prediction(room, params.features, params.graphics)
#     @trace(broadcasted_normal(c3_means, params.obs_noise), :c3)

#     return c3_means
# end


"""
Randomly samples a new piece of furniture
"""
@gen function furnish(r::Room, weights::Matrix{Float64})

    # first sample a vertex to add furniture to
    # - baking in a prior about number of immediate neighbors
    g = pathgraph(r)
    candidates = valid_spaces(r)
    nc = length(candidates)
    ws = weights[candidates]
    if isempty(ws) || iszero(sum(ws))
        return Set(Tile[])
    end
    # ws = fill(1.0 / nc, nc)
    ws = ws ./ sum(ws)
    vi = @trace(categorical(ws), :vertex)
    v = candidates[vi]
    ns = neighbors(g, v)
    nns = length(ns)
    # then pick a subset of neighbors if any
    # defined as a mbrfs
    p = 1.0 / nns
    mbrfs = map(n -> BernoulliElement{Any}(p, id, (n,)), ns)
    mbrfs = RFSElements{Any}(mbrfs)
    others = @trace(rfs(mbrfs), :neighbors)
    f = Set{Tile}(vcat(v, others))
    return f
end

"""
Adds a randomly generated piece of furniture
"""
@gen (static) function furniture_step(t::Int, r::Room, weights::Matrix{Float64})
    f = @trace(furnish(r, weights), :furniture)
    new_r = add(r, f)
    return new_r
end

furniture_chain = Gen.Unfold(furniture_step)


"""
Move a piece of furniture
"""
@gen function reorganize(r::Room)
    # pick a random furniture block, this will prefer larger pieces
    g = pathgraph(r)
    vs = @>> g vertices filter(v -> istype(g, v, :furniture))
    n = length(vs)
    ps = fill(1.0/n, n)
    vi = @trace(categorical(ps), :block)
    v = vs[vi]
    f = connected(g, v)
    # find the valid moves and pick one at random
    # each move will be one unit
    moves = valid_moves(r, f)

    inds = CartesianIndices(steps(r))
    move_probs = moves ./ sum(moves)
    move_id = @trace(categorical(move_probs), :move)
    move = move_map[move_id]
    new_r = shift_furniture(r, f, move)
end


export furnish, furniture_step, furniture_chain, reorganize
