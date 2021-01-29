
@gen (static) function flip(p::Float64)::Bool
    f = @trace(bernoulli(p), :flip)
    return f
end

flip_map = Gen.Map(flip)

@gen (static) function beta_flip(ab::Tuple{Float64, Float64})
    a, b = ab
    result = @trace(beta(a, b), :bflip)
    return result
end

beta_map = Gen.Map(beta_flip)

@gen (static) function tracker_state(beta_weights)
    bws = @trace(beta_map(beta_weights), :state)
    return bws
end

tracker_state_map = Gen.Map(tracker_state)

"""
Take a room and place a random set of objects in an xy plane
"""
@gen (static) function room_from_state(state::Vector{Float64},
                                       params::ModelParams)::Room
    occupied = @trace(flip_map(state),  :furniture)
    result = add_from_state_flip(params, occupied)
    return result
end

room_map = Gen.Map(room_from_state)

@gen (static) function model(params::ModelParams)
    # sampler trackers
    active = @trace(flip_map(params.tracker_ps), :active)
    trackers = define_trackers(params, active)
    # a collection of local states
    states = @trace(tracker_state_map(trackers), :trackers)
    # a global room matrix
    global_state = consolidate_local_states(states)
    gs = fill(global_state, params.instances)
    ps = fill(params, params.instances)

    room_instances = @trace(room_map(gs, ps), :instances)

    # mean and variance of observation
    viz = graphics_from_instances(room_instances, params)
    # viz = (mean, cov diagonal)
    @trace(broadcasted_normal(viz[1], viz[2]), :viz)

    # compute in sensitivity for effeciency
    # affordances = map(afforandace, room_instances, ps)

    return room_instances
end


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


export furnish, furniture_step, furniture_chain, reorganize, model
