export model

## Helpers

@gen (static) function flip(p::Float64)::Bool
    f = @trace(bernoulli(p), :flip)
    return f
end

@gen (static) function stability(ab::Tuple)
    bounds, probs = ab
    bw = @trace(uniform(bounds, probs), :sflip)
    return bw
end

@gen (static) function tracker_prior(params::ModelParams)
    level = @trace(categorical(params.level_weights), :level)
    # a vec of tuples for each tile
    lp = level_prior(params, level)
    state = @trace(broadcasted_uniform(lp), :state)
    return (level, state)
end

"""
Take a room and place a random set of objects in an xy plane
"""
@gen (static) function room_from_state(state::Matrix{Float64},
                                       params::ModelParams)::Room
    occupied = @trace(broadcasted_bernoulli(state),
                      :furniture)
    result = add_from_state_flip(params, occupied)
    return result
end

## Model

@gen (static) function model(params::ModelParams)
    # initialize trackers
    prior_args = tracker_prior_args(params)
    states = @trace(Gen.Map(tracker_prior)(prior_args), :trackers)

    # a global room matrix
    global_state = consolidate_local_states(params, states)
    (gs, ps) = room_from_state_args(params, global_state)
    room_instances = @trace(Gen.Map(room_from_state)(gs, ps),
                            :instances)
    result = (global_state, room_instances)

    # mean and variance of observation
    viz = graphics_from_instances(room_instances, params)
    pred = @trace(broadcasted_normal(viz[1], viz[2]), :viz)

    return result
end
