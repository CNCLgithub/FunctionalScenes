export model

#################################################################################
# Helpers
#################################################################################

@gen (static) function tile_flip(p::Float64)::Bool
    f = @trace(bernoulli(p), :flip)
    return f
end


#################################################################################
# Generative Model
#################################################################################

"""
Sample a level and tiles for a tracker.
Here `state` is a mxm matrix of bernoulli weights
"""
@gen (static) function tracker_prior(params::ModelParams, tid::Int64)
    level = {:level} ~ categorical(params.level_weights)
    # Sample state for each tile in the tracker
    lp = level_prior(params, tid, level)
    state = {:state} ~ broadcasted_uniform(lp)
    return (level, state)
end

"""
Given a global state, sample furniture according to bernoulli weights
"""
@gen (static) function room_from_state(state::Array{Float64, 3},
                                       params::ModelParams)::Room
    cleaned = clean_state(state)
    occupied = {:furniture} ~ Gen.Map(tile_flip)(cleaned)
    result = add_from_state_flip(params, occupied)
    return result
end

#################################################################################
# Generative Model
#################################################################################

@gen (static) function model(params::ModelParams)
    # initialize trackers
    (prior_args, tids) = tracker_prior_args(params)
    # a matrix of trackers
    states = {:trackers} ~ Gen.Map(tracker_prior)(prior_args, tids)

    # a global room matrix
    global_state = consolidate_local_states(params, states)
    (gs, ps) = room_from_state_args(params, global_state)

    # empirical estimation of multigranular predictions
    room_instances = {:instances} ~ Gen.Map(room_from_state)(gs, ps)

    # mean and variance of observation
    viz = graphics_from_instances(room_instances, params)
    pred = @trace(broadcasted_normal(viz[1], viz[2]), :viz)

    result = (collect(Matrix{Float64}, states), # tracker matrix
              global_state, # room matrix (for rendering?)
              room_instances) # empirical distribution
    return result
end
