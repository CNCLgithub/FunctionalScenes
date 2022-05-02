export qt_model

#################################################################################
# Generative Model
#################################################################################

@gen (static) function tile_flip(p::Float64)::Bool
    f::Bool = @trace(bernoulli(p), :flip)
    return f
end


"""
Given a global state, sample furniture according to bernoulli weights
"""
@gen (static) function room_from_gs(state::Matrix{Float64},
                                       params::QuadTreeModel)
    occupied = {:furniture} ~ Gen.Map(tile_flip)(state)
    result::GridRoom = add_from_state_flip(params, occupied)
    return result
end

@gen (static) function qt_model(params::QuadTreeModel)
    # initialize trackers
    qt = {:trackers} ~ quad_tree(params.start_node, 1)

    # a global room matrix
    global_state = consolidate_qt_states(params, qt)
    (gs, ps) = room_from_state_args(params, global_state)

    # empirical estimation of multigranular predictions
    instances = {:instances} ~ Gen.Map(room_from_gs)(gs, ps)

    # mean and variance of observation
    viz = graphics_from_instances(instances, params)
    pred = @trace(broadcasted_normal(viz[1], viz[2]), :viz)

    result::QuadTreeState = QuadTreeState(qt, global_state, instances)
    return result
end
