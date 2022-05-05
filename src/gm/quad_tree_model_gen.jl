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
@gen (static) function obst_gen(state::Matrix{Float64})
    occupied = {:obstacle} ~ Gen.Map(tile_flip)(state)
    # result::GridRoom = add_from_state_flip(params, occupied)
    return occupied
end

@gen (static) function qt_model(params::QuadTreeModel)
    # initialize trackers
    qt = {:trackers} ~ quad_tree(params.start_node, 1)

    # a global room matrix
    global_state = consolidate_qt_states(params, qt)

    # empirical estimation of multigranular predictions
    obstacles = {:instances} ~ Gen.Map(obst_gen)(fill(global_state,
                                                      params.instances))
    instances = instances_from_gen(params, obstacles)

    # mean and variance of observation
    viz = graphics_from_instances(instances, params)
    pred = @trace(broadcasted_normal(viz[1], viz[2]), :viz)

    # shortest path given qt uncertainty
    spath::Matrix{Bool}, lv::Vector{QTState} = qt_a_star(qt, params.dims[1],
                                                         params.entrance,
                                                         params.exit)

    result::QuadTreeState = QuadTreeState(qt, global_state, instances,
                                          spath, lv)
    return result
end
