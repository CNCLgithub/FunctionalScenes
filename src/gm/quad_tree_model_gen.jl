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
    return occupied
end


@gen function qt_model(params::QuadTreeModel)
    # initialize trackers
    qt = {:trackers} ~ quad_tree_prior(params.start_node, 1)
    # leaf nodes used for graphics and planning
    leaves::Vector{QTAggNode} = leaf_vec(qt)

    # a global room matrix
    global_state = project_qt(leaves, params.dims)

    # REVIEW: Maybe this doesn't need to be tracked?
    # empirical estimation of multigranular predictions
    # obstacles = {:instances} ~ Gen.Map(obst_gen)(fill(global_state,
    #                                                   params.instances))
    obstacles = Gen.Map(obst_gen)(fill(global_state,
                                       params.instances))
    instances = instances_from_gen(params, obstacles)

    # mean and variance of observation
    viz = stats_from_instances(instances, params)
    pred = @trace(broadcasted_normal(viz[1], viz[2]), :viz)

    # shortest path given qt uncertainty
    qtpath::QTPath = qt_a_star(leaves, params.dims[1],
                               params.entrance,
                               params.exit)

    result::QuadTreeState = QuadTreeState(qt, global_state, instances,
                                          viz[1], qtpath, leaves)
    return result
end
