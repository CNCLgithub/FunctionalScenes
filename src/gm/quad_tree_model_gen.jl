export qt_model

#################################################################################
# Generative Model
#################################################################################
@gen function qt_model(params::QuadTreeModel)
    # initialize trackers
    qt = {:trackers} ~ quad_tree_prior(params.start_node, 1)
    # leaf nodes used for graphics and planning
    leaves::Vector{QTAggNode} = leaf_vec(qt)

    # a global room matrix
    # this is no longer necessary but useful for visualization
    projected = project_qt(leaves, params.dims)

    # mean and variance of observation
    viz = stats_from_qt(leaves, params)
    pred = @trace(broadcasted_normal(viz[1], viz[2]), :viz)

    # shortest path given qt uncertainty
    qtpath::QTPath = qt_a_star(leaves, params.dims[1],
                               params.entrance,
                               params.exit)

    result::QuadTreeState = QuadTreeState(qt, projected,
                                          viz[1], qtpath, leaves)
    return result
end
