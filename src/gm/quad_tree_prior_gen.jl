export quad_tree_prior

@gen (static) function qt_production(n::QTProdNode)
    w = produce_weight(n)
    s = @trace(bernoulli(w), :produce)
    children::Vector{QTProdNode} = s ? produce_qt(n) : QTProdNode[]
    result = Production(n, children)
    return result
end

@gen function qt_aggregation(n::QTProdNode,
                             children::Vector{QTAggNode})
    local mu
    if isempty(children)
        # prior sharpens with node depth
        w = 1.0 - 0.2 * (n.level - 1)
        clamp(w, 0.2, 0.99)
        mu = @trace(beta(w, w), :mu)
        # mu = @trace(uniform(0., 1.), :mu)
    else
        mu = mean(weight.(children))
    end

    agg::QTAggNode = aggregate_qt(n, mu, children)
    return agg
end

const quad_tree_prior = Recurse(qt_production,
                                qt_aggregation,
                                4, # quad tree only can have 4 children
                                QTProdNode,# U (production to children)
                                QTProdNode,# V (production to aggregation)
                                QTAggNode) # W (aggregation to parents)
