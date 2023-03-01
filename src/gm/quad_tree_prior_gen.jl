export quad_tree_prior

@gen (static) function qt_production(n::QTNode)
    w = produce_weight(n)
    s = @trace(bernoulli(w), :produce)
    children::Vector{QTNode} = s ? produce_qt(n) : QTNode[]
    result = Production(n, children)
    return result
end

@gen function qt_aggregation(n::QTNode,
                             children::Vector{QTState})
    local mu
    if isempty(children)
        mu = @trace(uniform(0., 1.), :mu)
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
