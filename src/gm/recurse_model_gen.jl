export quad_tree

@gen (static) function qt_production(n::QTNode)
    w = produce_weight(n)
    s = @trace(bernoulli(w), :produce)
    children::Vector{QTNode} = s ? produce_qt(n) : QTNode[]
    result = Production(n, children)
    return result
end

@gen (static) function qt_aggregation(n::QTNode,
                                      children::Vector{QTState})
    y = @trace(uniform(0., 1.), :mu)
    agg_state::QTState = aggregate_qt(n, y, children)
    return agg_state
end

const quad_tree = Recurse(qt_production,
                          qt_aggregation,
                          4, # quad tree only can have 4 children
                          QTNode,  # U (passed from production to its children)
                          QTNode,  # V (passed from production to aggregation)
                          QTState) # W (passed from aggregation to its parents)
