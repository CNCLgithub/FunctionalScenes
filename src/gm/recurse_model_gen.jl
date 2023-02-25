export quad_tree

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

    agg_state::QTState = aggregate_qt(n, mu, children)
    return agg_state
end

const quad_tree = Recurse(qt_production,
                          qt_aggregation,
                          4, # quad tree only can have 4 children
                          QTNode,  # U (passed from production to its children)
                          QTNode,  # V (passed from production to aggregation)
                          QTState) # W (passed from aggregation to its parents)
