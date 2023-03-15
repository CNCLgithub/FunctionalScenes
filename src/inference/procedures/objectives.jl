export quad_tree_path,
    qt_path_cost

"""
Given a trace, returns the objective over paths
"""
function quad_tree_path(tr::Gen.Trace)
    get_retval(tr).path
end

function qt_path_cost(tr::Gen.Trace)::Float64
    qt_path = quad_tree_path(tr)
    c = 0
    for e in qt_path.edges
        c += qt_path.dm[src(e), dst(e)]
    end
    return c
end
