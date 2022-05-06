export quad_tree_path

"""
Given a trace, returns the objective over paths
"""
function quad_tree_path(tr::Gen.Trace)
    get_retval(tr).path
end

function qt_path_cost(tr::Gen.Trace)::Float64
    qt_path = quad_tree_path(tr)
    c = 0
    for i = 2:length(qt_path.vs)
        c += qt_path.dm[qt_path.vs[i-1], qt_path.vs[i]]
    end
    return c
end
