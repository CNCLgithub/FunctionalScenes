export quad_tree_path

"""
Given a trace, returns the objective over paths
"""
function quad_tree_path(tr::Gen.Trace)
    get_reval(tr).pg
end
