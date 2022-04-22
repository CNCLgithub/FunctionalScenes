using Gen
using FunctionalScenes
# using Graphs
using GraphRecipes
using Plots
using FunctionalScenes: QTNode, qt_production, qt_aggregation, quad_tree, graph_from_qt

function mytest()
    center = zeros(2)
    dims = [32., 32.]
    max_level = 3
    start_node = QTNode(center, dims, 1, max_level)
    display(start_node)
    trace, ls = Gen.generate(qt_production, (start_node,))
    trace, ls = Gen.generate(qt_aggregation, (start_node, QTState[]))
    # trace, ls = Gen.generate(fixed_qt_rec, (start_node, 2))
    trace, ls = Gen.generate(quad_tree, (start_node, 1))
    @time trace, ls = Gen.generate(quad_tree, (start_node, 1))
    st = get_retval(trace)
    # display(st)
    g = graph_from_qt(st)
    # display(get_choices(trace))
    display(g)
    display(graphplot(g, method = :buchheim))
    return nothing
end

mytest();
