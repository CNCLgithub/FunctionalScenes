using Gen
using FunctionalScenes
# using Graphs
# using GraphRecipes
# using Plots
using FunctionalScenes: QTNode, qt_production, qt_aggregation, quad_tree

function mytest()
    center = zeros(2)
    dims = [1., 1.]
    max_level = 3
    start_node = QTNode(center, dims, 1, max_level, 1)
    display(start_node)
    trace, ls = Gen.generate(qt_production, (start_node,))
    trace, ls = Gen.generate(qt_aggregation, (start_node, QTState[]))
    # trace, ls = Gen.generate(fixed_qt_rec, (start_node, 2))
    trace, ls = Gen.generate(quad_tree, (start_node, 1))


    cm = choicemap()
    fwd = choicemap()
    cm[(1, Val(:production)) => :produce] = true
    for i = 2 : 5
        cm[(i, Val(:production)) => :produce] = false
        fwd[(i, Val(:aggregation)) => :mu] = 0.5
    end
    @time trace, ls = Gen.generate(quad_tree, (start_node, 1), cm)
    st = get_retval(trace)
    # display(st)
    # g = graph_from_qt(st)
    # gs = Matrix{Float64}(undef, 32, 32)
    # FunctionalScenes.consolidate_qt_states!(gs, st)
    # display(gs)

    display(qt_a_star(st, 4, 2, 14))


    (new_trace, weight, retdiff, discard) = update(trace, fwd)
    display(weight)
    display(get_choices(new_trace))
    display(discard)
    # display(g)
    # display(graphplot(g, method = :buchheim))
    return nothing
end

mytest();
