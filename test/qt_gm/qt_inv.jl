using Gen
using Parameters
using FunctionalScenes
import Gen: get_child, get_parent



function mytest()


    # testing involution on prior
    center = zeros(2)
    dims = [1., 1.]
    max_level = 5
    start_node = QTProdNode(center, dims, 1, max_level, 1)
    display(start_node)

    cm = choicemap()
    # root node has children
    cm[(1, Val(:production)) => :produce] = true
    for i = 1:4
        # only one child of root has children
        cm[(i+1, Val(:production)) => :produce] = i == 1
        # child of 2 should not reproduce
        cm[(Gen.get_child(2, i, 4), Val(:production)) => :produce] = false
    end
    (trace, ls) = Gen.generate(quad_tree_prior, (start_node, 1), cm)

    # 1 -> 2 | [3,4,5] -> 6 | [7,8,9]
    node = 6 # first child of node 2
    translator = SymmetricTraceTranslator(qt_split_merge_proposal,
                                          (node,),
                                          # qt_sm_inv_manual)
                                          qt_involution)
    # @time (new_trace, w) = translator(trace, check = true)
    @time (new_trace, w) = translator(trace, check = false)

    # display(get_choices(new_trace))

    @show w

    # room_dims = (16, 16)
    # entrance = [8,9]
    # exits = [16*16 - 8]
    # r = GridRoom(room_dims, room_dims, entrance, exits)
    # r = add(r, Set(16 * 8 + 8))
    # r = expand(r, 2)

    # params = QuadTreeModel(;gt = r)

    # cm = choicemap()
    # cm[:trackers => (1, Val(:production)) => :produce] = true
    # cm[:trackers => (3, Val(:aggregation)) => :mu] = 0.1
    # for i = 1:4
    #     cm[:trackers => (i+1, Val(:production)) => :produce] = i == 1
    #     cm[:trackers => (Gen.get_child(2, i, 4), Val(:production)) => :produce] = false
    # end
    # display(cm)

    # trace, ll = generate(qt_model, (params,), cm)
    # st = get_retval(trace)
    # FunctionalScenes.viz_room(st.instances[1])
    # c = FunctionalScenes.qt_path_cost(trace)
    # cm2 = choicemap()
    # cm2[:trackers => (3, Val(:aggregation)) => :mu] = 0.9
    # new_trace,_ = update(trace, cm2)
    # ds = FunctionalScenes.downstream_selection(new_trace, 3)
    # new_trace,_ = regenerate(new_trace, ds)
    # @show trace[:trackers => (3, Val(:aggregation)) => :mu]
    # @show new_trace[:trackers => (3, Val(:aggregation)) => :mu]
    # @show sum(sum(trace[:instances]))
    # @show sum(sum(new_trace[:instances]))

    # # new_trace, w, d = vertical_move(trace, 3)
    # st = get_retval(new_trace)
    # FunctionalScenes.viz_room(st.instances[1])
    # @show c
    # @show FunctionalScenes.qt_path_cost(trace) - c
    return nothing
end

mytest();
