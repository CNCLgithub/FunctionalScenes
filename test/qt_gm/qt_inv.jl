using Gen
using JSON
using FunctionalScenes
import Gen: get_child

@gen function qt_split_merge_proposal(t::Gen.Trace, i::Int64)
    head::QTAggNode = get_retval(t)
    st::QTAggNode = FunctionalScenes.traverse_qt(head, i)
    w = FunctionalScenes.split_weight(st)
    @debug "proposal on node $(st.node.tree_idx)"
    @debug "split prob $(w)"
    # refine or coarsen?
    split = {:split} ~ bernoulli(w)
    if split
        # refer to tree_idx since st could
        # be from parent in "merge" backward (split)
        mu = t[(st.node.tree_idx, Val(:aggregation)) => :mu]
        {:split_kernel} ~ split_kernel(mu)
    end
    split ? split_move : merge_move
end

function qt_sm_inv_manual(t, u, uret, uarg)

    node = first(uarg)
    # populate constraints
    constraints = choicemap()
    bwd = choicemap()
    bwd[:split] = !u[:split]
    if u[:split]
        constraints[(node, Val(:production)) => :produce] = true
        println("splitting node $(node)")
        dof  = 4.0 * t[(node, Val(:aggregation)) => :mu]
        for i = 1:3
            c_mu = u[:split_kernel => :steps => i => :mu]
            dof -= c_mu
            cid = get_child(node, i, 4)
            println("assigning node $(cid) -> mu $(c_mu)")
            constraints[(cid, Val(:aggregation)) => :mu] = c_mu
            constraints[(cid, Val(:production)) => :produce] = false
        end
        cid = get_child(node, 4, 4)
        constraints[(cid, Val(:aggregation)) => :mu] = dof
        constraints[(cid, Val(:production)) => :produce] = false
    else
        mu = 0
        for i = 1:4
            cid = Gen.get_child(node, i, 4)
            c_mu = t[(cid, Val(:aggregation)) => :mu]
            mu += c_mu
            if i < 4
                bwd[:split_kernel => :steps => i => :mu] = c_mu
            end
        end
        constraints[(node, Val(:aggregation)) => :mu] = mu * 0.25
        constraints[(node, Val(:production)) => :produce] = false
    end

    # obtain new trace and discard, which contains the previous subtree
    (new_trace, weight, _, discard) = update(t, constraints)

    (new_trace, bwd, weight)
end


function mytest()


    # testing involution on prior
    center = zeros(2)
    dims = [1., 1.]
    max_level = 3
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

    node = 3
    translator = SymmetricTraceTranslator(qt_split_merge_proposal,
                                          (node,),
                                          qt_sm_inv_manual)
    (new_trace, w) = translator(trace)



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
