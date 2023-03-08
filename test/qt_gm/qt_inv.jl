using Gen
using JSON
using FunctionalScenes
import Gen: get_child, get_parent


function balanced_split_merge(t::Gen.Trace, tidx::Int64)::Bool
    head::QTAggNode = get_retval(t)
    # balanced if root node is terminal : Split <-> Merge
    tidx == 1 && return isempty(head.children)
    st = traverse_qt(head, tidx)
    # it's possible to not reach the node
    # in that case, not balanced?
    # REVIEW
    parent_idx = get_parent(tidx, 4)
    # st.node.tree_idx === tidx ||
    #     parent_idx === tidx \\
    #     return false
    @unpack level, max_level = st.node
    # cannot split or merge if max depth
    level == max_level && return false
    # balanced if node is terminal : Split <-> Merge
    # or if children are all terminal : Merge <-> Split
    isempty(st.children) || any(x -> !isempty(x.children), st.children)
end

"""
    split_weight(st::QTAggNode)

The probability of splitting node.
"""
function split_weight(st::QTAggNode)::Float64
    @unpack node, children = st
    @unpack level, max_level = node
    # cannot split further than max level
    level == max_level && return 0
    # no children -> split
    isempty(children) && return 1
    # if children but no gran children -> merge
    Float64(any(x -> !isempty(x.children), children))
end


@gen function qt_split_merge_proposal(t::Gen.Trace, i::Int64)
    head::QTAggNode = get_retval(t)
    st::QTAggNode = FunctionalScenes.traverse_qt(head, i)
    # `st` could be parent if t' is a result of merge
    # since the original `i` would have been merged with its
    # sibling in t
    ref_idx = st.node.tree_idx
    after_merge = ref_idx == get_parent(i, 4)
    println("target: $(i), actual; $(ref_idx)")
    @assert ref_idx == i || ref_idx == get_parent(i, 4)
    # assuming that `i` is referencing a "balanced" node
    # splitting
    if isempty(st.children)
        # refer to tree_idx since st could
        # be from parent in "merge" backward (split)
        mu = t[(ref_idx, Val(:aggregation)) => :mu]
        {:split_kernel} ~ split_kernel(mu)
    end
    ref_idx
end

function qt_sm_inv_manual(t, u, uret, uarg)

    node = first(uarg)
    # populate constraints
    constraints = choicemap()
    bwd = choicemap()
    bwd[:split] = !u[:split]
    if u[:split]
        # split node
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
        # Merge all siblings of `node`
        parent = get_parent(node, 4)
        mu = 0
        for i = 1:4
            cid = Gen.get_child(parent, i, 4)
            c_mu = t[(cid, Val(:aggregation)) => :mu]
            mu += c_mu
            if i < 4
                bwd[:split_kernel => :steps => i => :mu] = c_mu
            end
        end
        constraints[(parent, Val(:aggregation)) => :mu] = mu * 0.25
        constraints[(parent, Val(:production)) => :produce] = false
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
