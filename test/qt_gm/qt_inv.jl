using Gen
using Parameters
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

# """
#     split_weight(st::QTAggNode)

# The probability of splitting node.
# """
# function split_weight(st::QTAggNode)::Float64
#     @unpack node, children = st
#     @unpack level, max_level = node
#     # cannot split further than max level
#     level == max_level && return 0
#     # no children -> split
#     isempty(children) && return 1
#     # if children but no gran children -> merge
#     Float64(any(x -> !isempty(x.children), children))
# end

@gen function split_step(i::Int64,
                         temp::Float64,
                         n::Int64)
    hi::Float64 = min(1.0, temp)
    lo::Float64 = max(0.0, temp - (n - i))
    u_i::Float64 = {:mu} ~ uniform(lo, hi)
    res::Float64 = temp - u_i
    return res
end

@gen function split_kernel(mu::Float64)
    n = 4
    temp::Float64 = mu * n
    mus = {:steps} ~ Gen.Unfold(split_step)(3, temp, n)
    return mus
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
    split = isempty(st.children)
    # splitting
    if split
        # refer to tree_idx since st could
        # be from parent in "merge" backward (split)
        mu = t[(ref_idx, Val(:aggregation)) => :mu]
        {:split_kernel} ~ split_kernel(mu)
    end
    return split
end


@transform qt_involution (t, u) to (t_prime, u_prime) begin

    # Retrieve node and move type
    _, node = get_args(u)
    split = @read(u[], :discrete)

    if split
        # splitting node
        # update `t_prime` with samples from `u`
        # no backward choices (`u_prime`) as merge is deterministic
        @write(t_prime[(node, Val(:production)) => :produce],
               true, :discrete)
        mu = @read(t[(node, Val(:aggregation)) => :mu], :continuous)
        dof = 0.0
        # assigning to first 3 children
        for i = 1:3
            c_mu = @read(u[:split_kernel => :steps => i => :mu], :continuous)
            dof += c_mu
            cid = Gen.get_child(node, i, 4)
            @write(t_prime[(cid, Val(:aggregation)) => :mu],
                   c_mu, :continuous)
            @write(t_prime[(cid, Val(:production)) => :produce],
                   false, :discrete)
        end
        # computing residual for 4th child
        res = 4 * mu - dof
        cid = Gen.get_child(node, 4, 4)
        @write(t_prime[(cid, Val(:aggregation)) => :mu],
               res, :continuous)
        @write(t_prime[(cid, Val(:production)) => :produce],
                false, :discrete)

    else
        # Merge all children of `node`
        # update `t_prime` with the average of children in `t`
        # backward `u_prime` contains the original children in `t`
        mu = 0
        for i = 1:3
            cid = get_child(node, i, 4)
            @show cid
            cmu =  @read(t[(cid, Val(:aggregation)) => :mu],
                         :continuous)
            mu += cmu
            @write(u_prime[:split_kernel => :steps => i => :mu],
                cmu, :continuous)
        end
        cid = get_child(node, 4, 4)
        cmu =  @read(t[(cid, Val(:aggregation)) => :mu],
                        :continuous)
        mu += cmu
        mu *= 0.25
        @write(t_prime[(node, Val(:aggregation)) => :mu],
               mu, :continuous)
        @write(t_prime[(node, Val(:production)) => :produce],
               false, :discrete)
    end
end

function qt_sm_inv_manual(t, u, uret, uarg)

    node = first(uarg)
    split = uret
    # populate constraints
    constraints = choicemap()
    bwd = choicemap()
    if split
        println("Split involution")
        # split node
        constraints[(node, Val(:production)) => :produce] = true
        println("splitting node $(node)")
        resid  = 4.0 * t[(node, Val(:aggregation)) => :mu]
        # 3 children are sampled randomly
        for i = 1:3
            c_mu = u[:split_kernel => :steps => i => :mu]
            resid -= c_mu
            cid = get_child(node, i, 4)
            println("assigning node $(cid) -> mu $(c_mu)")
            constraints[(cid, Val(:aggregation)) => :mu] = c_mu
            constraints[(cid, Val(:production)) => :produce] = false
        end
        # 4th child is the residual
        cid = get_child(node, 4, 4)
        constraints[(cid, Val(:aggregation)) => :mu] = resid
        constraints[(cid, Val(:production)) => :produce] = false
    else
        println("Merge involution")
        # Merge all children of `node`
        mu = 0
        for i = 1:4
            cid = Gen.get_child(node, i, 4)
            @show cid
            c_mu = t[(cid, Val(:aggregation)) => :mu]
            mu += c_mu
            if i < 4
                bwd[:split_kernel => :steps => i => :mu] = c_mu
            end
        end
        constraints[(node, Val(:aggregation)) => :mu] = mu * 0.25
        constraints[(node, Val(:production)) => :produce] = false
    end

    # display(constraints)
    # display(bwd)
    # obtain new trace and discard, which contains the previous subtree
    (new_trace, weight, _, discard) = update(t, constraints)
    # @show weight
    # @show discard
    (new_trace, bwd, weight)
end


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
