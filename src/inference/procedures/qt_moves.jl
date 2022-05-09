export lateral_move, vertical_move


@gen function qt_node_random_walk(t::Gen.Trace, i::Int64)
    addr = :trackers => (i, Val(:aggregation)) => :mu
    mu::Float64 = t[addr]
    bounds::Vector{Float64} = [mu - 0.05, mu + 0.05]
    clamp!(bounds, 0., 1,)
    {addr} ~ uniform(bounds[1], bounds[2])
end

function lateral_move(t::Gen.Trace, i::Int64)
    (new_trace, w1) = apply_random_walk(t, qt_node_random_walk, (i,))
    downstream = downstream_selection(no_change, t, i)
    (new_trace, w2) = regenerate(new_trace, downstream)
    (new_trace, w1 + w2)
end

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

function split_weight(st::QTState)::Float64
    @unpack node, children = st
    @unpack level, max_level = node
    #  max level not balanced so is excluded from SM
    # root must split
    level == 1 && return 1.0
    # no children must split
    # if children, then assumed balanced and must merge
    Float64(isempty(st.children))
end

@gen function qt_split_merge_proposal(t::Gen.Trace, i::Int64)
    head::QTState = t[:trackers]
    st::QTState = traverse_qt(head, i)
    w = split_weight(st)
    @debug "proposal on node $(st.node.tree_idx)"
    @debug "split prob $(w)"
    # refine or coarsen?
    split = {:split} ~ bernoulli(w)
    if split
        # refer to tree_idx since st could 
        # be from parent in "merge" backward (split)
        mu = t[:trackers => (st.node.tree_idx, Val(:aggregation)) => :mu]
        {:split_kernel} ~ split_kernel(mu)
    end
    split ? split_move : merge_move
end

function adjust_reference(::Split, qargs::Tuple)
    # node = first(qargs)
    # (Gen.get_child(node, 1, 4),)
    qargs
end
function adjust_reference(::MoveDirection, qargs::Tuple)
    qargs
end

function my_inv(prev_model_trace, translator)
    forward_trace = simulate(translator.q, (prev_model_trace, translator.q_args...,))
    forward_score = get_score(forward_trace)
    forward_choices = get_choices(forward_trace)
    forward_retval::MoveDirection = get_retval(forward_trace)
    (new_model_trace, backward_choices, log_weight) = translator.involution(
        prev_model_trace, forward_choices, forward_retval, translator.q_args)
    new_q_args = adjust_reference(forward_retval, translator.q_args)
    (backward_score, backward_retval) =
        assess(translator.q, (new_model_trace, new_q_args...), backward_choices)
    log_weight += (backward_score - forward_score)
    return (new_model_trace, log_weight, forward_retval)
end

function qt_sm_inv_manual(t, u, uret, uarg)

    node = first(uarg)
    # populate constraints
    constraints = choicemap()
    bwd = choicemap()
    bwd[:split] = !u[:split]
    if u[:split]
        constraints[:trackers => (node, Val(:production)) => :produce] = true
        @debug "splitting node $(node)"
        dof  = 4.0 * t[:trackers => (node, Val(:aggregation)) => :mu]
        for i = 1:3
            c_mu = u[:split_kernel => :steps => i => :mu]
            dof -= c_mu
            cid = get_child(node, i, 4)
            @debug "assigning node $(cid) -> mu $(c_mu)"
            constraints[:trackers => (cid, Val(:aggregation)) => :mu] = c_mu
            constraints[:trackers => (cid, Val(:production)) => :produce] = false
        end
        cid = get_child(node, 4, 4)
        constraints[:trackers => (cid, Val(:aggregation)) => :mu] = dof
        constraints[:trackers => (cid, Val(:production)) => :produce] = false
    else
        mu = 0
        for i = 1:4
            cid = Gen.get_child(node, i, 4)
            c_mu = t[:trackers => (cid, Val(:aggregation)) => :mu]
            mu += c_mu
            if i < 4
                bwd[:split_kernel => :steps => i => :mu] = c_mu
            end
        end
        constraints[:trackers => (node, Val(:aggregation)) => :mu] = mu * 0.25
        constraints[:trackers => (node, Val(:production)) => :produce] = false
    end

    # obtain new trace and discard, which contains the previous subtree
    (new_trace, weight, _, discard) = update(t, constraints)

    (new_trace, bwd, weight)
end


@transform qt_split_merge_involution (t, u) to (t_prime, u_prime) begin


    _, node = get_args(u)
    split = @read(u[:split], :discrete)

    if split
        # splitting node
        @debug "splitting node $(node)"
        @write(t_prime[:trackers => (node, Val(:production)) => :produce],
               true, :discrete)
        mu = @read(t[:trackers => (node, Val(:aggregation)) => :mu], :continuous)
        @debug "split mean: $(mu)"
        dof = 0.0
        # assigning to first 3 children
        for i = 1:3
            c_mu = @read(u[:split_kernel => :steps => i => :mu], :continuous)
            dof += c_mu
            cid = Gen.get_child(node, i, 4)
            @debug "assigning node $(cid) -> mu $(c_mu)"
            @write(t_prime[:trackers => (cid, Val(:aggregation)) => :mu],
                   c_mu, :continuous)
            @write(t_prime[:trackers => (cid, Val(:production)) => :produce],
                   false, :discrete)
        end
        # computing residual for 4th child
        res = 4 * mu - dof
        cid = Gen.get_child(node, 4, 4)
        @debug "assigning node $(cid) -> mu $(res)"
        @write(t_prime[:trackers => (cid, Val(:aggregation)) => :mu],
               res, :continuous)
        @write(t_prime[:trackers => (cid, Val(:production)) => :produce],
                false, :discrete)

        # define u_prime
        @write(u_prime[:split], false, :discrete)

    else
        # merging nodes
        parent = Gen.get_parent(node, 4)

        @debug "MERGE: parent of $(node) is $(parent)"
        # compute average of children
        mu = 0.
        for i = 1:4
            cid = get_child(parent, i, 4)
            cmu =  @read(t[:trackers => (cid, Val(:aggregation)) => :mu],
                         :continuous)
            mu += cmu
            if i < 4
                @debug "assigning prop $(i) -> mu $(cmu)"
                @write(u_prime[:split_kernel => :steps => i => :mu],
                    cmu, :continuous)
            end
        end
        mu *= 0.25
        @debug "assigning node $(parent) -> mu $(mu)"
        @write(t_prime[:trackers => (parent, Val(:aggregation)) => :mu],
               mu, :continuous)

        # prevent productio of children
        @write(t_prime[:trackers => (parent, Val(:production)) => :produce],
               false, :discrete)
        @write(u_prime[:split], true, :discrete)
    end
end

is_involution!(qt_split_merge_involution)


function vertical_move_direction(t::Gen.Trace, t_prime::Gen.Trace,
                                 node::Int64)::MoveDirection
    p = first(get_args(t))
    vertical_move_direction(p, t, t_prime, node)
end

function vertical_move_direction(p::QuadTreeModel, t::Gen.Trace,
                                 t_prime::Gen.Trace, node::Int64)
    parent = Gen.get_parent(node, 4)
    p_addr = :trackers => (parent, Val(:production)) => :produce
    split_addr = :trackers => (node, Val(:production)) => :produce
    @debug "direction: node $(node)"
    # node now splits
    @debug "direction: $(split_addr)"
    (!(t[split_addr]) && t_prime[split_addr]) && return split_move
    # parent no longer splits
    @debug "direction: $(p_addr)"
    (t[p_addr] && !(t_prime[p_addr])) &&  return merge_move
    no_change
end

function balanced_split_merge(t::Gen.Trace, node::Int64)::Bool
    head::QTState = t[:trackers]
    # balanced if root node is terminal : Split <-> Merge
    node == 1 && return isempty(head.children)
    st = traverse_qt(head, node)
    @unpack level, max_level = st.node 
    # cannot split or merge if max depth
    level == max_level && return false
    # balanced if node is terminal : Split <-> Merge
    # or if children are all terminal : Merge <-> Split
    isempty(st.children) || all(x -> isempty(x.children), st.children)
end
function v_refine(::Split, tr::Gen.Trace, node::Int64)
    nt = tr
    result = 0.0
    for i = 1:4
        nt, w = lateral_move(nt, Gen.get_child(node, i, 4))
        result += w 
    end
    (nt, result)
end
function v_refine(::Merge, tr::Gen.Trace, node::Int64)
    lateral_move(tr, node)
end

function vertical_move(trace::Gen.Trace,
                       node::Int64)
    # determine if split-merge is defined
    #  - all siblings same level
    #  
    balanced_split_merge(trace, node) || return (trace, -Inf, no_change)
    # RJ-mcmc move over tracker resolution
    @debug "vertical kernel - $node"
    translator = SymmetricTraceTranslator(qt_split_merge_proposal,
                                          (node,),
                                          qt_sm_inv_manual)
                                          # qt_split_merge_involution)
    (new_trace, w1, direction) = my_inv(trace, translator)
    # (new_trace, w1) = translator(trace; check = true)
    # determine direction of move
    @debug "direction: $direction"
    # update instance addresses
    # downstream = downstream_selection(direction, new_trace, node)
    # (new_trace, w2) = regenerate(new_trace, downstream)
    isinf(w1) && error("-Inf in vertical move")
    (new_trace, w2) = v_refine(direction, new_trace, node)
    @debug "vm components w1, w2 : $(w1) + $(w2) = $(w1 + w2)"
    (new_trace, w1+w2, direction)
end
