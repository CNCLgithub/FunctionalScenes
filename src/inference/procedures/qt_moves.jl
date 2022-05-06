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
    model_args = get_args(t)
    argdiffs = map((_) -> NoChange(), model_args)
    (new_trace, w2) = regenerate(new_trace, model_args, argdiffs, downstream)
    (new_trace, w1 + w2)
end

# @gen function qt_subtree_production(t::Gen.Trace, i::Int64)
#     prop_addr = :trackers => (i, Val(:production)) => :produce

# end

@gen function split_step(i::Int64,
                         temp::Float64,
                         n::Int64)
    hi::Float64 = min(1.0, temp)
    lo::Float64 = max(0.0, temp - (n - i))
    u_i::Float64 = {:mu} ~ uniform(lo, hi)
    res::Float64 = temp - u_i
    # println("i $i, temp $temp, lo $lo, hi $hi")
    return res
end

@gen function split_kernel(mu::Float64)
    n = 4
    temp::Float64 = mu * n
    mus = {:steps} ~ Gen.Unfold(split_step)(3, temp, n)
    return mus
end

function split_weight(n::QTNode)::Float64
    @unpack level, max_level = n
    level == 1 && return 1.0
    level < max_level && return 0.5
    0.0
end

@gen function qt_split_merge_proposal(t::Gen.Trace, i::Int64)

    # prop_addr = :trackers => (i, Val(:production))
    head::QTState = t[:trackers]
    st::QTState = traverse_qt(head, i)
    # st = t[:trackers => (i, Val(:aggregation))]
    w = split_weight(st.node)
    s = @trace(bernoulli(w), :produce)

    # refine or coarsen?
    if ({:split} ~ bernoulli(w))
        mu = t[:trackers => (i, Val(:aggregation)) => :mu]
        {:split_kernel} ~ split_kernel(mu)
    end
end


@transform qt_split_merge_involution (t, u) to (t_prime, u_prime) begin


    _, node = get_args(u)
    split = @read(u[:split], :discrete)

    if split
        # splitting node
        @write(t_prime[:trackers => (node, Val(:production)) => :produce],
               true, :discrete)
        # computing residual for 4th child
        mu = @read(t[:trackers => (node, Val(:aggregation)) => :mu], :continuous)
        dof = 0.0
        # steps = @read(u[:split_kernel => :steps], :continuous)
        # res = mu - sum(steps)
        # assigning to first 3 children
        for i = 1:3
            c_mu = @read(u[:split_kernel => :steps => i => :mu], :continuous)
            dof += c_mu
            cid = Gen.get_child(node, i, 4)
            @write(t_prime[:trackers => (cid, Val(:aggregation)) => :mu],
                   c_mu, :continuous)
            @write(t_prime[:trackers => (cid, Val(:production)) => :produce],
                   false, :discrete)
        end
        # 4th child
        res = 4 * mu - dof
        cid = Gen.get_child(node, 4, 4)
        @write(t_prime[:trackers => (cid, Val(:aggregation)) => :mu],
               res, :continuous)
        @write(t_prime[:trackers => (cid, Val(:production)) => :produce],
                false, :discrete)

        # define u_prime
        @write(u_prime[:split], false, :discrete)

    else
        # merging nodes
        parent = Gen.get_parent(node, 4)

        # compute average of children
        mu = 0.
        for i = 1:4
            cid = get_child(parent, i, 4)
            cmu =  @read(t[:trackers => (cid, Val(:aggregation)) => :mu],
                         :continuous)
            mu += cmu
            if i < 4
                @write(u_prime[:split_kernel => :steps => i => :mu],
                    cmu, :continuous)
            end
        end
        mu *= 0.25
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
    # parent no longer splits
    (t[p_addr] && !(t_prime[p_addr])) &&  return merge_move
    # node now splits
    (!(t[split_addr]) && t_prime[split_addr]) && return split_move
    no_change
end

function vertical_move(trace::Gen.Trace,
                       node::Int64)
    # RJ-mcmc move over tracker resolution
    translator = SymmetricTraceTranslator(qt_split_merge_proposal,
                                          (node,),
                                          qt_split_merge_involution)
    (new_trace, w1) = translator(trace; check = false)
    # determine direction of move
    direction = vertical_move_direction(trace, new_trace, node)
    # # random walk over tracker state (bernoulli weights)
    # (new_trace, w2) = apply_random_walk(new_trace,
    #                                     qt_node_random_walk,
    #                                     (node,))
    # update instance addresses
    downstream = downstream_selection(direction, new_trace, node)
    (new_trace, w2) = regenerate(new_trace, downstream)
    (new_trace, w1 + w2, direction)
end
