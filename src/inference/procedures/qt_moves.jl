export lateral_move, vertical_move


@gen function qt_node_random_walk(t::Gen.Trace, i::Int64)
    addr = :trackers => (i, Val(:aggregation)) => :mu
    mu::Float64 = trace[addr]
    bounds::Vector{Float64} = [mu - 0.05, mu + 0.05]
    clamp!(bounds, 0., 1,)
    {addr} ~ uniform(bounds[1], bounds[2])
end

function lateral_move(t::Gen.Trace, i::Int64)
    apply_random_walk(t, qt_node_random_walk, (t, i))
end

# @gen function qt_subtree_production(t::Gen.Trace, i::Int64)
#     prop_addr = :trackers => (i, Val(:production)) => :produce

# end

@gen function split_step(i::Int64,
                         temp::Float64,
                         n::Int64)
    hi::Float64 = min(1.0, temp)
    lo::Float64 = max(0.0, temp - (n - i))
    u_i::Float64 = {:u} ~ uniform(lo, hi)
    res::Float64 = temp - u_i
    return res
end

@gen function split_kernel(mu::Float64)
    # n = prod(kdim)
    # k = n - 1
    temp::Float64 = mu * 4
    mus = {:steps} ~ Gen.Unfold(split_step)(3, temp, n)
end

@gen function qt_split_merge_proposal(t::Gen.Trace, i::Int64)

    # prop_addr = :trackers => (i, Val(:production))
    n = t[:trackers => (i, Val(:production))]
    w = produce_weight(n)
    s = @trace(bernoulli(w), :produce)
    # what moves are possible?
    w = produce_weight(n)

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
        @write(t_prime[:trackers => (i, Val(:production)) => :produce],
               true, :discrete)
        # computing residual for 4th child
        mu = @read(t[:trackers => (i, Val(:aggregation)) => :mu], :continuous)
        steps = @read(u[:split_kernel => :steps], :continuous)
        res = mu - sum(steps)
        # assigning to first 3 children
        for i = 1:3
            cid = Gen.get_child(node, i, 4)
            @copy(u[:split_kernel => :steps => i => :u],
                  t_prime[:trackers => (cid, Val(:aggregation)) => :mu])
            @write(t_prime[:trackers => (cid, Val(:production)) => :produce],
                   false, :discrete)
        end
        # 4th child
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
        for i = 1:4
            cid = get_child(parent, i, 4)

        end
    end

end

function vertical_move(t::Gen.Trace, i::Int64)
end
