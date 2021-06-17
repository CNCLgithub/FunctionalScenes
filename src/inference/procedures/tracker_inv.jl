export split_merge_proposal, split_merge_involution

@gen function split_merge_proposal(trace, tracker)

    params = first(get_args(trace))
    lvl = trace[:trackers => tracker => :level]

    # what moves are possible?
    w = refine_weight(params, lvl)

    # refine or coarsen?
    if ({:refine} ~ bernoulli(w))
        dims = level_dims(params, lvl + 1)
        median = {:median} ~ uniform(0., 1.0)
        # 8
        k = prod(dims) - prod(level_dims(params, lvl))
        w = k % 2 == 0 ? k / 2 : (k - 1) / 2

        {:lower} ~ broadcasted_normal(fill((epsilon, median - epsilon), w))
        {:upper} ~ broadcasted_normal(fill((median + epsilon, 1.0 - epsilon), w))
    end
end


epsilon = 1E-10

@transform split_merge_involution (t, u) to (t_prime, u_prime) begin

    params = first(get_args(t))
    _, tracker = get_args(u)
    lvl = @read(t[:trackers => tracker => :level], :discrete)
    state = @read(t[:trackers => tracker => :state], :continuous)
    state_dims = level_dims(params, lvl)
    state = reshape(state, state_dims)

    # refine or coarsen
    refine = @read(u[:refine], :discrete)

    if refine
        println("refining")
        next_lvl = lvl + 1
        # N - 1 values
        median = @read(u[:median], :continuous)
        lower = @read(u[:lower], :continuous)
        upper = @read(u[:upper], :continuous)
        dims = level_dims(params, next_lvl)
        n = prod(dims)
        k = n - 2 * length(lower)
        if k == 1
            next_state = [lower; median; upper]
        else
            next_state = [lower; median; median; upper]
        end
        # sd = sum(deltas)
        # ms = mean(state)
        # diff = (ms * n) - sd
        # if diff >= 1.0
        #     println("d > 1")
        #     deltas ./= sd
        #     deltas .*= (ms * (n-k) - epsilon)
        # else
        #     println("d < 0")
        #     deltas ./= sd
        #     deltas .*= (ms * (n - k) - epsilon)
        # end
        # x0 = ms * n - sum(deltas)
        # x0 = fill(x0 ./ k, k)
        # # 8x1 -> 3x3
        # next_state = reshape([deltas; x0], dims)
        @show mean(next_state)
        # # mean is now correct but could be out of range
        # next_state = clean_state(next_state)
        @write(u_prime[:refine], false, :discrete)
        # update t_prime
        @write(t_prime[:trackers => tracker => :level],
                next_lvl, :discrete)
        @write(t_prime[:trackers => tracker => :state],
                next_state, :continuous)

    else # coarsen
        println("coarsening")
        next_lvl = lvl - 1
        next_state = coarsen_state(params, state, next_lvl)
        @write(u_prime[:refine], true, :discrete)
        k = prod(size(next_state))
        #    27 -> 8
        @show size(state)
        @show size(next_state)
        @write(u_prime[:deltas], state[1:end - k], :continuous)
        # 36 -> 9 -> 1
        @write(t_prime[:trackers => tracker => :level],
                next_lvl, :discrete)
        @write(t_prime[:trackers => tracker => :state],
                next_state, :continuous)
    end

end


is_involution!(split_merge_involution)
