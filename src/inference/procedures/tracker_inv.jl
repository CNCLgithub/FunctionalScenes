export split_merge_proposal, split_merge_involution

@gen function split_merge_proposal(trace, tracker)

    params = first(get_args(trace))
    lvl = trace[:trackers => tracker => :level]

    # what moves are possible?
    w = refine_weight(params, lvl)

    # refine or coarsen?
    if ({:refine} ~ bernoulli(w))
        state = trace[:trackers => tracker => :state]
        n = prod(size(state))
        mu = mean(state)
        w_min = {:w_min} ~ uniform(0., mu)
        w_max = {:w_max} ~ uniform(mu, 1.0)
        bounds = [w_min, mu, w_max]
        args = (bounds, [0.5, 0.5])
        {:deltas} ~ broadcasted_piecewise_uniform()
        dims = level_dims(params, lvl + 1)
        window = {:window} ~ uniform(0., 1.0)
        # 8
        k = prod(dims) - prod(level_dims(params, lvl))
        w = Int64(k % 2 == 0 ? k / 2 : (k - 1) / 2)

        {:lower} ~ broadcasted_uniform(fill((epsilon, median - epsilon), w))
        {:upper} ~ broadcasted_uniform(fill((median + epsilon, 1.0 - epsilon), w))
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
        nmed = n - (length(lower) + length(upper))
        next_state = [lower; fill(median, nmed); upper]
        next_state = reshape(next_state, dims)
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
        @show size(state)
        @show size(next_state)
        med = Statistics.median(state)
        lower = state[state .< median]
        upper = state[state .> median]
        @write(u_prime[:median], med, :continuous)
        @write(u_prime[:lower], lower, :continuous)
        @write(u_prime[:upper], upper, :continuous)
        @write(t_prime[:trackers => tracker => :level],
                next_lvl, :discrete)
        @write(t_prime[:trackers => tracker => :state],
                next_state, :continuous)
    end

end


is_involution!(split_merge_involution)
