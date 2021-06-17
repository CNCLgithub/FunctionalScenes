export split_merge_proposal, split_merge_involution

@gen function split_merge_proposal(trace, tracker)

    params = first(get_args(trace))
    lvl = trace[:trackers => tracker => :level]

    # what moves are possible?
    w = refine_weight(params, lvl)

    # refine or coarsen?
    if ({:refine} ~ bernoulli(w))
        dims = level_dims(params, lvl + 1)
        k = prod(k) - 1
        # samples bounded on [0, 1]
        {:deltas} ~ broadcasted_uniform(fill((0., 1.), k))
    end
end

@transform split_merge_involution (t, u) to (t_prime, u_prime) begin

    params = first(get_args(t))
    _, tracker = get_args(u)
    lvl = @read(t[:trackers => tracker => :level], :discrete)
    state = @read(t[:trackers => tracker => :state], :continuous)
    state = reshape(state, level_dims(params, lvl))

    # refine or coarsen
    refine = @read(u[:refine], :discrete)

    if refine
        println("refining")
        next_lvl = lvl + 1
        deltas = @read(u[:deltas], :continuous)
        # 3x3 -> 6x6
        mus = refine_state(params, state, next_lvl)
        # 6x6 + 6x6; but mean is not gauranteed
        # need to remove values outside (0, 1)
        next_state = mus + reshape(deltas, size(mus))
        # correct for offset
        # (6x6 -> 3x3) - 3x3
        correction = state - coarsen_state(params, next_state, lvl)
        # 6x6
        correction = refine_state(params, correction, next_lvl)
        next_state += correction
        # # mean is now correct but could be out of range
        next_state = clean_state(next_state)
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
        deltas = refine_state(params, next_state, lvl)
        deltas = state - deltas
        @write(u_prime[:refine], true, :discrete)
        @write(t_prime[:trackers => tracker => :level],
                next_lvl, :discrete)

        @write(u_prime[:deltas], deltas, :continuous)
        @write(t_prime[:trackers => tracker => :state],
                next_state, :continuous)
    end

    @show lvl
    @show next_lvl


end
