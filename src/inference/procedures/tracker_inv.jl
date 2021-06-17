export split_merge_proposal, split_merge_involution

@gen function split_merge_proposal(trace, tracker)

    params = first(get_args(trace))
    lvl = trace[:trackers => tracker => :level]

    # what moves are possible?
    w = refine_weight(params, lvl)

    # refine or coarsen?
    if ({:refine} ~ bernoulli(w))
        dims = level_dims(params, lvl + 1)
        # samples bounded on [0, 1]
        sd = {:sd} ~ broadcasted_uniform(fill((0., 0.5), dims))
        mus = zeros(dims)
        {:deltas} ~ broadcasted_normal(mus, sd)
    end
end

@transform split_merge_involution (t, u) to (t_prime, u_prime) begin

    params = first(get_args(t))
    _, tracker = get_args(u)
    lvl = @read(t[:trackers => tracker => :level], :discrete)
    state = @read(t[:trackers => tracker => :state], :discrete)

    # refine or coarsen
    refine = @read(u[:refine], :discrete)

    if refine
        next_lvl = lvl + 1
        sds = @read(u[:sd], :discrete)
        deltas = @read(u[:deltas], :discrete)
        mus = refine_state(params, state, next_lvl)
        next_state = clean_state(mus + deltas) # need to remove values outside [0, 1]
        @write(u_prime[:refine], false, :discrete)
    else # coarsen
        next_lvl = lvl - 1
        next_state, sd = coarsen_state(params, state, next_lvl)
        @write(u_prime[:sd], sd, :discrete)
        @write(u_prime[:deltas], zeros(size(sd)), :discrete)
        @write(u_prime[:refine], true, :discrete)
    end

    # update t_prime
    @write(t_prime[:trackers => tracker => :level],
            next_lvl, :discrete)
    @write(t_prime[:trackers => tracker => :state],
            next_state, :discrete)
end
