export split_merge_proposal, split_merge_involution

@gen function refinement_step(t::Int64, temp::Float64, n::Int64, upper::Float64)
    u = min(upper, temp)
    l = max(0.0, temp - upper*(n - t))
    x = {:x} ~ uniform(l, u)
    return temp - x
end

@gen function refine_kernel(mu::Float64, kdim::Tuple, upper::Float64)
    n = prod(kdim)
    k = n - 1
    temp = mu * n
    {:inner} ~ Gen.Unfold(refinement_step)(k, temp, n, upper)
end

@gen function split_merge_proposal(trace, tracker)

    params = first(get_args(trace))
    lvl = trace[:trackers => tracker => :level]

    # what moves are possible?
    w = refine_weight(params, lvl)

    # refine or coarsen?
    if ({:refine} ~ bernoulli(w))
        upper_bound = params.tracker_ps[tracker]
        cur_dims = level_dims(params, lvl)
        up_dims = level_dims(params, lvl + 1)
        state = trace[:trackers => tracker => :state]
        kdim = Int64.(up_dims ./ cur_dims)
        kdim = fill(kdim, size(state))
        upper_bounds = fill(upper_bound, size(state))
        {:outer} ~ Gen.Map(refine_kernel)(state, kdim, upper_bounds)
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
    T = eltype(state)

    # refine or coarsen
    refine = @read(u[:refine], :discrete)

    if refine
        # println("refining")
        next_lvl = lvl + 1

        lower_dims = size(state)
        upper_dims = level_dims(params, next_lvl)
        kernel_dims = Int64.(upper_dims ./ lower_dims)

        upper_ref = CartesianIndices(upper_dims) # dimensions of refine state
        lower_ref = CartesianIndices(lower_dims) # dimensions of coarse state
        kernel_ref = CartesianIndices(kernel_dims) # dimensions of coarse state

        upper_lref = LinearIndices(upper_dims)
        lower_lref = LinearIndices(lower_dims)
        kernel_lref = LinearIndices(kernel_dims)

        kp = prod(kernel_dims) # number of elements in kernel

        # iterate over coarse state kernel
        next_state = zeros(T, upper_dims)
        for lower in lower_ref
            # map together kernel steps
            i = lower_lref[lower]
            c = CartesianIndex((Tuple(lower) .- (1, 1)) .* kernel_dims)
            # @show c
            # iterate over scalars for each kernel sweep
            _sum = 0.
            for inner in kernel_ref
                # @show inner
                j = kernel_lref[inner] # index of inner
                idx = c + inner # cart index in refined space
                if j < kp # still retreiving from prop
                    val = @read(u[:outer => i => :inner => j => :x],
                                :continuous)
                    _sum += val
                else # solving for the final value
                    val = state[lower] * kp - _sum
                end
                next_state[idx] = val
            end
        end
        # update t_prime
        @write(t_prime[:trackers => tracker => :level],
               next_lvl, :discrete)
        @write(t_prime[:trackers => tracker => :state],
                next_state, :continuous)
        @write(u_prime[:refine], false, :discrete)

    else # coarsen
        # println("coarsening")
        next_lvl = lvl - 1
        next_state = coarsen_state(params, state, next_lvl)

        # forward write
        @write(t_prime[:trackers => tracker => :level],
                next_lvl, :discrete)
        @write(t_prime[:trackers => tracker => :state],
                next_state, :continuous)

        # reverse refinement trip
        @write(u_prime[:refine], true, :discrete)
        full_dims = size(state)
        outer_dims = size(next_state)
        outer_ref = CartesianIndices(outer_dims) # dimensions of coarse state
        outer_lref = LinearIndices(outer_ref) # linear variant of above
        kdim = Int64.(full_dims ./ outer_dims) # scaling factor
        kp = prod(kdim) # number of elements in kernel
        kern = CartesianIndices(kdim) # kernel coordinates
        lkern = LinearIndices(kern) # linear variance of above
        for outer in outer_ref
            i = outer_lref[outer]
            c = CartesianIndex((Tuple(outer) .- (1, 1)) .* kdim)
            for inner in kern[1 : end-1]
                idx = c + inner # cartesian coordinates in full state
                j = lkern[inner]
                @write(u_prime[:outer => i => :inner => j => :x],
                       state[idx],
                       :continuous)
            end
        end
    end

end


is_involution!(split_merge_involution)
