export tracker_kernel


@gen function random_walk_proposal(trace, tracker)
    addr = :trackers => tracker => :state
    mus = trace[addr]
    args = Array{Tuple{Float64, Float64}}(undef, size(mus))
    for i in eachindex(mus)
        bounds = (mus[i] - 0.05, mus[i] + 0.05)
        args[i] = clamp.(bounds, 0., 1.0)
    end
    {addr} ~ broadcasted_uniform(args)
end


# =( bug in Gen I think
# @kern function tracker_kernel(trace, args1, args2)
#     trace ~ mh(trace, split_merge_proposal, args1, split_merge_involution)
#     trace ~ mh(trace, random_walk_proposal, args1)
#     trace ~ mh(trace, args2)
# end

function apply_random_walk(trace::Gen.Trace, proposal, proposal_args)
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = update(trace,
        model_args, argdiffs, fwd_choices)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = Gen.assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    (new_trace, weight)
end

function ddp_init_kernel(trace::Gen.Trace, prop_args, selected)

    (new_trace, w1) = apply_random_walk(trace,
                                        dd_state_proposal,
                                        prop_args)
    (new_trace, w2) = regenerate(new_trace, selected)
    (new_trace, w1 + w2)
end
function tracker_kernel(trace::Gen.Trace,
                        translator::Gen.SymmetricTraceTranslator,
                        tracker::Int64,
                        selected)
    (new_trace, w1) = translator(trace; check = false)
    # @debug "w1: $(w1)"
    (new_trace, w2) = apply_random_walk(new_trace,
                                        random_walk_proposal,
                                        (tracker,))
    # @debug "w2: $(w2)"
    (new_trace, w3) = regenerate(new_trace, selected)
    # @debug "w3: $(w3)"
    (new_trace, w1 + w2 + w3)
end
