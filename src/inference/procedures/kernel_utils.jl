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
    # @debug "initial trace score $(get_score(trace))"
    # (new_trace, w1) = apply_random_walk(trace,
    #                                     dd_state_proposal,
    #                                     prop_args)
    (new_trace, _) = metropolis_hastings(trace, dd_state_proposal, prop_args,
                                         dd_state_transform)
    # @debug "w1: $(w1)"
    # (new_trace, w2) = regenerate(new_trace, selected)
    # @debug "w2: $(w2)"
    (new_trace, 0.)
end

abstract type MoveDirection end
struct Merge end
struct Split end
struct NoChange end
const merge_move = Merge()
const split_move = Split()
const no_change  = NoChange()

function downstream_selection(t::Gen.Trace, node::Int64)
    params = first(get_args(t))
    head::QTState = t[:trackers]
    st::QTState = traverse_qt(head, node)
    # associated rooms samples
    idxs = node_to_idx(st.node, params.dims[1])
    v = Vector{Pair}(undef, length(idxs) * params.instances)
    lis = LinearIndices((params.instances, length(idxs)))
    for i = 1:params.instances, j in idxs
        v[lis[i, j]] = :instances => i => :obstacle => j => :flip
    end
    select(v...)
end

function downstream_selection(::Union{NoChange,Split}, t::Gen.Trace, node::Int64)
    downstream_selection(t, node)
end

function downstream_selection(::Merge, t::Gen.Trace, node::Int64)
    p = Gen.get_parent(node, 4)
    downstream_selection(t, p)
end

function all_downstream_selection(t::Gen.Trace)::Gen.Selection
    p = first(get_args(t))
    all_downstream_selection(p)
end