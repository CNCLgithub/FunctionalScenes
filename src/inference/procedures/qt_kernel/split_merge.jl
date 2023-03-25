export split_merge_move, balanced_split_merge

include("split_merge_gen.jl")

function construct_translator(::MoveDirection, node::Int64)
    error("not implemented")
end
function construct_translator(::Split, node::Int64)
    SymmetricTraceTranslator(qt_split_merge_proposal,
                             (node,),
                             qt_involution)
end
function construct_translator(::Merge, node::Int64)
    SymmetricTraceTranslator(qt_split_merge_proposal,
                             (Gen.get_parent(node, 4),),
                             qt_involution)
end

# this is just a copy of the `translator` method from Gen
# to expose internal scores.
function mytransform(translator::SymmetricTraceTranslator{TraceTransformDSLProgram},
                     prev_model_trace::Trace; check=false, observations=EmptyChoiceMap())

    # simulate from auxiliary program
    forward_proposal_trace =
        simulate(translator.q, (prev_model_trace, translator.q_args...,))

    # apply trace transform
    (new_model_trace, backward_proposal_trace, log_abs_determinant) =
        Gen.run_transform(translator, prev_model_trace, forward_proposal_trace)

    # compute log weight
    prev_model_score = get_score(prev_model_trace)
    new_model_score = get_score(new_model_trace)
    forward_proposal_score = get_score(forward_proposal_trace)
    backward_proposal_score = get_score(backward_proposal_trace)
    log_weight = new_model_score - prev_model_score +
        backward_proposal_score - forward_proposal_score + log_abs_determinant

    # @show new_model_score
    # @show prev_model_score
    # @show backward_proposal_score
    # @show forward_proposal_score
    # @show log_abs_determinant

    if check
        check_observations(get_choices(new_model_trace), observations)
        (prev_model_trace_rt, forward_proposal_trace_rt, _) =
            run_transform(translator, new_model_trace, backward_proposal_trace)
        check_round_trip(prev_model_trace, prev_model_trace_rt,
                         forward_proposal_trace, forward_proposal_trace_rt)
    end

    return (new_model_trace, log_weight)
end

function split_merge_move(trace::Gen.Trace,
                          node::Int64,
                          direction::MoveDirection)
    # RJ-mcmc move over tracker resolution
    @debug "SM kernel - $node"
    translator = construct_translator(direction, node)
    (new_trace, w1) = translator(trace, check = false)
    # (new_trace, w1) = mytransform(translator, trace, check = false)
    if isinf(w1)
        @show node => direction
        @show get_depth(node)
        @show trace[:trackers => (node, Val(:production)) => :produce]
        @show trace[:trackers => (node, Val(:aggregation)) => :mu]
        if direction == split_move
            @show new_trace[:trackers => (node, Val(:production)) => :produce]
            for i = 1:4
                child = Gen.get_child(node, i, 4)
                @show new_trace[:trackers => (child, Val(:production)) => :produce]
                @show new_trace[:trackers => (child, Val(:aggregation)) => :mu]
            end
        end
        error("-Inf in SM move")
    end
    (new_trace, w1)
    # (new_trace, w2) = rw_move(direction, new_trace, node)
    # @debug "vm components w1, w2 : $(w1) + $(w2) = $(w1 + w2)"
    # (new_trace, w1+w2, direction)
end

function balanced_split_merge(t::Gen.Trace, tidx::Int64)::Bool
    qt = get_retval(t).qt
    st = traverse_qt(qt, tidx)
    # it's possible to not reach the node
    # in that case, not balanced?
    # REVIEW
    st.node.tree_idx === tidx || return false

    # Root node cannot merge
    st.node.tree_idx === 1 && return false

    # cannot split or merge if max depth
    st.node.level == st.node.max_level && return false
    # balanced if node is terminal : Split <-> Merge
    # and if siblings are all terminal : Merge <-> Split
    parent_idx = Gen.get_parent(tidx, 4)
    parent_st = traverse_qt(qt, parent_idx)
    all(x -> isempty(x.children), parent_st.children)
end
