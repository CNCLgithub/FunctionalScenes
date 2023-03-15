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

function split_merge_move(trace::Gen.Trace,
                          node::Int64,
                          direction::MoveDirection)
    # RJ-mcmc move over tracker resolution
    @debug "SM kernel - $node"
    translator = construct_translator(direction, node)
    (new_trace, w1) = translator(trace, check = false)
    # downstream = downstream_selection(direction, new_trace, node)
    # (new_trace, w2) = regenerate(new_trace, downstream)
    isinf(w1) && error("-Inf in SM move")
    (new_trace, w2) = rw_move(direction, new_trace, node)
    @debug "vm components w1, w2 : $(w1) + $(w2) = $(w1 + w2)"
    (new_trace, w1+w2, direction)
end

function balanced_split_merge(t::Gen.Trace, tidx::Int64)::Bool
    head::QTAggNode = get_retval(t).qt
    st = traverse_qt(head, tidx)
    # it's possible to not reach the node
    # in that case, not balanced?
    # REVIEW
    st.node.tree_idx === tidx || return false

    # cannot split or merge if max depth
    st.node.level == st.node.max_level && return false
    # balanced if node is terminal : Split <-> Merge
    # and if siblings are all terminal : Merge <-> Split
    parent_idx = tidx == 1 ? tidx : Gen.get_parent(tidx, 4)
    parent_st = tidx == 1 ? head : traverse_qt(head, parent_idx)
    all(x -> isempty(x.children), parent_st.children)
end
