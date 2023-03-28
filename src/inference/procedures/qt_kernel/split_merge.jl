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
