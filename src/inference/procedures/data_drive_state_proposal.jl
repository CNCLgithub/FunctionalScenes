export DataDrivenState, dd_state_proposal

@with_kw struct DataDrivenState
    # neural network details
    config_path::String
    device::PyObject = _load_device()
    nn::PyObject = _init_dd_state(config_path, device)

    # proposal variables
    var::Float64 = 0.05

end

function _init_dd_state(config_path::String, device::PyObject)
    @show config_path
    functional_scenes.og_proposal.init_dd_state(config_path, device)::PyObject
end


@gen function dd_state_proposal(tr::Gen.Trace, params::DataDrivenState, img::Array{Float64, 4})

    mparams = first(Gen.get_args(tr))
    dims = mparams.dims

    @unpack nn, device, var = params
    # raw state output from python
    # _state = @pycall functional_scenes.og_proposal.dd_state(nn, img, device)::P
    # @show _state.shape
    # state = Array{Float64, 3}(_state)
    state = @pycall functional_scenes.og_proposal.dd_state(nn, img, device)::Matrix{Float64}
    # reverse!(state, dims = 1)
    # state = Matrix{Float64}(state')
    # clamp!(state, 0., 1.0)
    # state = clean_state(state)
    viz_ddp_state(state)

    qts::QuadTreeState = get_retval(tr)
    tree_ids = Vector{Int64}(undef, length(qts.lv))
    for i = 1:length(qts.lv)
        node = gts[i].node
        tree_ids[i] = node.tree_idx
        # agg_addr = :trackers => (node.tree_idx, Val(:aggregation)) => :mu
        agg_addr = :trackers => node.tree_idx => :mu
        sidxs = node_to_idx(node, dims[1])
        mu = mean(state[sidxs])
        bounds::Vector{Float64} = [mu - var, mu + var]
        clamp!(bounds, 0., 1,)
        {agg_addr} ~ uniform(bounds[1], bounds[2])
    end
    tree_ids
end
function dd_proposal_involution(trace, fwd_choices::ChoiceMap, fwd_ret::Tuple, proposal_args::Tuple)
    #fwd_ret is tree_ids
    tree_ids = first(fwd_ret)
    model_args = get_args(trace)

    # populate constraints
    constraints = choicemap()
    # populate backward assignment
    bwd_choices = choicemap()
    @inbounds for i = 1:length(tree_ids)
        t_addr = :trackers => (tree_ids[i], Val(:aggregation)) => :mu
        p_addr = :trackers => tree_ids[i]  => :mu
        constraints[t_addr] = fwd_choices[p_addr]
        bwd_choices[p_addr] = trace[t_addr]
    end

    # obtain new trace and discard, which contains the previous subtree
    (new_trace, weight, _, discard) = update(trace, model_args, (NoChange(),), constraints)

    (new_trace, bwd_choices, weight)
end
