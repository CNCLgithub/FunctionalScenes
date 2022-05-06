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

@gen function qt_mu_proposal(tid, mu, var)
    bounds::Vector{Float64} = clamp.([mu - var, mu + var], 0., 1.)
    # clamp!(bounds, 0., 1,)
    {:mu} ~ uniform(bounds[1], bounds[2])
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

    qt::QuadTreeState = get_retval(tr)
    lv = qt.lv
    nlv = length(lv)
    tree_ids = Vector{Int64}(undef, nlv)
    mus = Vector{Float64}(undef, nlv)
    for i = 1:nlv
        node = lv[i].node
        tree_ids[i] = node.tree_idx
        agg_addr = (node.tree_idx, Val(:aggregation)) => :mu
        # agg_addr = :trackers => node.tree_idx => :mu
        sidxs = node_to_idx(node, dims[1])
        mus[i] = mean(state[sidxs])
        # @trace(qt_mu_proposal(tree_ids[i], mus[i], var), :trackers)
    end
    @trace(Map(qt_mu_proposal)(tree_ids, mus, fill(var, nlv)),
                               :subtree)
    return tree_ids
end

@transform dd_state_transform (t, u) to (t_p, u_p) begin
    tree_ids = @read(u[], :discrete)
    for i = 1:length(tree_ids)
        tid = tree_ids[i]
        t_addr = :trackers => (tid, Val(:aggregation)) => :mu
        p_addr = :subtree => i => :mu
        @copy(u[p_addr], t_p[t_addr])
        @copy(t[t_addr], u_p[p_addr])
    end
end

function _dd_state_involution(trace, fwd_choices::ChoiceMap, tree_ids, proposal_args::Tuple)
    #fwd_ret is tree_ids
    model_args = get_args(trace)

    # populate constraints
    display(fwd_choices)
    constraints = choicemap()
    # populate backward assignment
    bwd_choices = choicemap()
    # @inbounds for i = 1:length(tree_ids)
    #     t_addr = :trackers => (tree_ids[i], Val(:aggregation))
    #     p_addr = :subtree =>  i => (tree_ids[i], Val(:aggregation))
    #     # p_addr = :trackers => tree_ids[i]  => :mu
    #     set_submap!(constraints, t_addr, get_submap(fwd_choices, p_addr))
    #     # set_submap!(bwd_choices, p_addr, get_submap(trace, t_addr))
    #     # constraints[t_addr] = fwd_choices[p_addr]
    #     # bwd_choices[p_addr] = trace[t_addr]
    # end
    set_submap!(constraints, :trackers, get_submap(fwd_choices, :trackers))
    display(constraints)
    choices = get_choices(trace)
    display(get_submap(choices, :trackers))
    display(get_submap(constraints, :trackers))
    # (new_trace, weight, _, discard) = update(trace, model_args, (NoChange(),), constraints)
    # results = update(trace, model_args, (NoChange(),), constraints)
    # w, rv = assess(get_gen_fn(trace), model_args, constraints)
    (t, w) = generate(get_gen_fn(trace), model_args, constraints)
    @show w
    results = update(trace, constraints)

    for i = 1:length(results)
        @show typeof(results[i])
    end
    set_submap!(bwd_choices, :subtree, get_submap(discard, :tree))
    # @inbounds for i = 1:length(tree_ids)
    #     t_addr = :trackers => (tree_ids[i], Val(:aggregation)) => :mu
    #     p_addr = :trackers => tree_ids[i]  => :mu
    #     constraints[t_addr] = fwd_choices[p_addr]
    #     bwd_choices[p_addr] = trace[t_addr]
    # end

    # populate backward assignment
    # bwd_choices = choicemap()
    # for i = 1:length(tree_ids)
    #     t_addr = :trackers => (tree_ids[i], Val(:aggregation)) => :mu
    #     p_addr = (tree_ids[i], Val(:aggregation)) => :mu
    #     # constraints[t_addr] = fwd_choices[p_addr]
    #     bwd_choices[p_addr] = trace[t_addr]
    # end
    # set_submap!(bwd_choices, :subtree, get_submap(discard, :trackers))
    # obtain new trace and discard, which contains the previous subtree
    # display(constraints)
    # cm = get_choices(trace)
    # display(get_submap(cm, :trackers))
    # error()
    # (new_trace, weight) = results[[1,2]]
    # (new_trace, weight, _, discard) = update(trace, model_args, (NoChange(),), constraints)

    (results[1], bwd_choices, results[2])
end
