@gen function qt_mu_proposal(mu, var)
    low::Float64 = max(0., mu - var)
    high::Float64 = min(1., mu + var)
    {:mu} ~ uniform(low, high)
end

@gen function dd_state_proposal(tr::Gen.Trace, params::DataDrivenState, img::Array{Float64, 4})

    mparams = first(Gen.get_args(tr))
    dims = mparams.dims

    @unpack nn, device, var = params
    state = @pycall fs_py.dd_state(nn, img, device)::Matrix{Float64}

    println("DDP")
    display_mat(state)

    qt::QuadTreeState = get_retval(tr)
    lv = qt.lv
    nlv = length(lv)
    tree_ids = Vector{Int64}(undef, nlv)
    mus = Vector{Float64}(undef, nlv)
    for i = 1:nlv
        node = lv[i].node
        tree_ids[i] = node.tree_idx
        sidxs = node_to_idx(node, dims[1])
        mus[i] = mean(state[sidxs])
    end
    @trace(Map(qt_mu_proposal)(mus, fill(var, nlv)),
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
