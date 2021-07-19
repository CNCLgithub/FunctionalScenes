export DataDrivenState, dd_state_proposal

@with_kw struct DataDrivenState
    # neural network details
    vae_weights::String
    ddp_weights::String
    device::PyObject = _load_device()
    nn::PyObject = _init_dd_state(vae_weights, ddp_weights, device)

    # proposal formatting
    level::Int64 = 2 # should be 3x3
    var::Float64 = 0.05

end

function _init_dd_state(vae::String, ddp::String, device::PyObject)
    functional_scenes.init_dd_state(vae, ddp, device)::PyObject
end


@gen function dd_state_proposal(tr::Gen.Trace, params::DataDrivenState, img::Array{Float64, 4})

    mparams = first(Gen.get_args(tr))

    @unpack nn, device, var, level = params
    # raw state output from python
    state = @pycall functional_scenes.dd_state(nn, img, device)::Array{Float64, 4}
    state = state[1, 1, :, :]
    state = Matrix{Float64}(state')
    clamp!(state, 0., 1.0)
    # state = clean_state(state)
    viz_ddp_state(state)

    # format state for proposal to gm trace
    @unpack levels, tracker_ps, dims, tracker_size, n_trackers = mparams

    state_ref = CartesianIndices((dims..., n_trackers))

    for i = 1:n_trackers

        lvl = tr[:trackers => i => :level]

        tracker_spread = tracker_ps[i] - 1E-5
        # extract info from `state`
        vs = state_to_room(mparams, vec(state_ref[:, :, i]))
        dat = reshape(state[vs], dims)

        ldims = level_dims(mparams, lvl)
        mus = coarsen_state(dat, ldims)
        args = Array{Tuple{Float64, Float64}}(undef, ldims)
        for i in eachindex(mus)
            prop_bounds = (mus[i] - var, mus[i] + var)
            args[i] = clamp.(prop_bounds, 0., tracker_spread)
        end

        @trace(broadcasted_uniform(args),
               :trackers => i => :state)
    end

end
