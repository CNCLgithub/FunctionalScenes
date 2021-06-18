

@with_kw struct DataDrivenState
    # neural network details
    nn_weights::String
    device = _load_device()
    nn = functional_scenes.init_dd_state(nn_weights, device)

    # proposal formatting
    level::Int64 = 2 # should be 3x3
    var::Float64 = 0.1

end


@gen function dd_state_proposal(tr::Gen.Trace, params::DataDrivenState, img::Array{UInt8, 4})

    mparams = Gen.get_args(tr)

    @unpack nn, level = params
    # raw state output from python
    state = @pycall functional_scenes.dd_state(nn, img)::Array{Float64, 4}

    # format state for proposal to gm trace
    @unpack levels, dims, tracker_size, n_trackers = mparams


    cm = Gen.ChoiceMap()

    # deterministically sets level to 2
    level_weights = zeros(levels)
    level_weights[level] = 1.0

    state_ref = CartesianIndices(tracker_size, n_trackers)

    for i = 1:n_trackers
        @trace(categorical(level_weights),
               :trackers => i => :level)

        # extract info from `state`
        vs = state_to_room(mparams, state_ref[:, i])
        dat = state[vs]

        ldims = level_dims(mparams, level)
        mus, vars = mat_resize(dat, ldims)

        # some number of steps..
        @trace(broadcasted_normal(mus, vars),
               :trackers => i => :state)
    end

end
