using Base.Iterators: take

export AttentionMH

#################################################################################
# Attention MCMC
#################################################################################

@with_kw struct AttentionMH <: Gen_Compose.MCMC

    #############################################################################
    # Inference Paremeters
    #############################################################################

    # chain length
    samples::Int64 = 10


    #############################################################################
    # Data driven proposal
    #############################################################################

    ddp::Function = ddp_init_kernel
    ddp_args::Tuple 

    #############################################################################
    # Attention
    #############################################################################

    # Goal driven belief
    objective::Function = quad_tree_path
    # destance metrics for two task objectives
    distance::Function = batch_compare_og

    # address schema for IOT sensitivity
    nodes::Int64

    # smoothing relative sensitivity for each tracker
    smoothness::Float64 = 1.0

    # number of steps for init kernel
    init_cycles::Int64 = 10
    # number of steps per node
    lateral_cycles::Int64 = 10
    vertical_cycles::Int64 = 10
end


function load(::Type{AttentionMH}, path::String; kwargs...)
    loaded = read_json(path)
    AttentionMH(; nodes = nodes, selections = selections, loaded...,
                kwargs...)
end


mutable struct AMHTrace <: MCMCTrace
    current_trace::Gen.Trace
    initialized::Bool
    sensitivities::Vector{Float64}
    weights::Vector{Float64}
end

function update_weights!(state::AMHTrace, proc::AttentionMH)
    state.weights = softmax(proc.smoothness .* state.sensitivities)
    return nothing
end


function update_structure!(state::AMHTrace, proc::AttentionMH, d::Split,
                           idx::Int64)
    @inbounds for i = 1 : 4
        c = Gen.get_child(idx, i, 4)
        state.sensitivities[c] = state.sensitivities[idx]
    end
    state.sensitivies[idx] = 0.0
    return nothing
end

function update_structure!(state::AMHTrace, proc::AttentionMH, d::Merge,
                           idx::Int64)
    parent = Gen.get_parent(idx, 4)
    state.sensitivities[parent] = 0.0
    @inbounds for i = 1 : 4
        c = Gen.get_child(idx, i, 4)
        state.sensitivities[parent] += state.sensitivities[c]
        state.sensitivities[c] = 0.0
    end
    return nothing
end

function kernel_init!(state::AMHTrace, proc::AttentionMH)

    t = state.current_trace
    _t  = t
    # loop through each node in initial trace
    nodes = proc.get_nodes(t)
    n = length(nodes)
    sensitivities = Vector{Float64}(undef, n)
    lls = Vector{Float64}(undef, init_cycles)
    distances = Vector{Float64}(undef, init_cycles)
    for i = 1:n
        node = nodes[i]
        for j = 1:proc.init_cycles
            _t, lls[j] = lateral_move(t, node)
            distances[j] = distance(objective(t), objective(_t))
            if log(rand()) < lls[i]
                @debug "accepted"
                t = _t
            end
        end
        # clamp!(lls, -Inf, 0.)
        # compute expectation over sensitivities
        @debug "distances: $(distances)"
        e_dist = mean(distances) # sum(distances .* exp.((lls .- logsumexp(lls))))
        state.sensitivities[i] = isinf(e_dist) ? 0. : e_dist # clean up -Inf
    end

    state.current_trace = t
    state.initialized = true
    update_weights!(state, proc)
    return nothing
end

function kernel_move!(state::AMHTrace, proc::AttentionMH)

    # select node to rejuv
    idx = categorical(state.weights)

    @debug "Attending to tracker $(idx)"

    @unpack lateral_cycles, vertical_cycles, objective, distance = proc

    # lateral moves
    t = state.current_trace
    lls = Vector{Float64}(undef, lateral_cycles)
    distances = Vector{Float64}(undef, lateral_cycles)
    for i = 1:lateral_cycles
        _t, lls[j] = lateral_move(t, idx)
        distances[j] = distance(objective(t), objective(_t))
        if log(rand()) < lls[i]
            @debug "accepted"
            t = _t
        end
    end

    # clamp!(lls, -Inf, 0.)
    # compute expectation over sensitivities
    @debug "distances: $(distances)"
    e_dist = mean(distances) # sum(distances .* exp.((lls .- logsumexp(lls))))
    state.sensitivities[idx] = isinf(e_dist) ? 0. : e_dist # clean up -Inf

    # attempt vertical moves
    accepted = false
    for i = 1 : proc.vertical_cycles
        _t, _ls, direction = vertical_move(t, node)
        if log(rand()) < _ls
            accepted = true
            t = _t
            break
        end
    end

    # update trace
    state.current_trace = t
    # need to update metrics given structure change
    accepted && update_structure!(state, proc, direction, idx)
    # finally update weights for next step
    update_weights!(state, proc)
    return idx
end

function Gen_Compose.initialize_procedure(proc::AttentionMH,
                                          query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    trace,_ = proc.ddp(trace, proc.ddp_args...)

    n = proc.nodes
    sensitivities = zeros(proc.nodes)
    weights = fill(1.0/proc.nodes, proc.nodes)
    AMHTrace(trace,
             false
             sensitivities,
             weights)
end

function Gen_Compose.mc_step!(state::AMHTrace,
                              proc::AttentionMH,
                              query::StaticQuery)

    # initialize kernel by exploring nodes
    state.initialized || kernel_init!(state, proc)

    # proposal
    idx = kernel_move!(state, proc)

    # viz_render(state.current_trace)
    # viz_compute_weights(state.weights)
    # viz_sensitivity(state.current_trace, state.sensitivities)
    # viz_global_state(state.current_trace)
    # viz_ocg(state.current_objective)
    println("current score $(get_score(state.current_trace))")
    # return packaged aux_state
    aux_state = Dict(
        :weights => state.weights,
        :sensitivities => deepcopy(state.sensitivities),
        :node => idx)
end
