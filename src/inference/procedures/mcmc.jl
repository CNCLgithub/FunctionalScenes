using Base.Iterators: take

export AttentionMH

@with_kw struct AttentionMH <: Gen_Compose.MCMC
    # inference parameters
    samples::Int64 = 10
    smoothness::Float64 = 1.0
    explore::Float64 = 0.30 # probability of sampling random tracker
    burnin::Int64 = 10 # number of steps per tracker

    # data driven initialization
    ddp::Function = dd_init_kernel
    ddp_args::Tuple 

    # task objective
    objective::Function = batch_og
    # destance metrics for two task objectives
    distance::Function = batch_compare_og

    # adress schema
    nodes::Vector{Symbol}
    selections::LittleDict{Symbol, Gen.Selection}
end


function load(::Type{AttentionMH}, selections, path::String; kwargs...)
    nodes = @>> keys(selections) collect(Symbol)
    loaded = read_json(path)
    AttentionMH(; nodes = nodes, selections = selections, loaded...)
end


mutable struct AMHTrace <: MCMCTrace
    current_trace::Gen.Trace
    current_objective
    counters::Vector{Int64}
    sensitivities::Vector{Float64}
    weights::Vector{Float64}
end

function Gen_Compose.initialize_procedure(proc::AttentionMH,
                                          query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    trace,_ = proc.ddp(trace, proc.ddp_args...)

    n = length(proc.nodes)
    sensitivities = zeros(n)
    weights = fill(1.0/n, n)
    counters = zeros(n)
    AMHTrace(trace,
             proc.objective(trace),
             counters,
             sensitivities,
             weights)
end

function Gen_Compose.mc_step!(state::AMHTrace,
                              proc::AttentionMH,
                              query::StaticQuery)

    # draw proposal
    addr = kernel_move!(state, proc)
    update_weights!(state, proc)

    # viz_render(state.current_trace)
    viz_compute_weights(state.weights)
    viz_sensitivity(state.current_trace, state.sensitivities)
    viz_global_state(state.current_trace)
    viz_ocg(state.current_objective)
    println("current score $(get_score(state.current_trace))")
    # return packaged aux_state
    aux_state = Dict(
        :objective => state.current_objective,
        :weights => state.weights,
        :sensitivities => deepcopy(state.sensitivities),
        :addr => addr)
end

function update_sensitivity!(state::AMHTrace, addr, ll::Float64, dist::Float64)
    v = state.sensitivities[addr]
    println(v)
    v = ll >= 0.0  ? dist : exp(ll)*dist + (1 - exp(ll))*v
    # v = log(exp(log(dist) + ll) + exp(v))
    # v = log(dist + exp(v))
    # v -= log(2)
    println(" ll $(ll) -> dist $(dist) -> $(v)")
    state.sensitivities[addr] = v
    return nothing
end

function update_weights!(state::AMHTrace, proc::AttentionMH)
    state.weights = softmax(proc.smoothness .* state.sensitivities)
    return nothing
end

function sweep(proc::AttentionMH,
               state::AMHTrace)

end

function kernel_move!(state::AMHTrace, proc::AttentionMH)

    @unpack current_trace, current_objective, weights = state

    # select tracker to rejuv
    # chance to ingnore weights
    n = length(weights)
    unvisited = findfirst(state.counters .== 0)
    if !isnothing(unvisited)
        idx = unvisited
    else
        if bernoulli(proc.explore)
            weights = fill(1.0/n, n)
        end
        idx = categorical(weights)
    end

    # update record of attended trackers
    state.counters[idx] += 1
    addr = proc.nodes[idx]

    @unpack burnin, objective, selections, distance = proc
    selection = selections[addr]

    translator = Gen.SymmetricTraceTranslator(split_merge_proposal,
                                              (idx,),
                                              split_merge_involution)
    distances = zeros(burnin)
    lls = Vector{Float64}(undef, burnin)
    for i = 1:proc.burnin
        (_trace, weight) = tracker_kernel(current_trace, translator, idx, selection)
        @debug "kernel proposal weight $(weight)"
        if log(rand()) < weight
            # accept
            @debug "accepted"
            new_objective = objective(_trace)
            distances[i] = distance(current_objective, new_objective)
            current_objective = new_objective
            current_trace = _trace
        end

    end

    # compute expectation over sensitivities
    @debug "distances: $(distances)"
    exp_dist = mean(distances)

    # update references
    state.sensitivities[idx] = exp_dist
    @show state.sensitivities
    state.current_trace = current_trace
    state.current_objective = current_objective

    # return attended tracker
    return addr
end
