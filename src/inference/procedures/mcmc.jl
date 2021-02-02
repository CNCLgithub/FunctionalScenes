using Base.Iterators: take

export AttentionMH

@with_kw struct AttentionMH <: Gen_Compose.MCMC
    # inference parameters
    samples::Int64 = 10
    smoothness::Float64 = 1.0
    explore::Float64 = 0.30
    burnin::Int64 = 10

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
    sensitivities::LittleDict{Symbol, Float64}
    weights::LittleDict{Symbol, Float64}
end

function Gen_Compose.initialize_procedure(proc::AttentionMH,
                                          query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    n = length(proc.nodes)
    # sensitivities = fill(-Inf, n)
    sensitivities = zeros(n)
    sensitivities = LittleDict(proc.nodes,
                               sensitivities)
    weights = fill(1.0/n, n)
    weights = LittleDict(proc.nodes,
                         weights)
    AMHTrace(trace,
             proc.objective(trace),
             zeros(n),
             sensitivities,
             weights)
end

function Gen_Compose.mc_step!(state::AMHTrace,
                              proc::AttentionMH,
                              query::StaticQuery)

    prev_objective = state.current_objective
    # draw proposal
    addr = ancestral_kernel_move!(proc, state)
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
        :sensitivities => state.sensitivities,
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
    ks = Tuple(proc.nodes)
    new_weights = @>> values(state.sensitivities) collect(Float64)
    infs = findall(isinf, new_weights)
    if length(infs) != length(new_weights)
        min_s = @>> new_weights Base.filter((!isinf)) minimum
        new_weights[infs] .= min_s
    end
    new_weights = softmax(proc.smoothness .* new_weights)
    state.weights = LittleDict(ks, Tuple(new_weights))
    return nothing
end

function sweep(proc::AttentionMH,
               state::AMHTrace)

end

function ancestral_kernel_move!(proc::AttentionMH,
                               state::AMHTrace)


    trace = state.current_trace
    prev_objective = state.current_objective
    weights = @>> values(state.weights) collect(Float64)
    # select addr to rejuv
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
    state.counters[idx] += 1
    addr = proc.nodes[idx]
    println(addr)
    selection = proc.selections[addr]

    current_trace = trace
    distances = Vector{Float64}(undef, proc.burnin)
    lls = Vector{Float64}(undef, proc.burnin)
    for i = 1:proc.burnin
        (new_tr, lls[i], _) = regenerate(trace, selection)
        new_objective = proc.objective(new_tr)
        distances[i] = proc.distance(prev_objective, new_objective)
        accepted = log(rand()) < lls[i]
        current_trace = accepted ? new_tr : current_trace
    end

    lws = exp.(lls .- logsumexp(lls))
    exp_dist = sum(distances .* lws)
    state.sensitivities[addr] = exp_dist
    state.current_trace = current_trace
    state.current_objective = proc.objective(current_trace)
    return addr
end
