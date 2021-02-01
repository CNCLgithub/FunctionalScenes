using Base.Iterators: take

export AttentionMH

@with_kw struct AttentionMH <: Gen_Compose.MCMC
    # inference parameters
    samples::Int64 = 10
    smoothness::Float64 = 1.0
    explore::Float64 = 0.30

    # task objective
    objective::Function = batch_og
    # destance metrics for two task objectives
    distance::Function = wsd

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
    visited::Vector{Bool}
    sensitivities::LittleDict{Symbol, Float64}
    weights::LittleDict{Symbol, Float64}
end

function Gen_Compose.initialize_procedure(proc::AttentionMH,
                                          query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    n = length(proc.nodes)
    sensitivities = fill(-Inf, n)
    sensitivities = LittleDict(proc.nodes,
                               sensitivities)
    weights = fill(1.0/n, n)
    weights = LittleDict(proc.nodes,
                         weights)
    AMHTrace(trace,
             proc.objective(trace),
             fill(false, n),
             sensitivities,
             weights)
end

function Gen_Compose.mc_step!(state::AMHTrace,
                              proc::AttentionMH,
                              query::StaticQuery)

    prev_objective = state.current_objective
    # draw proposal
    (new_tr, ll, addr) = ancestral_kernel_move(proc, state)
    # compare objectives and update sensitivity record
    new_objective = proc.objective(new_tr)
    dist = proc.distance(prev_objective, new_objective)
    update_sensitivity!(state, addr, ll, dist)
    update_weights!(state, proc)

    accepted = log(rand()) < ll
    if accepted
        state.current_trace = new_tr
        state.current_objective = new_objective
    end
    # viz_render(state.current_trace)
    # viz_compute_weights(state.weights)
    viz_sensitivity(state.current_trace, state.sensitivities)
    viz_global_state(state.current_trace)
    viz_ocg(state.current_objective)
    println("current score $(get_score(state.current_trace))")
    # return packaged aux_state
    aux_state = Dict(
        :objective => new_objective,
        :weights => state.weights,
        :sensitivities => state.sensitivities,
        :addr => addr)
end

function update_sensitivity!(state::AMHTrace, addr, ll::Float64, dist::Float64)
    v = state.sensitivities[addr]
    println(v)
    v = log(exp(log(dist) + clamp(ll, -Inf, 0.)) + exp(v))
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
    # if length(infs) != length(new_weights)
    #     min_s = @>> new_weights Base.filter((!isinf)) minimum
    #     new_weights[infs] .= min_s
    # end
    new_weights = softmax(proc.smoothness .* new_weights)
    state.weights = LittleDict(ks, Tuple(new_weights))
    return nothing
end


function ancestral_kernel_move(proc::AttentionMH,
                               state::AMHTrace)

    trace = state.current_trace
    weights = @>> values(state.weights) collect(Float64)
    # select addr to rejuv
    # chance to ingnore weights
    n = length(weights)
    unvisited = findall(!, state.visited)
    if !isempty(unvisited)
        _ws = zeros(size(weights))
        _ws[unvisited] .= 1.0 / length(unvisited)
        weights = _ws
    end
    if bernoulli(proc.explore)
        weights = 1.0 .- weights
        weights = softmax(weights)
    end
    idx = categorical(weights)
    state.visited[idx] = true
    addr = proc.nodes[idx]
    selection = proc.selections[addr]


    println(addr)
    (new_tr, ll, _) = regenerate(trace, selection)
    (new_tr, ll, addr)
end
