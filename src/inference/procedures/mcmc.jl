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
    objective::Function = qt_path_cost
    # destance metrics for two task objectives
    distance::Function = (x, y) -> norm(x - y)

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
    AttentionMH(; loaded...,
                kwargs...)
end


mutable struct AttentionAux <: AuxillaryState
    initialized::Bool
    sensitivities::Array{Float64}
    weights::Array{Float64}
    node::Int64
end

function update_weights!(chain::StaticMHChain, proc::AttentionMH)
    @unpack auxillary = chain
    auxillary.weights = softmax(proc.smoothness .* vec(auxillary.sensitivities))
    chain.auxillary = auxillary
    return nothing
end


# function update_structure!(state::AMHTrace, proc::AttentionMH, ::Split,
#                            idx::Int64)
#     @inbounds for i = 1 : 4
#         c = Gen.get_child(idx, i, 4)
#         state.sensitivities[c] = state.sensitivities[idx]
#     end
#     state.sensitivies[idx] = 0.0
#     return nothing
# end

# function update_structure!(state::AMHTrace, proc::AttentionMH, ::Merge,
#                            idx::Int64)
#     parent = Gen.get_parent(idx, 4)
#     state.sensitivities[parent] = 0.0
#     @inbounds for i = 1 : 4
#         c = Gen.get_child(idx, i, 4)
#         state.sensitivities[parent] += state.sensitivities[c]
#         state.sensitivities[c] = 0.0
#     end
#     return nothing
# end

function kernel_init!(chain::StaticMHChain, proc::AttentionMH)
    @unpack state = chain
    @unpack init_cycles, objective, distance = proc
    t = state
    st::QuadTreeState = get_retval(t)
    lls = Vector{Float64}(undef, init_cycles)
    distances = Vector{Float64}(undef, init_cycles)
    # loop through each node in initial trace
    _t  = t
    accept_ct = 0
    for i = 1:length(st.lv)
        node = st.lv[i].node
        accept_ct = 0
        e_dist = 0
        println("INIT KERNEL: node $(node.tree_idx)")
        for j = 1:init_cycles
            _t, lls[j] = lateral_move(t, node.tree_idx)
            # distances[j] = distance(objective(t), objective(_t))
            if log(rand()) < lls[j]
                e_dist += distance(objective(t), objective(_t))
                t = _t
                accept_ct += 1
            end
        end
        # clamp!(lls, -Inf, 0.)
        # compute expectation over sensitivities
        # e_dist = sum(distances .* exp.((lls .- logsumexp(lls))))
        sidx = node_to_idx(node, size(st.gs, 1))
        chain.auxillary.sensitivities[sidx] .= isinf(e_dist) ? 0. : e_dist # clean up -Inf
        println("\t avg distance: $(e_dist)")
        println("\t acceptance ratio: $(accept_ct/init_cycles)")
    end

    chain.state = t
    chain.auxillary.initialized = true
    update_weights!(chain, proc)
    return nothing
end

function kernel_move!(chain::StaticMHChain, proc::AttentionMH)
    @unpack state, auxillary = chain
    t = state
    params = first(get_args(t))
    st::QuadTreeState = get_retval(t)
    # select node to rejuv
    room_idx = categorical(auxillary.weights)
    node = ridx_to_leaf(st, room_idx, params.dims[1]).node

    @debug "Attending to node $(node.tree_idx)"

    println("ATTENTION KERNEL: node $(node.tree_idx)")
    @unpack lateral_cycles, vertical_cycles, objective, distance = proc

    # lateral moves
    lls = Vector{Float64}(undef, lateral_cycles)
    distances = Vector{Float64}(undef, lateral_cycles)
    accept_ct = 0
    e_dist = 0
    for i = 1:lateral_cycles
        _t, lls[i] = lateral_move(t, node.tree_idx)
        # distances[i] = distance(objective(t), objective(_t))
        if log(rand()) < lls[i]
            e_dist += distance(objective(t), objective(_t))
            t = _t
            accept_ct += 1
        end
    end
    # clamp!(lls, -Inf, 0.)
    # compute expectation over sensitivities
    # @debug "distances: $(distances)"
    # e_dist = sum(distances .* exp.((lls .- logsumexp(lls))))
    sidx = node_to_idx(node, size(st.gs, 1))
    auxillary.sensitivities[sidx] .= isinf(e_dist) ? 0. : e_dist # clean up -Inf
    auxillary.node = node.tree_idx

    println("\t avg distance: $(e_dist)")
    println("\t lateral acceptance ratio: $(accept_ct/lateral_cycles)")

    # attempt vertical moves
    accept_ct = 0
    for i = 1 : proc.vertical_cycles
        _t, _ls, direction = vertical_move(t, node.tree_idx)
        @debug "vmove log score: $(_ls)" 
        if direction == no_change
            @debug "node not balanced for split-merge"
            break
        end
        if log(rand()) < _ls
            t = _t
            accept_ct += 1
            break
        end
    end
    println("\t accepted vertical move: $(accept_ct == 1)")

    # update trace
    chain.state = t
    chain.auxillary = auxillary
    # need to update metrics given structure change
    # accepted && update_structure!(state, proc, direction, idx)
    # finally update weights for next step
    update_weights!(chain, proc)
    return nothing
end

function viz_chain(chain::StaticMHChain)
    @unpack auxillary, state = chain
    params = first(get_args(state))
    trace_st = get_retval(state)
    #  viz_room(params.gt)
    viz_grid(auxillary.sensitivities, "Attention")
    viz_grid(trace_st.gs, "Inferred state")
end

function Gen_Compose.initialize_chain(proc::AttentionMH,
                                      query::StaticQuery)
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           query.observations)
    cm = choicemap()
    cm[:trackers => (1, Val(:production)) => :produce] = true
    for i = 1:4
        cm[:trackers => (i+1, Val(:production)) => :produce] = true
        for j = 1:4
            cm[:trackers => (Gen.get_child(i+1, j, 4), Val(:production)) => :produce] = false
        end 
    end
    trace, _ = update(trace, cm)
    trace,_ = proc.ddp(trace, proc.ddp_args...)
    dims = first(get_args(trace)).dims
    n = prod(dims)
    sensitivities = zeros(dims)
    weights = fill(1.0/n, n)
    aux = AttentionAux(false,
                       sensitivities,
                       weights,
                       0)
    StaticMHChain(query,  proc, trace, aux)
end

function Gen_Compose.mc_step!(chain::StaticMHChain,
                              proc::AttentionMH,
                              i::Int)

    @debug "mc step $(i)"

    @unpack auxillary = chain
    # initialize kernel by exploring nodes
    auxillary.initialized || kernel_init!(chain, proc)

    # proposal
    kernel_move!(chain, proc)

    viz_chain(chain)
    println("current score $(get_score(chain.state))")
    return nothing
end
