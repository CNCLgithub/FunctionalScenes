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
    rw_cycles::Int64 = 10
    sm_cycles::Int64 = 10
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


function Gen_Compose.initialize_chain(proc::AttentionMH,
                                      query::StaticQuery)
    # Intialize using DDP
    cm = query.observations
    tracker_cm = generate_qt_from_ddp(proc.ddp_args...)
    set_submap!(cm, :trackers,
                get_submap(tracker_cm, :trackers))
    trace,_ = Gen.generate(query.forward_function,
                           query.args,
                           cm)
    # initialize auxillary state
    dims = first(get_args(trace)).dims
    n = prod(dims)
    sensitivities = zeros(dims)
    weights = fill(1.0/n, n)
    aux = AttentionAux(false,
                       sensitivities,
                       weights,
                       0)
    # initialize chain
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

function update_weights!(chain::StaticMHChain, proc::AttentionMH)
    @unpack auxillary = chain
    auxillary.weights = softmax(proc.smoothness .* vec(auxillary.sensitivities))
    chain.auxillary = auxillary
    return nothing
end

function kernel_init!(chain::StaticMHChain, proc::AttentionMH)
    @unpack state = chain
    @unpack init_cycles, objective, distance = proc
    # current trace
    t = state
    # objective of current trace
    obj_t = objective(t)
    # loop through each node in initial trace
    st::QuadTreeState = get_retval(t)
    _t  = t
    accept_ct = 0
    for i = 1:length(st.qt.leaves)
        node = st.qt.leaves[i].node
        accept_ct = 0
        e_dist = 0
        # println("INIT KERNEL: node $(node.tree_idx)")
        for j = 1:init_cycles
            _t, alpha = rw_move(t, node.tree_idx)
            w = abs(0.5 - exp(clamp(alpha, -Inf, 0)))
            obj_t_prime = objective(_t)
            e_dist += distance(obj_t, objective(_t)) * w
            if log(rand()) < alpha
                t = _t
                obj_t = obj_t_prime
                accept_ct += 1
            end
        end
        e_dist /= init_cycles
        # map node to sensitivity matrix
        sidx = node_to_idx(node, max_leaves(st.qt))
        chain.auxillary.sensitivities[sidx] .= e_dist
        # println("\t avg distance: $(e_dist)")
        # println("\t acceptance ratio: $(accept_ct/init_cycles)")
    end

    chain.state = t
    chain.auxillary.initialized = true
    update_weights!(chain, proc)
    return nothing
end

function kernel_move!(chain::StaticMHChain, proc::AttentionMH)
    @unpack state, auxillary = chain
    @unpack rw_cycles, sm_cycles, objective, distance = proc
    # current trace
    t = state
    obj_t = objective(t)
    params = first(get_args(t))
    st::QuadTreeState = get_retval(t)

    # select node to rejuv
    room_idx = categorical(auxillary.weights)
    node = room_to_leaf(st, room_idx, params.dims[1]).node

    println("ATTENTION KERNEL: node $(node.tree_idx), prob $(auxillary.weights[room_idx])")

    # RW moves
    accept_ct = 0
    e_dist = 0
    for j = 1:rw_cycles
        _t, alpha = rw_move(t, node.tree_idx)
        w = abs(0.5 - exp(clamp(alpha, -Inf, 0)))
        obj_t_prime = objective(_t)
        e_dist += distance(obj_t, obj_t_prime) * w
        # @show alpha
        if log(rand()) < alpha
            t = _t
            obj_t = obj_t_prime
            accept_ct += 1
        end
    end
    e_dist /= rw_cycles
    sidx = node_to_idx(node, max_leaves(st.qt))
    auxillary.sensitivities[sidx] .*= 0.5
    auxillary.sensitivities[sidx] .+= 0.5 * e_dist
    auxillary.node = node.tree_idx

    println("\t avg distance: $(e_dist)")
    println("\t rw acceptance ratio: $(accept_ct/rw_cycles)")

    # SM moves
    # split randomly
    # or deterministically if node is root
    # (can still merge lvl 2 into root)
    can_split = node.max_level > node.level
    accept_ct = 0
    if can_split
        is_balanced = balanced_split_merge(t, node.tree_idx)
        moves = is_balanced ? [split_move, merge_move] : [split_move]
        for i = 1 : proc.sm_cycles
            move = rand(moves)
            _t, _w = split_merge_move(t, node.tree_idx, move)
            if log(rand()) < _w
                t = _t
                accept_ct += 1
                break
            end
        end
    end
    println("\t accepted SM move: $(accept_ct == 1)")

    # update trace
    chain.state = t
    chain.auxillary = auxillary
    # finally update weights for next step
    update_weights!(chain, proc)
    return nothing
end

function viz_chain(chain::StaticMHChain)
    @unpack auxillary, state = chain
    params = first(get_args(state))
    trace_st = get_retval(state)
    # println("Attention")
    # s = size(auxillary.sensitivities)
    # display_mat(reshape(auxillary.weights, s))
    println("Inferred state")
    display_mat(trace_st.qt.projected)
    # println("Predicted Image")
    # display_img(trace_st.img_mu)
end

function display_selected_node(sidx, dims)
    bs = zeros(dims)
    bs[sidx] .= 1
    println("Selected node")
    display_mat(bs)
end
