using Base.Iterators: take
using Base.Order: ReverseOrdering, Reverse
using DataStructures: PriorityQueue

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
    sensitivities::Matrix{Float64}
    queue::PriorityQueue{Int64, Float64, ReverseOrdering}
    node::Int64
end

const AMHChain = Gen_Compose.MHChain{StaticQuery, AttentionMH}

function Gen_Compose.initialize_chain(proc::AttentionMH,
                                      query::StaticQuery,
                                      n::Int)
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
    ndims = prod(dims)
    sensitivities = zeros(dims)
    queue = init_queue(trace)
    aux = AttentionAux(true,
                       sensitivities,
                       queue,
                       0)
    # initialize chain
    AMHChain(query, proc, trace, aux, 1, n)
end


function Gen_Compose.step!(chain::AMHChain)

    @debug "mc step $(i)"

    aux = auxillary(chain)
    # # initialize kernel by exploring nodes
    # aux.initialized || kernel_init!(chain)

    # proposal
    kernel_move!(chain)

    viz_chain(chain)
    println("current score $(get_score(chain.state))")
    return nothing
end

#################################################################################
# Helpers
#################################################################################

function update_weights!(chain::AMHChain)
    aux = auxillary(chain)
    proc = estimator(chain)
    aux.weights = softmax(proc.smoothness .* vec(aux.sensitivities))
    chain.auxillary = aux
    return nothing
end

function kernel_move!(chain::AMHChain)
    state = estimate(chain)
    proc = estimator(chain)
    aux = auxillary(chain)
    @unpack rw_cycles, sm_cycles, objective, distance = proc
    # current trace
    t = state
    obj_t = objective(t)
    params = first(get_args(t))
    st::QuadTreeState = get_retval(t)

    # select node to rejuv
    node, gr = first(aux.queue)
    println("ATTENTION KERNEL: node $(node); prev gr $(gr)")

    # RW moves - first stage
    accept_ct::Int64 = 0
    delta_pi::Float64 = 0.0
    delta_s::Float64 = 0.0
    for j = 1:rw_cycles
        _t, alpha = rw_move(t, node)
        obj_t_prime = objective(_t)
        delta_pi += distance(obj_t, obj_t_prime)
        delta_s += exp(clamp(alpha, -Inf, 0.))
        if log(rand()) < alpha
            t = _t
            obj_t = obj_t_prime
            accept_ct += 1
        end
    end

    # if RW acceptance ratio is high, add more
    # otherwise, ready for SM
    accept_ratio = accept_ct / rw_cycles
    addition_rw_cycles = (delta_pi > 0) * floor(Int64, sm_cycles * accept_ratio)
    for j = 1:addition_rw_cycles
        _t, alpha = rw_move(t, node)
        obj_t_prime = objective(_t)
        delta_pi += distance(obj_t, obj_t_prime)
        delta_s += exp(clamp(alpha, -Inf, 0.))
        if log(rand()) < alpha
            t = _t
            obj_t = obj_t_prime
            accept_ct += 1
        end
    end

    # compute goal-relevance
    total_cycles = rw_cycles + addition_rw_cycles
    delta_pi /= total_cycles
    delta_s /= total_cycles
    goal_relevance = delta_pi * delta_s

    # update aux state
    prod_node = traverse_qt(st.qt, node).node
    sidx = node_to_idx(prod_node, max_leaves(st.qt))
    aux.sensitivities[sidx] .= goal_relevance
    aux.queue[node] = goal_relevance
    aux.node = node

    println("\t delta pi: $(delta_pi)")
    println("\t delta S: $(delta_s)")
    println("\t goal relevance: $(goal_relevance)")
    println("\t rw acceptance ratio: $(accept_ratio)")


    # SM moves
    remaining_sm = sm_cycles - addition_rw_cycles
    can_split = prod_node.max_level > prod_node.level
    accept_ct = 0
    if can_split
        is_balanced = balanced_split_merge(t, node)
        moves = is_balanced ? [split_move, merge_move] : [split_move]
        for i = 1 : remaining_sm
            move = rand(moves)
            _t, _w = split_merge_move(t, node, move)
            if log(rand()) < _w
                t = _t
                accept_ct += 1
                update_queue!(aux, node, move)
                break
            end
        end
    end
    println("\t accepted SM move: $(accept_ct == 1)")

    display_selected_node(sidx, size(aux.sensitivities))

    # update trace
    chain.state = t
    chain.auxillary = aux
    return nothing
end

function init_queue(tr::Gen.Trace)
    st = get_retval(tr)
    q = PriorityQueue{Int64, Float64, ReverseOrdering}(Reverse)
    # go through the current set of terminal nodes
    # and intialize priority
    for n = st.qt.leaves
        q[n.node.tree_idx] = 0.1 * area(n.node)
    end
    return q
end

function update_queue!(aux::AttentionAux, node::Int64, move::Split)
    prev_val = aux.queue[node] * 0.25
    # copying parent's (node) value to children
    for i = 1:4
        cid = Gen.get_child(node, i, 4)
        aux.queue[cid] = prev_val
    end
    delete!(aux.queue, node)
    return nothing
end

function update_queue!(aux::AttentionAux, node::Int64, move::Merge)
    # merge to parent, averaging siblings relevance
    parent = Gen.get_parent(node, 4)
    prev_val = 0.
    for i = 1:4
        cid = Gen.get_child(parent, i, 4)
        prev_val += aux.queue[cid]
        delete!(aux.queue, cid)
    end
    aux.queue[parent] = prev_val
    return nothing
end

function viz_chain(chain::AMHChain)
    @unpack auxillary, state = chain
    params = first(get_args(state))
    trace_st = get_retval(state)
    # println("Attention")
    # s = size(auxillary.sensitivities)
    # display_mat(reshape(auxillary.weights, s))
    println("Inferred state")
    display_mat(trace_st.qt.projected)
    println("Estimated path")
    path = Matrix{Float64}(ex_path(chain))
    display_mat(path)
    # println("Predicted Image")
    # display_img(trace_st.img_mu)
end

function display_selected_node(sidx, dims)
    bs = zeros(dims)
    bs[sidx] .= 1
    println("Selected node")
    display_mat(bs)
end
