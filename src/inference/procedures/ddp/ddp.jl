export DataDrivenState, dd_state_proposal

@with_kw struct DataDrivenState
    # neural network details
    config_path::String
    device::PyObject = _load_device()
    nn::PyObject = _init_dd_state(config_path, device)

    # proposal variables
    var::Float64 = 0.05

end


function _init_dd_state(config_path::String, device::PyObject)
    nn = @pycall fs_py.init_dd_state(config_path, device)::PyObject
    nn.to(device)
    return nn
end


# proposal and involution here
include("gen.jl")

# Function used to initialize the chain under `mcmc::kernel_init!`
function ddp_init_kernel(trace::Gen.Trace, prop_args::Tuple)
    translator = SymmetricTraceTranslator(dd_state_proposal,
                                          prop_args,
                                          dd_state_transform)
    (new_trace, w1) = translator(trace)
    # st = get_retval(new_trace)
    # # REVIEW: probably not neccessary
    # w2 = 0
    # for i = 1:length(st.lv)
    #     node = st.lv[i].node.tree_idx
    #     s = downstream_selection(new_trace, node)
    #     (new_trace, _w) = regenerate(new_trace, s)
    #     w2 += _w
    # end
    # @debug "w1 $w1  + w2 $w2 = $(w1 + w2)"
    # (new_trace, w1 + w2)
end

function generate_qt_from_ddp(ddp_params::DataDrivenState, img, model_params,
                              min_depth::Int64 = 1)
    @unpack nn, device, var = ddp_params

    max_depth::Int64 = 4
    pimg = permutedims(img, (3,1,2))
    x = @pycall torch.tensor(pimg, device = device)::PyObject
    x = @pycall x.unsqueeze(0)::PyObject
    x = @pycall nn.determ_forward(x)::PyObject
    state = @pycall x.detach().cpu().numpy()::Matrix{Float64}
    # setting unseen corners to prior
    # TODO: this could be acheived with a better training dataset
    state[1:6, 1:6] .= 0.5
    state[end-6:end, 1:6] .= 0.5
    println("Data-driven state")
    display_mat(state)
    head = model_params.start_node
    d = model_params.dims[2]
    # Iterate through QT
    cm = choicemap()
    queue = [model_params.start_node]
    while !isempty(queue)
        head = pop!(queue)
        idx = node_to_idx(head, d)
        mu = mean(state[idx])
        sd = std(state[idx], mean = mu)
        # split = sd > ddp_params.var && head.level < head.max_level
        # restricting depth of nn
        split = head.level < min_depth || (sd > ddp_params.var && head.level < max_depth)
        cm[:trackers => (head.tree_idx, Val(:production)) => :produce] = split
        if split
            # add children to queue
            append!(queue, produce_qt(head))
        else
            # terminal node, add aggregation choice
            cm[:trackers => (head.tree_idx, Val(:aggregation)) => :mu] = mu
        end
    end
    return cm
end
