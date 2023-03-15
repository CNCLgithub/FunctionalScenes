
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
    @pycall fs_py.init_dd_state(config_path, device)::PyObject
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
