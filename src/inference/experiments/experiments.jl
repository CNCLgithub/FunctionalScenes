export run_inference, resume_inference, query_from_params, proc_from_params

function run_inference(query::StaticQuery,
                       proc::AttentionMH)

    results = static_monte_carlo(proc, query,
                                 buffer_size = proc.samples)
end
function run_inference(query::StaticQuery,
                       proc::AttentionMH,
                       path::String)

    results = static_monte_carlo(proc, query,
                                 buffer_size = proc.samples,
                                 path = path)
end
function resume_inference(path::String, proc::AttentionMH)
    Gen_Compose.resume_mc_chain(path, proc.samples)
end

function ex_choicemap(c::StaticMHChain)
    ex_choicemap(c.state)
end
function ex_choicemap(tr::Gen.Trace)
    s = Gen.complement(select(:viz, :instances))
    choices = get_choices(tr)
    get_selected(choices, s)
end

function ex_global_state(c::StaticMHChain)
     st = get_retval(c.state)
     deepcopy(st.qt.projected)
end

function ex_img_mu(c::StaticMHChain)
    st = get_retval(c.state)
    deepcopy(st.img_mu)
end

function ex_attention(c::StaticMHChain)
    @unpack auxillary = c
    Dict(:weights => deepcopy(auxillary.weights),
         :sensitivities => deepcopy(auxillary.sensitivities),
         :node => deepcopy(auxillary.node))
end

function query_from_params(room::GridRoom, path::String; kwargs...)

    _lm = Dict{Symbol, Any}(
        :trackers => ex_choicemap,
        :global_state => ex_global_state,
        :img_mu => ex_img_mu,
        :attention => ex_attention
    )
    latent_map = LatentMap(_lm)

    gm_params = load(QuadTreeModel, path; gt = room,
                     kwargs...)

    obs = create_obs(gm_params)
    # add_init_constraints!(obs, gm_params, room)

    viz_room(room)

    # compiling further observations for the model
    query = Gen_Compose.StaticQuery(latent_map,
                                    qt_model,
                                    (gm_params,),
                                    obs)
end

# function proc_from_params(gt::Room,
#                           model_params::ModelParams, proc_path::String,
#                           vae::String, ddp::String; kwargs...)

#     # init ddp
#     img = image_from_instances([gt], model_params)
#     ddp_params = DataDrivenState(;vae_weights = vae,
#                                  ddp_weights = ddp)
#     all_selection = FunctionalScenes.all_selections_from_model(model_params)
#     ddp_args = ((ddp_params, img), all_selection)

#     selections = FunctionalScenes.selections(model_params)
#     proc = FunctionalScenes.load(AttentionMH, selections, proc_path;
#                                  ddp_args = ddp_args,
#                                  kwargs...)
# end
