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

function ex_choicemap(tr::Gen.Trace)
    s = Gen.complement(select(:viz))
    choices = get_choices(tr)
    get_selected(choices, s)
end

function query_from_params(room::Room, path::String; kwargs...)

    # TODO Extract attention stats
    _lm = Dict{Symbol, Any}(
        :trace => ex_choicemap
    )

    latent_map = LatentMap(_lm)

    gm_params = load(ModelParams, path; gt = room,
                  kwargs...)
    obs = create_obs(gm_params, room)


    # compiling further observations for the model
    query = Gen_Compose.StaticQuery(latent_map,
                                    model,
                                    (gm_params,),
                                    obs)
end

function proc_from_params(model_params::ModelParams, proc_path::String, 
                          vae::String, ddp::String; kwargs...)
    selections = FunctionalScenes.selections(model_params)
    all_selection = FunctionalScenes.all_selections_from_model(model_params)
    proc = FunctionalScenes.load(AttentionMH, selections,
                                 args[att_mode]["params"])

export run_inference, query_from_params
