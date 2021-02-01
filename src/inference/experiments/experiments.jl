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

export run_inference, query_from_params
