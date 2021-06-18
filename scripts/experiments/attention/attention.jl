using CSV
using JLD2
using FileIO
using ArgParse
using DataFrames
using FunctionalScenes
using FunctionalScenes: shift_furniture

experiment = "2e_1p_30s_matchedc3"

function parse_commandline(vs)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--gm"
        help = "Generative Model params"
        arg_type = String
        default = "$(@__DIR__)/gm.json"

        "--proc"
        help = "Inference procedure params"
        arg_type = String
        default = "$(@__DIR__)/proc.json"

        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "--viz", "-v"
        help = "Whether to render masks"
        action = :store_true

        "--move", "-m"
        help = "Which scene to run"
        arg_type = Symbol
        required = false

        "--furniture", "-f"
        help = "Which scene to run"
        arg_type = Int64
        required = false

        "scene"
        help = "Which scene to run"
        arg_type = Int64
        required = true


        "chain"
        help = "The number of chains to run"
        arg_type = Int
        required = true

        "attention", "A"
        help = "Using trackers"
        action = :command

        "naive", "N"
        help = "Naive approximation"
        action = :command


    end

    @add_arg_table! s["attention"] begin
        "--params"
        help = "Attention params"
        arg_type = String
        default = "$(@__DIR__)/attention.json"

    end
    @add_arg_table! s["naive"] begin
        "--params"
        help = "Attention params"
        arg_type = String
        default = "$(@__DIR__)/naive.json"
    end

    return parse_args(vs, s)
end

load_base_scene(path::String) = load(path)["r"]


function load_moved_scene(base_p::String, args)
    move = args["move"]
    furn = args["furniture"]
    base = load_base_scene(base_p)
    room = shift_furniture(base,
                           furniture(base)[furn],
                           move)
end



function main(cmd)
    args = parse_commandline(cmd)
    att_mode = args["%COMMAND%"]

    base_path = "/experiments/$(experiment)_$(att_mode)"
    scene = args["scene"]
    path = joinpath(base_path, "$(scene)")

    base_p = "/scenes/$(experiment)/$(scene).jld2"
    if isnothing(args["move"])
        room = load_base_scene(base_p)
    else
        room = load_moved_scene(base_p, args)
        move = args["move"]
        furn = args["furniture"]
        path = "$(path)_$(furniture)_$(move)"
    end

    query = query_from_params(room, args["gm"],
                              img_size = (240, 360),
                              dims = (6,6),
                              )

    selections = FunctionalScenes.selections(first(query.args))
    proc = FunctionalScenes.load(AttentionMH, selections,
                                 args[att_mode]["params"])

    try
        isdir(base_path) || mkpath(base_path)
        isdir(path) || mkpath(path)
    catch e
        println("could not make dir $(path)")
    end
    c = args["chain"]
    out = joinpath(path, "$(c).jld2")
    if isfile(out) && !args["restart"]
        println("chain $c complete")
        return
    end

    println("running chain $c")
    results = run_inference(query, proc, out )
    FunctionalScenes.viz_gt(query)
    # if (args["viz"])
    #     visualize_inference(results, gt_causal_graphs, gm_params, att, path;
    #                         render_tracker_masks=true)
    # end

    # df = MOT.analyze_chain(results)
    # df[!, :scene] .= args["scene"]
    # df[!, :chain] .= c
    # CSV.write(joinpath(path, "$(c).csv"), df)

    return nothing
end


df = DataFrame(CSV.File("/scenes/$(experiment).csv"))
for i = 1:30
    cmd = ["$(i)", "1", "A"]
    main(cmd);
    for r in eachrow(df[df.id  .== i, :])
        cmd = [
            "-f=$(r.furniture)",
            "-m=$(r.move)",
            "$(i)", "1", "A",
               # ("furniture", "$(r.furniture)"),
               # ("move", "$(r.move)"),
        ]

        display(cmd)
        main(cmd);
    end
end
