using CSV
using Gen: get_retval
using JSON
using JLD2
using FileIO
using ArgParse
using DataFrames
using FunctionalScenes
using FunctionalScenes: shift_furniture

using Random

dataset = "ccn_2023_exp"

function parse_commandline(c)
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

        "--ddp"
        help = "DDP config"
        arg_type = String
        default = "/project/scripts/nn/configs/og_decoder.yaml"

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

        "door"
        help = "door"
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

    return parse_args(c, s)
end

function load_base_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
end


function load_moved_scene(base_p::String, args)
    move = args["move"]
    furn = args["furniture"]
    door = args["door"]
    base = load_base_scene(base_p)
    room = shift_furniture(base,
                           furniture(base)[furn],
                           move)
end



function main(c)
    args = parse_commandline(c)
    att_mode = args["%COMMAND%"]

    base_path = "/spaths/datasets/$(dataset)/scenes"
    scene = args["scene"]
    door = args["door"]
    base_p = joinpath(base_path, "$(scene)_$(door).json")

    out_path = "/spaths/experiments/$(dataset)/$(scene)_$(door)"

    if isnothing(args["move"])
        room = load_base_scene(base_p)
    else
        room = load_moved_scene(base_p, args)
        move = args["move"]
        furn = args["furniture"]
        door = args["door"]
        out_path = "$(out_path)_$(furniture)_$(move)"
    end

    # Load query (identifies the estimand)
    query = query_from_params(room, args["gm"])

    # Load estimator - Adaptive MCMC
    model_params = first(query.args)
    ddp_params = DataDrivenState(;config_path = args["ddp"],
                                 var = 0.2)
    gt_img = render_mitsuba(room, model_params.scene, model_params.sparams,
                            model_params.skey, model_params.spp)
    proc = FunctionalScenes.load(AttentionMH, args[att_mode]["params"];
                                 ddp_args = (ddp_params, gt_img, model_params))

    try
        isdir("/spaths/experiments/$(dataset)") || mkpath("/spaths/experiments/$(dataset)")
        isdir(out_path) || mkpath(out_path)
    catch e
        println("could not make dir $(out_path)")
    end

    # save the gt image for reference
    save_img_array(gt_img, "$(out_path)/gt.png")

    for c = 1:args["chain"]
        Random.seed!(c)
        out = joinpath(out_path, "$(c).jld2")
        
        if isfile(out) && args["restart"]
            println("chain $c restarting")
            rm(out)
        end
        local results 
        if isfile(out)
            println("resuming chain $c")
            results = resume_inference(out, proc)
        else
            println("starting chain $c")
            results = run_inference(query, proc, out )
        end
        save_img_array(get_retval(results.state).img_mu,
                       "$(out_path)/img_mu.png")

    end
    return nothing
end



function outer()
    args = Dict("scene" => 2)
    # args = parse_outer()
    i = args["scene"]
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    # scene | door | chain | attention
    cmd = ["--restart", "$(i)","1", "1", "A"]
    # cmd = ["--restart", "$(i)", "2", "1", "A"]
    main(cmd);
    # for r in eachrow(df[df.id  .== i, :])
    #     cmd = [
    #         "-f=$(r.furniture)",
    #         "-m=$(r.move)",
    #         "$(i)", "$(r.door)", "1", "A",
    #     ]

    #     display(cmd)
    #     main(cmd);
    # end
end



function parse_outer()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "scene"
        help = "Which scene to run"
        arg_type = Int64
        required = true

    end

    return parse_args(s)
end

# main();
outer();
