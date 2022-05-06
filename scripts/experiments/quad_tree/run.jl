using CSV
using JSON
using JLD2
using FileIO
using ArgParse
using DataFrames
using FunctionalScenes
using FunctionalScenes: shift_furniture

using Random
Random.seed!(1235)

dataset = "vss_pilot_11f_32x32_restricted"

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

    return parse_args(vs, s)
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
    base = load_base_scene(base_p, door)
    room = shift_furniture(base,
                           furniture(base)[furn],
                           move)
end



function main(cmd)
    args = parse_commandline(cmd)
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

    query = query_from_params(room,
                              args["gm"])

    model_params = first(query.args)
    proc = FunctionalScenes.proc_from_params(room, model_params,
                                             args[att_mode]["params"],
                                             args["ddp"])

    try
        isdir("/spaths/experiments/$(dataset)") || mkpath("/spaths/experiments/$(dataset)")
        isdir(out_path) || mkpath(out_path)
    catch e
        println("could not make dir $(out_path)")
    end
    c = args["chain"]
    out = joinpath(out_path, "$(c).jld2")

    if isfile(out) && !args["restart"]
        println("chain $c complete")
        return
    end

    println("running chain $c")
    results = run_inference(query, proc, out )
    FunctionalScenes.viz_gt(room)
    return nothing
end



function outer()
    args = Dict("scene" => 1)
    # args = parse_outer()
    i = args["scene"]
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    cmd = ["$(i)","1", "1", "A"]
    main(cmd);
    cmd = ["$(i)", "2", "1", "A"]
    main(cmd);
    for r in eachrow(df[df.id  .== i, :])
        cmd = [
            "-f=$(r.furniture)",
            "-m=$(r.move)",
            "$(i)", "$(r.door)", "1", "A",
        ]

        display(cmd)
        main(cmd);
    end
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

outer();
