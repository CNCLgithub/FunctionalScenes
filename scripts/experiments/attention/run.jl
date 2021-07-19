using CSV
using JLD2
using FileIO
using ArgParse
using DataFrames
using FunctionalScenes
using FunctionalScenes: shift_furniture

experiment = "1_exit_22x40_doors"

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

        "--vae"
        help = "DDP VAE weights"
        arg_type = String
        default = "/checkpoints/vae"

        "--ddp"
        help = "DDP decoder weights"
        arg_type = String
        default = "/checkpoints/ddp"

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

load_base_scene(path::String, door::Int64) = load(path)["rs"][door]


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

    base_path = "/experiments/$(experiment)_$(att_mode)"
    scene = args["scene"]
    door = args["door"]
    path = joinpath(base_path, "$(scene)_$(door)")

    base_p = "/scenes/$(experiment)/$(scene).jld2"
    if isnothing(args["move"])
        room = load_base_scene(base_p, args["door"])
    else
        room = load_moved_scene(base_p, args)
        move = args["move"]
        furn = args["furniture"]
        door = args["door"]
        path = "$(path)_$(move)_$(furniture)_$(move)"
    end

    tracker_ps = ones(3, 6)
    tracker_ps[:, 1:2] .= 0.01
    tracker_ps = vec(tracker_ps)
    query = query_from_params(room, args["gm"],
                              img_size = (240, 360),
                              dims = (6,6),
                              tracker_ps = tracker_ps
                              )

    model_params = first(query.args)
    proc = FunctionalScenes.proc_from_params(room, model_params,
                                             args[att_mode]["params"],
                                             args["vae"], args["ddp"];)

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
    FunctionalScenes.viz_gt(room)
    return nothing
end



function outer()
    # args = Dict("scene" => 1)
    args = parse_outer()
    i = args["scene"]
    df = DataFrame(CSV.File("/scenes/$(experiment).csv"))
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
