using CSV
using Lazy
using JSON
using FileIO
using ArgParse
using DataFrames
using FunctionalScenes

cycles_args = Dict(
    :mode => "full",
    # :mode => "none",
    :navigation => false
)

function render_stims(df::DataFrame, name::String;
                      threads = Sys.CPU_THREADS)
    out = "/spaths/datasets/$(name)/render_cycles"
    isdir(out) || mkdir(out)
    for r in eachrow(df)
        base_p = "/spaths/datasets/$(name)/scenes/$(r.id)_$(r.door).json"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)
        p = "$(out)/$(r.id)_$(r.door)"
        render(base, p;
               cycles_args...)
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               Symbol(r.move))
        p = "$(out)/$(r.id)_$(r.door)_$(r.furniture)_$(r.move)"
        render(room, p;
               cycles_args...)
    end
end

function main()
    # args = Dict(
    #   "dataset" => "vss_pilot",
    #    "scene" => 0,
    #    "threads" => Sys.CPU_THREADS
    #)
    args = parse_commandline()

    name = args["dataset"]
    src = "/spaths/datasets/$(name)"
    df = DataFrame(CSV.File("$(src)/scenes.csv"))
    seeds = unique(df.id)
    if args["scene"] == 0
        seeds = unique(df.id)
    else
        seeds = [args["scene"]]
        df = df[df.id .== args["scene"], :]
    end

    render_stims(df, name,
                 threads = args["threads"]
                 )
    return nothing
end



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "dataset"
        help = "Which scene to run"
        arg_type = String
        required = true

        "scene"
        help = "Which scene to run"
        arg_type = Int64
        required = true


        "--threads"
        help = "Number of threads for cycles"
        arg_type = Int64
        default = 4
    end
    return parse_args(s)
end



main();
