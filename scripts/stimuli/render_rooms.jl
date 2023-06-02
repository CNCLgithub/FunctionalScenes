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
    :navigation => false,
    :template => "/spaths/datasets/vss_template.blend"
)

function render_stims(df::DataFrame, name::String;
                      threads = Sys.CPU_THREADS)
    out = "/spaths/datasets/$(name)/render_cycles"
    isdir(out) || mkdir(out)
    for r in eachrow(df)
        base_p = "/spaths/datasets/$(name)/scenes/$(r.scene)_$(r.door).json"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)
        p = "$(out)/$(r.scene)_$(r.door)"
        render(base, p;
               cycles_args...)
        room = shift_furniture(base,
                               furniture(base)[r.furniture],
                               Symbol(r.move))
        p = "$(out)/$(r.scene)_$(r.door)_shifted"
        render(room, p;
               cycles_args...)
    end
end

function main()
    cmd = ["pathcost_6.0", "0"]
    args = parse_commandline(;x=cmd)

    name = args["dataset"]
    src = "/spaths/datasets/$(name)"
    df = DataFrame(CSV.File("$(src)/scenes.csv"))
    seeds = unique(df.scene)
    if args["scene"] == 0
        seeds = unique(df.scene)
    else
        seeds = [args["scene"]]
        df = df[df.scene .== args["scene"], :]
    end

    render_stims(df, name,
                 threads = args["threads"]
                 )
    return nothing
end



function parse_commandline(;x=ARGS)
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
    return parse_args(x, s)
end



main();
