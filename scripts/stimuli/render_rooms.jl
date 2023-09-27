using CSV
using JSON
using FileIO
using ArgParse
using DataFrames
using FunctionalScenes
using FunctionalScenes: render_mitsuba
using Images:colorview, RGB

cycles_args = Dict(
    :mode => "full",
    # :mode => "none",
    :navigation => false,
    :template => "/spaths/datasets/vss_template.blend",
)

function load_from_disk(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    from_json(GridRoom, base_s)
end

function render_stims(df::DataFrame, name::String;
                      threads = Sys.CPU_THREADS)
    out = "/spaths/datasets/$(name)/render_cycles"
    isdir(out) || mkdir(out)
    for r in eachrow(df), door = 1:2
        base_p = "/spaths/datasets/$(name)/scenes/$(r.scene)_$(door).json"
        base = load_from_disk(base_p)
        p = "$(out)/$(r.scene)_$(door)"
        # original room
        render(base, p;
               cycles_args...)

        # removed one obstacle cluster
        f = furniture(base)[r.fidx]
        rem = remove(base, f)
        p = "$(out)/$(r.scene)_$(door)_removed"
        render(rem, p;
               cycles_args...)
    end
end

function main()
    cmd = ["diffusion_n_block", "0"]
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
                 threads = args["threads"],
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
