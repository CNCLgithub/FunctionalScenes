using CSV
using Lazy
using JSON
using FileIO
using Images
using ArgParse
using DataFrames
using UnicodePlots
using FunctionalScenes

cycles_args = Dict(
    :mode => "full",
    # :mode => "none",
    :navigation => false
)

function debug_viz(r::GridRoom, p)
    d = data(r)
    m = fill(RGB{Float32}(0, 0, 0), steps(r))
    m[p] .= 1
    m[d .== obstacle_tile] .= RGB{Float32}(1, 0, 0)
    m[d .== wall_tile] .= RGB{Float32}(0, 0, 1)
    m[p] .= RGB{Float32}(0, 1, 0)
    m = imresize(m, (400, 400))
    return m
end
function render_debug(ids, name::String)
    out = "/spaths/datasets/$(name)/render_debug"
    isdir(out) || mkdir(out)
    for id in ids
        base_p = "/spaths/datasets/$(name)/scenes/$(id).json"
        p = "$(out)/$(id)"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)
        room_img = debug_viz(base, safe_shortest_paths(base))
        save("$(out)/$(id)_print.png", room_img)
    end
end
function render_stims(ids, name::String;
                      threads = Sys.CPU_THREADS)
    out = "/spaths/datasets/$(name)/render_cycles"
    isdir(out) || mkdir(out)
    # for r in eachrow(df)
    for id in ids
        base_p = "/spaths/datasets/$(name)/scenes/$(id).json"
        p = "$(out)/$(id)"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)
        render(base, p;
               cycles_args...)
    end
end

function main()
    args = Dict(
        "dataset" => "ecog_pilot",
        "scene" => 0,
        "threads" => Sys.CPU_THREADS
    )
    # args = parse_commandline()

    name = args["dataset"]
    src = "/spaths/datasets/$(name)"
    df = DataFrame(CSV.File("$(src)/scenes.csv"))
    if args["scene"] == 0
        ids = unique(df.id)
        # ids = [1, 2, 11, 14]
    else
        df = df[df.id .== args["scene"], :]
        ids = df.id
    end

    # render_debug(ids, name)
    render_stims(ids, name,
                 threads = args["threads"]
                 )
    return nothing
end



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--dataset"
        help = "Which scene to run"
        arg_type = String
        default = "ecog_pilot"

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
