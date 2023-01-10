using CSV
using Lazy
using JSON
using FileIO
using Images
using ArgParse
using DataFrames
using UnicodePlots
using FunctionalScenes
import Base: println

cycles_args = Dict(
    :mode => "full",
    # :mode => "none",
    :navigation => false
)

function Base.println(tiles::Matrix{Tile})
    for row in eachrow(tiles)
      println(row)
    end
  end

function rem_front_wall(r::GridRoom)
    newdata = deepcopy(data(r))
    newdata[:, 1:2] .= floor_tile
    newroom = GridRoom(r, newdata)

    # can be commented out, i just like seeing the rooms - Chloe 01/02/2023
    println(newdata)

    return newroom
end

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

        println("ROOM NUMBER: $(id)")
        room = from_json(GridRoom, base_s)

        # remove front wall here
        newroom = rem_front_wall(room)

        room_img = debug_viz(newroom, safe_shortest_paths(newroom))
        save("$(out)/$(id)_print.png", room_img)
    end
end

function render_stims(ids, name::String;
                      threads = Sys.CPU_THREADS)
    out = "/spaths/datasets/$(name)/render_cycles"
    isdir(out) || mkdir(out)

    for id in ids
        base_p = "/spaths/datasets/$(name)/scenes/$(id).json"
        p = "$(out)/$(id)"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)

        # remove front wall here
        println("ROOM NUMBER: $(id)")
        newroom = rem_front_wall(base)

        render(newroom, p;
               cycles_args...)
    end
end

# essentially the same function as `main()``, just wrapped in 3 loops to set params
function set_params()
    quant_ls = collect(range(0.25, stop=0.75, step=0.25))
    temp_ls = collect(range(0.25, stop=4, step=0.25))
    max_f_ls = collect(range(4, stop=10, step=1))

    for t in eachindex(temp_ls)
        for m in eachindex(max_f_ls)
            for q in eachindex(quant_ls)
                args = Dict(
                    "dataset" => "ecog_pilot_max-f=$(max_f_ls[m])_temp=$(temp_ls[t])_quant=$(quant_ls[q])",
                    "scene" => 0,
                    "threads" => Sys.CPU_THREADS
                )

                name = args["dataset"]
                src = "/spaths/datasets/$(name)"
                df = DataFrame(CSV.File("$(src)/scenes.csv"))
                if args["scene"] == 0
                    ids = unique(df.id)
                else
                    df = df[df.id .== args["scene"], :]
                    ids = df.id
                end
            
                # render_debug(ids, name)
                render_stims(ids, name,
                             threads = args["threads"]
                             )
            end
        end
    end
    return nothing
end

# essentially the same function as `main()``, just wrapped in 1 loop to get selected scenes based on params
function select_scenes()
    scenes_ls = ["ecog_pilot_max-f=5_temp=1.0_quant=0.25", "ecog_pilot_max-f=5_temp=1.5_quant=0.5", "ecog_pilot_max-f=5_temp=2.25_quant=0.75", "ecog_pilot_max-f=5_temp=3.0_quant=0.75", "ecog_pilot_max-f=6_temp=1.0_quant=0.5", "ecog_pilot_max-f=6_temp=3.5_quant=0.25", "ecog_pilot_max-f=7_temp=2.0_quant=0.25"]

    for s in eachindex(scenes_ls)
        args = Dict(
            "dataset" => "$(scenes_ls[s])",
            "scene" => 0,
            "threads" => Sys.CPU_THREADS
        )

        name = args["dataset"]
        src = "/spaths/datasets/$(name)"
        df = DataFrame(CSV.File("$(src)/scenes.csv"))
        if args["scene"] == 0
            ids = unique(df.id)
        else
            df = df[df.id .== args["scene"], :]
            ids = df.id
        end
    
        # render_debug(ids, name)
        render_stims(ids, name,
                        threads = args["threads"]
                        )
    end
    return nothing
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

    render_debug(ids, name)
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
# set_params();
# select_scenes();