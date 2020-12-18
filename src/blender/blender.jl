using JSON


function floor(r::Room)
    dx,dy = bounds(r)
    Dict(:position => [0,0,0],
         :orientation => [0,0,0],
         :shape => :Plane,
         :dims => [dx, dy, 0],
         :appearance => :white)
end

function tile(t, coords, space)
    dx,dy = space
    dz = sqrt(dx^2 + dy^2)
    dz *= t == :wall ? 3.0 : 1.0
    pos = [coords..., dz / 2.0]
    Dict(:position => pos,
         :orientation => [0,0,0],
         :shape => :Block,
         :dims => [dx,dy,dz],
         :appearance => :blue)
end


function spot(coords, space; exit::Bool=false)
    dx,dy = space
    dz = 0.01
    pos = [coords..., dz]
    Dict(:position => pos,
         :orientation => [0,0,0],
         :shape => :Plane,
         :dims => [dx,dy, 0],
         :appearance => exit ? :red : :green)
end

function tiles(r::Room)
    g = pathgraph(r)
    space = bounds(r) ./ steps(r)
    offset = 0.5 .* steps(r) .* space .+ (0.5 .* space)
    vs = @>> g vertices filter(v -> !isfloor(g, v))
    cis = CartesianIndices(steps(r))
    vcis = Tuple.(cis[vs])
    coords = map(c -> c .* space .- offset, vcis)
    display(space)
    display(offset)
    display(coords)
    types = @>> vs lazymap(v -> get_prop(g, v, :type))
    @>> zip(types, coords) lazymap(xy -> tile(xy..., space)) collect
end

function task(r::Room)
    g = pathgraph(r)
    cis = CartesianIndices(steps(r))
    space = bounds(r) ./ steps(r)
    offset = 0.5 .* steps(r) .* space .+ (0.5 .* space)
    transform = c -> c .* space .- offset
    ents = @>> Tuple.(cis[entrance(r)]) lazymap(transform)
    ents = @>> ents lazymap(x -> spot(x, space)) collect

    exts = @>> Tuple.(cis[exits(r)]) lazymap(transform)
    exts = @>> exts lazymap(x -> spot(x, space; exit=true)) collect

    vcat(ents, exts)
end

function render(r::Room, script::String, template::String, out::String;
                blender="blender", threads=Sys.CPU_THREADS,
                )
    scene = Dict(:floor => floor(r),
                 :objects => vcat(tiles(r), task(r))
                 ) |> json

    isdir(out) || mkpath(out)

    scene_out = joinpath(out, "scene.json")
    open(scene_out, "w") do f
        write(f, scene)
    end
    cmd = `$(blender) --verbose 2 -noaudio --background $(template) -P $(script) -t $(threads) -- --scene $(scene_out) --out $(out)`
    run(cmd)
end

export render
