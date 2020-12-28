using JSON

function camera(r::Room)
    # camera is placed over entrance
    space, transform = lattice_to_coord(r)
    cis = CartesianIndices(steps(r))
    pos = @>> Tuple.(cis[entrance(r)]) lazymap(transform) first
    pos = [pos..., 2.0]
    # exts = @>> Tuple.(cis[exits(r)]) lazymap(transform) mean(dims = 2)
    Dict(:position => pos,
         :orientation => [0.5 * pi, 0., 0.])
end

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
         # :appearance => t == :wall ? :blue : :white)
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

function lattice_to_coord(r::Room)
    space = bounds(r) ./ steps(r)
    offset = 0.5 .* steps(r) .* space .+ (0.5 .* space)
    (space, c -> c .* space .- offset)
end

function tiles(r::Room)
    space, transform = lattice_to_coord(r)
    g = pathgraph(r)
    vs = @>> g vertices filter(v -> !isfloor(g, v))
    cis = CartesianIndices(steps(r))
    vcis = Tuple.(cis[vs])
    coords = map(transform, vcis)
    types = @>> vs lazymap(v -> get_prop(g, v, :type))
    @>> zip(types, coords) lazymap(xy -> tile(xy..., space)) collect
end

function task(r::Room)
    space, transform = lattice_to_coord(r)
    cis = CartesianIndices(steps(r))
    ents = @>> Tuple.(cis[entrance(r)]) lazymap(transform)
    ents = @>> ents lazymap(x -> spot(x, space)) collect

    exts = @>> Tuple.(cis[exits(r)]) lazymap(transform)
    exts = @>> exts lazymap(x -> spot(x, space; exit=true)) collect

    vcat(ents, exts)
end


function plot_path(r::Room)
    space, transform = lattice_to_coord(r)
    cis = CartesianIndices(steps(r))
    paths = navigability(r)
    # vs = @>> paths lazymap(p -> lazymap(src, p)) flatten collect
    vs = vcat(paths...)
    vs = @>> Tuple.(cis[vs]) lazymap(transform)
    @>> vs lazymap(x -> spot(x, space)) collect
end

function translate(r::Room, paths::Bool)::String
    ps = paths ? plot_path(r) : []
    Dict(:floor => floor(r),
         :camera => camera(r),
         :objects => vcat(tiles(r), task(r), ps),
         ) |> json
end

default_script = joinpath(@__DIR__, "render.py")
default_template = joinpath(@__DIR__, "template.blend")

function render(r::Room, out::String;
                navigation = false,
                script = default_script, template = default_template,
                mode = "none", blender="blender", threads=Sys.CPU_THREADS)

    isdir(out) || mkpath(out)
    scene_out = joinpath(out, "scene.json")

    scene = translate(r, navigation)
    open(scene_out, "w") do f
        write(f, scene)
    end
    cmd = `$(blender) --verbose 2 -noaudio --background $(template) -P $(script) -t $(threads) -- --scene $(scene_out) --mode $(mode) --out $(out)`
    run(cmd)
end

export translate, render
