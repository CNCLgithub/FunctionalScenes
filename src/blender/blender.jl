using JSON

const tile_height = 5.0

function light(pos)
    Dict(:position => [pos..., 0.99 * tile_height],
         :orientation => [0., 0., 0.5 * pi],
         :intensity => 300.0)
end

function lights(r::Room)
    space, transform = lattice_to_coord(bounds(r), (2, 3))
    cis = CartesianIndices((2,3))
    @>> Tuple.(cis) map(transform) map(light) vec
end


function camera(r::Room)
    # camera is placed over entrance
    space, transform = lattice_to_coord(r)
    cis = CartesianIndices(steps(r))
    pos = @>> Tuple.(cis[entrance(r)]) lazymap(transform) first
    pos = pos .- ((0.5, 1.0) .* space)
    pos = [pos..., 0.9 * tile_height]
    # exts = @>> Tuple.(cis[exits(r)]) lazymap(transform) mean(dims = 2)
    orientation = [0.45 * pi, 0., 0.]
    # pos = [0,0,60]
    # orientation = [0, 0, 0.5 * pi]
    Dict(:position => pos,
         :orientation => orientation)
end

function floor(r::Room; ceiling = false)
    dx,dy = bounds(r)
    z = ceiling ? tile_height : 0.0
    rx = ceiling ? pi : 0.0
    Dict(:position => [0,0,z],
         :orientation => [rx,0,0],
         :shape => :Plane,
         :dims => [dx, dy, 0],
         :appearance => :white)
end

function tile(t, coords, space)
    dx,dy = space
    dz = t == :wall ? tile_height : sqrt(dx^2 + dy^2)
    pos = [coords..., dz / 2.0]
    Dict(:position => pos,
         :orientation => [0,0,0],
         :shape => :Block,
         :dims => [dx,dy,dz],
         :appearance => t == :wall ? :white : :blue)
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

function lattice_to_coord(bounds::Tuple, steps::Tuple)
    space = bounds ./ steps
    offset = 0.5 .* steps .* space .+ (0.5 .* space)
    (space, c -> c .* space .- offset)
end

function lattice_to_coord(r::Room)
    lattice_to_coord(bounds(r), steps(r))
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

function translate(r::Room, paths::Bool)
    ps = paths ? plot_path(r) : []
    Dict(:floor => floor(r),
         :ceiling => floor(r, ceiling = true),
         :lights => lights(r),
         :camera => camera(r),
         :objects => vcat(tiles(r), task(r), ps),
         )
end

default_script = joinpath(@__DIR__, "render.py")
default_template = joinpath(@__DIR__, "template.blend")

function render(r::Room, out::String;
                navigation = false,
                script = default_script, template = default_template,
                mode = "none", blender="blender", threads=Sys.CPU_THREADS)

    isdir(out) || mkpath(out)
    scene_out = joinpath(out, "scene.json")

    scene = translate(r, navigation) |> json
    open(scene_out, "w") do f
        write(f, scene)
    end
    cmd = `$(blender) --verbose 2 -noaudio --background $(template) -P $(script) -t $(threads) -- --scene $(scene_out) --mode $(mode) --out $(out)`
    run(cmd)
end

export translate, render
