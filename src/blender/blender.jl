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
    pos = pos  .+ ((0.5, -1.25) .* space)
    pos = [pos..., 0.75 * tile_height]
    # exts = @>> Tuple.(cis[exits(r)]) lazymap(transform) mean(dims = 2)
    orientation = [0.475 * pi, 0., 0.]
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
         # :appearance => ceiling ? :white : :blue)
         :appearance => :white)
end

function tile(t, coords, space, sphere)
    dx,dy = space
    if t == :wall
        dz = tile_height
        shape = :Block
    else
        # dx, dy = sphere ? 0.9 .* (dx, dy) : (dx, dy)
        # dx, dy = (0.8, 0.6) .* (dx, dy)
        dz = 0.35 * tile_height # sqrt(dx^2 + dy^2)
        shape = sphere ? :Ball : :Block
    end
    pos = [coords..., dz / 2.0]
    Dict(:position => pos,
         :orientation => [0,0,0],
         :shape => shape,
         :dims => [dx,dy,dz],
         # :appearance =>  :blue)
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

function tiles(r::Room; spheres::Bool = false)
    space, transform = lattice_to_coord(r)
    g = pathgraph(r)
    vs = @>> g vertices filter(v -> !isfloor(g, v))
    cis = CartesianIndices(steps(r))
    vcis = Tuple.(cis[vs])
    coords = map(transform, vcis)
    types = @>> vs lazymap(v -> get_prop(g, v, :type))
    @>> zip(types, coords) lazymap(xy -> tile(xy..., space, spheres)) collect
end

function cubify(r::Room)

    g = pathgraph(r)
    cis = CartesianIndices(steps(r))
    dims = reverse(steps(r))
    walls = zeros(4, dims...)
    furniture = zeros(4, dims...)
    for v in vertices(g)
        idx = reverse(Tuple(cis[v]))
        if istype(g,v,:wall)
            walls[:, idx...] .= 1.0
        elseif istype(g,v,:furniture)
            furniture[1, idx...] = 1.0
        end
    end
    return (furniture, walls)

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
    vs = safe_shortest_paths(r)
    vs = @>> Tuple.(cis[vs]) lazymap(transform)
    @>> vs lazymap(x -> spot(x, space)) collect
end

function translate(r::Room, paths::Bool;
                   cubes::Bool=false,
                   spheres::Bool=false)
    ps = paths ? plot_path(r) : []
    d = Dict(:floor => floor(r),
             :ceiling => floor(r, ceiling = true),
             :lights => lights(r),
             :camera => camera(r))
    if cubes
        furn, walls = cubify(r)
        d[:walls] = walls
        d[:furniture] = furn
    else
        d[:objects] = vcat(tiles(r, spheres=spheres),
                           ps)
                           # task(r), ps)
    end
    return d
end

default_script = joinpath(@__DIR__, "render.py")
default_template = joinpath(@__DIR__, "template.blend")

function render(r::Room, out::String;
                navigation = false,
                spheres = false,
                script = default_script, template = default_template,
                mode = "none", blender="blender", threads=Sys.CPU_THREADS)

    isdir(out) || mkpath(out)
    scene_out = joinpath(out, "scene.json")

    scene = translate(r, navigation, spheres = spheres) |> json
    open(scene_out, "w") do f
        write(f, scene)
    end
    cmd = `$(blender) --verbose 2 -noaudio --background $(template) -P $(script) -t $(threads) -- --scene $(scene_out) --mode $(mode) --out $(out) --resolution $(720,480)`
    run(cmd)
end

export translate, render
