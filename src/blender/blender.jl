using JSON

const tile_height = 5.0
const obstacle_height = 0.3 * tile_height

function light(pos)
    Dict(:position => [pos..., 0.95 * tile_height],
         :orientation => [0., 0., 0.5 * pi],
         :intensity => 150.0)
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
    pos = transform.(Tuple.(cis[entrance(r)]))
    # REVIEW: may need adjustment along y (forward-back)
    # NOTE: push back slightly to maximize visability of scene
    y = pos[1][2] - 1.75 * space[2]
    x = mean(first.(pos)) + 0.5
    # center of x-y for entrances
    camera_height = 0.85 * tile_height
    pos = [x, y, camera_height]
    orientation = [0.475 * pi, 0., 0.]
    Dict(:position => pos,
         :orientation => orientation)
end

function plain(r::Room; ceiling = false)
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

function tile(::Wall, x::Float64, y::Float64, dx::Float64, dy::Float64)
    dz = tile_height
    pos = [x, y, dz / 2.0]
    Dict(:position => pos,
         :orientation => [0,0,0],
         :shape => :Block,
         :dims => [dx,dy,dz],
         :appearance => :white)
end

function tile(::Obstacle, x::Float64, y::Float64,
              dx::Float64, dy::Float64)
    dz = 0.3 * tile_height
    pos = [x, y, dz / 2.0]
    Dict(:position => pos,
         :orientation => [0,0,0],
         :shape => :Block,
         :dims => [dx,dy,dz],
         :appearance => :blue)
end


function tile(::Floor, x::Float64, y::Float64,
              dx::Float64, dy::Float64)
    dz = 0.01
    pos = [x, y, dz]
    Dict(:position => pos,
         :orientation => [0,0,0],
         :shape => :Plane,
         :dims => [dx,dy, dz],
         :appearance => :green)
end

function lattice_to_coord(bounds::Tuple, steps::Tuple)
    space = bounds ./ steps
    offset = 0.5 .* steps .* space .+ (0.5 .* space)
    (space, c -> c .* space .- offset)
end

function lattice_to_coord(r::GridRoom)
    lattice_to_coord(bounds(r), steps(r))
end

function tiles(r::Room)
    tiles(r, Int64[])
end

function tiles(r::Room, p::Vector{Int64})
    d = data(r)
    vs = union(get_tiles(r, obstacle_tile),
               get_tiles(r, wall_tile))
    nv = length(vs)
    np = length(p)
    n = nv + np
    # map from room space to cartesian coordinates
    (dx, dy), transform = lattice_to_coord(r)
    cis = CartesianIndices(steps(r))
    result = Vector{Dict}(undef, nv + np)
    @inbounds for i = 1:n
        v = i <= nv ? vs[i] : p[i - nv]
        ci = Tuple(cis[v])
        x, y = transform(ci)
        result[i] =  tile(d[v], x, y, dx, dy)
    end
    return result
end

"""

Creates a voxel map Z*Y*X
"""
function cubify(r::GridRoom)

    d = data(r)
    cis = CartesianIndices(steps(r))
    dx, dy = steps(r)
    c_th = Int64(ceil(tile_height))
    c_oh = Int64(ceil(obstacle_height))
    vdims = (c_th * 2 + 1, dy, dx)
    floor_voxels = zeros(vdims)
    floor_voxels[c_th + 1, :, :] .= 1.
    wall_voxels = zeros(vdims)
    obs_voxels = zeros(vdims)
    for i in 1:length(d)
        (x, y) = Tuple(cis[i])
        if d[i] == wall_tile
            wall_voxels[c_th+1:end, y, x] .= 1.0
        elseif d[i] == obstacle_tile
            obs_voxels[c_th+1:c_th+c_oh, y, x] .= 1.0
        end
    end
    Dict(:floor_voxels => floor_voxels,
         :wall_voxels => wall_voxels,
         :obstacle_voxels => obs_voxels,
         :voxel_dim => maximum(vdims) * 0.5)
end

function translate(r::Room, paths::Vector{Int64};
                   cubes::Bool=false)
    d = Dict(:floor => plain(r),
             :ceiling => plain(r, ceiling = true),
             :lights => lights(r),
             :camera => camera(r))
    if cubes
        merge!(d, cubify(r))
    else
        d[:objects] = tiles(r, paths)
    end
    return d
end

default_script = joinpath(@__DIR__, "render.py")
default_template = joinpath(@__DIR__, "template.blend")

function render(r::Room, out::String;
                navigation = false,
                script = default_script, template = default_template,
                mode = "none", blender="blender", threads=Sys.CPU_THREADS)

    isdir(out) || mkpath(out)
    scene_out = joinpath(out, "scene.json")

    paths = navigation ? safe_shortest_paths(r) : Int64[]
    scene = translate(r, paths) |> json
    open(scene_out, "w") do f
        write(f, scene)
    end
    cmd = `$(blender) --verbose 2 -noaudio --background $(template) -P $(script) -t $(threads) -- --scene $(scene_out) --mode $(mode) --out $(out) --resolution $(720,480)`
    run(cmd)
end

export translate, render, cubify
