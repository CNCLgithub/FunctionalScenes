import Statistics: mean
import ImageFiltering: Kernel, imfilter

export navigability, compare, occupancy_grid, diffuse_og

function k_shortest_paths(r::Room, k::Int64, ent::Int64, ext::Int64)
    g = pathgraph(r)
    yen_k_shortest_paths(g, ent, ext, weights(g), k).paths
end

function shortest_path_length(g, s, e)
    s = a_star(g, s, e)
    length(s)
end

function average_path_length(g, s, e; n = 5)
    s = yen_k_shortest_paths(g, s, e, weights(g), n)
    l = @>> s.paths map(length) mean
    isnan(l) ? Inf : l
end

function build_subpath(g::AbstractGraph, gd::Array{Int64}, v::Int64)::Vector{Int64}
    path = @>> g nv zeros BitVector
    build_subpath!(path, g, gd, v)
    findall(path)
end

function build_subpath!(path::BitVector, g::AbstractGraph, gd::Array{Int64}, v::Int64)
    # add neighbors that are 1 unit closer to the src
    path[v] = 1
    d_next = gd[v] - 1
    subpath = @>> v begin
        neighbors(g)
        # only consider new tiles that are 1 unit closer
        filter(x -> !path[x] && gd[x] == d_next)
        collect(Int64)
    end

    # left fold
    # terminates once `subpath` is empty due to reaching src
    @>> subpath begin
        foreach(x -> build_subpath!(path, g, gd, x))
    end

    return nothing
end

function all_shortest_paths(r::Room)
    all_shortest_paths(pathgraph(r), first(entrance(r)), first(exits(r)))
end

function all_shortest_paths(g::AbstractGraph, start::Tile, stop::Tile)
    # get subgraph of just floor
    floor, vmap = @> g begin
        filter_vertices(:type, :floor)
        (@>> induced_subgraph(g))
    end
    ent_i = findfirst(vmap .== start)
    exit_i = findfirst(vmap .== stop)

    # compute geodesic distances to the entrance for each tile
    gd = gdistances(floor, ent_i)
    # we could filter out any vertices that have a gd > the exit but
    # that would require another level of vertex mapping

    # walk through
    vs = Int64[]
    start_d = gd[exit_i]

    # No path from entrance to exit
    start_d <= length(floor) || return vs

    floor_path = build_subpath(floor, gd, exit_i)
    # map back to pathgraph
    vmap[floor_path]
end

function safe_shortest_path(r::Room, start::Tile, stop::Tile)
    dx, dy = steps(r)
    g = pathgraph(r)
    _, coords = lattice_to_coord(r)
    cis = CartesianIndices(steps(r))
    ec = coords(Tuple(cis[stop]))
    vs = connected(g, start) |> collect |> sort
    vcis = Tuple.(cis[vs])
    # index of the closest point to the exit
    vi = @>> vcis map(v -> norm(ec .- coords(v))) argmin
    all_shortest_paths(g, start, vs[vi])
    # yen_k_shortest_paths(g, start, vs[vi], weights(g), k).paths
end

function safe_shortest_path(r::Room, e::Tile)
    ent = entrance(r) |> first
    safe_shortest_path(r, ent, e)
end

function total_length(g::PathGraph)
    @>> g begin
        navigability
        map(length)
        sum
    end
end
function cf_total_length(g::PathGraph)
    @>> furniture(r) begin
        map(f -> @>> remove(g,f) total_length)
        argmax
    end
end


function diffuse_og(r::Room)::Matrix{Float64}
    @>> furniture(r) map(f -> diffuse_og(r,f)) mean
end

function diffuse_og(r::Room, f::Furniture)::Matrix{Float64}
    @>> f begin
        valid_moves(r)
        x -> (!iszero).(x)
        findall
        map(m -> shift_furniture(r,f,m))
        x -> vcat(x..., r)
        map(occupancy_grid)
        mean
    end
end

function _wsd(a,b)
    v = rand(size(a, 2))
    v = v ./ norm(v)
    OptimalTransport.pot.wasserstein_1d(a * v, b * v)
end

function wsd(a::Matrix{Float64}, b::Matrix{Float64}; n::Int64 = 500)::Float64
    d = 0.
    for _ in 1:n
        d += _wsd(a,b)
    end
    d / n
end


function occupancy_grid(r::Room;
                        decay::Float64 = 0.0,
                        sigma::Float64 = 1.0)
    @>> r begin
        exits
        # k paths for each exit
        map(e -> safe_shortest_path(r,e))
        # grids for exit x paths -> one grid per exit
        map(ps -> occupancy_grid(r, ps, decay=decay, sigma=sigma))
        # average grids across exits
        mean
    end
end

function occupancy_grid(r::Room, ps::Vector{Vector{Tile}};
                        decay::Float64 = 0.0,
                        sigma::Float64 = 1.0)
    @>> ps begin
        # og for each path
        map(p -> occupancy_grid(r, p;
                                decay = decay,
                                sigma = sigma))
        # average across paths
        mean
    end
end

function occupancy_grid(r::Room, p::Vector{Tile};
                        decay = -1, sigma = 1)
    g = pathgraph(r)
    m = zeros(steps(r))
    mask = zeros(steps(r))
    lp = length(p)
    iszero(lp) && return m
    for (i,v) in enumerate(p)
        if !isfloor(g, v)
            println("$(v) => $(get_prop(g, v, :type))")
            ns = connected(g, v)
            new_r = clear_room(r)
            new_r = add(new_r, ns)
            display(new_r)

            @>> ns collect filter(v -> istype(g, v, :floor)) println
            error("invalid path")
        end
        # m[v] = exp(decay * (i - 1))  + exp(decay * (lp - i))
        m[v] = exp(decay * (i - 1))
        # m[v] = exp(decay * (lp - i))
    end
    gf = Kernel.gaussian(sigma)
    m = imfilter(m, gf, "symmetric")
    floor = @>> g vertices Base.filter(v -> isfloor(g, v)) collect
    mask[floor] .= 1
    mm = maximum(m)
    mm = iszero(mm) ? m : m ./ mm
    m .* mask
end

function navigability(r::Room)
    g = pathgraph(r)
    ent = first(entrance(r))
    ds = desopo_pape_shortest_paths(g, first(entrance(r)))
    @>> r exits enumerate_paths(ds)
end


# the edit distance in paths
compare(a::Room, b::Room) = @>> map(symdiff, navigability(a), navigability(b)) map(length) sum
# function compare(a::Room, b::Room)
#     og_a = occupancy_grid(a)
# end
