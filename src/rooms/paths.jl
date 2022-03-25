import Statistics: mean
import ImageFiltering: Kernel, imfilter

export navigability, compare, occupancy_grid, diffuse_og, safe_shortest_paths

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

function build_subpath(g::AbstractGraph, gd::Array{Int64}, v)::Vector{Int64}
    # initialize a bit vector (`falses`) for each tile being on a
    # shortest path to the entrance
    path = Vector{Bool}(falses(nv(g)))
    build_subpath!(path, g, gd, v)
    findall(path)
end

function build_subpath!(path::Vector{Bool}, g::AbstractGraph{T}, gd::Array{T},
                        vs::Vector{T}) where {T<:Int}
    @>> vs foreach(v -> build_subpath!(path, g, gd, v))
end

function build_subpath!(path::Vector{Bool}, g::AbstractGraph{T}, gd::Array{T},
                        v::T) where {T<:Int}
    # add neighbors that are 1 unit closer to the src
    path[v] = true
    d_next = gd[v] - 1

    subpath = @>> v begin
        neighbors(g)
        # only consider new tiles that are 1 unit closer
        filter(x -> !path[x] && gd[x] == d_next)
        # left fold
        # terminates once `subpath` is empty due to reaching src
        foreach(x -> build_subpath!(path, g, gd, x))
    end
    nothing
end

function all_shortest_paths(r::Room)
    all_shortest_paths(pathgraph(r), entrance(r), exits(r))
end

# function all_shortest_paths(g::AbstractGraph, start::Int64, stop::Int64)
# end

function all_shortest_paths(g::AbstractGraph{T}, start::Vector{T},
                            stop::Vector{T}) where {T}
    gd = gdistances(g, start)
    # No path from entrance to exit
    maximum(gd[stop]) <= length(g) || return T[]
    build_subpath(g, gd, stop)
end

function safe_shortest_paths(r::GridRoom)
    g = pathgraph(r)
    ents = entrance(r)
    exs = exits(r)
    # closest points to the exit
    # that are still reachable from the entrance
    vs = neighborhood(g, first(ents), length(g))
    gs = @> r begin
        clear_room
        pathgraph
        gdistances(exs)
        getindex(vs)
    end
    min_d = minimum(gs)
    safe_exits = vs[gs .== min_d]
    all_shortest_paths(g, ents, safe_exits)
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

function occupancy_position(r::Room)::Matrix{Float64}
    g = pathgraph(r) # r is the room
    grid = zeros(steps(r))
    vs = @>> g vertices Base.filter(v -> istype(g, v, :furniture))
    grid[vs] .= 1.0
    grid
end

function occupancy_grid(r::Room;
                        decay::Float64 = 0.0,
                        sigma::Float64 = 1.0)
    @>> r begin
        safe_shortest_paths
        # grids for exit x paths -> one grid per exit
        path -> occupancy_grid(r, path;
                               decay=decay, sigma=sigma)
    end
end

function occupancy_grid(r::Room, ps::Vector{Vector{Int64}};
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

function occupancy_grid(r::Room, p::Vector{Int64};
                        decay::Float64 = -1.,
                        sigma::Float64 = 1.0)
    g = pathgraph(r)
    m = zeros(steps(r))
    lp = length(p)
    iszero(lp) && return m
    gs = @>> r entrance gdistances(g)

    for v in p
        is_floor(r, v) || error("invalid path at tile $(v)")
        # m[v] = exp(decay * (i - 1))  + exp(decay * (lp - i))
        m[v] = exp(decay * gs[v])
        # m[v] = exp(decay * (lp - i))
    end
    # apply gaussian blur
    gf = Kernel.gaussian(sigma)
    m = imfilter(m, gf, "symmetric")
    mm = maximum(m)
    mm = iszero(mm) ? m : m ./ mm # normalize to [0,1]
    # remove blur along walls and obstacles
    mask = zeros(steps(r))
    mask[get_tiles(r, floor_tile)] .= 1
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
