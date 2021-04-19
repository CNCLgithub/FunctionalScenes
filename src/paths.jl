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

function safe_shortest_path(r::Room, start::Tile, stop::Tile)
    dx, dy = steps(r)
    g = pathgraph(r)
    _, coords = lattice_to_coord(r)
    cis = CartesianIndices(steps(r))
    ec = coords(Tuple(cis[stop]))
    vs = connected(g, start) |> collect |> sort
    vcis = Tuple.(cis[vs])
    vi = @>> vcis map(v -> norm(ec .- coords(v))) argmin
    ds = desopo_pape_shortest_paths(g, start)
    enumerate_paths(ds, vs[vi])
end

function safe_shortest_path(r::Room, e::Tile)
    ent = entrance(r) |> first
    forward = safe_shortest_path(r, ent, e)
    # backward = safe_shortest_path(r, e, ent)
    # union(forward, backward)
    forward
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


function occupancy_grid(r::Room; decay = 0.0, sigma = 1.0) # ::Matrix{Float64}
    es = exits(r)
    paths = @>> es map(e -> safe_shortest_path(r,e))
    f = p -> occupancy_grid(r, p, decay=decay, sigma=sigma)
    @>> paths map(f)
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
#     na = @>> a navigability map(x -> map(src, x)) flatten
#     nb = @>> b navigability map(x -> map(src, x)) flatten
#     length(setdiff(na,nb))
# end

