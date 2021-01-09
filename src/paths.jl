import Statistics: mean
import ImageFiltering: Kernel, imfilter

function shortest_path_length(g, s, e)
    s = a_star(g, s, e)
    length(s)
end

function average_path_length(g, s, e; n = 5)
    s = yen_k_shortest_paths(g, s, e, weights(g), n)
    l = @>> s.paths map(length) mean
    isnan(l) ? Inf : l
end


function safe_shortest_path(r::Room, e::Tile)
    dx, dy = steps(r)
    g = pathgraph(r)
    ent = entrance(r) |> first
    _, coords = lattice_to_coord(r)
    ec = coords(e)
    vs = connected(g, ent)
    vi = @>> vs begin
        map(v -> norm(ec .- coords(v)))
        argmin
    end
    ds = desopo_pape_shortest_paths(g, ent)
    enumerate_paths(ds, vs[vi])
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

function occupancy_grid(r::Room; decay = -1, sigma = 0)
    f = p -> occupancy_grid(r, p, decay=decay, sigma=sigma)
    @>> navigability(r) map(f) mean
end
function occupancy_grid(r::Room, p::Vector{Tile};
                        decay = -1, sigma = 1)
    g = pathgraph(r)
    m = zeros(steps(r))
    mask = zeros(steps(r))
    lp = length(p)
    iszero(lp) && return m
    for (i,v) in enumerate(p)
        isfloor(g, v) || break
        m[v] = exp(decay * (i - 1)) + exp(decay * (lp - i))
    end
    gf = Kernel.gaussian(sigma)
    m = imfilter(m, gf, "symmetric")
    floor = @>> g vertices filter(v -> isfloor(g, v)) collect
    mask[floor] .= 1
    og = m .* mask
    og ./ sum(og)
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

export navigability, compare, occupancy_grid, diffuse_og
