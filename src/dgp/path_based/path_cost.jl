using ImageFiltering

export PathProcedure, AStarPath, astar_path,
    path_density, distance_to_path

abstract type PathProcedure end

@with_kw struct AStarPath <: PathProcedure
    obstacle_cost::Float64 = 1.0
    floor_cost::Float64 = 0.1
    wall_cost_ratio::Float64 = 10.0
    wall_cost::Float64 = obstacle_cost * wall_cost_ratio
end

function nav_graph(r::GridRoom, params::AStarPath)
    @unpack obstacle_cost, floor_cost, wall_cost = params
    d = data(r)
    ws = fill(floor_cost, size(d))
    ws[d .== obstacle_tile] .+= obstacle_cost
    ws[d .== wall_tile] .+= wall_cost

    n = length(d)
    row = size(d, 1)
    adm = fill(false, (n, n))
    dsm = fill(Inf, (n, n))

    @inbounds for i = 1:(n-1), j = (i+1):n
        delta = abs(i - j)
        delta == 1 || delta == row || continue
        adm[i, j] = adm[j, i] = true
        dsm[i, j] = dsm[j, i] = ws[i] + ws[j]
    end
    (ws, adm, dsm)
end

function Graphs.a_star(r::GridRoom, params::AStarPath)
    ent = first(entrance(r))
    ext = first(exits(r))
    g = pathgraph(r)
    w, ad, dm = nav_graph(r, params)
    # g = SimpleGraph(ad)
    h = x -> round(cart_dist(x, ext, size(data(r), 1)))
    path = a_star(g, ent, ext, weights(g), h)
    path, w, ad, dm
end

@with_kw struct NoisyPath <: PathProcedure
    obstacle_cost::Float64 = 1.0
    floor_cost::Float64 = 0.1
    wall_cost_ratio::Float64 = 10.0
    wall_cost::Float64 = obstacle_cost * wall_cost_ratio
    kernel_sigma::Float64 = 1.0
    kernel_width::Int64 = 7
    kernel_alpha::Float64 = 2.0
    kernel_beta::Float64 = 0.9
end

function nav_graph(r::GridRoom, params::NoisyPath)
    nav_graph(r, params, params.kernel_sigma)
end
function nav_graph(r::GridRoom, params::NoisyPath, sigma::Float64)
    @unpack obstacle_cost, floor_cost, wall_cost, kernel_width = params
    d = data(r)
    ws = fill(floor_cost, size(d))
    ws[d .== obstacle_tile] .+= obstacle_cost
    ws[d .== wall_tile] .+= wall_cost
    noisy_ws = imfilter(ws, Kernel.gaussian([sigma, sigma],
                                            [kernel_width, kernel_width]))
    noisy_ws[d .== wall_tile] .+= wall_cost

    n = length(d)
    row = size(d, 1)
    adm = fill(false, (n, n))
    dsm = fill(Inf, (n, n))

    @inbounds for i = 1:(n-1), j = (i+1):n
        delta = abs(i - j)
        delta == 1 || delta == row || continue
        adm[i, j] = adm[j, i] = true
        dsm[i, j] = dsm[j, i] = noisy_ws[i] + noisy_ws[j]
    end
    (noisy_ws, adm, dsm)
end

function cart_dist(src, trg, n)
    delta_x = ceil(src / n) - ceil(trg / n)
    delta_y = src % n - trg % n
    sqrt(delta_x^2 + delta_y^2)
end

function avg_location(lvs, n::Int64)
    center = zeros(2)
    @inbounds for v in lvs
        center[1] += ceil(v / n)
        center[2] += v % n
    end
    center ./= length(lvs)
    return center
end

function Graphs.a_star(r::GridRoom, params::NoisyPath, sigma::Float64)
    ent = first(entrance(r))
    ext = first(exits(r))
    # _, ad, dm = nav_graph(r, params, sigma)
    # g = simplegraph(ad)
    h = x -> cart_dist(x, ext, size(data(r), 1))
    path = a_star(g, ent, ext, dm, h)
    path, dm
end

function path_cost(path)
    length(path)
end

function path_cost(path, dm)
    c::Float64 = 0.0
    @inbounds for step in path
        c += dm[src(step), dst(step)]
    end
    return c
end


function path_cost(path::Vector{Int64}, dm)
    c::Float64 = 0.0
    @inbounds for step in path
        c += dm[step]
    end
    return c
end

function kernel_from_linear(i::Int64, m::Matrix{Float64}, w::Int64)
    ny= size(m, 1)
    offset = Int64((w-1) / 2)
    result::Float64 = 0.
    for y = -offset:offset
        yoffset = y * ny
        for x = -offset:offset
            xoffset = x + i
            result += get(m, yoffset + xoffset, 1.0)
        end
    end
    result / (w ^ 2)
end

function path_density(m::Matrix{Float64}, path::Vector{T}, w::Int64) where
    {T <: Edge}
    # m = Matrix{Float64}(data(r) .== obstacle_tile)
    d::Float64 = 0.0
    @inbounds for e = path
        d += kernel_from_linear(dst(e), m, w)
    end
    return d
end

function distance_to_path(r::GridRoom, vs, pmat::Matrix{Float64})
    n = steps(r)[2]
    n = size(pmat, 1)
    loc = avg_location(vs, n)
    d = 0
    spmat = sum(pmat)
    @inbounds for x = 1:size(pmat,2), y = 1:size(pmat, 1)
        d += (pmat[y,x] / spmat) *
            sqrt( (ceil(x/n) - loc[1])^2 + (y % n - loc[2])^2)
    end
    return d
end


function distance_to_path(r::GridRoom, vs, path::Array{T}) where {T<:Edge}
    n = steps(r)[2]
    loc = avg_location(vs, n)
    ne = length(path)
    d::Float64 = 0.0
    for e in path
        v = dst(e)
        x = ceil(v / n), y = v % n
        d += sqrt((x - loc[1])^2 + (y - loc[2])^2)
    end
    return d / ne
end

# function nearest_k_distance(vs, pmat)
#     n = size(pmat, 1)
#     loc = avg_location(vs, n)
#     d = Inf
#     ws = Matrix{Float64}(undef, size(pmat))
#     sum_pmat = sum(pmat)
#     @inbounds for x = 1:size(pmat,2), y = 1:size(pmat, 1)
#         ws[y, x] = sqrt( (ceil(x/n) - loc[1])^2 + (y % n - loc[2])^2) /
#             (pmat[y,x] / sum_pmat)
#     end

#     return d
# end

function astar_path(room::GridRoom, params::AStarPath;
                    kernel::Int64 = 5,
                    samples::Int64 = 50)
    pmat = zeros(steps(room))
    path, w, _... = Graphs.a_star(room, params)
    c = path_density(w, path, kernel)
    @inbounds for step in path
        pmat[dst(step)] += 1.0
    end
    # c::Float64 = 0.0
    # pmat = zeros(steps(room))
    # for _ = 1:samples
    #     sample = reorganize(room)
    #     path, w, _... = Graphs.a_star(sample, params)
    #     c += path_density(w, path, kernel)
    #     @inbounds for step in path
    #         pmat[dst(step)] += (1.0 / samples)
    #     end
    # end
    # c /= samples
    (c, pmat)
end
