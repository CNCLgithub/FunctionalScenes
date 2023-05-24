using ImageFiltering

export NoisyPath, noisy_path, path_cost, distance_to_path

@with_kw struct NoisyPath
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
    (adm, dsm)
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
    ad, dm = nav_graph(r, params, sigma)
    g = SimpleGraph(ad)
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

function distance_to_path(vs, pmat)
    n = size(pmat, 1)
    loc = avg_location(vs, n)
    d = 0
    for x = 1:size(pmat,2), y = 1:size(pmat, 1)
        # occupancy weight * neg log of l2 distance
        # d += pmat[y,x] * log( sqrt( (ceil(x/n) - loc[1])^2 + (y % n - loc[2])^2))
        d += pmat[y,x] * sqrt( (ceil(x/n) - loc[1])^2 + (y % n - loc[2])^2)
    end
    return d
end

function noisy_path(room::GridRoom, params::NoisyPath;
                    samples::Int64 = 300)
    c::Float64 = 0.0
    pmat = zeros(steps(room))
    for _ = 1:samples
        # var = inv_gamma(1.2, 0.8)
        var = gamma(params.kernel_alpha, params.kernel_beta)
        path, dm = Graphs.a_star(room, params, sqrt(var))
        c += path_cost(path, dm)
        @inbounds for step in path
            pmat[dst(step)] += 1.0
        end
    end
    (c / samples, pmat ./ samples)
end
