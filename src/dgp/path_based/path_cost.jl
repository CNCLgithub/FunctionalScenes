using ImageFiltering
using Random

export PathProcedure, AStarPath, NoisyPath, path_analysis,
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
    ws[d .== obstacle_tile] .= obstacle_cost
    ws[d .== wall_tile] .= wall_cost

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

function path_procedure(r::GridRoom, params::AStarPath)
    ent = first(entrance(r))
    ext = first(exits(r))
    g = pathgraph(r)
    w, ad, dm = nav_graph(r, params)
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
end

function nav_graph(r::GridRoom, params::NoisyPath)
    nav_graph(r, params, params.kernel_sigma)
end
function nav_graph(r::GridRoom, params::NoisyPath, sigma::Float64)
    @unpack obstacle_cost, floor_cost, wall_cost, kernel_width = params
    d = data(r)
    ws = fill(floor_cost, size(d))
    ws[d .== obstacle_tile] .= obstacle_cost
    ws[d .== wall_tile] .= obstacle_cost
    noisy_ws = imfilter(ws, Kernel.gaussian([sigma, sigma],
                                            [kernel_width, kernel_width]))
    noisy_ws[d .== wall_tile] .= wall_cost

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

path_procedure(r::GridRoom, params::NoisyPath) = path_procedure(r, params,
                                                                params.kernel_sigma)

function path_procedure(r::GridRoom, params::NoisyPath, sigma::Float64)
    ent = first(entrance(r))
    ext = last(exits(r))
    w, ad, dm = nav_graph(r, params, sigma)
    g = SimpleGraph(ad)
    h = x -> cart_dist(x, ext, size(data(r), 1))
    path = a_star(g, ent, ext, dm, h)
    path, w, ad, dm
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
    mx = @inbounds m[1] # HACK: should be wall tile
    result::Float64 = 0.
    for y = -offset:offset
        yoffset = y * ny
        for x = -offset:offset
            xoffset = x + i
            scale = exp(-sqrt(x^2  + y^2)/sqrt(w))
            result += scale * get(m, yoffset + xoffset, mx)
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
    # d::Float64 = 0.0
    # for e in path
    #     v = dst(e)
    #     x = ceil(v / n)
    #     y = v % n
    #     d += sqrt((x - loc[1])^2 + (y - loc[2])^2)
    # end
    # return d / ne
    d::Float64 = Inf
    for e in path
        v = dst(e)
        x = ceil(v / n)
        y = v % n
        d  = min(sqrt((x - loc[1])^2 + (y - loc[2])^2), d)
    end
    return d
end

function diffusion!(
    m::Array{Int64},
    g::AbstractGraph{T},
    p::Real,
    n::Integer,
    terminal::Set{T},
    node_weights::Vector,
    initial_infections::Vector{T}
    ) where {T}

    # Initialize
    infected_vertices = BitSet(initial_infections)

    # Run simulation
    for step in 2:n
        new_infections = Set{T}()

        @inbounds for i in infected_vertices
            outn = outneighbors(g, i)
            outd = length(outn)
            cur_dis = node_weights[i]
            for n in outn
                n_dis = node_weights[n]
                local_p = cur_dis >= n_dis ? 1.0 : p
                if rand() < local_p
                    push!(new_infections, n)
                end
            end
        end

        # Record only new infections
        setdiff!(new_infections, infected_vertices)
        for v in new_infections
            m[v] += 1
        end

        # Kill of terminal infections
        setdiff!(new_infections, terminal)

        # Add new to master set of infected
        union!(infected_vertices, new_infections)
    end

    return nothing
end


function obstacle_diffusion(room::GridRoom, f::Furniture,
                            path::Array{T},
                            p::Float64, n::Int64) where {T<:Edge}
    # diffusion on target furniture
    vs = dst.(path)
    m = zeros(Int64, steps(room))
    g = pathgraph(clear_room(room))
    clear_gds = gdistances(g, last(vs))
    fs = furniture(room)
    terminal = union(fs...)
    diffusion!(m, g, p, n, terminal, clear_gds, vs)
    # estimate the "cost" incurred by each obstacle
    cost_of_f::Float64 = 0.0
    @inbounds for v in f
        cost_of_f += m[v]
    end
    # diffusion across all furniture
    # tot = sum(m)
    tot::Int64 = 0
    gt_c::Float64 = 0
    @inbounds for fi in fs
        f_s = 0
        for v in fi
            f_s += m[v]
        end
        if f_s > cost_of_f
            gt_c +=1
        end
        tot += f_s
    end
    frac = iszero(tot) ? 0. : cost_of_f / tot
    sm = sum(m)
    path_covered = iszero(sm) ? 0. : tot / sm
    (cost_of_f, frac, gt_c, tot, Matrix{Float64}(m .> 0.))
end

function path_analysis(room::GridRoom, params::PathProcedure,
                       f::Furniture;
                       kernel::Int64 = 4, p::Float64 = 0.5,
                       n::Int64 = 3)
    pmat = zeros(steps(room))
    path, w, _... = path_procedure(room, params)
    c, fc, mc, sc, m = obstacle_diffusion(room, f, path, p, n)
    result = Dict{Symbol, Any}(
        :density => path_density(w, path, kernel),
        :diffusion_ct => c,
        :diffusion_ct_max => mc,
        :diffusion_ct_gt => mc,
        :diffusion_prop => fc,
        :diffusion_tot => sc,
        :path_dist => distance_to_path(room, f, path),
        :path_length => length(path)
                  )
    return (m, result)
end
