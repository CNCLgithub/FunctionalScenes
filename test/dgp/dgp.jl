using Gen
using JSON
using Graphs
import Graphs: a_star
using ImageFiltering
using FileIO
using FunctionalScenes
using Profile
using StatProfilerHTML

function nav_graph(r::GridRoom, w::Float64)
    d = data(r)
    ws = zeros(size(d))
    ws[d .== obstacle_tile] .= 1.0
    noisy_ws = imfilter(ws, Kernel.gaussian(w))
    noisy_ws[d .== wall_tile] .= Inf
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

function Graphs.a_star(r::GridRoom, w::Float64)

    ent = first(entrance(r))
    ext = first(exits(r))

    ad, dm = nav_graph(r, w)
    g = SimpleGraph(ad)

    path = a_star(g, ent, ext, dm)
end

function mytest()
    name = "ccn_2023_exp"
    base_p = "/spaths/datasets/$(name)/scenes/1_1.json"
    local base_s
    open(base_p, "r") do f
        base_s = JSON.parse(f)
    end
    room = from_json(GridRoom, base_s)
    path = a_star(room, 0.1)
    ps = Int64[]
    push!(ps, src(path[1]))
    for step in path
        push!(ps, dst(step))
    end
    viz_room(room, ps)
    # Profile.init(;n = 100000, delay = 1E-5)
    # Profile.clear()
    # @profilehtml a_star(room, 0.01)
end

mytest();
