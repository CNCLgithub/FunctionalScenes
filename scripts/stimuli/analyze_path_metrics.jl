using CSV
using JSON
using FileIO
using Graphs
using DataFrames
using SparseArrays
using FunctionalScenes
using OptimalTransport
# using MetaGraphsNext

function og_sinkhorn(p::Matrix{Float64}, q::Matrix{Float64})
    c = pairwise(td_cost, a_k, b_k)
    # ot = sinkhorn_unbalanced(a_w, b_w, c, λ, λ, ε;
    #                          maxiter=1_000)
    ot = sinkhorn(a_w, b_w, c, ε;
                  atol = 1E-4,
                  maxiter=10_000)
    # ot = log.(ot)
    # d = logsumexp(ot .+ log.(c))
    d = OptimalTransport.sinkhorn_cost_from_plan(ot, c, ε;
                                                 regularization=false)
end

function fuzzy_astar(r::GridRoom, o_weight::Float64)
    d = data(r)
    s = steps(r)
    g =  grid(s)
    ws = convert.(Float64, adjacency_matrix(g))
    is,js,_ = findnz(ws)
    n = length(is)
    vs = fill(Inf, n)
    # vs = zeros(n)
    for x = 1:n
        i = is[x]
        j = js[x]
        i == j && continue
        vs[x] = (d[i] == floor_tile && d[j] == floor_tile) ? 1. : o_weight
    end
    ws = sparse(is, js, vs)
    ent = first(entrance(r))
    ext = first(exits(r))
    path_state = dijkstra_shortest_paths(g, [ent], ws)
    return enumerate_paths(path_state, ext)
end

function main()

    dataset = "vss_pilot_11f_32x32_restricted"
    obstacle_weight = 1.5
    dpath = "/spaths/datasets/$(dataset)"
    # scenes = [1, 6, 10, 25]
    scenes = [13]

    df = DataFrame(CSV.File("$(dpath)/scenes.csv"))
    df = filter(r -> in(r.id, scenes), df)
    # df = df[df.id .== scene, :]

    display(df)

    for r in eachrow(df)
        println("\n LOOKING AT SCENE: $(r.id)")

        base_p = "$(dpath)/scenes/$(r.id)_$(r.door).json"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)
        shifted = shift_furniture(base,
                                  furniture(base)[r.furniture],
                                  Symbol(r.move))
        base_paths = safe_shortest_paths(base)
        # base_paths = fuzzy_astar(base, obstacle_weight)
        # base_paths = fuzzy_astar(base, obstacle_weight)
        base_og = occupancy_grid(base, base_paths;
                                 decay = 0.,
                                 sigma = 0.)
        # shifted_paths = fuzzy_astar(shifted, obstacle_weight)
        shifted_paths = safe_shortest_paths(shifted)
        shifted_og = occupancy_grid(shifted, shifted_paths;
                                    decay = 0.,
                                    sigma = 0.)
        println("n steps base: $(length(base_paths))")
        println("n steps shifted: $(length(shifted_paths))")
        FunctionalScenes.viz_room(base, base_og)
        FunctionalScenes.viz_room(shifted, shifted_og)
    end

end;


main();
