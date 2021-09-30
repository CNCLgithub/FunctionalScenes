using FunctionalScenes
using LightGraphs,SimpleWeightedGraphs
using Lazy: @>, @>>, lazymap, flatten

#using FunctionalScenes:a_star_path


function a_star_path(r::Room, trackers)

    tracker_nvs = map(length, trackers)
    tracker_dims = map(size, trackers)
    g = merge(trackers, tracker_nvs, tracker_dims)

    total_vs = sum(tracker_nvs)
    bs = vcat(trackers...)
    edge_mx = repeat(bs'; outer = (total_vs, 1))

    ent = @>> r entrance first
    ext = @>> r exits first
    a_star_path(g, edge_mx, ent, ext)
end

 
function generate_trackers(room::Room, scale::Int64 = 6, dims::Vector{Int64} = [1,3,6])
    #tracker_col,tracker_row = map(x -> floor(Int64,x), steps(room).*1/scale)
    tracker_col = floor(Int64,steps(room)[1] * 1/scale)
    tracker_row = floor(Int64,steps(room)[2] * 1/scale)
    trackers = Matrix{SimpleWeightedGraph{Int64, Float64}}(undef,tracker_col,tracker_row)

    @inbounds for row_ind = 1:tracker_row, col_ind = 1:tracker_col
        tracker_dim = rand(dims)
        tracker_weight = rand(Float64,tracker_dim^2)
        trackers[row_ind,col_ind] = SimpleWeightedGraph(grid([tracker_dim,tracker_dim]),scale/tracker_dim)
    end
    return trackers
end


function test()
template_room = Room((18,12), (18,12), [9], [210])
desc = exits(template_room)

trackers_a = generate_trackers(template_room)
trackers_b = generate_trackers(template_room)

paths_a = a_star_path(template_room, trackers_a)
paths_b = a_star_path(template_room, trackers_b)
display(paths_a)

points_a = transform(template_room, trackers_a, paths_a)
points_b = transform(template_room, trackers_b, paths_b)

# distance matrix
dm = pairwise(norm,eachrow(points_a),eachrow(points_b))

# number of vertices in paths
na = size(points_a,1)
nb = size(points_b,1)

# discrete measures
measure_a = fill(1.0/na,na)
measure_b = fill(1.0/nb,nb)
λ = 1.0
ε = 0.01

ot = sinkhorn_unbalanced(measure_a, measure_b, dm, λ, λ, ε)
d = sum(ot .* dm)


end

display(test())
