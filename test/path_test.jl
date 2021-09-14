using FunctionalScenes
using LightGraphs,SimpleWeightedGraphs
using Distances
using OptimalTransport

import  FunctionalScenes:a_star_paths,transforms
 
function generate_trackers(room::Room, scale::Int64 = 6, dims::Vector{Int64} = [1,3,6])
	#tracker_col,tracker_row = map(x -> floor(Int64,x), steps(room).*1/scale)
	tracker_row = floor(Int64,steps(room)[1] * 1/scale)
	tracker_col = floor(Int64,steps(room)[2] * 1/scale)
	trackers = Matrix{SimpleWeightedGraph{Int64, Float64}}(undef,tracker_row,tracker_col)

	@inbounds for row_ind = 1:tracker_row, col_ind = 1:tracker_col
		tracker_dim = rand(dims)
		tracker_weight = rand(Float64,tracker_dim^2)
		trackers[row_ind,col_ind] = SimpleWeightedGraph(grid([tracker_dim,tracker_dim]),scale/tracker_dim) 
	end
	return trackers
end


#function test()
template_room = Room((18,12), (18,12), [1], [210])
desc = exits(template_room)

trackers_a = generate_trackers(template_room)
trackers_b = generate_trackers(template_room)
#display(trackers_a)

#tracker_nvs = map(nv, trackers_a)
#tracker_dims = map(x -> round(Int64,sqrt(nv(x))), trackers_a)
#display(tracker_nvs)
#display(tracker_dims)
#display(merge(trackers_a, tracker_nvs, tracker_dims))

paths_a = a_star_paths(template_room, trackers_a)
paths_b = a_star_paths(template_room, trackers_b)
#display(paths_a)

points_a = transforms(trackers_a, paths_a)
points_b = transforms(trackers_b, paths_b)
#display(points_a)

# distance matrix
dm = pairwise(Euclidean(), points_a, points_b, dims = 1)
#display(dm)

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
display(d)


#end

#display(test())
