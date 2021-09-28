using FunctionalScenes
using LightGraphs,SimpleWeightedGraphs
using Distances
using OptimalTransport
using Plots

import  FunctionalScenes:a_star_paths,transforms
 
function generate_trackers(room::Room, scale::Int64 = 6, dims::Vector{Int64} = [1,3,6])
	tracker_row = floor(Int64,steps(room)[1] * 1/scale)
	tracker_col = floor(Int64,steps(room)[2] * 1/scale)
	#trackers = Matrix{SimpleWeightedGraph{Int64, Float64}}(undef,tracker_row,tracker_col)
        trackers = Matrix{Matrix{Float64}}(undef,tracker_row,tracker_col)

	@inbounds for col_ind = 1:tracker_col, row_ind = 1:tracker_row
		tracker_dim = rand(dims)
		tracker_weight = rand(Float64,tracker_dim^2)
		#trackers[row_ind,col_ind] = SimpleWeightedGraph(grid([tracker_dim,tracker_dim]),scale/tracker_dim) 
	        trackers[row_ind,col_ind] = reshape(tracker_weight, (tracker_dim, tracker_dim)) 
        end
	return trackers
end

# transform every tracker to 6*6 matrix for heatmap visualization
# TODO: generalize this function if the scale is not 6
function bern_plots(trackers::Matrix{Matrix{Float64}}, scale::Int64 = 6)
        tracker_row, tracker_col = size(trackers)
        bern_weights = Matrix{Float64}(undef, tracker_row * scale, tracker_col * scale)
        
        @inbounds for col_ind = 1:tracker_col, row_ind = 1:tracker_row
               tracker = trackers[row_ind,col_ind]
               row_startind = 1 + scale * (row_ind - 1)
               row_endind = scale * row_ind
	       col_startind = 1 + scale * (col_ind - 1)
               col_endind = scale * col_ind

               if size(tracker)[1] == 3
                      # for each element in 3*3 matrix, turn it into a 2*2 matrix
                      bern_weights[row_startind:row_endind, col_startind:col_endind] = repeat(tracker, inner = (2,2), outer = (1,1))
               elseif size(tracker)[1] == 1
                      # turn this float into a 6*6 matrix 
                      bern_weights[row_startind:row_endind, col_startind:col_endind]  = reshape(repeat(tracker,36),(6,6))
               else 
                      bern_weights[row_startind:row_endind, col_startind:col_endind]  = tracker
               end 
        end
        return bern_weights
end


#function test()
template_room = Room((18,12), (18,12), [1], [216])
desc = exits(template_room)

trackers_a = generate_trackers(template_room)
trackers_b = generate_trackers(template_room)

#tracker_nvs = map(nv, trackers_a)
#tracker_dims = map(x -> round(Int64,sqrt(nv(x))), trackers_a)

paths_a = a_star_paths(template_room, trackers_a)
paths_b = a_star_paths(template_room, trackers_b)

points_a = transforms(trackers_tograph(trackers_a), paths_a)
points_b = transforms(trackers_tograph(trackers_b), paths_b)

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
display(ot)
#end

ga = bern_plots(trackers_a)
gb = bern_plots(trackers_b)
row_axis_a = points_a[:,1]
col_axis_a = points_a[:,2]
row_axis_b = points_b[:,1]
col_axis_b = points_b[:,2]

# plots for multi-res path
plotly()
plot_a = heatmap(1:1:12, 1:1:18, ga, color = :Greys_9,size=(600,900)) 
scatter!(row_axis_a, col_axis_a, legend = false)
plot!(row_axis_a, col_axis_a, legend = false)
scatter!([row_axis_a[1],row_axis_a[na]],[col_axis_a[1],col_axis_a[na]], legend=false)

plot_b = heatmap(1:1:12, 1:1:18, gb, color = :Greys_9,size=(600,900), legend = false)
scatter!(row_axis_b, col_axis_b, legend = false)
plot!(row_axis_b, col_axis_b, legend = false)
scatter!([row_axis_b[1],row_axis_b[nb]],[col_axis_b[1],col_axis_b[nb]], legend=false)

plot_ref = plot(plot_a, plot_b, layout = (1, 2), legend = false)
savefig(plot_ref,"test/test_fig/fig.png")

# plots for sinkhorn and cost matrix
sinkhorn_plot = heatmap(
    collect(1:na),
    collect(1:nb),
    ot';
    title="Entropically regularised optimal transport",
    titlefont=font(10,"Computer Modern"),
    size=(na*30,nb*30),
    xlabel="path_a",ylabel= "path_b"
)

cost_plot = heatmap(
    collect(1:na),
    collect(1:nb),
    dm';
    title="Cost Matrix",
    titlefont=font(10,"Computer Modern"),
    size=(na*30,nb*30),
    xlabel="path_a",ylabel= "path_b"
)

savefig(sinkhorn_plot,"test/test_fig/sinkhorn_plot.png")
savefig(cost_plot,"test/test_fig/cost_plot.png")
