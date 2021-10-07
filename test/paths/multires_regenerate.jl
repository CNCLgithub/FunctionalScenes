using Gen
using FunctionalScenes
using FunctionalScenes: select_from_model
import FunctionalScenes: a_star_paths,transforms, trackers_merge
using LightGraphs,SimpleWeightedGraphs
using Distances
using OptimalTransport
using Plots
using Lazy: @>>

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

function div_weight(r::Room, trackers::Matrix{Matrix{Float64}}, 
                    path_nodes::Vector{Int64},
                    offset::CartesianIndex{2} = CartesianIndex(2, 2),
                    scale::Int64 = 6)
    offset = Tuple(offset) .* 2
    trackers_row, trackers_col = map(x->fld(x,scale),(steps(r).- offset))
    #trackers_row = fld(steps(r)[1], scale)
    #trackers_col = fld(steps(r)[2], scale)
    trackers_graphs = trackers_tograph(trackers)
    trackers_weights = Vector{Float64}()

    @inbounds for col_ind = 1:trackers_col, row_ind = 1:trackers_row
        trackers_weights = [trackers_weights; vec(trackers[row_ind, col_ind])]
    end

    tracker_nvs = map(nv, trackers_graphs)
    tracker_dims = map(x -> round(Int64,sqrt(nv(x))), trackers_graphs)
    g = trackers_merge(trackers_graphs, tracker_nvs, tracker_dims)

    total_vs = sum(tracker_nvs)
    edge_mx = repeat(trackers_weights'; outer = (total_vs, 1))
    dist_mx = edge_mx .* weights(g)
    
    div_weight = Vector{Float64}(undef,length(path_nodes))
    div_weight[1] = dist_mx[path_nodes[1], path_nodes[1]]
    @inbounds for ind = 2:length(path_nodes)
        div_weight[ind] = dist_mx[path_nodes[ind-1], path_nodes[ind]]
    end
    
    normed_weight = div_weight .* 1/sum(div_weight)
    return normed_weight
end



#function test()

    room_dims = (22,40)
    entrance = [11]
    exits = [640]
    r = Room(room_dims, room_dims, entrance, exits)
    #offset =  CartesianIndex(2, 2)
    weights_r = ones(steps(r))
    new_r = last(furniture_chain(10, r, weights_r))
    params = ModelParams(;
                         gt = new_r,
                         dims = (6, 6),
                         img_size = (240, 360),
                         instances = 10,
                         template = r,
                         feature_weights = "/spaths/datasets/alexnet_places365.pth.tar",
                         base_sigma = 10.0)

    @show params

    constraints = choicemap()
    # constraints = choicemap((:trackers => 1 => :level, 3))
    trace, ll = generate(model, (params,), constraints)

    println("ORIGINAL TRACE")
    display(trace[:trackers => 1 => :state])
    @show ll

    # mh(trace, split_merge_proposal, (1,), split_merge_involution)

    println("\n\n\nNEW TRACE")

    selection = select_from_model(params, 1)
    (new_trace, log_weight) = regenerate(trace, selection)

    # translator = Gen.SymmetricTraceTranslator(split_merge_proposal, (1,), split_merge_involution)
    # (new_trace, log_weight) = tracker_kernel(trace, translator, 1, selection)

    display(new_trace[:trackers => 1 => :state])
    @show log_weight


    trackers_a = @>> trace begin
        get_retval
        first
    end
    trackers_b = @>> new_trace begin
        get_retval
        first
    end


    paths_a = a_star_paths(r, trackers_a)
    paths_b = a_star_paths(r, trackers_b)

    points_a = transforms(trackers_tograph(trackers_a), paths_a)
    points_b = transforms(trackers_tograph(trackers_b), paths_b)

    # distance matrix
    dm = pairwise(Euclidean(), points_a, points_b, dims = 1)
    #display(dm)

    # number of vertices in paths
    na = size(points_a,1)
    nb = size(points_b,1)

    # discrete measures
    #measure_a = fill(1.0/na,na)
    #measure_b = fill(1.0/nb,nb)
    measure_a = div_weight(r, trackers_a, paths_a)
    measure_b = div_weight(r, trackers_b, paths_b)
    #display(measure_a)
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
    display(ga)

    #pyplot()
    plotlyjs()
    plot_a = heatmap(1:1:36, 1:1:18, ga', color = :Greys_9,size=(900,450), legend = false)
    scatter!(row_axis_a, col_axis_a, legend = false)
    plot!(row_axis_a, col_axis_a, legend = false)
    scatter!([row_axis_a[1],row_axis_a[na]],[col_axis_a[1],col_axis_a[na]], legend=false)
    #savefig(plot_a,"test/paths/multires_a.png")

    plot_b = heatmap(1:1:36, 1:1:18, gb', color = :Greys_9,size=(1800,900), legend = false)
    scatter!(row_axis_b, col_axis_b, legend = false)
    plot!(row_axis_b, col_axis_b, legend = false)
    scatter!([row_axis_b[1],row_axis_b[nb]],[col_axis_b[1],col_axis_b[nb]], legend=false)

    #Base.invokelatest(Plots.plot)
    plot_ref = plot(plot_a, plot_b, layout = (2, 1), legend = false)
    savefig(plot_ref,"test/paths/multires_regenerate.png")
#end
#savefig(test(),"test/paths/multires_regenerate.png")
#pyplot()
#test();
