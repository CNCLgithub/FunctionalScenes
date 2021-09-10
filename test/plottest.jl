# Tests multi-resolution planning

using FunctionalScenes
using LightGraphs
using SimpleWeightedGraph
using Plots

# TODO add `Plots` tp test deps

m1 = SimpleWeightedGraph(grid([6,6]),1.0)
b1 = rand(Float64, 36)

m2 = SimpleWeightedGraph(grid([3,3]),2.0)
b2 = rand(Float64, 9)

m3 = SimpleWeightedGraph(grid([1,1]),6.0)
b3 = rand(Float64, 1)

m4 = SimpleWeightedGraph(grid([1,1]),6.0)
b4 = rand(Float64, 1)

m5 = SimpleWeightedGraph(grid([6,6]),1.0)
b5 = rand(Float64, 36)

m6 = SimpleWeightedGraph(grid([6,6]),1.0)
b6 = rand(Float64, 36)

nvs = [[36,9,1]  [1,36,36]]
dims = [[6,3,1] [1,6,6]]
g = merge([[m1,m2,m3] [m4,m5,m6]],nvs,dims)

b = [b1;b2;b3;b4;b5;b6]
bernweight = ones(sum(nvs),sum(nvs))
nvs_b = [fill(1.0,36);fill(2.0,9);fill(6.0,1);fill(6.0,1);fill(1.0,36);fill(1.0,36)]
@inbounds for i in 1:sum(nvs), j in 1:sum(nvs)
    bernweight[i,j] = (b[i]*nvs_b[i] + b[j]*nvs_b[j])/(nvs_b[i]+nvs_b[j]) #weighted average
end

src_node = map(src, a_star_path(g,bernweight,1,119))
src_node = [src_node;119]
nvs_vec = [36,9,1,1,36,36]
dims_vec = [6,3,1,1,6,6]

row_axis = transform(src_node,nvs_vec,dims_vec,3)[1]
col_axis = transform(src_node,nvs_vec,dims_vec,3)[2]
#display(transform(src_node,nvs_vec,dims_vec,3))
plot_ref = heatmap(1:1:12,1:1:18, gg, color = :Greys_9,size=(600,900)) 
scatter!(row_axis,col_axis, legend=false)
plot!(row_axis,col_axis, legend=false)
scatter!([row_axis[1],row_axis[src_len]],[col_axis[1],col_axis[src_len]], legend=false)
savefig(plot_ref,"/spaths/fn_1")
