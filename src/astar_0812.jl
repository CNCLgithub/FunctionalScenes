using LightGraphs, MetaGraphs, SimpleWeightedGraphs
using Profile, StatProfilerHTML

# first step: join by column

# join every tracker within the column in sequence
function merge_by_column(column::Vector{SimpleWeightedGraph{Int64, Float64}})
    dim = map(x -> floor(Int64,sqrt(nv(x))),column)
    num = size(column,1)
    num === 1 && return column[1]
    merge_by_column!(merge_by_column(column[1:(num-1)]),
               column[num],
               dim[num-1],
               dim[num])
end

function merge_by_column!(up::SimpleWeightedGraph{Int64, Float64},
                          down::SimpleWeightedGraph{Int64, Float64},
                          updim::Int64, downdim::Int64)
    # index of the touching face of tracker on top
    up_ind = collect((nv(up) - (updim-1)*updim):updim:nv(up))
    # index of the touching face of tracker on below
    dowm_ind = (nv(up) + 1) .+ collect(0:downdim:(downdim-1)*downdim)
    index_merge!(blockdiag(up, down), up_ind, dowm_ind)
end 

# second step: join by first row

# join the columns by their first row in sequence
function merge_by_firstrow(columns::Vector{SimpleWeightedGraph{Int64, Float64}}, 
                           nvs::Matrix{Int64}, dims::Matrix{Int64})
    num = size(columns,1)
    num === 1 && return columns[1]
    merge_by_firstrow!(merge_by_firstrow(columns[1:(num-1)],nvs, dims),
                       columns[num], nvs, dims, (num-1))
end   

function merge_by_firstrow!(left::SimpleWeightedGraph{Int64, Float64}, 
                            right::SimpleWeightedGraph{Int64, Float64}, 
                            nvs::Matrix{Int64}, dims::Matrix{Int64}, col_ind::Int64)
    #tracker to join on left and right column
    leftdim = dims[1,col_ind] 
    rightdim =  dims[1,(col_ind+1)]

    # index of touching face for left and right tracker
    left_ind_k = sum(nvs[:,1:(col_ind-1)]) + leftdim * (leftdim-1)
    left_ind = collect((1 + left_ind_k):(leftdim + left_ind_k))
    right_ind_k = sum(nvs[:,1:col_ind])
    right_ind = collect((1 + right_ind_k):(rightdim + right_ind_k))
    index_merge!(blockdiag(left, right),left_ind,right_ind)
end

# third step: add the rest of row-edges
function merge_by_row(graph::SimpleWeightedGraph{Int64, Float64}, 
                      nvs::Matrix{Int64}, dims::Matrix{Int64}, 
                      total_column::Int64, total_row::Int64)
    @inbounds for row_ind = 2:total_row, col_ind = 1:(total_column-1)
        leftdim = dims[row_ind, col_ind]
        rightdim = dims[row_ind, (col_ind+1)]
        left_ind_k = sum(nvs[:,1:(col_ind-1)]) + sum(nvs[1:row_ind, col_ind]) - leftdim
        left_ind = collect((1 + left_ind_k):(leftdim + left_ind_k))
        right_ind_k = sum(nvs[:,1:col_ind]) + sum(nvs[1:(row_ind-1),1:(col_ind+1)])
        right_ind = collect((1 + right_ind_k):(rightdim + right_ind_k))
        index_merge!(graph,left_ind,right_ind)
    end
end

# Helper function

# add edges given index
function index_merge!(graph::SimpleWeightedGraph{Int64, Float64},ind1::Vector{Int64},
                      ind2::Vector{Int64};
                      weight_numerator::Int64 = 6)
    dim1 = size(ind1,1)
    dim2 = size(ind2,1)
    if dim1 === dim2
        @inbounds for k = 1:dim1
            add_edge!(graph, ind1[k], ind2[k], weight_numerator/dim2) # weight of added edge
        end
    elseif dim1 > dim2
        # for each node in dim2, it is linked to multiple nodes in dim1
        num_nodes = floor(Int64, dim1 / dim2)
        @inbounds for k = 1:dim2, j = ((k-1)*num_nodes+1):k*num_nodes
            add_edge!(graph, ind1[j],ind2[k], weight_numerator/dim2)
        end
    else
        num_nodes = floor(Int64, dim2 / dim1)
        @inbounds for k = 1:dim1, j in ((k-1)*num_nodes+1):k*num_nodes
            add_edge!(graph, ind1[k],ind2[j],weight_numerator/dim1)
        end
    end
    graph
end   

# function that implement column merge and row merge together
# pass in the matrix of trackers
function merge(trackers::Matrix{SimpleWeightedGraph{Int64, Float64}},
               nvs::Matrix{Int64}, dims::Matrix{Int64})

    @assert !isempty(trackers) "Tracker matrix is empty"

    total_row, total_column = size(nvs)
    # swap column and row merge
    columns = Vector{SimpleWeightedGraph{Int64, Float64}}(undef,
                                                          total_column)
    # TODO: can we use @simb?
    @inbounds for i = 1:total_column
        columns[i] = merge_by_column(trackers[:, i])
    end

    total_row === 1 && return columns
    rowmerge = merge_by_firstrow(columns,nvs,dims)
    merge_by_row(rowmerge, nvs, dims, total_column, total_row)
    rowmerge
end

# planning graph type
const PathGraph = SimpleWeightedGraph{Int64, Float64}

# after merge, apply a* algorithm to find the shortest path
function a_star_path(g::PathGraph, bernweight::Matrix{Float64},
                     ent::Int64, ext::Int64)
    dist_mx = bernweight .* weights(g)
    g = MetaGraph(g)
    a_star(g, ent, ext, dist_mx)
end

#test
m1 = SimpleWeightedGraph(grid([6,6]),1.0)
m2 = SimpleWeightedGraph(grid([3,3]),2.0)
m3 = SimpleWeightedGraph(grid([1,1]),6.0)
m4 = SimpleWeightedGraph(grid([1,1]),6.0)
m5 = SimpleWeightedGraph(grid([6,6]),1.0)
m6 = SimpleWeightedGraph(grid([6,6]),1.0)

nvs = [[36,9,1]  [1,36,36]]
dims = [[6,3,1] [1,6,6]]


Profile.clear()
# Profile.init()
Profile.init(; n = 10^8, delay = 1E-9)
 @profilehtml merge([[m1,m2,m3] [m4,m5,m6]],nvs,dims)
 @profilehtml merge([[m1,m2,m3] [m4,m5,m6]],nvs,dims)
 g = merge([[m1,m2,m3] [m4,m5,m6]],nvs,dims)

# assume the following bernoulli weights
# distance between source and destination depend on the weighted average of funiture probability between source and desc vertices
b = rand(Float64, sum(nvs))
bernweight = ones(sum(nvs),sum(nvs))
nvs_b = [fill(1.0,36);fill(2.0,9);fill(6.0,1);fill(6.0,1);fill(1.0,36);fill(1.0,36)]
@inbounds for i in 1:sum(nvs), j in 1:sum(nvs)
    bernweight[i,j] = (b[i]*nvs_b[i] + b[j]*nvs_b[j])/(nvs_b[i]+nvs_b[j]) #weighted average
end

#display(size(merge_by_column([m1,m2])))
#dist_mx = bernweight .* weights(g)
#display(enumerate_paths(dijkstra_shortest_paths(g, 1, dist_mx),119))
#display(a_star(MetaGraph(g),1,119,dist_mx))
display(a_star_path(g,bernweight,1,119))
