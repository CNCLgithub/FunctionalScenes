using SimpleWeightedGraphs

export a_star_paths, transforms, trackers_tograph, trackers_merge

# convert matrix of matrices of floats (Bernoulli weights for each tracker) to matrix of matrices of SimpleWeightedGraph
function trackers_tograph(trackers::Matrix{Matrix{Float64}}, scale::Int64 = 6)
    trackers_row, trackers_col = size(trackers)
    trackers_graphs = Matrix{SimpleWeightedGraph{Int64, Float64}}(undef, trackers_row, trackers_col)
    @inbounds for col_ind = 1:trackers_col, row_ind = 1:trackers_row
        tracker_dim = size(trackers[row_ind, col_ind])[1]
        trackers_graphs[row_ind,col_ind] = SimpleWeightedGraph(grid([tracker_dim,tracker_dim]),scale/tracker_dim)
    end
    return trackers_graphs
end

function a_star_paths(r::Room, trackers::Matrix{Matrix{Float64}}, scale::Int64 = 6)
    trackers_row = fld(steps(r)[1], scale)
    trackers_col = fld(steps(r)[2], scale)
    trackers_graphs = trackers_tograph(trackers)
    trackers_weights = Vector{Float64}()  
    display(trackers_row)
    #nw = @>> trackers begin
    #    map(length)
    #    sum
    #end
    #trackers_weights = Vector{Float64}(undef, nw)
    
    @inbounds for col_ind = 1:trackers_col, row_ind = 1:trackers_row
	trackers_weights = [trackers_weights; vec(trackers[row_ind, col_ind])]
    end 

    #c = 0
    #@inbounds for i = 1:length(trackers)
    #    nt = length(trackers[i])
    #    trackers_weights[c:(c + nt)] = trackers[i]
    #    c += nt
    #end

    tracker_nvs = map(nv, trackers_graphs)
    tracker_dims = map(x -> round(Int64,sqrt(nv(x))), trackers_graphs)
    g = trackers_merge(trackers_graphs, tracker_nvs, tracker_dims)
    
    total_vs = sum(tracker_nvs)
    edge_mx = repeat(trackers_weights'; outer = (total_vs, 1))

    ent = @>> r entrance first
    ext = @>> r exits first
    tracker_ent = inv_transforms(trackers_graphs, ent)
    tracker_ext = inv_transforms(trackers_graphs, ext)

    a_star_path(g, edge_mx, tracker_ent, tracker_ext)
end

# after merge, apply a* algorithm to find the shortest path
function a_star_path(g::SimpleWeightedGraph{Int64, Float64},
                     bernweight::Matrix{Float64},
                     ent::Int64, ext::Int64)
    dist_mx = bernweight .* weights(g)
    g = MetaGraph(g)
    paths = a_star(g, ent, ext, dist_mx)
    nodes = [map(src,paths); ext]
    return nodes
end

# function that implement column merge and row merge together
# pass in the matrix of trackers
function trackers_merge(trackers::Matrix{SimpleWeightedGraph{Int64, Float64}},
               nvs::Matrix{Int64}, dims::Matrix{Int64})

    @assert !isempty(trackers) "Tracker matrix is empty"

    #print(nvs)
    total_row, total_column = size(nvs)
    # swap column and row merge
    columns = Vector{SimpleWeightedGraph{Int64, Float64}}(undef,
                                                          total_column)
 
    # TODO: can we use @simb?
    @inbounds for i = 1:total_column
        columns[i] = merge_by_column(trackers[:, i])
    end

    rowmerge = merge_by_firstrow(columns,nvs,dims)
    total_row === 1 && return rowmerge
    merge_by_row(rowmerge, nvs, dims, total_column, total_row)
    rowmerge
end

# first step: join by column
# join every tracker within the column in sequence
function merge_by_column(column::Vector{SimpleWeightedGraph{Int64, Float64}})
    #dim = map(x -> floor(Int64,sqrt(nv(x))),column)
    dim = map(x -> round(Int64, sqrt(nv(x))), column)
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
        right_ind_k = sum(nvs[:,1:col_ind]) + sum(nvs[1:(row_ind-1),(col_ind+1)])
        right_ind = collect((1 + right_ind_k):(rightdim + right_ind_k))
        index_merge!(graph,left_ind,right_ind)
    end
end

# Helper function
# add edges between given indices
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
        #num_nodes = floor(Int64, dim1 / dim2)
    num_nodes = round(Int64, dim1 / dim2)
        @inbounds for k = 1:dim2, j = ((k-1)*num_nodes+1):k*num_nodes
            add_edge!(graph, ind1[j],ind2[k], weight_numerator/dim2)
        end
    else
        #num_nodes = floor(Int64, dim2 / dim1)
    num_nodes = round(Int64, dim2 / dim1)
        @inbounds for k = 1:dim1, j in ((k-1)*num_nodes+1):k*num_nodes
            add_edge!(graph, ind1[k],ind2[j],weight_numerator/dim1)
        end
    end
    graph
end

# inverse-transform function from cartesian index to tracker index for entrance and destination
function inv_transforms(trackers::Matrix{SimpleWeightedGraph{Int64, Float64}}, 
                        node_ind::Int64, scale::Int64 = 6)

    # size and dimension for each tracker
    tracker_sizes = map(nv,trackers)
    tracker_dims = map(x -> round(Int64, sqrt(x)), tracker_sizes)
    nrow = size(trackers)[1]
    display(tracker_dims)
    
    # transformation from cartesian to tracker index
    col_ind = fld((node_ind-1),(nrow * scale)) + 1 # start from index 1
    row_ind = node_ind - nrow * scale * (col_ind - 1) # start from index 1
    tracker_col_ind  = fld((col_ind - 1), scale) + 1 # start from index 1
    tracker_row_ind  = fld((row_ind - 1), scale) + 1 # start from index 1
    within_col_ind = (col_ind - 1) - scale * (tracker_col_ind - 1) + 1 # start from index 1
    within_row_ind = (row_ind - 1) - scale * (tracker_row_ind - 1) + 1 # start from index 1
    
    display(col_ind)    
    # find the position within tracker
    tracker_dim = tracker_dims[tracker_row_ind, tracker_col_ind]
    tracker_within_col_ind = div((within_col_ind - 1), scale/tracker_dim) + 1 
    tracker_within_row_ind = div((within_row_ind - 1), scale/tracker_dim) + 1  

    # starting point for new index
    starting_col_ind = sum(tracker_sizes[:,1:(tracker_col_ind-1)])
    starting_row_ind = sum(tracker_sizes[1:(tracker_row_ind-1),tracker_col_ind])
    starting_ind = starting_col_ind + starting_row_ind

    # extra index within the tracker
    extra_ind = tracker_dim * (tracker_within_col_ind - 1) + tracker_within_row_ind
    new_ind = round(Int64, starting_ind + extra_ind)
    return new_ind
end

function transforms(trackers::Matrix{SimpleWeightedGraph{Int64, Float64}},
                    path_nodes::Vector{Int64}, scale::Int64 = 6)

    # size and dimension for each tracker
    tracker_sizes = vec(map(nv,trackers))
    tracker_dims = map(x -> round(Int64, sqrt(x)), tracker_sizes)
    nrow = size(trackers)[1]

    nvs_sum = cumsum(tracker_sizes)
    src_len = length(path_nodes)
    row_axis = Vector{Float64}(undef,src_len)
    col_axis = Vector{Float64}(undef,src_len)

    @inbounds for i in 1:src_len
        ind = path_nodes[i] .- nvs_sum
        tracker_ind= round(Int64, findfirst(y->y<=0,ind)) - 1 # start from 0
        within_ind = round(Int64, path_nodes[i] - sum(tracker_sizes[1:tracker_ind])) - 1 # start from 0
        row_start = scale * fld(tracker_ind,nrow)
        col_start = scale * round(Int64,mod(tracker_ind,nrow))

        tracker_dim = tracker_dims[(tracker_ind+1)]

        # upper-left index of the partition within tracker (transform to starting index of 1)
        row_within_start = fld(within_ind,tracker_dim) * scale/tracker_dim + 1
        col_within_start = mod(within_ind,tracker_dim) * scale/tracker_dim  + 1

        # get to the center position of the partition within the tracker
        row_within = row_within_start + 0.5* (scale/tracker_dim-1)
        col_within = col_within_start + 0.5* (scale/tracker_dim-1)

        row_axis[i] = row_start + row_within
        col_axis[i] = col_start + col_within
    end
    hcat(row_axis,col_axis)
end

