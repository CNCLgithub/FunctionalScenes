using SimpleWeightedGraphs

export a_star_path


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

# after merge, apply a* algorithm to find the shortest path
function a_star_path(g::SimpleWeightedGraph{Int64, Float64},
                     bernweight::Matrix{Float64},
                     ent::Int64, ext::Int64)
    dist_mx = bernweight .* weights(g)
    g = MetaGraph(g)
    a_star(g, ent, ext, dist_mx)
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

    rowmerge = merge_by_firstrow(columns,nvs,dims)
    total_row === 1 && return rowmerge
    merge_by_row(rowmerge, nvs, dims, total_column, total_row)
    rowmerge
end

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
        right_ind_k = sum(nvs[:,1:col_ind]) + sum(nvs[1:(row_ind-1),(col_ind+1)])
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


function transform(trackers::Matrix{SimpleWeightedGraph{Int64, Float64}}
		   path_nodes::Vector{Int64}, scale::Int64 = 6)
	
    # size and dimension for each tracker
    tracker_sizes = vec(map(nv,trackers))
    tracker_dims = map(x -> floor(Int64, sqrt(x)), tracker_sizes)
    nrow = size(trackers)[1]

    nvs_sum = cumsum(tracker_sizes)
    src_len = length(path_nodes)
    row_axis = Vector{Float64}(undef,src_len)
    col_axis = Vector{Float64}(undef,src_len)

    @inbounds for i in 1:src_len
        ind = path_nodes[i] .- nvs_sum
        tracker_ind= floor(Int64, findfirst(y->y<=0,ind)) - 1 # start from 0
        within_ind = floor(Int64, path_nodes[i] - sum(tracker_sizes[1:tracker_ind])) - 1 # start from 0
        row_start = scale * floor(Int64,tracker_ind/nrow)
        col_start = scale * floor(Int64,mod(tracker_ind,nrow))

        tracker_dim = tracker_dims[(tracker_ind+1)]

        # upper-left index of the partition within tracker (transform to starting index of 1)
        row_within_start = floor(Int64,within_ind/tracker_dim) * scale/tracker_dim + 1
        col_within_start = mod(within_ind,tracker_dim) * scale/tracker_dim  + 1

        # get to the center position of the partition within the tracker
        row_within = row_within_start + 0.5* (scale/tracker_dim-1)
        col_within = col_within_start + 0.5* (scale/tracker_dim-1)

        row_axis[i] = row_start + row_within
        col_axis[i] = col_start + col_within
    end
    return [row_axis,col_axis]
end

# transform to cartesian indexes x and y
function transform(path_nodes::Vector{Int64},
                   tracker_sizes::Vector{Int64},
                   tracker_dims::Vector{Int64},
                   room_dim::Tuple{Int64, Int64};
                   scale::Int64 = 6)
    nvs_sum = cumsum(tracker_sizes)
    src_len = length(path_nodes)
    row_axis = Vector{Float64}(undef,src_len)
    col_axis = Vector{Float64}(undef,src_len)

    @inbounds for i in 1:src_len
        ind = path_nodes[i] .- nvs_sum
        tracker_ind= floor(Int64, findfirst(y->y<=0,ind)) - 1 # start from 0
        within_ind = floor(Int64, path_nodes[i] - sum(tracker_sizes[1:tracker_ind])) - 1 # start from 0
        row_start = scale * floor(Int64,tracker_ind/nrow)
        col_start = scale * floor(Int64,mod(tracker_ind,nrow))

        tracker_dim = tracker_dims[(tracker_ind+1)]

        # upper-left index of the partition within tracker (transform to starting index of 1)
        row_within_start = floor(Int64,within_ind/tracker_dim) * scale/tracker_dim + 1
        col_within_start = mod(within_ind,tracker_dim) * scale/tracker_dim  + 1

        # get to the center position of the partition within the tracker
        row_within = row_within_start + 0.5* (scale/tracker_dim-1)
        col_within = col_within_start + 0.5* (scale/tracker_dim-1)

        row_axis[i] = row_start + row_within
        col_axis[i] = col_start + col_within
    end
    return [row_axis,col_axis]
end
