export QTNode, QTState

struct QTNode
    center::SVector{2, Float64}
    dims::SVector{2, Float64}
    level::Int64
    max_level::Int64
end
area(n::QTNode) = prod(n.dims)

# const QTBranch = SVector{4, QTNode}
# const QTLeaf = SVector{0, QTNode}

struct QTState
    mu::Float64
    u::Float64
    k::Int64
    node::QTNode
    # children::Union{QTBranch, QTLeaf}
    children::Vector{QTState}
end

weight(st::QTState) = st.mu
dof(st::QTState) = st.u
node(st::QTState) = st.node
Base.length(st::QTState) = st.k

function produce_weight(n::QTNode)::Float64
    @unpack level, max_level = n
    # maximum depth, do not split
    # otherwise uniform
    level == max_level ? 0.0 : 0.5
end

const sqrt_v = SVector{2, Float64}(fill(sqrt(2), 2))
const q1 = SVector{2, Float64}([-1.0,  1.0])
const q2 = SVector{2, Float64}([-1.0, -1.0])
const q3 = SVector{2, Float64}([ 1.0,  1.0])
const q4 = SVector{2, Float64}([ 1.0, -1.0])

# split
"""
Splits the node into 4 children, centered in 4 quadrants
with respect to the center of the parent.
"""
function produce_qt(n::QTNode)::Vector{QTNode}
    @unpack center, dims, level, max_level = n
    # calculate new centers
    dc = 0.25 .* dims .* sqrt_v
    # offset by 1/2 the diagonal
    c11 = center + (q1 .* dc) # top left
    c21 = center + (q2 .* dc) # bottom left
    c12 = center + (q3 .* dc) # top right
    c22 = center + (q4 .* dc) # bottom right
    # half xy dimensions (1/4 area)
    new_dims = 0.5 * dims
    children = Vector{QTNode}(undef, 4)
    children[1] = QTNode(c11, new_dims, level + 1, max_level)
    children[2] = QTNode(c12, new_dims, level + 1, max_level)
    children[3] = QTNode(c21, new_dims, level + 1, max_level)
    children[4] = QTNode(c22, new_dims, level + 1, max_level)
    # SVector{4, QTNode}(children)
    children
end

# merge
function aggregate_qt(n::QTNode, y::Float64,
                      children::Vector{QTState})::QTState
    if isempty(children)
        mu = y
        u  = 0.0
        k = 1
    else
        mu = mean(weight.(children))
        # equal area => mean of variance
        u = sqrt(mean(dof.(children).^2))
        k = sum(length.(children)) + 1
    end
    QTState(mu, u, k, n, children)
end

function adj_matrix(st::QTState)
    am = Matrix{Bool}(falses(st.k + 1, st.k + 1))
    if st.k == 1
        am[1,2] = true
    end
    adj_matrix!(am, st, 1, 1)
    return am
end

function adj_matrix!(am::Matrix{Bool},
                     st::QTState,
                     node::Int64,
                     node_index::Int64)
    c_index = node_index + 1
    isempty(st.children) && return c_index
    @show node => node_index
    # TODO: inbounds
    for i = 1:4
        # child in gen space
        c = get_child(node, i, 4)
        # child in graph space
        # add edge to adj matrix
        @show node => node_index =>  c_index
        am[node_index, c_index] = true
        # recurse through children
        c_index = adj_matrix!(am, st.children[i], c, c_index)
    end
    println("ret c_index: $(c_index)")
    return c_index
end
function graph_from_qt(st::QTState)
    SimpleDiGraph(adj_matrix(st))
end
