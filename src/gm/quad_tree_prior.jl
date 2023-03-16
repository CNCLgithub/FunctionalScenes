using Base.Iterators: product

export QTProdNode, QTAggNode


#################################################################################
# Production
#################################################################################

"""

A production node in the quad tree.
Defines a spatially oriented rectangle

# Properties

- center: The XY center of the node
- dims: The xy extents of the node
- level: The number of splits with `level==1` denoting no splits
- max_level: The maximum number of splits allowed
- tree_idx: The Gen-trace index of the node in the production trace
"""
struct QTProdNode
    center::SVector{2, Float64}
    dims::SVector{2, Float64}
    level::Int64
    max_level::Int64
    tree_idx::Int64
end

"""
    area(n)

The area of the node.
"""
area(n::QTProdNode) = prod(n.dims)

# helper constants
const _slope =  SVector{2, Float64}([1., -1.])
const _intercept =  SVector{2, Float64}([0.5, 0.5])

"""
    pos_to_idx(pos, n)

Maps R^2 position of QTProdNode to a linear index in nxn

# Arguments
- pos: XY position
- n  : The
"""
function pos_to_idx(pos::SVector{2, Float64}, n::Int64)
    # cartesian coordinates
    c = @. ceil(Int64, n * (pos * _slope + _intercept))
    # linear index
    ceil(Int64, n*(c[1] - 1) + c[2])
end

"""

Maps R^2 position of QTProdNode to a linear index in nxn
"""
function node_to_idx(n::QTProdNode, d::Int64)
    @unpack center, level, max_level, dims = n
    if level == max_level
        # single index
        idx = [pos_to_idx(center, d)]
    else
        # offset from each boundry
        fac = (0.5 - (1.0 / exp2(max_level - level + 1)))
        lower = center - fac * dims
        upper = center + fac * dims
        steps = Int64(exp2(max_level - level))
        # xy ordering to match julia col-wise
        # doesn't actually matter
        xs = LinRange(lower[1], upper[1], steps)
        ys = LinRange(upper[2], lower[2], steps)
        idx = Vector{Int64}(undef, steps^2)
        for (i,(y,x)) in enumerate(product(ys, xs))
            sv = SVector{2, Float64}([x, y])
            idx[i] = pos_to_idx(sv, d)
        end
    end
    return idx
end

function contains(n::QTProdNode, p::SVector{2, Float64})
    @unpack center, dims = n
    lower = center - 0.5 * dims
    upper = center + 0.5 * dims
    all(lower .<= p) && all(p .<= upper)
end

"""

Maps a linear index in nxn to a R^2 plane [-0.5, 0.5]

# Arguments
- i: linear index
- d: number of columns
"""
function idx_to_node_space(i::Int64, d::Int64)
    c = [i / d, ((i - 1) % d) + 1.0]
    c .*= 1.0/d
    c .+= -_intercept
    c .*= _slope
    SVector{2, Float64}(c)
end

function dist(x::QTProdNode, y::QTProdNode)
    norm(x.center - y.center)
end

"""
    contact(a, b)

True if a wall of each node are touching.
> Note: Both nodes can be of different area (granularity)
"""
function contact(a::QTProdNode, b::QTProdNode)
    d = dist(a,b) - (0.5*a.dims[1]) - (0.5*b.dims[1])
    contact(a, b, d)
end
function contact(a::QTProdNode, b::QTProdNode, d::Float64)
    aa = area(a)
    ab = area(b)
    # if same size, then contact is simply distance
    aa == ab && return isapprox(d, 0.)
    # otherwise must account for diagonal
    big, small = aa > ab ? (a, b) : (b, a)
    dlevel = small.level - big.level
    thresh = 0.5 * big.dims[1] * (1 - exp2(-(dlevel + 1)))
    d < thresh
end

# REVIEW: Some way to parameterize weights?
function produce_weight(n::QTProdNode)::Float64
    @unpack level, max_level = n
    # maximum depth, do not split
    # otherwise uniform
    level == max_level && return 0.0
    return 0.5
end

const sqrt_v = SVector{2, Float64}(fill(sqrt(2), 2))
const q1 = SVector{2, Float64}([-1.0,  1.0])
const q2 = SVector{2, Float64}([-1.0, -1.0])
const q3 = SVector{2, Float64}([ 1.0,  1.0])
const q4 = SVector{2, Float64}([ 1.0, -1.0])

"""
Splits the node into 4 children, centered in 4 quadrants
with respect to the center of the parent.
"""
function produce_qt(n::QTProdNode)::Vector{QTProdNode}
    @unpack center, dims, level, max_level = n
    # calculate new centers
    dc = 0.25 .* dims # .* sqrt_v
    # offset by 1/2 the diagonal
    c11 = center + (q1 .* dc) # top left
    c21 = center + (q2 .* dc) # bottom left
    c12 = center + (q3 .* dc) # top right
    c22 = center + (q4 .* dc) # bottom right
    # half xy dimensions (1/4 area)
    new_dims = 0.5 * dims
    children = Vector{QTProdNode}(undef, 4)
    children[1] = QTProdNode(c11, new_dims, level + 1, max_level,
                         Gen.get_child(n.tree_idx, 1, 4))
    children[2] = QTProdNode(c12, new_dims, level + 1, max_level,
                         Gen.get_child(n.tree_idx, 2, 4))
    children[3] = QTProdNode(c21, new_dims, level + 1, max_level,
                         Gen.get_child(n.tree_idx, 3, 4))
    children[4] = QTProdNode(c22, new_dims, level + 1, max_level,
                         Gen.get_child(n.tree_idx, 4, 4))
    # REVIEW: Way to used static vector?
    # SVector{4, QTProdNode}(children)
    children
end


#################################################################################
# Aggregation
#################################################################################

struct QTAggNode
    mu::Float64
    u::Float64
    k::Int64
    leaves::Int64
    node::QTProdNode
    children::Vector{QTAggNode}
end

weight(st::QTAggNode) = st.mu
dof(st::QTAggNode) = st.u
leaves(st::QTAggNode) = st.leaves
node(st::QTAggNode) = st.node
Base.length(st::QTAggNode) = st.k

function contains(st::QTAggNode, p::SVector{2, Float64})
    contains(st.node, p)
end

"""
    leaf_vec(s)

Returns all of the leaf `QTAggNode`s from root `s`.
"""
function leaf_vec(s::QTAggNode)::Vector{QTAggNode}
    v = Vector{QTAggNode}(undef, s.leaves)
    add_leaves!(v, s)
    return v
end

function add_leaves!(v::Vector{QTAggNode}, s::QTAggNode)
    heads::Vector{QTAggNode} = [s]
    i::Int64 = 1
    while !isempty(heads)
        head = pop!(heads)
        if isempty(head.children)
            v[i] = head
            i += 1
        else
            append!(heads, head.children)
        end
    end
    return nothing
end

"""

Aggregates quad tree production nodes into a tree, keeping track of
DOF.
"""
function aggregate_qt(n::QTProdNode, y::Float64,
                      children::Vector{QTAggNode})::QTAggNode
    if isempty(children)
        u  = 0.0
        k = 1
        l = 1
    else
        # equal area => mean of variance
        u = sqrt(mean(dof.(children).^2))
        k = sum(length.(children)) + 1
        l = sum(leaves.(children))
    end
    QTAggNode(y, u, k, l, n, children)
end

# struct QuadTree
#     g::SimpleDiGraph{Int64, Float64}
#     d::Vector{QTAggNode}
# end

# function QuadTree(root::QTAggNode)
#     @unpack n = root
#     d = Vector{QTAggNode}(undef, n)
#     adj = fill(false, (n, n))
#     @inbounds for i = 1:n
#         pass
#     end
# end

#################################################################################
# Misc.
#################################################################################

"""
    get_depth(n::Int64)

Returns the depth of a node in a quad tree,
using Gen's `Recurse` indexing system.
"""
function get_depth(n::Int64)
    n == 1 && return 1
    p = Gen.get_parent(n, 4)
    d = 2
    while p != 1
        p = Gen.get_parent(p, 4)
        d += 1
    end
    d
end


"""
    traverse_qt(root, dest)

Returns the quad tree node at index `dest`.
"""
function traverse_qt(root::QTAggNode, dest::Int64)
    # No children or root is destination
    (isempty(root.children) || dest == 1) && return root
    d = get_depth(dest) - 1
    path = Vector{Int64}(undef, d)
    idx = dest
    # REVIEW: More direct mapping of dest -> Node?
    # build path to root from dest
    @inbounds for i = 1:d
        path[d - i + 1] = Gen.get_child_num(idx, 4)
        idx = Gen.get_parent(idx, 4)
    end
    # traverse qt using `path`
    for i = 1:d
        # in the context of split-merge,
        # for the backward of a merge, t_prime will not
        # have a child at the last step
        isempty(root.children) && break
        root = root.children[path[i]]
    end
    root
end


# function inh_adj_matrix(st::QTAggNode)
#     am = Matrix{Bool}(falses(st.k + 1, st.k + 1))
#     if st.k == 1
#         am[1,2] = true
#     end
#     inh_adj_matrix!(am, st, 1, 1)
#     return am
# end

# function inh_adj_matrix!(am::Matrix{Bool},
#                      st::QTAggNode,
#                      node::Int64,
#                      node_index::Int64)
#     c_index = node_index + 1
#     isempty(st.children) && return c_index
#     @show node => node_index
#     # TODO: inbounds
#     for i = 1:4
#         # child in gen space
#         c = get_child(node, i, 4)
#         # child in graph space
#         # add edge to adj matrix
#         @show node => node_index =>  c_index
#         am[node_index, c_index] = true
#         # recurse through children
#         c_index = adj_matrix!(am, st.children[i], c, c_index)
#     end
#     println("ret c_index: $(c_index)")
#     return c_index
# end
# function inheritance_graph(st::QTAggNode)
#     SimpleDiGraph(inh_adj_matrix(st))
# end
