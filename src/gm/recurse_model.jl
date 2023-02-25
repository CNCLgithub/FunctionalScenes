using Base.Iterators: product

export QTNode, QTState

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
struct QTNode
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
area(n::QTNode) = prod(n.dims)

# helper constants
const _slope =  SVector{2, Float64}([1., -1.])
const _intercept =  SVector{2, Float64}([0.5, 0.5])

"""
    pos_to_idx(pos, n)

Maps R^2 position of QTNode to a linear index in nxn

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

Maps R^2 position of QTNode to a linear index in nxn
"""
function node_to_idx(n::QTNode, d::Int64)
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

function contains(n::QTNode, p::SVector{2, Float64})
    @unpack center, dims = n
    lower = center - 0.5 * dims
    upper = center + 0.5 * dims
    all(lower .<= p) && all(p .<= upper)
end

"""

Maps a linear index in nxn to a R^2 plane [-0.5, 0.5]
"""
function idx_to_node_space(i::Int64, d::Int64)
    c = [i / d, ((i - 1) % d) + 1.0]
    c .*= 1.0/d
    c .+= -_intercept
    c .*= _slope
    SVector{2, Float64}(c)
end

function dist(x::QTNode, y::QTNode)
    norm(x.center - y.center) - (0.5*x.dims[1]) - (0.5*y.dims[1])
end

function contact(a::QTNode, b::QTNode)
    d = dist(a,b)
    contact(a, b, d)
end
function contact(a::QTNode, b::QTNode, d::Float64)
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

struct QTState
    mu::Float64
    u::Float64
    k::Int64
    leaves::Int64
    node::QTNode
    children::Vector{QTState}
end

weight(st::QTState) = st.mu
dof(st::QTState) = st.u
leaves(st::QTState) = st.leaves
node(st::QTState) = st.node
Base.length(st::QTState) = st.k

function leaf_vec(s::QTState)::Vector{QTState}
    v = Vector{QTState}(undef, s.leaves)
    add_leaves!(v, 1, s)
    return v
end

function add_leaves!(v::Vector{QTState}, i::Int64, s::QTState)
    if isempty(s.children)
        v[i] = s
        return i + 1
    end
    for c in s.children
        i = add_leaves!(v, i, c)
    end
    return i
end

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


function traverse_qt(head::QTState, dst::Int64)
    (isempty(head.children) || dst == 1) && return head
    d = get_depth(dst) - 1
    path = Vector{Int64}(undef, d)
    idx = dst
    @inbounds for i = 1:d
        path[d - i + 1] = Gen.get_child_num(idx, 4)
        idx = Gen.get_parent(idx, 4)
    end
    for i = 1:d
        # in the context of split-merge,
        # for the backward of a merge, t_prime will not
        # have a child at the last step
        isempty(head.children) && break
        head = head.children[path[i]]
    end
    head
end

function produce_weight(n::QTNode)::Float64
    @unpack level, max_level = n
    # maximum depth, do not split
    # otherwise uniform
    level == max_level && return 0.0
    return 0.5
    # level == 1 && return 0.9
    # return 0.3
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
    dc = 0.25 .* dims # .* sqrt_v
    # offset by 1/2 the diagonal
    c11 = center + (q1 .* dc) # top left
    c21 = center + (q2 .* dc) # bottom left
    c12 = center + (q3 .* dc) # top right
    c22 = center + (q4 .* dc) # bottom right
    # half xy dimensions (1/4 area)
    new_dims = 0.5 * dims
    children = Vector{QTNode}(undef, 4)
    children[1] = QTNode(c11, new_dims, level + 1, max_level,
                         Gen.get_child(n.tree_idx, 1, 4))
    children[2] = QTNode(c12, new_dims, level + 1, max_level,
                         Gen.get_child(n.tree_idx, 2, 4))
    children[3] = QTNode(c21, new_dims, level + 1, max_level,
                         Gen.get_child(n.tree_idx, 3, 4))
    children[4] = QTNode(c22, new_dims, level + 1, max_level,
                         Gen.get_child(n.tree_idx, 4, 4))
    # SVector{4, QTNode}(children)
    children
end

# merge
function aggregate_qt(n::QTNode, y::Float64,
                      children::Vector{QTState})::QTState
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
    QTState(y, u, k, l, n, children)
end

function inh_adj_matrix(st::QTState)
    am = Matrix{Bool}(falses(st.k + 1, st.k + 1))
    if st.k == 1
        am[1,2] = true
    end
    inh_adj_matrix!(am, st, 1, 1)
    return am
end

function inh_adj_matrix!(am::Matrix{Bool},
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
function inheritance_graph(st::QTState)
    SimpleDiGraph(inh_adj_matrix(st))
end

function nav_graph(st::QTState)
    adj = Matrix{Bool}(falses(st.leaves, st.leaves))
    ds = fill(Inf, st.leaves, st.leaves)
    lv = leaf_vec(st)
    for i = 1:(st.leaves-1), j = (i+1):st.leaves
        x = lv[i]
        y = lv[j]
        d = dist(x.node, y.node)
        # spatial distance of x <-> y
        dw = 0.5 * (x.node.dims[1] + y.node.dims[1])
        # average obstacle cost of x <-> y
        bw = 0.5 * (weight(x) + weight(y))
        w = dw * bw
        # println(i => j => d)
        # println(x.node.center => y.node.center => d)
        if contact(x.node, y.node, d)
            adj[i, j] = true
            adj[j, i] = true
            ds[i, j] = w
            ds[j, i] = w
        end
    end
    (adj, ds, lv)
end


function contains(st::QTState, p::SVector{2, Float64})
    contains(st.node, p)
end
