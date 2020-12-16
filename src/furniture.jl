
Furniture = Vector{Tile}

function add(r::Room, f::Furniture)::Room
    g = copy(pathgraph(r))
    # assign furniture status
    foreach(v -> set_prop!(g, v, :type, :furniture), f)
    # ensure that furniture is only connected to itself
    # returns a list of edges connected each furniture vertex
    es = @>> f lazymap(v -> @>> v neighbors(g) lazymap(n -> Edge(v, n))) flatten
    # removes any edge that is no longer valid in the neighborhood (ie :furniture <-> :floor)
    @>> es filter(e -> !matched_type(g, e)) foreach(e -> rem_edge!(g, e))
    Room(r.steps, r.bounds, r.entrance, r.exits, g)
end

function shift_tile(r::Room, t::Tile, m::Symbol)::Tile
    rows = first(steps(r))
    idx = copy(t)
    if m == :up
        idx += - 1
    elseif m == :down
        idx += 1
    elseif m == :left
        idx -= rows
    else
        idx += rows
    end
    return idx
end

function swap_tiles(g, p::Tuple{Tile, Tile})
    x,y = p
    new_g = copy(g)
    set_prop!(new_g, x, :type, get_prop(g, y, :type))
    set_prop!(new_g, y, :type, get_prop(g, x, :type))
    return new_g
end

function shift_furniture(r::Room, f::Furniture, move::Symbol)
    f = sort(f, rev = in(move, [:down, :right]))
    g = pathgraph(r)
    # shift tiles
    new_tiles = @>> f lazymap(v -> shift_tile(r, v, move)) zip(f)
    new_g = @>> new_tiles foldl((x,y) -> swap_tiles(x,y); init=g)
    # shift edges
    es = @>> f lazymap(v -> @>> v neighbors(g) lazymap(n -> Edge(v, n))) flatten
    new_edges = @>> es lazymap(e -> Edge(shift_tile(r, src(e), move),
                                         shift_tile(r, dst(e), move)))
    @>> es foreach(e -> rem_edge!(new_g, e))
    @>> new_edges foreach(e -> add_edge!(new_g, e))
    Room(r.steps, r.bounds, r.entrance, r.exits, new_g)
end

function valid_spaces(r)
    g = pathgraph(r)
    # cannot block entrance
    e = entrance(r)
    en = neighbors(g, e)
    enn = @>> en lazymap(v -> @>> v neighbors(g)) flatten
    special = [e, en..., enn..., exits(r)...]
    vs = @> g vertices setdiff(special)
    vs = @>> vs filter(v -> istype(g, v, :floor))
    ns = @>> vs lazymap(v -> @>> v neighbors(g))
    nns = @>> ns lazymap(length)
    (vs, ns, nns)
end

@dist function labelled_categorical(xs)
    n = length(xs)
    probs = fill(1.0 / n, n)
    index = categorical(probs)
    xs[index]
end

@dist function id(x)
    probs = ones(1)
    xs = fill(x, 1)
    index = categorical(probs)
    xs[index]
end


"""
Randomly samples a new piece of furniture
"""
@gen function furniture(r::Room)::Furniture

    # first sample a vertex to add furniture to
    # - baking in a prior about number of immediate neighbors
    (vs, ns, nns) = valid_spaces(r)
    # ws = nns ./ sum(nns)
    ws = fill(1.0 / length(nns), length(nns))
    vi = @trace(categorical(ws), :vertex)
    v = vs[vi]
    # then pick a subset of neighbors if any
    # defined as a mbrfs
    p = 1.0 / nns[vi]
    mbrfs = map(n -> BernoulliElement{Any}(p, id, (n,)), ns[vi])
    mbrfs = RFSElements{Any}(mbrfs)
    others = @trace(rfs(mbrfs), :neighbors)
    f = [vs[vi], others...]
    return f
end

"""
Adds a randomly generated piece of furniture
"""
@gen function furniture_step(t::Int, r::Room)
    f = @trace(furniture(r), :furniture)
    new_r = add(r, f)
    return new_r
end

furniture_chain = Gen.Unfold(furniture_step)


const move_map = [:up, :down, :left, :right]

function valid_moves(r::Room, f::Furniture)
    g = pathgraph(r)
    vs = vertices(g)
    rows, cols = steps(r)
    moves = Matrix{Tile}(undef, length(f), 4)
    moves[:, 1] = f .- 1
    moves[:, 2] = f .+ 1
    moves[:, 3] = f .- rows
    moves[:, 4] = f .+ rows
    valid = @>> moves eachcol lazymap(m -> @>> setdiff(m, f) filter(v -> !isfloor(g, v, )))
    valid = @>> valid lazymap(isempty)
    collect(Float64, valid)
end


function connected(g, v::Tile)::Vector{Tile}
    @>> v bfs_tree(g) edges collect induced_subgraph(g) last
end

"""
Move a piece of furniture
"""
@gen function reorganize(r::Room)
    # pick a random furniture block, this will prefer larger pieces
    g = pathgraph(r)
    vs = @>> g vertices filter(v -> istype(g, v, :furniture))
    n = length(vs)
    ps = fill(1.0/n, n)
    vi = @trace(categorical(ps), :block)
    v = vs[vi]
    f = connected(g, v)
    # find the valid moves and pick one at random
    # each move will be one unit
    moves = valid_moves(r, f)
    move_probs = moves ./ sum(moves)
    move_id = @trace(categorical(move_probs), :move)
    move = move_map[move_id]
    new_r = shift_furniture(r, f, move)
end


export add, Furniture, furniture, furniture_step, furniture_chain, reorganize
