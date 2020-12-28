Furniture = Set{Tile}

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


function patch!(r::Room, t::Tile, move::Symbol)
    g = pathgraph(r)
    spots = setdiff(move_map, [move])
    @>> spots begin
        lazymap(m -> shift_tile(r, t, m))
        flatten
        filter(v -> has_vertex(g, v))
        lazymap(v -> Edge(t, v))
        filter(e -> matched_type(g, e))
        foreach(e -> add_edge!(g, e))
    end
    return nothing
end

function shift_furniture(r::Room, f::Furniture, move::Symbol)
    f = sort(collect(f), rev = in(move, [:down, :right]))
    g = pathgraph(r)
    new_g = copy(g)
    # shift tiles
    new_tiles = @>> f begin
        lazymap(v -> shift_tile(r, v, move))
        zip(f)
        foreach(xy -> swap_tiles!(new_g, xy))
    end
    # edges
    es = @>> f begin
        lazymap(v -> @>> v neighbors(g) lazymap(n -> Edge(v, n)))
        flatten
    end
    # remove old edges
    foreach(e -> rem_edge!(new_g, e), es)
    # add shifted edges
    shifted = @>> es lazymap(e -> Edge(shift_tile(r, src(e), move),
                                       shift_tile(r, dst(e), move)))
    foreach(e -> add_edge!(new_g, e), shifted)

    new_r = Room(r.steps, r.bounds, r.entrance, r.exits, new_g)
    # patch edges to tiles that are now empty
    foreach(t -> patch!(new_r, t, move), f)
    return new_r
end

function valid_spaces(r::Room)
    g = pathgraph(r)
    # cannot block entrance
    e = first(entrance(r))
    en = neighbors(g, e)
    enn = @>> en lazymap(v -> @>> v neighbors(g)) flatten
    special = [e, en..., enn..., exits(r)...]
    vs = @> g vertices setdiff(special)
    vs = @>> vs filter(v -> istype(g, v, :floor))
    ns = @>> vs lazymap(v -> @>> v neighbors(g))
    nns = @>> ns lazymap(length)
    (vs, ns, nns)
end

function furniture(r::Room)::Array{Furniture}
    g = pathgraph(r)
    vs = @>> g vertices filter(v -> istype(g, v, :furniture))
    @>> vs map(v -> connected(g, v)) unique
end


"""
Randomly samples a new piece of furniture
"""
@gen function furnish(r::Room)

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
    f = Set([vs[vi], others...])
    return f
end

"""
Adds a randomly generated piece of furniture
"""
@gen (static) function furniture_step(t::Int, r::Room)
    f = @trace(furnish(r), :furniture)
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
    valid = @>> moves begin
        eachcol
        lazymap(m -> @>> setdiff(m, f) filter(v -> has_vertex(g, v)) filter(v -> !isfloor(g, v)))
        lazymap(isempty)
    end
    collect(Float64, valid)
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

    inds = CartesianIndices(steps(r))
    move_probs = moves ./ sum(moves)
    move_id = @trace(categorical(move_probs), :move)
    move = move_map[move_id]
    new_r = shift_furniture(r, f, move)
end


export add, Furniture, furniture, furnish, furniture_step, furniture_chain, reorganize
