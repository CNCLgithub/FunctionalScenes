const Furniture = Set{Tile}


function furniture(r::Room)::Array{Furniture}
    g = pathgraph(r)
    vs = @>> g begin
        connected_components
        filter(c -> istype(g, first(c), :furniture))
        map(Set{Tile})
        collect()
    end
end

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
    # shift tiles
    shifted = @>> f map(v -> shift_tile(r, v, move)) collect(Tile) Set
    # add shifted tiles to room
    new_r = add(r, shifted)
    new_g = pathgraph(new_r)
    # clear newly empty tiles
    @>> setdiff(f, shifted) foreach(v -> set_prop!(new_g, v, :type, :floor))
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
    # enn = collect(Tile, enn)
    special = vcat(e, en, enn, exits(r))
    # floor tiles that are not special
    vs = @>> g begin
        vertices
        filter(v -> isfloor(g, v) && !isempty(neighbors(g, v)))
        collect(Tile)
    end
    vs = setdiff(vs, special)
end



"""
Randomly samples a new piece of furniture
"""
@gen function furnish(r::Room)

    # first sample a vertex to add furniture to
    # - baking in a prior about number of immediate neighbors
    g = pathgraph(r)
    candidates = valid_spaces(r)
    nc = length(candidates)
    if isempty(candidates)
        return Set([])
    end
    # ws = nns ./ sum(nns)
    ws = fill(1.0 / nc, nc)
    vi = @trace(categorical(ws), :vertex)
    v = candidates[vi]
    ns = neighbors(g, v)
    nns = length(ns)
    # then pick a subset of neighbors if any
    # defined as a mbrfs
    p = 1.0 / nns
    mbrfs = map(n -> BernoulliElement{Any}(p, id, (n,)), ns)
    mbrfs = RFSElements{Any}(mbrfs)
    others = @trace(rfs(mbrfs), :neighbors)
    f = Set{Tile}(vcat(v, others))
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

function valid_move(g::PathGraph, f::Furniture, m)
    @>> setdiff(m, f) begin
        map(v -> has_vertex(g, v) && isfloor(g, v))
        all
    end
end

function valid_moves(r::Room, f::Furniture)
    g = pathgraph(r)
    vs = vertices(g)
    rows, cols = steps(r)
    moves = Matrix{Tile}(undef, length(f), 4)
    moves[:, 1] = f .- 1
    moves[:, 2] = f .+ 1
    moves[:, 3] = f .- rows
    moves[:, 4] = f .+ rows
    @>> moves eachcol lazymap(m -> valid_move(g,f,m)) collect(Float64)
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
