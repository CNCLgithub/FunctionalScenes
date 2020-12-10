
Furniture = Vector{Int64}

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

function valid_spaces(r)
    g = pathgraph(r)
    # cannot block entrance
    e = entrance(r)
    special = [e, neighbors(g, e)..., exits(r)...]
    vs = @> g vertices setdiff(special)
    vs = @>> vs filter(v -> isfloor(g, v))
    ns = @>> vs lazymap(v -> @>> v neighbors(g))
    nns = @>> ns lazymap(length)
    (vs, ns, nns)
end

@dist function id(x)
    probs = ones(1)
    xs = fill(x, 1)
    index = categorical(probs)
    xs[index]
end


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

@gen function furniture_step(t::Int, r::Room)
    f = @trace(furniture(r), :furniture)
    new_r = add(r, f)
    return new_r
end

furniture_chain = Gen.Unfold(furniture_step)

export add, Furniture, furniture, furniture_step, furniture_chain
