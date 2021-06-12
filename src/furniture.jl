export add, remove, Furniture, furniture, strongly_connected

const Furniture = Set{Tile}

furniture(r::Room)::Array{Furniture} = furniture(pathgraph(r))

function furniture(g::PathGraph)::Array{Furniture}
    vs = @>> g begin
        connected_components
        filter(c -> istype(g, first(c), :furniture))
        map(Set{Tile})
        collect()
    end
end

"""
Adds the furniture of `src` to `dest`
"""
function add(src::Room, dest::Room)::Room
    @>> src begin
        furniture
        fs -> foldl(add, fs; init = dest)
    end
end

function add(r::Room, f::Furniture)::Room
    g = deepcopy(pathgraph(r))
    # assign furniture status
    foreach(v -> set_prop!(g, v, :type, :furniture), f)
    # ensure that furniture is only connected to itself
    # returns a list of edges connected each furniture vertex
    @>> f begin
        collect(Tile)
        foreach(v -> @>> v begin
                    neighbors(g)
                    collect(Tile) # must collect for compete iteration
                    # removes any extraneous edges
                    foreach(x -> !in(x, f) && rem_edge!(g, Edge(x, v)))
                end)
    end
    Room(r.steps, r.bounds, r.entrance, r.exits, g)
end

function remove(r::Room, f::Furniture)::Room
    g = copy(pathgraph(r))
    foreach(v -> set_prop!(g, v, :type, :floor), f)
    new_r = Room(r.steps, r.bounds, r.entrance, r.exits, g)
    foreach(t -> patch!(new_r, t, move_map), f)
    return new_r
end

function clear_room(r::Room)::Room
    g = pathgraph(r)
    @>> g begin
        vertices
        Base.filter(v -> istype(g,v,:furniture))
        x -> Furniture(x)
        remove(r)
    end
end

function patch!(r::Room, t::Tile, moves::Vector{Symbol})
    g = pathgraph(r)
    @>> moves begin
        lazymap(m -> shift_tile(r, t, m))
        flatten
        filter(v -> has_vertex(g, v))
        lazymap(v -> Edge(t, v))
        filter(e -> matched_type(g, e))
        foreach(e -> add_edge!(g, e))
    end
    return nothing
end


function shift_furniture(r::Room, f::Furniture, move::Int64)
    shift_furniture(r, f, move_map[move])
end

function shift_furniture(r::Room, f::Furniture, move::Symbol)
    f = sort(collect(f), rev = in(move, [:down, :right]))
    fs = furniture(r)

    fid = findfirst(x -> !isempty(intersect(f, x)), fs)
    return shift_furniture(r, fid, move)
end


function shift_furniture(r::Room, fid::Int64, move::Symbol)
    fs = furniture(r)
    f = fs[fid] |> collect
    # shift tiles
    shifted = @>> f map(v -> shift_tile(r, v, move)) collect(Tile) Set
    fs[fid] = shifted

    source = pathgraph(r)

    new_r = Room(r.steps, r.bounds, r.entrance, r.exits)
    g = pathgraph(new_r)

    @>> source begin
        vertices
        collect
        Base.filter(v -> isfloor(source, v))
        foreach(v -> set_prop!(g, v, :type, :floor))
    end
    wall_tiles = @>> source begin
        vertices
        collect
        Base.filter(v -> get_prop(source, v, :type) == :wall)
    end
    foreach(v -> set_prop!(g, v, :type, :wall), wall_tiles)
    @>> wall_tiles begin
        foreach(v -> @>> neighbors(g, v) collect map(n -> rem_edge!(g, Edge(v, n))))
    end

    @>> source begin
        edges
        collect
        Base.filter(e -> wall_edge(source, e))
        foreach(e -> add_edge!(g, e))
    end

    # add furniture
    Base.reduce(add, fs; init = new_r)
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

function strongly_connected(r::Room, f::Furniture, move::Symbol)

    r_cart = CartesianIndices(steps(r))
    f_inds = collect(Int64, f)
    shifted = @>> f_inds begin
        map(v -> shift_tile(r, v, move))
        collect(Tile)
    end

    c1 = CartesianIndex(0, 1)
    c2 = CartesianIndex(1, 0)

    connected = Int64[]
    for (i, other) in enumerate(furniture(r))

        other == f && continue
        other_inds = collect(Int64, other)
        gs = gdistances(grid(steps(r)), other_inds)

        !any(d -> d == 1, gs[f_inds]) && continue
        # return if moving breaks connection
        !any(d -> d == 1, gs[shifted]) && return Int64[]

        push!(connected, i)
    end

    return connected
end
