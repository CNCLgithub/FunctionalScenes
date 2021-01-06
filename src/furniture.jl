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
    es = @>> f begin
        collect
        map(v -> @>> v neighbors(g) map(n -> Edge(v, n)))
        x -> vcat(x...)
    end
    # removes any edge that is no longer valid in the neighborhood (ie :furniture <-> :floor)
    foreach(e -> !matched_type(g, e) && rem_edge!(g, e), es)
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

export add, Furniture, furniture
