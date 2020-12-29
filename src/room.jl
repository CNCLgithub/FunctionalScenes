

"""
Defines a room
"""
struct Room
    steps::Tuple{Int64, Int64}
    bounds::Tuple{Float64, Float64}
    entrance::Vector{Tile}
    exits::Vector{Tile}
    graph::PathGraph
end

pathgraph(r::Room) = r.graph
bounds(r::Room) = r.bounds
steps(r::Room) = r.steps
entrance(r::Room) = r.entrance
exits(r::Room) = r.exits


# helpers
istype(g,v,t) = get_prop(g, v, :type) == t
isfloor(g,v) = istype(g,v,:floor)
iswall(g,v) = (length ∘ neighbors)(g, v) < 4
matched_type(g, e::Edge) = get_prop(g, src(e), :type) ==
    get_prop(g, dst(e), :type)

"""
Builds a room given ...
"""
function Room(steps::Tuple{T,T}, bounds::Tuple{G,G},
              ent::Vector{T}, exits::Vector{T}) where {T<:Int, G<:Real}
    # initialize grid
    g = PathGraph(grid(steps))

    # distinguish between walls and floor
    @>> g vertices foreach(v -> set_prop!(g, v, :type,
                                          iswall(g,v) ? :wall : :floor))

    # set entrances and exits
    # these are technically floors but are along the border
    @>> ent foreach(v -> set_prop!(g, v, :type, :floor))
    @>> exits foreach(v -> set_prop!(g, v, :type, :floor))

    # walls and non-walls cannot be connected
    togo = @>> g edges collect filter(e -> !matched_type(g, e))
    foreach(e -> rem_edge!(g, e), togo)

    return Room(steps, bounds, ent, exits, g)
end


function expand(r::Room, factor::Int64)::Room
    g = pathgraph(r)
    s = Tuple(collect(steps(r)) .* factor)
    rows, cols = s
    new_g = MetaGraph(grid(s))

    a = CartesianIndices(steps(r))
    c = LinearIndices(s)
    kern = CartesianIndices((0:factor-1, 0:factor-1))

    mp = @> g vertices reshape(steps(r)) repeat(inner=(factor,factor))
    # display(@>> g vertices filter(v -> !has_prop(g, v, :type)))
    # copy over tile info
    sp = v -> set_prop!(new_g, v, :type,  get_prop(g, mp[v], :type))
    @>> new_g vertices foreach(sp)
    # display(@>> new_g vertices filter(v -> !has_prop(new_g, v, :type)))
    # consolidate objects
    @>> new_g edges collect filter(e -> !matched_type(new_g, e)) foreach(e -> rem_edge!(new_g, e))
    # update entrances and exits
    ent = [first(c[a[entrance(r)] * factor .- kern])]
    exs = @>> r exits lazymap(v -> first(c[collect(a[v] * factor .- kern)])) collect

    Room(s, bounds(r) .* factor, ent, exs, new_g)
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

type_map = Dict{Symbol, Char}(
    :entrance => '◉',
    :exit => '◎',
    :wall => '■',
    :floor => '□',
    :furniture => '◆'
)

print_row(i) = print("$(String(i))")

function Base.show(io::IO, m::MIME"text/plain", r::Room)
    g = pathgraph(r)
    types = @>> pathgraph(r) vertices lazymap(v -> get_prop(g, v, :type)) collect(Symbol)
    types[entrance(r)] .= :entrance
    types[exits(r)] .= :exit
    grid = @>> types lazymap(t -> type_map[t]) collect(Char)
    grid = reshape(grid, steps(r))
    for i in eachrow(grid)
        print("\n")
        print(String(i))
    end
    # show(io, m, grid)
end


export Room, pathgraph, entrance, exits, bounds, steps
