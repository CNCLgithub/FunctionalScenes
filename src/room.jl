
"""
Defines a room
"""
struct Room
    steps::Tuple{Int64, Int64}
    bounds::Tuple{Float64, Float64}
    entrance::Int64
    exits::Vector{Int64}
    graph::MetaGraph{Int64, Float64}
end

pathgraph(r::Room) = r.graph
bounds(r::Room) = r.bounds
steps(r::Room) = r.steps
entrance(r::Room) = r.entrance
exits(r::Room) = r.exits


# helpers

isfloor(g, v) = get_prop(g, v, :type) == :floor
iswall(g,v) = (length ∘ neighbors)(g, v) < 4
matched_type(g, e) = get_prop(g, src(e), :type) ==
    get_prop(g, dst(e), :type)

"""
Builds a room given ...
"""
function Room(steps::Tuple{T,T}, bounds::Tuple{G,G},
              entrance::T, exits::Vector{T}) where {T<:Int, G<:Real}
    # initialize grid
    g = MetaGraph(grid(steps))

    # distinguish between walls and floor
    @>> g vertices foreach(v -> set_prop!(g, v, :type,
                                          iswall(g,v) ? :wall : :floor))

    # set entrances and exits
    # these are technically floors but are along the border
    set_prop!(g, entrance, :type, :floor)
    @>> exits foreach(v -> set_prop!(g, v, :type, :floor))

    # walls and non-walls cannot be connected
    @>> g edges foreach(e -> set_prop!(g, e, :matched,
                                       matched_type(g, e)))
    @>> filter_edges(g, :matched, false) foreach(e -> rem_edge!(g, e))

    return Room(steps, bounds, entrance, exits, g)
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
    types[entrance(r)] = :entrance
    types[exits(r)] .= :exit
    grid = @>> types lazymap(t -> type_map[t]) collect(Char)
    grid = reshape(grid, steps(r))
    for i in eachrow(grid)
        print("\n")
        print(String(i))
    end
    # show(io, m, grid)
end


export Room, pathgraph
