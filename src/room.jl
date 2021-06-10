
const Tile = Int64
const PathGraph = MetaGraphs.MetaGraph{Int64, Float64}

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
wall_edge(g, e::Edge) = (get_prop(g, src(e), :type) == :wall) &&
    (get_prop(g, dst(e), :type) == :wall)


"""
Builds a room given ...
"""
function Room(steps::Tuple{T,T}, bounds::Tuple{G,G},
              ent::Vector{T}, exs::Vector{T}) where {T<:Int, G<:Real}
    # initialize grid
    g = PathGraph(grid(steps))

    # distinguish between walls and floor
    @>> g vertices foreach(v -> set_prop!(g, v, :type,
                                          iswall(g,v) ? :wall : :floor))

    # set entrances and exits
    # these are technically floors but are along the border
    @>> ent foreach(v -> set_prop!(g, v, :type, :floor))
    @>> exs foreach(v -> set_prop!(g, v, :type, :floor))

    # walls and non-walls cannot be connected
    togo = @>> g edges collect filter(e -> !matched_type(g, e))
    foreach(e -> rem_edge!(g, e), togo)

    return Room(steps, bounds, ent, exs, g)
end


function expand(r::Room, factor::Int64)::Room
    g = pathgraph(r)
    s = Tuple(collect(steps(r)) .* factor)
    rows, cols = s

    # will copy properties and edges from `src`
    new_g = MetaGraph(SimpleGraph(prod(s)))

    # indices of src
    src_c = CartesianIndices(steps(r))
    # indices of dest
    dest_c = CartesianIndices(s)
    dest_l = LinearIndices(dest_c)
    # mapping for src v -> dest vs
    kern = CartesianIndices((1:factor, 1:factor))

    # map for dest v -> src v
    mp = @> g begin
        vertices
        # form m x n lattice of src
        reshape(steps(r))
        # scale up to dest
        repeat(inner=(factor,factor))
    end

    for src in vertices(g)
        # vs in dest that correspond to the same v in src
        sister_vs = dest_l[src_c[src] .+ kern]
        sister_vs = up_scale_inds(src_c, dest_c, factor, src)

        # neighbors of src mapped to potential neighbors in dest
        src_ns = @>> neighbors(g, src) collect(Tile)
        dest_n_candidates = up_scale_inds(src_c, dest_c, factor, src_ns)

        for dest in sister_vs
            # copy type
            set_prop!(new_g, dest, :type,
                    get_prop(g, src, :type))

            # connect sister vs according to grid structure
            dc = dest_c[dest]
            for sv in sister_vs
                svc = dest_c[sv]
                # sister is either left,right,above,below
                if !is_next_to(dc, svc)
                    continue
                end
                add_edge!(new_g, Edge(dest, sv))
            end

            # connect to neighbhors from src
            for sn in dest_n_candidates
                snc = dest_c[sn]
                if !is_next_to(dc, snc)
                    continue
                end
                add_edge!(new_g, Edge(dest, sn))
            end
        end
    end

    # update entrances and exits
    ent = @>> r begin
        entrance
        first
        up_scale_inds(src_c, dest_c, factor)
        first
        x -> [x]
    end
    exs = @>> r begin
        exits
        first
        up_scale_inds(src_c, dest_c, factor)
        first
        x -> [x]
    end
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
    :furniture => '◆',
    :path => '○'
)

print_row(i) = print("$(String(i))")

function Base.show(io::IO, m::MIME"text/plain", r::Room)
    Base.show(io,m,(r, Tile[]))
end
function Base.show(io::IO, m::MIME"text/plain", t::Tuple{Room, Vector{Tile}})
    r, paths = t
    g = pathgraph(r)
    types = @>> r begin
        pathgraph 
        vertices 
        map(v -> get_prop(g, v, :type)) 
        collect(Symbol)
    end
    types[entrance(r)] .= :entrance
    types[exits(r)] .= :exit
    types[paths] .= :path
    grid = @>> types map(t -> type_map[t]) collect(Char)
    grid = reshape(grid, steps(r))
    for i in eachrow(grid)
        print("\n")
        print(String(i))
    end
    # show(io, m, grid)
end


export Room, pathgraph, entrance, exits, bounds, steps
