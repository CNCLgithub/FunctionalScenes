export GridRoom, pathgraph, data, entrance, exits, bounds, steps, expand,
    from_json

#################################################################################
# Type aliases
#################################################################################

# Scenes contain a lattice graph over the paths in the room
const PathGraph = SimpleGraph{Int64}


#################################################################################
# GridRoom
#################################################################################

struct GridRoom <: Room
    steps::Tuple{Int64, Int64}
    bounds::Tuple{Float64, Float64}
    entrance::Vector{Int64}
    exits::Vector{Int64}
    graph::PathGraph
    data::Matrix{Tile}
end

pathgraph(r::GridRoom) = r.graph
data(r::GridRoom) = r.data
bounds(r::GridRoom) = r.bounds
steps(r::GridRoom) = r.steps
entrance(r::GridRoom) = r.entrance
exits(r::GridRoom) = r.exits


# helpers
# istype(g,v,t) = get_prop(g, v, :type) == t
# isfloor(g,v) = istype(g,v,:floor)
# iswall(g,v) = (length ∘ neighbors)(g, v) < 4
matched_tile(d::Array{Tile}, e::AbstractEdge) = d[src(e)] == d[dst(e)]
floor_edge(d::Array{Tile}, e::AbstractEdge) = (d[src(e)] == floor_tile) &&
    (d[dst(e)] == floor_tile)
# wall_edge(g, e::Edge) = (get_prop(g, src(e), :type) == :wall) &&
#     (get_prop(g, dst(e), :type) == :wall)
get_tiles(r::GridRoom, t::Tile) = findall(data(r) .== t)


function GridRoom(steps, bounds)
    # initialize grid
    g = PathGraph(grid(steps))
    d = fill(floor_tile, steps)
    GridRoom(steps, bounds, Int64[], Int64[], g, d)
end

"""
Builds a room given ...
"""
function GridRoom(steps, bounds, ent, exs)

    g = PathGraph(grid(steps))
    d = Matrix{Tile}(undef, steps)
    d[:, :] .= floor_tile

    # add walls
    d[:, 1] .= wall_tile
    d[:, end] .= wall_tile
    d[1, :] .= wall_tile
    d[end, :] .= wall_tile

    # set entrances and exits
    # these are technically floors but are along the border
    d[ent] .= floor_tile
    d[exs] .= floor_tile

    # walls and non-walls cannot be connected
    prune_edges!(g, d)

    GridRoom(steps, bounds, ent, exs, g, d)
end


# TODO: Generalize?
function prune_edges!(g, d)
    for e in collect(edges(g))
        !matched_tile(d, e) && rem_edge!(g, e)
    end
    return nothing
end

function expand(r::GridRoom, factor::Int64)::GridRoom
    s = steps(r) .* factor
    # "expand" by `factor`
    d = data(r)
    sd = repeat(d, inner = (factor, factor))
    sg = PathGraph(grid(s))

    prune_edges!(sg, sd)

    cis = CartesianIndices(d)
    slis = LinearIndices(s)
    sents = similar(entrance(r))
    # update entrances and exits
    @inbounds for (i, v) in enumerate(entrance(r))
        sents[i] = slis[(cis[v] - unit_ci) * factor + unit_ci]
    end
    sexits = similar(exits(r))
    @inbounds for (i, v) in enumerate(exits(r))
        sexits[i] = slis[(cis[v] - unit_ci) * factor + unit_ci]
    end
    GridRoom(s, bounds(r) .* factor, sents, sexits, sg, sd)
end


const _char_map = Dict{Symbol, String}(
    :entrance => "◉",
    :exit => "◎",
    :path => "○"
)

print_row(i) = print("$(String(i))")

function Base.show(io::IO, m::MIME"text/plain", r::GridRoom)
    Base.show(io,m,(r, Int64[]))
end
function Base.show(io::IO, m::MIME"text/plain",
                   t::Tuple{GridRoom, Vector{Int64}})
    r, paths = t
    rd = repr.(r.data)
    rd[entrance(r)] .= _char_map[:entrance]
    rd[exits(r)] .= _char_map[:exit]
    rd[paths] .= _char_map[:path]
    rd[:, 1:(end-1)] .*= "\t"
    rd[:, end] .*= "\n"
    s::String = @>> rd permutedims join
    println(io,s)
end

JSON.lower(r::GridRoom) = Dict(
    steps  => steps(r),
    bounds => bounds(r),
    entrance => entrance(r),
    exits => exits(r),
    data   => convert.(Symbol, data(r))
)

# FIXME: several janky statements
function from_json(::Type{GridRoom}, jd::Dict)
    s = Tuple(collect(Int64, jd["steps"]))
    b = Tuple(collect(Float64, jd["bounds"]))
    g = PathGraph(grid(s))
    d = Matrix{Tile}(undef, s)
    for i = 1:s[1], j = 1:s[2]
        msg = Symbol(jd["data"][j][i])
        d[i, j] = convert(Tile, msg)
    end
    prune_edges!(g, d)
    en = collect(Int64, jd["entrance"])
    ex = collect(Int64, jd["exits"])
    GridRoom(s,b,en,ex, g, d)
end


# Debug visualization

function viz_room(r::GridRoom, ocg::Matrix{Float64})
    d = data(r)
    c = deepcopy(ocg)
    c[d .== obstacle_tile] .= -1.
        println(spy(c,
                    title = "Obstacles + path"))

end
function viz_room(r::GridRoom)
    d = data(r)
    c = occupancy_position(r)
    println(heatmap(c,
                    border = :none,
                    colorbar_border = :none,
                    colormap = :inferno
                    ))
end


function viz_ocg(ocg; title = "occupancy grid")
    # ocg = mean(ocg)
    println(heatmap(ocg,
                    title = title,
                    border = :none,
                    colorbar_border = :none,
                    # colormap = :inferno,
                    xfact = 2.0,
                    yfact = 2.0
                    ))
end
