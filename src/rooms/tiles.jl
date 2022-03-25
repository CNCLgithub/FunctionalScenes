export Tile, Floor, Wall, Obstacle, floor_tile, wall_tile, obstacle_tile

abstract type Tile end

"""
    navigable(::Tile)

Whether a tile is navigable
"""
function navigable(::Tile)::Bool
    # ... [implementation sold separately] ...
end

#TODO: understand why this is needed
Base.length(::Tile)  = 1
Base.iterate(t::Tile)  = (t, nothing)
Base.iterate(t::Tile, ::Nothing)  = nothing


struct Floor <: Tile end
const floor_tile = Floor()
navigable(::Floor) = true
Base.show(io::IO, ::Floor) = Base.print(io, '□')
# Base.show(io::IO, m::MIME"text/plain", ::Floor) = Base.show(io, m, '□')

struct Wall <: Tile end
const wall_tile = Wall()
navigable(::Wall) = false
Base.show(io::IO, ::Wall) = Base.print(io, '■')
# Base.show(io::IO, m::MIME"text/plain", ::Wall) = Base.show(io, m, '■')

struct Obstacle <: Tile end
const obstacle_tile = Obstacle()
navigable(::Obstacle) = false
Base.show(io::IO, ::Obstacle) = Base.print(io, '◆')
# Base.show(io::IO, m::MIME"text/plain", ::Obstacle) = Base.show(io, m, '◆')

Base.convert(::Type{Symbol}, ::Floor) = :floor
Base.convert(::Type{Symbol}, ::Wall) = :wall
Base.convert(::Type{Symbol}, ::Obstacle) = :obstacle

const tile_d = Dict{Symbol, Tile}(
    :wall => wall_tile,
    :floor => floor_tile,
    :obstacle => obstacle_tile,
)
Base.convert(::Type{Tile}, s::Symbol) = tile_d[s]
Base.convert(::Type{Tile}, s::String) = tile_d[Symbol(s)]
