export Tile, Floor, Wall, Obstacle

abstract type Tile end

"""
    navigable(::Tile)

Whether a tile is navigable
"""
function navigable(::Tile)::Bool
    # ... [implementation sold separately] ...
end



struct Floor <: Tile end
const floor_tile = Floor()
navigable(::Floor) = true
Base.show(io::IO, m::MIME"text/plain", ::Floor) = Base.show(io, m, '□')

struct Wall <: Tile end
const wall_tile = Wall()
navigable(::Wall) = false
Base.show(io::IO, m::MIME"text/plain", ::Wall) = Base.show(io, m, '■')

struct Obstacle <: Object end
const obstacle_tile = Obstacle()
navigable(::Obstacle) = false
Base.show(io::IO, m::MIME"text/plain", ::Obstacle) = Base.show(io, m, '◆')
