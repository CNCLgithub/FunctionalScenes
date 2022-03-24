export Room

abstract type Room end

# TODO: Documentation
function pathgraph(::Room) end
function entrance(::Room) end
function exits(::Room) end

include("tiles.jl")
include("room.jl")
include("moves.jl")
include("furniture.jl")
include("paths.jl")
include("multires_path.jl")
