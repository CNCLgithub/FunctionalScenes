export Move, Left, Right, Down, Up, left_move, right_move, down_move, up_move,
    move_map, move_d

#################################################################################
# Movement
#################################################################################

abstract type Move end
struct Left <: Move end
struct Right <: Move end
struct Down <: Move end
struct Up <: Move end

Base.convert(::Type{Symbol}, ::Left) = :left
Base.convert(::Type{Symbol}, ::Right) = :right
Base.convert(::Type{Symbol}, ::Down) = :down
Base.convert(::Type{Symbol}, ::Up) = :up

const left_move = Left()
const right_move = Right()
const down_move = Down()
const up_move = Up()

const move_map = [up_move, down_move, left_move, right_move]

const move_d = Dict(
    :up => up_move,
    :down => down_move,
    :left => left_move,
    :right => right_move,
)
