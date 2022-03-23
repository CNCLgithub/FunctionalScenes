export Move, Left, Right, Down, Up, left_move, right_move, down_move, up_move

#################################################################################
# Movement
#################################################################################

abstract type Move end
struct Left <: Move end
struct Right <: Move end
struct Down <: Move end
struct Up <: Move end

const left_move = Left()
const right_move = Right()
const down_move = Down()
const up_move = Up()
