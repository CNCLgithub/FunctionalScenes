using Images: colorview, RGB
using ImageInTerminal

export viz_room

function Base.display(r::GridRoom)
    viz_room(r)
end

function viz_room(r::GridRoom, p::Array{Int64})
    d = data(r)
    m = fill(RGB{Float32}(0, 0, 0), steps(r))
    m[d .== obstacle_tile] .= RGB{Float32}(1, 0, 0)
    m[d .== wall_tile] .= RGB{Float32}(0, 0, 1)
    m[p] .= RGB{Float32}(0, 1, 0)
    display(rotr90(m, 3))
end

function viz_room(r::GridRoom)
    viz_room(r, Int64[])
end
