# using Images: colorview, RGB
using Colors, Images
using ImageInTerminal

export viz_room

function draw_room(r::GridRoom, p::Array{T}) where {T<:Edge}
    draw_room(r, map(dst, p))
end

function draw_room(r::GridRoom, p::Array{Int64})
    d = data(r)
    m = fill(RGB{Float32}(0, 0, 0), steps(r))
    m[d .== obstacle_tile] .= RGB{Float32}(1, 0, 0)
    m[d .== wall_tile] .= RGB{Float32}(0, 0, 1)
    m[p] .= RGB{Float32}(0, 1, 0)
    rotr90(m, 3)
end

function draw_room(r::GridRoom, p::Matrix{Float64})
    d = data(r)
    reds = zeros(size(p))
    reds[d .== obstacle_tile] .= 1
    blues = zeros(size(p))
    blues[d .== wall_tile] .= 1
    m = colorview(RGB{Float64}, reds, p, blues)
    rotr90(m, 3)
end

function viz_room(r::GridRoom, p::Array)
    m = draw_room(r, p)
    display(m)
end

function viz_room(r::GridRoom)
    viz_room(r, Int64[])
end

function Base.display(r::GridRoom)
    viz_room(r)
end
