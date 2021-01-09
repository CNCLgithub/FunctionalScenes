using CSV
using Lazy
using Images, FileIO
using DataFrames
using LinearAlgebra:norm

import FunctionalScenes: Room, furniture, shift_furniture, navigability,
    occupancy_grid, diffuse_og

function compare_pixels(a::String, b::String)
    x = load(a)
    y = load(b)
    norm(x .- y)
end

function move_type(d::Real)
    if d == -1
        :up
    elseif d == 1
        :down
    elseif d < -1
        :left
    elseif d > 1
        :right
    else
        error("Not a valid move")
    end
end

function relative_path(path)
    lp = length(path)
    p = Vector{Symbol}(undef, max(0, lp - 1))
    for i = 2:lp
        d = path[i] - path[i-1]
        p[i-1] = move_type(d)
    end
    return p
end


function lv_distance(a, b)
    size_x = length(a) + 1
    size_y = length(b) + 1
    (isempty(a) || isempty(b)) && (return max(size_x, size_y) - 1)
    matrix = zeros((size_x, size_y))
    matrix[:, 1] = 0:(size_x-1)
    matrix[1, :] = 0:(size_y-1)
    for x = 2:size_x, y = 2:size_y
        if a[x-1] == b[y-1]
            matrix[x,y] = min(
                matrix[x-1, y] + 1,
                matrix[x-1, y-1],
                matrix[x, y-1] + 1
            )
        else
            matrix[x,y] = min(
                matrix[x-1,y] + 1,
                matrix[x-1,y-1] + 1,
                matrix[x,y-1] + 1
            )
        end
    end
    return last(matrix)
end

function compare_og(a,b)
    paths_a = navigability(a) # paths to each exit
    paths_b = navigability(b)
    d = 0
    for i = 1:length(paths_a) # for each exit
        d += norm(occupancy_grid(a, paths_a[i], decay = -1.0) .-
                  occupancy_grid(a, paths_b[i], decay = -1.0))
        d += norm(occupancy_grid(b, paths_b[i], decay = -1.0) .-
                  occupancy_grid(b, paths_a[i], decay = -1.0))
        d += norm(occupancy_grid(a, reverse(paths_a[i]), decay = -1.0) .-
                  occupancy_grid(a, reverse(paths_b[i]), decay = -1.0))
        d += norm(occupancy_grid(b, reverse(paths_b[i]), decay = -1.0) .-
                  occupancy_grid(b, reverse(paths_a[i]), decay = -1.0))
    end
    0.5 * d
end

function compare_rooms(base_p::String, fid, move)
    base = load(base_p)["r"]
    f = furniture(base)[fid]
    room = shift_furniture(base, f, Symbol(move))
    paths_a = @>> navigability(base) map(p -> relative_path(p))
    paths_b = @>> navigability(room) map(p -> relative_path(p))
    lvd = sum(map(lv_distance, paths_a, paths_b))
    ogd = norm(diffuse_og(base) .- diffuse_og(room))
    # ogd = compare_og(base, room)
    (lvd, ogd)
end

function main(exp::String)
    df = DataFrame(CSV.File("/scenes/$(exp).csv"))
    new_df = DataFrame(id = Int64[], furniture = Int64[],
                       move = String[], pixeld = Float64[],
                       lvd = Float64[], ogd = Float64[])

    for r in eachrow(df)
        base = "/renders/$(exp)/$(r.id).png"
        img = "/renders/$(exp)/$(r.id)_$(r.furniture)_$(r.move).png"
        pixeld = compare_pixels(base, img)

        base = "/scenes/$(exp)/$(r.id).jld2"
        lvd, ovd = compare_rooms(base, r.furniture, r.move)

        push!(new_df, (r.id, r.furniture, r.move, pixeld, lvd, ovd))
    end
    isdir("/experiments/$(exp)") || mkdir("/experiments/1exit")
    CSV.write("/experiments/$(exp)/covariates.csv", new_df)
end


# main("pilot");
main("1exit");
