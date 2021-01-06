using CSV
using Lazy
using Images, FileIO
using DataFrames
using LinearAlgebra:norm

import FunctionalScenes: Room, furniture, shift_furniture, navigability

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

function compare_rooms(base_p::String, fid, move)
    base = load(base_p)["r"]
    f = furniture(base)[fid]
    room = shift_furniture(base, f, Symbol(move))
    paths_a = @>> navigability(base) map(p -> relative_path(p))
    paths_b = @>> navigability(room) map(p -> relative_path(p))
    min_a_i = @>> paths_a map(x -> isempty(x) ? Inf : length(x)) argmin
    # max_a_i = @>> paths_a map(x -> isempty(x) ? Inf : length(x)) argmax
    # lv_distance(paths_a[max_a_i], paths_b[max_a_i])
    lv_distance(paths_a[min_a_i], paths_b[min_a_i])
    sum(map(lv_distance, paths_a, paths_b))
end

function main()
    df = DataFrame(CSV.File("/scenes/1exit.csv"))
    new_df = DataFrame(id = Int64[], furniture = Int64[],
                       move = String[], pixeld = Float64[],
                       lvd = Float64[])

    for r in eachrow(df)
        base = "/renders/1exit/$(r.id).png"
        img = "/renders/1exit/$(r.id)_$(r.furniture)_$(r.move).png"
        pixeld = compare_pixels(base, img)

        base = "/scenes/1exit/$(r.id).jld2"
        lvd = compare_rooms(base, r.furniture, r.move)

        push!(new_df, (r.id, r.furniture, r.move, pixeld, lvd))
    end
    isdir("/experiments/1exit") || mkdir("/experiments/1exit")
    CSV.write("/experiments/1exit/covariates.csv", new_df)
end


main();
