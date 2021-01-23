using CSV
using Lazy
using Images, FileIO
using DataFrames
using LinearAlgebra:norm
using OptimalTransport

import FunctionalScenes: Room, furniture, shift_furniture, navigability,
    occupancy_grid, diffuse_og, safe_shortest_path

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
    es = exits(a)
    paths_a = @>> es map(e -> safe_shortest_path(a,e))
    paths_b = @>> es map(e -> safe_shortest_path(b,e))
    # perhaps look at rooms wholistically
    # decay 0 seemed to help
    # f = (x,y) -> norm(g(a, x) .- g(b,y))
    g = (r,p) -> occupancy_grid(r,p, decay = 0.0, sigma = 1.0)
    f = (x,y) -> wsd(g(a, x), g(b,y))
    map(f, paths_a, paths_b) |> sum
end

function _wsd(a,b)
    v = rand(size(a, 2))
    v = v ./ norm(v)
    OptimalTransport.pot.wasserstein_1d(a * v, b * v)
end
function wsd(a::Matrix{Float64}, b::Matrix{Float64}; n::Int64 = 100)::Float64
    d = 0.
    for _ in 1:n
        d += _wsd(a,b)
    end
    d / n
end

function compare_rooms(base_p::String, fid, move)
    base = load(base_p)["r"]
    f = furniture(base)[fid]
    room = shift_furniture(base, f, Symbol(move))
    es = exits(base)
    paths_a = @>> es map(e -> safe_shortest_path(base,e)) map(relative_path)
    paths_b = @>> es map(e -> safe_shortest_path(room,e)) map(relative_path)
    # paths_a = @>> navigability(base) map(p -> relative_path(p))
    # paths_b = @>> navigability(room) map(p -> relative_path(p))
    lvd = sum(map(lv_distance, paths_a, paths_b))
    # ogd = norm(diffuse_og(base) .- diffuse_og(room))
    ogd = compare_og(base, room)
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
        # pixeld = 0

        base = "/scenes/$(exp)/$(r.id).jld2"
        lvd, ovd = compare_rooms(base, r.furniture, r.move)

        push!(new_df, (r.id, r.furniture, r.move, pixeld, lvd, ovd))
    end
    isdir("/experiments/$(exp)") || mkdir("/experiments/$(exp)")
    CSV.write("/experiments/$(exp)/covariates.csv", new_df)
end


# main("pilot");
# main("1exit");
main("2e_1p_30s");
