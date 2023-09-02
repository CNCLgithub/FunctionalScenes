using CSV
using Lazy
using JSON
using DataFrames
using Statistics
using FunctionalScenes
using LinearAlgebra: norm
using FunctionalCollections

# using Profile
# using StatProfilerHTML

function move_change(r::GridRoom,
                     f::Furniture,
                     m::Move,
                     base_og)
    shifted = shift_furniture(r,f,m)
    shifted_og = occupancy_grid(shifted; sigma = 0., decay = 0.)
    # compute impact of shift
    d = norm(base_og - shifted_og)
end

function paired_change_in_path(x)
    # door 1 has no change in path
    min_i = argmin(x[:, :d])
    c1 = x[min_i, :door] == 1 && x[min_i, :d] == 0.0
    # door 2 changes
    max_i = argmax(x[:, :d])
    c2 = x[max_i, :door] == 2 && x[max_i, :d] < 17.0 && x[max_i, :d] > 13.0
    c1 && c2
end

function examine_furniture(df::DataFrame)::DataFrame
    gdf = @>> DataFrames.groupby(df, :furniture) begin
        # look find change for one but not another door
        filter(paired_change_in_path)
    end
    isempty(gdf) && return DataFrame()
    # pick furniture, move pair with biggest difference
    selected_i = @> gdf begin
        combine(:d => maximum)
        sortperm(:d_maximum)
        last
    end
    gdf[selected_i]
end

function build(door_conditions::Vector{GridRoom},
               move::Move;
               max_f::Int64 = 9,
               max_size::Int64 = 5,
               factor::Int64 = 1,
               pct_open::Float64 = 0.3,
               side_buffer::Int64 = 1)

    # assuming all rooms have the same entrance and dimensions
    rex = first(door_conditions)
    # rex = expand(first(door_conditions), factor)
    dims = steps(rex)

    # prevent furniture generated in either:
    # -1 out of sight
    # -2 blocking entrance exit
    # -3 hard to detect spaces next to walls
    weights = Matrix{Bool}(zeros(dims))
    # ensures that there is no furniture near the observer
    start_x = Int64(ceil(last(dims) * pct_open))
    stop_x = last(dims) - 2 # nor blocking the exit
    # buffer along sides
    start_y = side_buffer + 1
    stop_y = first(dims) - side_buffer
    weights[start_y:stop_y, start_x:stop_x] .= 1.0
    vmap = PersistentVector(vec(weights))

    # generate furniture once and then apply to
    # each door condition
    with_furn = furniture_gm(rex, vmap, max_f, max_size)
    n_v = prod(steps(with_furn))
    exp_bases = map(r -> add(with_furn, r),
                    door_conditions)
    fs = furniture(with_furn)
    results = DataFrame(door = Int64[],
                        furniture = Int64[],
                        move = Symbol[],
                        d = Float64[])
    for (i, ri) in enumerate(exp_bases)
        base_og = occupancy_grid(ri; sigma = 0., decay = 0.)
        for (j, fj) in enumerate(fs)
            # REVIEW: see if only changing rear obstacles makes a difference
            # ignore obstacles in the front
            (minimum(fj) < (0.3 * n_v)) && continue
            valid_move(ri, fj, move) || continue
            d = move_change(ri, fj, move, base_og)
            # door, furn, distance
            push!(results, [i, j, move, d])
        end
    end
    # see if any furniture pieces fit the criterion
    results = examine_furniture(results)
    (exp_bases, results)
end



function main()
    name = "09_02_2023"
    dataset_out = "/spaths/datasets/$(name)"
    isdir(dataset_out) || mkdir(dataset_out)

    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)


    # Parameters
    room_dims = (16, 16)
    entrance = [8, 9]
    door_rows = [5, 12]
    inds = LinearIndices(room_dims)
    doors = inds[door_rows, room_dims[2]]

    # number of trials
    n = 6

    # will only consider these moves
    moves = [down_move, up_move]
    n_moves = length(moves)

    # number of trials per move condition
    max_count = Int64(ceil(n / n_moves))

    # empty rooms with doors
    templates = @>> doors begin
        map(d -> GridRoom(room_dims, room_dims, entrance, [d]))
        collect(GridRoom)
    end

    # will store summary of generated rooms here
    df = DataFrame(id = Int64[],
                   flip = Bool[],
                   door = Int64[],
                   furniture = Int64[],
                   move = Symbol[],
                   d = Float64[])
    for (idx, move) in enumerate(moves)
        i = 1
        while i <= max_count
            # generate a room pair
            (pair, _df) = build(templates, move, factor = 2)
            # no valid pair generated, try again or finish
            isempty(_df) && continue

            # valid pair found, organize and store
            id = (idx - 1) * max_count + i
            _df[!, :id] .= id
            _df[!, :flip] .= (i-1) % 2
            append!(df, _df)

            # save scenes as json
            open("$(scenes_out)/$(id)_1.json", "w") do f
                _d = pair[1] |> json
                write(f, _d)
            end
            open("$(scenes_out)/$(id)_2.json", "w") do f
                _d = pair[2] |> json
                write(f, _d)
            end

            print("move $(idx): $(i)/$(max_count)\r")
            i += 1
        end
    end
    @show df
    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)

    # saving rooms

    return nothing
end

main();
