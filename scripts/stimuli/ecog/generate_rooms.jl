using CSV
using Lazy
using JSON
using DataFrames
using Statistics
using FunctionalScenes
using LinearAlgebra: norm
using FunctionalCollections

using Random
Random.seed!(1235)
# using Profile
# using StatProfilerHTML

# function move_change(r::GridRoom,
#                      f::Furniture,
#                      m::Move,
#                      base_og)
#     shifted = shift_furniture(r,f,m)
#     shifted_og = occupancy_grid(shifted; sigma = 0., decay = 0.)
#     # compute impact of shift
#     d = norm(base_og - shifted_og)
# end

# function paired_change_in_path(x)
#     # door 1 has no change in path
#     min_i = argmin(x[:, :d])
#     c1 = x[min_i, :door] == 1 && x[min_i, :d] == 0.0
#     # door 2 changes
#     max_i = argmax(x[:, :d])
#     c2 = x[max_i, :door] == 2 && x[max_i, :d] < 17.0 && x[max_i, :d] > 13.0
#     c1 && c2
# end

# function examine_furniture(r::GridRoom)::DataFrame


#     gdf = @>> DataFrames.groupby(df, :furniture) begin
#         # look find change for one but not another door
#         filter(paired_change_in_path)
#     end
#     isempty(gdf) && return DataFrame()
#     # pick furniture, move pair with biggest difference
#     selected_i = @> gdf begin
#         combine(:d => maximum)
#         sortperm(:d_maximum)
#         last
#     end
#     gdf[selected_i]
# end

function build(template::GridRoom;
               temp::Float64 = 1,
               max_f::Int64 = 5,
               max_size::Int64 = 5,
               factor::Int64 = 2,
               pct_open::Float64 = 0.15,
               side_buffer::Int64 = 2)

    p = recurse_path_gm(template, temp)
    template = fix_shortest_path(template, p)
    dims = steps(template)

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
    weights[start_y:stop_y, start_x:stop_x] .= true

    # prevent obstacles on path
    weights[p] .= false

    vmap = PersistentVector(vec(weights))


    # generate furniture once and then apply to
    # each door condition
    with_furn = furniture_gm(template, vmap, max_f, max_size)
    with_furn = expand(with_furn, factor)
    # perform path analysis
    # results = examine_furniture(with_furn)
    (with_furn, nothing)
end



function main()
    name = "ecog_pilot"
    dataset_out = "/spaths/datasets/$(name)"
    isdir(dataset_out) || mkdir(dataset_out)

    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)


    # Parameters
    room_dims = (16, 16)
    entrance = [8, 9]
    door_rows = [5]
    inds = LinearIndices(room_dims)
    doors = inds[door_rows, room_dims[2]]
    t_factor = 1.0

    # number of trials
    n = 30
    template = GridRoom(room_dims, room_dims, entrance, doors)

    # will store summary of generated rooms here
    df = DataFrame(id = Int64[],
                   flip = Bool[],
                   temp = Float64[])
    # generate rooms
    i = 1
    while i <= n
        # generate a room pair
        temp = t_factor * i / n
        (r, _) = build(template; temp = temp)
        _df = DataFrame(id = i,
                        flip = (i - 1) % 2,
                        temp = temp)
        # invalid room generated, try again or finish
        # isempty(_df) && continue

        # valid pair found, organize and store
        append!(df, _df)

        # save scenes as json
        open("$(scenes_out)/$(i).json", "w") do f
            _d = r |> json
            write(f, _d)
        end

        print("step: $(i)/$(n)\r")
        i += 1
    end
    @show df
    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)
    return nothing
end

main();
