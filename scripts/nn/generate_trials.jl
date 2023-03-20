using JSON
using Images
using Lazy: @>>
using FunctionalScenes
using FunctionalCollections: PersistentVector

IMG_RES = (120, 180)
SPP = 16

function occupancy_position(r::GridRoom)::Matrix{Float64}
    grid = zeros(steps(r))
    grid[FunctionalScenes.data(r) .== obstacle_tile] .= 1.0
    grid
end

function build(r::GridRoom;
               max_f::Int64 = 11,
               max_size::Int64 = 5,
               factor::Int64 = 1,
               pct_open::Float64 = 0.3,
               side_buffer::Int64 = 1)

    dims = steps(r)
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
    with_furn = furniture_gm(r, vmap, max_f, max_size)
    with_furn = expand(with_furn, factor)
end

function render(r::GridRoom)
    @time img = render_mitsuba(r, IMG_RES, SPP)
end

function save_trial(dpath::String, i::Int64, r::GridRoom,
                    img, og)
    out = "$(dpath)/$(i)"
    isdir(out) || mkdir(out)

    open("$(out)/room.json", "w") do f
        rj = r |> json
        write(f, rj)
    end
    open("$(out)/scene.json", "w") do f
        r2 = translate(r, Int64[]; cubes = false)
        r2j = r2 |> json
        write(f, r2j)
    end
    img = map(clamp01nan, img)
    save_img_array(img, "$(out)/render.png")
    # occupancy grid saved as grayscale image
    save("$(out)/og.png", og)
    return nothing
end

function main()
    # Parameters
    name = "ccn_2023_ddp_train_11f_32x32"
    n = 1000
    # name = "ccn_2023_ddp_test_11f_32x32"
    # n = 25
    room_dims = (16, 16)
    entrance = [8, 9]
    door_rows = [5, 12]
    inds = LinearIndices(room_dims)
    doors = inds[door_rows, room_dims[2]]

    # empty rooms with doors
    templates = @>> doors begin
        map(d -> GridRoom(room_dims, room_dims, entrance, [d]))
        collect(GridRoom)
    end

    # will store summary of generated rooms here
    m = Dict(
        :n => n,
        :templates => templates,
        :og_shape => (32, 32),
    )
    out = "/spaths/datasets/$(name)"
    isdir(out) || mkdir(out)

    for i = 1:n
        t = templates[i % 2 + 1]
        r = build(t, factor = 2)
        r_img = render(r)
        r_og = occupancy_position(r)
        save_trial(out, i, r, r_img, r_og)
    end
    
    m[:img_mu] = zeros(3)
    m[:img_sd] = ones(3)

    open("$(out)_manifest.json", "w") do f
        write(f, m |> json)
    end
    return nothing
end

main();
