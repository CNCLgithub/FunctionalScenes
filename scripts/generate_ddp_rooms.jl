using CSV
using JLD2
using Lazy: @>, @>>
using FunctionalScenes
import FunctionalScenes: expand, furniture, labelled_categorical, entrance, exits


function build(templates::Vector{Room};
               k::Int64 = 16, factor::Int64 = 1,
               pct_open::Float64 = 0.5)
    println("generating room")
    # randomly sample a door
    # assuming all rooms have the same entrance and dimensions
    r = labelled_categorical(templates)
    r = expand(r, factor)
    weights = ones(steps(r))
    # populate with furniture
    last(furniture_chain(k, r, weights))
end


function create(room_dims::Tuple{Int64, Int64},
                entrance::Int64,
                doors::Vector{Int64};
                n::Int64 = 8)

    r = Room(room_dims, room_dims, [entrance], Int64[])

    templates = @>> doors begin
        map(d -> Room(room_dims, room_dims, [entrance], [d]))
        collect(Room)
    end
    # generate n rooms randomly from templates
    @>> collect(1:n) begin
        map(i -> build(templates, factor = 2))
        # make sure the result is Vector{Room}
        collect(Room)
    end
end

function saver(out::String, rs::Vector{Room})
    @save "$(out)/rooms.jld2" rs
end


function main()
    name = "train_ddp_1_exit_22x40_doors"
    room_dims = (11,20)
    entrance = 6
    inds = LinearIndices(room_dims)
    doors = [3, 9]
    doors = @>> doors map(d -> inds[d, room_dims[2]]) collect(Int64)
    n = 1000
    rooms = create(room_dims, entrance, doors; n = n)
    out = "/scenes/$(name)"
    isdir(out) || mkdir(out)
    saver(out, rooms)
    return nothing
end

main();
