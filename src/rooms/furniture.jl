export Furniture, furniture, add, remove, clear_room, valid_move, shift_furniture

#################################################################################
# Furniture
#################################################################################

# TODO: Why `Set`?
# Some set based comparisons later
const Furniture = Set{Int64}

function furniture(r::Room)::Vector{Furniture} end

# TODO: why use an `Array{Set}` rather than `Array`
function furniture(r::GridRoom)
    g = pathgraph(r)
    d = r.data
    vs = @>> g begin
        connected_components
        filter(c -> d[first(c)] == obstacle_tile)
        map(Furniture) # maybe try `collect(Furniture)` ?
    end
end


"""
Adds the furniture of `src` to `dest`
"""
#TODO: Try out "flattened" update scheme (avoid `fold`)
function add(src::GridRoom, dest::GridRoom)::GridRoom
    omap = src.data .== obstacle_tile
    d = deepcopy(dest.data)
    d[omap] .= obstacle_tile
    g = deepcopy(dest.graph)
    prune_edges!(g, d)
    GridRoom(dest.steps, dest.bounds, dest.entrance,
             dest.exits, g, d)
end

function add(r::GridRoom, f::Furniture)::GridRoom
    g = @> r steps grid PathGraph
    d = deepcopy(r.data)
    d[f] .= obstacle_tile
    prune_edges!(g, d)
    GridRoom(r.steps, r.bounds, r.entrance,
             r.exits, g, d)
end

function remove(r::GridRoom, f::Furniture)::GridRoom
    g = @> r steps grid PathGraph
    d = deepcopy(r.data)
    d[f] .= floor_tile
    prune_edges!(g, d)
    GridRoom(r.steps, r.bounds, r.entrance,
             r.exits, g, d)
end

function clear_room(r::Room)::Room
    g = @> r steps grid PathGraph
    d = deepcopy(r.data)
    d[d .== obstacle_tile] .= floor_tile
    prune_edges!(g, d)
    GridRoom(r.steps, r.bounds, r.entrance,
             r.exits, g, d)
end



function valid_move(::Move, ::Room, ::Furniture)::Bool end

valid_move(::Left, r::GridRoom, f::Furniture) = all(f .> first(steps(r)))
valid_move(::Right, r::GridRoom, f::Furniture) = all(f .<= (prod(steps) - first(steps(r))))
valid_move(::Down, r::GridRoom, f::Furniture) = all((f .% first(steps(r))) .> 0)
valid_move(::Up, r::GridRoom, f::Furniture) = all((f .% first(steps(r))) .!= 1)


function shift_furniture(r::Room, f::Furniture, move::Int64)
    shift_furniture(r, f, move_map[move])
end

function shift_furniture(r::GridRoom, f::Furniture, m::Move)
    # Check to see if move is valid
    @assert valid_move(m, r, f)
    @assert all(r.data[f] .== obstacle_tile)
    d = deepcopy(r.data)
    # apply move
    moved_f = move(r, f, m)
    # update data and graph
    to_clear = setdiff(f, moved_f)
    d[to_clear] .= floor_tile
    to_add = setdiff(moved_f, f)
    d[to_add] .= obstacle_tile

    g = @> r steps grid PathGraph
    prune_edges!(g, d)
    GridRoom(r.steps, r.bounds, r.entrance,
             r.exits, g, d)
end

function move(r::Room, f::Furniture, m::Move)::Furniture
    nf = collect(Int64, f)
    unsafe_move!(nf, m, r)
    Set{Int64}(nf)
end

# function unsafe_move!(::Furniture, ::Move, ::Room) end

unsafe_move!(f::Vector{Int64}, ::Up, r::GridRoom) = f .-= 1
unsafe_move!(f::Vector{Int64}, ::Down, r::GridRoom) = f .+= 1
unsafe_move!(f::Vector{Int64}, ::Left, r::GridRoom) = f .-= first(steps(r))
unsafe_move!(f::Vector{Int64}, ::Right, r::GridRoom) = f .+= first(steps(r))
