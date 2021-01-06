function swap_tiles!(g::PathGraph, p::Tuple{Tile, Tile})
    x,y = p
    a = get_prop(g, y, :type)
    b = get_prop(g, x, :type)
    set_prop!(g, x, :type, a)
    set_prop!(g, y, :type, b)
    return nothing
end

function connected(g::PathGraph, v::Tile)::Set{Tile}
    s = @>> v bfs_tree(g) edges collect induced_subgraph(g) last Set
    isempty(s) ? Set([v]) : s
end

# TODO: implement me!
function bitmap_render(r::Room)::Matrix{Bool}
    ratio = [33, 40] # rows by columns (y, x)
    bm = fill(false, ratio...)
    serialized = translate(r)
    camera = serialized[:camera]
    camera_pos = camera[:pos] # xyz of camera
    camera_rot = camera[:orientation] # xyz rot in radians
    objects = serialized[:objects] # both walls and furniture
    furniture = filter(x -> x[:appearance] == :blue, objects)
    for f in furniture
        pos = f[:position] # x,y,z
        dims = f[:dims]
        # take camera_pos, camera_rot, and pos to
        f_bm = nothing # implement me!
        bm = bm .| f_bm
    end
end
