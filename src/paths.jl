
function shortest_path_length(g, s, e)
    s = a_star(g, s, e)
    length(s)
end

function average_path_length(g, s, e; n = 5)
    s = yen_k_shortest_paths(g, s, e, weights(g), n)
    l = @>> s.paths map(length) mean
    isnan(l) ? Inf : l
end

function navigability(r::Room)
    g = pathgraph(r)
    ent = first(entrance(r))
    # to_ent = v -> a_star(g, ent, v)
    # @>> exits(r) map(to_ent)
    ds = desopo_pape_shortest_paths(g, first(entrance(r)))
    @>> r exits enumerate_paths(ds)
end

# the edit distance in paths
compare(a::Room, b::Room) = @>> map(symdiff, navigability(a), navigability(b)) map(length) sum
# function compare(a::Room, b::Room)
#     na = @>> a navigability map(x -> map(src, x)) flatten
#     nb = @>> b navigability map(x -> map(src, x)) flatten
#     length(setdiff(na,nb))
# end

export navigability, compare
