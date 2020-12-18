

function average_path_length(g, s, e; n = 5)
    s = yen_k_shortest_paths(g, s, e, weights(g), n)
    l = @>> s.paths map(length) mean
    isnan(l) ? Inf : l
end

function navigability(r::Room)::Vector{Float64}
    g = pathgraph(r)
    ents = entrance(r)
    to_ent = v -> @>> ents map(e -> average_path_length(g, v, e)) mean
    @>> exits(r) map(to_ent) vec
end


compare(a::Room, b::Room) = norm(navigability(a) .- navigability(b))

export navigability, compare
