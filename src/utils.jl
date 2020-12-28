Tile = Int64

function connected(g::MetaGraphs.MetaGraph{Int64,Float64},
                   v::Tile)::Set{Tile}
    s = @>> v bfs_tree(g) edges collect induced_subgraph(g) last Set
    isempty(s) ? Set([v]) : s
end
