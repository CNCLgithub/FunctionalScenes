using FunctionalScenes
using Gen
using Graphs
using FunctionalCollections
using Test

# r = Room((4,10), (4, 10), [2], [38]);
# r = add(r, Set([18]));
# f = first(furniture(r));
# r2 = shift_furniture(r, f, :down);
# rg = pathgraph(r);
# f2 = first(furniture(r2));
# r2g = pathgraph(r2);


@testset "GridRoom" begin
    r = GridRoom((5,3), (5,3), [2], [12]);
    # ■       ■       ■
    # ◉       □       ◎
    # ■       □       ■
    # ■       □       ■
    # ■       ■       ■
    g = pathgraph(r)
    @test Set(neighbors(g, 7)) == Set([2, 8, 12])
    @test Set(neighbors(g, 8)) == Set([7,9])
    @test Set(neighbors(g, 9)) == Set([8])

end;

@testset "Adding" begin
    r = GridRoom((5,3), (5,3), [2], [12]);
    g = pathgraph(r)
    @test Set(neighbors(g, 8)) == Set([7, 9])
    r = add(r, Set([7]))
    g = pathgraph(r)
    @test Set(neighbors(g, 8)) == Set([9])
    r = add(r, Set([8]))
    g = pathgraph(r)
    r = add(r,Set([1]))
    @test Set(neighbors(g, 9)) == Set([])
end;


@testset "Removing" begin
    x = GridRoom((4,10), (4, 10), [2], [38]);
    y = add(x, Set([14,15]));
    @test navigability(x) != navigability(y);
    @test Set(neighbors(pathgraph(x), 14)) !=
        Set(neighbors(pathgraph(y), 14))
end;

@load_generated_functions

@testset "Furnishing" begin
    r = GridRoom((5,3), (5,3), [2], [12]);
    r = add(r, Set([7]))
    g = pathgraph(r)
    vs = valid_spaces(r)
    @test findall(vs) ==  [8, 9];

    # gs = GrowState(8, vs, g)
    # result = FunctionalScenes.fixed_depth_grow(gs, 1)
    # display(result)
    tr, _ = Gen.generate(furnish, (r, vs, 2))
    display(get_choices(tr))
    display(get_retval(tr))
    # @test furnish(r, vmap, 1) == [8, 9];
end;

# @testset "Reorganizing" begin
#     @test first(furniture(r2)) == Set([19])
#     @test Set(neighbors(r2g, 18)) == Set([14, 22])
#     @test valid_moves(r, f) == [0.0, 1.0, 1.0, 1.0]
#     @test valid_moves(r2, f2) == [1.0, 0.0, 1.0, 1.0]
# end;

# @testset "Navigation" begin
#     @test compare(r, r2) > 0
# end;


# @testset "Showing" begin
#     r = Room((10,10), (10,10), [5], [22]);
#     p = k_shortest_paths(r, 5, 1, 1)
#     @show (r, first(p))
# end;

# @testset "gdistances" begin
#     r = Room((10,10), (10,10), [5], [95]);
#     paths = FunctionalScenes.all_shortest_paths(r)
#     @show paths
# end;
