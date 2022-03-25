using FunctionalScenes
using Gen
using Graphs
using Random
using FunctionalCollections
using Test
using Profile
using StatProfilerHTML

# Profile.init(delay = 0.0001,
#              n = 10^5)
# Random.seed!(0)



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
    r = GridRoom((10,10), (10,10), [5], []);
    r = add(r, Set([13]))
    g = pathgraph(r)
    vs = valid_spaces(r)
    # @test findall(vs) ==  [8, 9];

    # gs = GrowState(8, vs, g)
    # result = FunctionalScenes.fixed_depth_grow(gs, 1)
    # display(result)
    # @time tr, _ = Gen.generate(furnish, (r, vs, 1))

    Random.seed!(123)
    # @time tr, _ = Gen.generate(furnish, (r, vs, 1))
    # display(get_choices(tr))
    # display(get_retval(tr))

    tr, _ = Gen.generate(furniture_gm, (r, vs, 10, 5))
    # @time tr, _ = Gen.generate(furniture_gm, (r, vs, 10, 5))
    # display(get_choices(tr))
    # display(get_retval(tr))
    # @test furnish(r, vs, 1) == Set([8, 9]);
    @test true
end;

@testset "Reorganizing" begin
    r = GridRoom((4,10), (4, 10), [2], [38]);
    r = add(r, Set([18]));
    f = first(furniture(r));
    r2 = shift_furniture(r, f, down_move);
    rg = pathgraph(r);
    f2 = first(furniture(r2));
    r2g = pathgraph(r2);
    # make sure obstacle moved  18 ->  19
    @test first(furniture(r2)) == Set([19])
    # make sure free tile at 18 is now connected to neighbors
    @test Set(neighbors(r2g, 18)) == Set([14, 22])
    @test valid_moves(r, f) == [0, 1, 1, 1]
    @test valid_moves(r2, f2) == [1, 0, 1, 1]
end;

@testset "Navigation" begin
    r = GridRoom((4,10), (4, 10), [2], [38]);
    r = add(r, Set([18]));
    paths = safe_shortest_paths(r)
    # obstacle shouldn't show up in paths
    @test !in(18, paths)


    r = GridRoom((4,10), (4, 10), [2], [38]);
    r = add(r, Set([18, 19]));
    paths = safe_shortest_paths(r)
    # path should stop at blockage
    @test Set(paths) == Set([2, 6, 10, 14])

    r = GridRoom((4,10), (4, 10), [2], [38]);
    r = add(r, Set([18]));
    paths = safe_shortest_paths(r)
    og = occupancy_grid(r, decay = 0., sigma = 0.)

    # REVIEW is this a valid test?
    @test sum(og) == length(paths)

end;


# @testset "Showing" begin
#     r = Room((10,10), (10,10), [5], [22]);
#     p = k_shortest_paths(r, 5, 1, 1)
#     @show (r, first(p))
# end;

