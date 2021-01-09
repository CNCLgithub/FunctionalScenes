using FunctionalScenes
using LightGraphs, MetaGraphs
import FunctionalScenes: expand, furniture, valid_moves, shift_furniture, move_map, valid_spaces
using Test

r = Room((4,10), (4, 10), [2], [38]);
r = add(r, Set([18]));
f = first(furniture(r));
r2 = shift_furniture(r, f, :down);
rg = pathgraph(r);
f2 = first(furniture(r2));
r2g = pathgraph(r2);


@testset "Room" begin
    r = Room((5,3), (5,3), [2], [12]);
    # ■■■
    # ◉□◎
    # ■□■
    # ■□■
    # ■■■
    g = pathgraph(r)
    @test Set(neighbors(g, 7)) == Set([2, 8, 12])
    @test Set(neighbors(g, 8)) == Set([7,9])
    @test Set(neighbors(g, 9)) == Set([8])
end;

@testset "Adding" begin
    r = Room((5,3), (5,3), [2], [12]);
    g = pathgraph(r)
    @test Set(neighbors(g, 8)) == Set([7, 9])
    r = add(r, Set([7]))
    g = pathgraph(r)
    @test Set(neighbors(g, 8)) == Set([9])
    r = add(r, Set([8]))
    g = pathgraph(r)
    @test Set(neighbors(g, 9)) == Set([])
end;


@testset "Removing" begin
    x = Room((4,10), (4, 10), [2], [38]);
    y = add(x, Set([14,15]));
    z = remove(y, Set([14,15]));
    @test navigability(x) != navigability(y);
    @test navigability(y) != navigability(z);
    @test navigability(x) == navigability(z);
end;

@testset "Furnishing" begin
    r = Room((5,3), (5,3), [2], [12]);
    r = add(r, Set([7]))
    g = pathgraph(r)
    @test valid_spaces(r) ==  [8, 9];
    ws = ones(steps(r));
    @test furnish(r, ws) == Set([8, 9]);
end;

@testset "Reorganizing" begin
    @test first(furniture(r2)) == Set([19])
    @test Set(neighbors(r2g, 18)) == Set([14, 22])
    @test valid_moves(r, f) == [0.0, 1.0, 1.0, 1.0]
    @test valid_moves(r2, f2) == [1.0, 0.0, 1.0, 1.0]
end;

@testset "Navigation" begin
    @test compare(r, r2) > 0
end;
