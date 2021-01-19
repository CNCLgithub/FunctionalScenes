using FunctionalScenes
using LightGraphs, MetaGraphs
import FunctionalScenes: expand, furniture, valid_moves, shift_furniture, move_map, valid_spaces, furniture_prior,bitmap_render,coordinates_to_camera,coordinates_to_pixels
using Luxor
using Test
using Images

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
    p = furniture_prior(r,5.0)
    @test p[1,1] > -4
    q = coordinates_to_camera([20,20,20],[0,0,30],[pi/4,0,0])
    @test q[1] == 20.0
    j = bitmap_render(r)
    imgg = Gray.(j)
    #save("test1.png",colorview(Gray,j))
    
end;

@testset "Adding" begin
    r = Room((5,3), (5,3), [2], [12]);
    g = pathgraph(r)
    @test Set(neighbors(g, 8)) == Set([7, 9])
    r = add(r, Set([7]))
    g = pathgraph(r)
    j = bitmap_render(r)
    save("test1.png",colorview(Gray,j))
    @test Set(neighbors(g, 8)) == Set([9])
    r = add(r, Set([8]))
    g = pathgraph(r)
    j = bitmap_render(r)
    #println(sum(j))
    save("test2.png",colorview(Gray,j))
    r = add(r,Set([1]))
    j = bitmap_render(r)
    save("test3.png",colorview(Gray,j))
    @test Set(neighbors(g, 9)) == Set([])
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
