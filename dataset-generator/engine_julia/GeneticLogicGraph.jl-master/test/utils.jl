using GeneticLogicGraph
using Test
using Graphs
using ModelingToolkit

const GLG = GeneticLogicGraph

@testset "Pruning adjacency matrices" begin
    expected = SimpleDiGraph(falses(3, 3))
    @test GLG.prune(expected, []) == expected
    @test GLG.prune(expected, [1]) == expected
    @test GLG.prune(expected, [1, 2]) == expected
    @test GLG.prune(expected, [1, 2, 3]) == expected

    graph = SimpleDiGraph(Edge.([(1, 2), (2, 3)]))
    @test GLG.prune(graph, []) == expected

    expected = SimpleDiGraph(3)
    add_edge!(expected, Edge(1, 2))
    add_edge!(expected, Edge(2, 3))
    
    @test GLG.prune(graph, [3]) == expected

    graph = SimpleDiGraph(Edge.([(1, 2), (2, 3), (2, 5)]))
    expected = SimpleDiGraph(5)
    add_edge!(expected, Edge(1, 2))
    add_edge!(expected, Edge(2, 5))

    @test GLG.prune(graph, [5]) == expected
    @test GLG.prune(graph, [3, 5]) == graph
    @test GLG.prune(graph, [5, 3]) == graph

    graph = SimpleDiGraph(Edge.([(1, 1), (2, 3), (2, 5)]))
    expected = SimpleDiGraph(5)
    @test GLG.prune(graph, [2]) == expected
    @test GLG.prune(graph, [4]) == expected

    expected = SimpleDiGraph(5)
    add_edge!(expected, Edge(1, 1))
    @test GLG.prune(graph, [1]) == expected
end

@testset "Initial conditions" begin
    I = InputSpecies(1; name=:I)

    @variables t
    v = @variables I₊monomer(t)
    @test any(isequal(v[1]), keys(randu0(I)))

    C = ConstantSpecies(1; name=:C)
    v = @variables C₊monomer(t)
    @test any(isequal(v[1]), keys(randu0(C)))
end
