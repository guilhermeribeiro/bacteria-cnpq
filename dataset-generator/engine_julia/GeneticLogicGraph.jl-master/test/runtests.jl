using GeneticLogicGraph
using Test
using Random

Random.seed!(17)

@testset verbose=true "GeneticLogicGraph.jl" begin
    @testset "Graph utilities" begin
        include("utils.jl")
    end

    @testset "Product model definitions" begin
        include("products.jl")
    end

    @testset "Promoter model definitions" begin
        include("promoters.jl")
    end

    @testset "Reaction generating functions" begin
        include("reactions.jl")
    end
end
