using GeneticLogicGraph
using ModelingToolkit
using Catalyst
using Test

const GLG = GeneticLogicGraph

@testset "Promoter region definition" begin
    @named model = PromoterRegion(1.0)

    @test length(equations(model)) == 0
    @test length(states(model)) == 1
    @test length(parameters(model)) == 1

    @variables t promoter(t)
    @test isequal(t, ModelingToolkit.get_iv(model))
    @test any(isequal(promoter), states(model))

    @parameters λ
    @test any(isequal(λ), parameters(model))
    @test ModelingToolkit.get_defaults(model)[λ] == 1.0

    @test GLG.component_type(model) == GLG.PromoterRegion
end

@testset "Regulated Promoter definition" begin
    @named ligand  = InputSpecies(1.0)
    @named bound   = PromoterRegion(1.0)
    @named unbound = PromoterRegion(2.0)
    @named model   = RegulatedPromoter(bound, unbound, ligand, 1.0, 1.0)

    @test length(equations(model)) == 2
    @test length(states(model)) == 2
    @test length(ModelingToolkit.get_states(model)) == 0
    @test length(parameters(model)) == 4
    @test length(ModelingToolkit.get_ps(model)) == 2
    @test length(ModelingToolkit.get_systems(model)) == 2
    @test GLG.component_type(ModelingToolkit.get_systems(model)[1]) == GLG.PromoterRegion
    @test GLG.component_type(ModelingToolkit.get_systems(model)[2]) == GLG.PromoterRegion

    @variables t
    @test isequal(t, ModelingToolkit.get_iv(model))

    @parameters k₁ k₀
    @test any(isequal(k₁), parameters(model))
    @test any(isequal(k₀), parameters(model))
    @test ModelingToolkit.get_defaults(model)[k₁] == 1.0
    @test ModelingToolkit.get_defaults(model)[k₀] == 1.0

    @test GLG.component_type(model) == GLG.RegulatedPromoter
end

@testset "Regulated Promoter constructors" begin
    @named ligand  = InputSpecies(1.0)
    @named bound   = PromoterRegion(1.0)
    @named unbound = PromoterRegion(2.0)
    @named model   = RegulatedPromoter(bound, unbound, ligand, 1.0, 1.0)

    @test model == RegulatedPromoter(1.0, 2.0, ligand, 1.0, 1.0; name=:model)
    @test model == RegulatedPromoter(bound, unbound, ligand, 1.0, 1.0; name=:model)
end

@testset "Initially random conditions - conservation law for promoters" begin
    @named ligand  = InputSpecies(1.0)
    @named bound   = PromoterRegion(1.0)
    @named unbound = PromoterRegion(2.0)
    @named model   = RegulatedPromoter(bound, unbound, ligand, 1.0, 1.0)

    @test sum(last.(collect(randu0(model)))) == 1
end
