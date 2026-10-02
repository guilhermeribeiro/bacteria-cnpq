using GeneticLogicGraph
using ModelingToolkit
using Catalyst
using Test

const GLG = GeneticLogicGraph


@testset "InputSpecies definition" begin
    @named model = InputSpecies(1.0)

    @test length(equations(model)) == 1
    @test length(states(model)) == 1
    @test length(parameters(model)) == 1

    @variables t monomer(t)
    @test isequal(t, ModelingToolkit.get_iv(model))
    @test any(isequal(monomer), states(model))
    @test any(isequal(monomer), ModelingToolkit.outputs(model))

    @parameters λ
    @test any(isequal(λ), parameters(model))
    @test ModelingToolkit.get_defaults(model)[λ] == 1.0

    @test GLG.component_type(model) == GLG.InputSpecies
end

@testset "Monomer definition" begin
    @named model = Monomer(1.0)

    @test length(states(model)) == 2
    @test length(parameters(model)) == 1

    @variables t rna(t) monomer(t)
    @test isequal(t, ModelingToolkit.get_iv(model))
    @test any(isequal(rna), states(model))
    @test any(isequal(monomer), states(model))
    @test any(isequal(rna), ModelingToolkit.inputs(model))
    @test any(isequal(monomer), ModelingToolkit.outputs(model))

    @parameters λ
    @test any(isequal(λ), parameters(model))
    @test ModelingToolkit.get_defaults(model)[λ] == 1.0
    @test GLG.component_type(model) == GLG.Monomer
end

@testset "Dimer definition" begin
    @named model = Dimer(1.0, 2.0, 3.0)

    @test length(equations(model)) == 2
    @test length(states(model)) == 3
    @test length(parameters(model)) == 3

    @variables t rna(t) monomer(t) dimer(t)
    @test isequal(t, ModelingToolkit.get_iv(model))
    @test any(isequal(rna), states(model))
    @test any(isequal(monomer), states(model))
    @test any(isequal(dimer), states(model))
    @test any(isequal(rna), ModelingToolkit.inputs(model))
    @test any(isequal(dimer), ModelingToolkit.outputs(model))

    @parameters λ k₁ k₋₁
    @test any(isequal(λ), parameters(model))
    @test any(isequal(k₁), parameters(model))
    @test any(isequal(k₋₁), parameters(model))
    @test ModelingToolkit.get_defaults(model)[λ] == 1.0
    @test ModelingToolkit.get_defaults(model)[k₁] == 2.0
    @test ModelingToolkit.get_defaults(model)[k₋₁] == 3.0

    @test GLG.component_type(model) == GLG.Dimer
end
