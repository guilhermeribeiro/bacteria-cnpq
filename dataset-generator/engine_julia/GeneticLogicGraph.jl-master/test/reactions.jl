using GeneticLogicGraph
using ModelingToolkit
using Catalyst
using Test

const GLG = GeneticLogicGraph


@testset "Connections between promoters" begin
    @named input = InputSpecies(1.0)
    @named monomer = Monomer(1.0)
    @named dimer = Dimer(1.0, 1.0, 1.0)

    @named simplepromoter = PromoterRegion(1.0)
    @named promoter1 = RegulatedPromoter(1.0, 1.0, input, 1.0, 1.0)
    @named promoter2 = RegulatedPromoter(1.0, 1.0, monomer, 1.0, 1.0)
    @named promoter3 = RegulatedPromoter(1.0, 1.0, dimer, 1.0, 1.0)

    rxs = [
        Reaction(promoter1.bound.λ, [promoter1.bound.promoter], [promoter1.bound.promoter, monomer.rna]),
        Reaction(promoter1.unbound.λ, [promoter1.unbound.promoter], [promoter1.unbound.promoter, monomer.rna]),
    ]
    @test rxs == GLG.transcription(promoter1, promoter2)

    rxs = [
        Reaction(promoter1.bound.λ, [promoter1.bound.promoter], [promoter1.bound.promoter, dimer.rna]),
        Reaction(promoter1.unbound.λ, [promoter1.unbound.promoter], [promoter1.unbound.promoter, dimer.rna]),
    ]
    @test rxs == GLG.transcription(promoter1, promoter3)
    
    rxs = [
        Reaction(simplepromoter.λ, [simplepromoter.promoter], [simplepromoter.promoter, monomer.rna]),
    ]
    @test rxs == GLG.transcription(simplepromoter, promoter2)

    rxs = [
        Reaction(simplepromoter.λ, [simplepromoter.promoter], [simplepromoter.promoter, dimer.rna]),
    ]
    @test rxs == GLG.transcription(simplepromoter, promoter3)

    rxs = [
        Reaction(promoter2.bound.λ, [promoter2.bound.promoter], [promoter2.bound.promoter, monomer.rna]),
        Reaction(promoter2.unbound.λ, [promoter2.unbound.promoter], [promoter2.unbound.promoter, monomer.rna]),
    ]
    @test rxs == GLG.transcription(promoter2, promoter2)

    rxs = [
        Reaction(promoter1.bound.λ, [promoter1.bound.promoter], [promoter1.bound.promoter, monomer.rna, dimer.rna]),
        Reaction(promoter1.unbound.λ, [promoter1.unbound.promoter], [promoter1.unbound.promoter, monomer.rna, dimer.rna]),
    ]
    @test rxs == GLG.transcription(promoter1, promoter2, promoter3)
end

@testset "Connections between other components (empty reaction lists)" begin
    @named input = InputSpecies(1.0)
    @named monomer = Monomer(1.0)
    @named dimer = Dimer(1.0, 1.0, 1.0)

    @test Reaction[] == GLG.transcription(input, monomer)
    @test Reaction[] == GLG.transcription(monomer, input)
    @test Reaction[] == GLG.transcription(dimer, monomer)
    @test Reaction[] == GLG.transcription(monomer, dimer)
    @test Reaction[] == GLG.transcription(input, input)
    @test Reaction[] == GLG.transcription(monomer, monomer)
    @test Reaction[] == GLG.transcription(dimer, dimer)
end

@testset "Translation for products" begin
    @named monomer = Monomer(1.0)
    @named dimer = Dimer(1.0, 1.0, 1.0)

    expected = [Reaction(monomer.λ, [monomer.rna], [monomer.rna, monomer.monomer])]
    @test expected == GLG.translation(monomer)

    expected = [Reaction(dimer.λ, [dimer.rna], [dimer.rna, dimer.monomer])]
    @test expected == GLG.translation(dimer)
end

@testset "Translation in other components (empty reaction lists)" begin
    @named input = InputSpecies(1.0)
    @named constant = ConstantSpecies(1)
    @named promoter = PromoterRegion(1)
    @named regulatedpromoter = RegulatedPromoter(1, 1, input, 1, 1)

    @test Reaction[] == GLG.translation(input)
    @test Reaction[] == GLG.translation(constant)
    @test Reaction[] == GLG.translation(promoter)
    @test Reaction[] == GLG.translation(regulatedpromoter)
end

@testset "Degradation of mRNA for products" begin
    @named monomer = Monomer(1.0)
    @named dimer = Dimer(1.0, 1.0, 1.0)
    @parameters α

    expected = [Reaction(α, [monomer.rna], nothing)]
    @test expected == GLG.mrna_degradation(monomer, α)

    expected = [Reaction(α, [dimer.rna], nothing)]
    @test expected == GLG.mrna_degradation(dimer, α)
end
