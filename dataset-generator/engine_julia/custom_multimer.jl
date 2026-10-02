using ModelingToolkit
using Catalyst
using GeneticLogicGraph

abstract type CustomMultimer <: GeneticLogicGraph.Species end

function Multimer(translation, binding, unbinding, n::Int; name)
    @parameters λ=translation      
    @parameters k₋₁=unbinding      
    @parameters k₁=binding         
    
    @variables t
    @species rna(t) [
        description="Abundance of mRNA for the monomer",
        mrna=true,
        dilute=true,
        input=true
    ]
    @species monomer(t) [
        description="Abundance of monomers",
        protein=true,
        dilute=true,
    ]
    @species multimer(t) [
        description="Abundance of multimers",
        protein=true,
        dilute=true,
        output=true
    ]
    rxs = [
        Reaction(k₁, [monomer], [multimer], [n], [1]),
        Reaction(k₋₁, [multimer], [monomer], [1], [n]),
    ]
    opts = Dict(:name => name, :connection_type => (CustomMultimer, ))
    return ReactionSystem(rxs, t, [rna, monomer, multimer], [λ, k₋₁, k₁]; opts...)
end

function GeneticLogicGraph.translation(::Type{CustomMultimer}, x::ReactionSystem)
    return GeneticLogicGraph._translation(x)
end

function GeneticLogicGraph.mrna_degradation(::Type{CustomMultimer}, x::ReactionSystem, α)
    return GeneticLogicGraph._mrna_degradation(x, α)
end

function GeneticLogicGraph.protein_degradation(::Type{CustomMultimer}, x::ReactionSystem, α)
    return GeneticLogicGraph._protein_degradation(x, α)
end

function GeneticLogicGraph.randu0(::Type{CustomMultimer}, x)
    return merge(GeneticLogicGraph.randu0(GeneticLogicGraph.Monomer, x), Dict(x.multimer => rand(0:4)))
end

function GeneticLogicGraph.zerou0(::Type{CustomMultimer}, x)
    return merge(GeneticLogicGraph.zerou0(GeneticLogicGraph.Monomer, x), Dict(x.multimer => 0))
end

function GeneticLogicGraph.output(::Type{CustomMultimer}, x::ReactionSystem)
    return x.multimer
end
