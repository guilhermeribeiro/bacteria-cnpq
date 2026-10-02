module GeneticLogicGraph

using Symbolics
using SymbolicUtils: Symbolic
using ModelingToolkit
using Catalyst
using Distributions
using JumpProcesses
using Graphs


struct Dilute end
Symbolics.option_to_metadata_type(::Val{:dilute}) = Dilute
isdilutable(x)::Bool = getmetadata(x, Dilute, false)
export isdilutable

struct mRNA end
Symbolics.option_to_metadata_type(::Val{:mrna}) = mRNA
ismrna(x)::Bool = getmetadata(x, mRNA, false)
export ismrna

struct Protein end
Symbolics.option_to_metadata_type(::Val{:protein}) = Protein
isprotein(x)::Bool = getmetadata(x, Protein, false)
export isprotein

abstract type Component end
function ConcreteSystemType(::Type{T}) where {T<:Component}
    return ReactionSystem{
        Nothing,
        Catalyst.NetworkProperties{Int, Term{Real, Base.ImmutableDict{DataType, Any}}}
    }
end

abstract type Species <: Component end
abstract type InputSpecies <: Species end
abstract type ConstantSpecies <: Species end
abstract type Monomer <: Species end
abstract type Dimer <: Species end
promote_rule(::Type{InputSpecies}, ::Type{Monomer}) = Species
promote_rule(::Type{InputSpecies}, ::Type{Dimer}) = Species
promote_rule(::Type{ConstantSpecies}, ::Type{Monomer}) = Species
promote_rule(::Type{ConstantSpecies}, ::Type{Dimer}) = Species
promote_rule(::Type{Monomer}, ::Type{Dimer}) = Species
component_type(x) = first(x.connection_type)
component_args(x) = x.connection_type[2:end]

translation(::Type{<:Component}, args...) = Reaction[]
function translation(x::ReactionSystem, args...)
    return translation(component_type(x), x, args...)
end

transcription(::Type{<:Component}, args...) = Reaction[]
function transcription(x::ReactionSystem, args::Vararg{<:ReactionSystem})
    T1 = component_type(x)
    T2 = promote_type(component_type.(args)...)
    return transcription(T1, T2, x, args...)
end

mrna_degradation(::Type{<:Component}, args...) = Reaction[]
function mrna_degradation(x::ReactionSystem, args...)
    return mrna_degradation(component_type(x), x, args...)
end

protein_degradation(::Type{<:Component}, args...) = Reaction[]
function protein_degradation(x::ReactionSystem, args...)
    return protein_degradation(component_type(x), x, args...)
end

randu0(x::ReactionSystem, args...) = randu0(component_type(x), x, args...)
zerou0(x::ReactionSystem, args...) = zerou0(component_type(x), x, args...)

export InputSpecies
export ConstantSpecies
export Monomer
export Dimer

include("utils.jl")
export prune

include("products.jl")
export randu0
export translation
export mrna_degradation
export protein_degradation

abstract type PromoterRegion <: Component end
abstract type RegulatedPromoter <: PromoterRegion end

export PromoterRegion
export RegulatedPromoter

include("promoters.jl")
export randu0
export transcription

include("callbacks.jl")
export MinTimestepError
export make_doubling_callback
export make_species_limit_callback
export make_mintimestep_callback
export make_positive_domain_callback

abstract type Circuit end
export Circuit
include("circuit.jl")

end # module GeneticLogicGraph
