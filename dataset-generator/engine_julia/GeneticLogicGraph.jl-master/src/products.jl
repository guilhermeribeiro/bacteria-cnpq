"""
    InputSpecies(production, degradation; name)

Construct a model of a chemical species which is created from nothing.

InputSpecies can be used as "sources" in larger models. They have no
transcription or translation reactions, only production and
degradation.

# Example
```julia-repl
julia> using ModelingToolkit

julia> @named IPTG = InputSpecies(1, 0.1)
```
"""
function InputSpecies(production; name)
    @parameters λ=production  [description="Rate of production for this species"]
    @variables t
    @species monomer(t) [
        description="The abundance of the species",
        dilute=true,
        protein=true,
        output=true
    ]
    rxs = [
        Reaction(λ, nothing, [monomer]),
    ]
    opts = Dict(:name => name, :connection_type => (InputSpecies, ))
    return ReactionSystem(rxs, t, [monomer], [λ]; opts...)
end

"""
    ConstantSpecies(level; name)

Construct a model of a chemical species which is has constant abundance.

ConstantSpecies can be used as "sources" in larger models. They are
not produced or degraded or diluted.

# Example
```julia-repl
julia> using ModelingToolkit

julia> @named IPTG = ConstantSpecies(1)
```
"""
function ConstantSpecies(level; name)
    @variables t
    @species monomer(t)=level [
        description="The abundance of the species",
        dilute=false,
        output=true
    ]
    opts = Dict(:name => name, :connection_type => (ConstantSpecies, ))
    return ReactionSystem(Reaction[], t, [monomer], []; opts...)
end

function Monomer(translation; name)
    @parameters λ=translation [description="Rate of protein synthesis (translation) from mRNA"]
    @variables t
    @species rna(t) [
        description="Abundance of mRNA for the monomer",
        dilute=true,
        mrna=true,
        input=true
    ]
    @species monomer(t) [
        description="Abundance of monomers",
        dilute=true,
        protein=true,
        output=true
    ]    
    opts = Dict(:name => name, :connection_type => (Monomer, ))
    return ReactionSystem(Reaction[], t, [rna, monomer], [λ]; opts...)
end

function Dimer(translation, binding, unbinding; name)
    @parameters λ=translation      [description="Rate of protein synthesis (translation) from mRNA"]
    @parameters k₋₁=unbinding      [description="Rate of dimer unbinding"]
    @parameters k₁=binding         [description="Rate of monomer dimerisation"]
    
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
    @species dimer(t) [
        description="Abundance of dimers",
        protein=true,
        dilute=true,
        output=true
    ]
    rxs = [
        Reaction(k₁, [monomer], [dimer], [2], [1]),
        Reaction(k₋₁, [dimer], [monomer], [1], [2]),
    ]
    opts = Dict(:name => name, :connection_type => (Dimer, ))
    return ReactionSystem(rxs, t, [rna, monomer, dimer], [λ, k₋₁, k₁]; opts...)
end


function _translation(x::ReactionSystem)
    [Reaction(x.λ, [x.rna], [x.rna, x.monomer])]
end

function translation(::Type{Monomer}, x::ReactionSystem)
    return _translation(x)
end

function translation(::Type{Dimer}, x::ReactionSystem)
    return _translation(x)
end

function _mrna_degradation(x::ReactionSystem, α)
    return [Reaction(α, [x.rna], nothing)]
end

function mrna_degradation(::Type{Monomer}, x::ReactionSystem, α)
    return _mrna_degradation(x, α)
end

function mrna_degradation(::Type{Dimer}, x::ReactionSystem, α)
    return _mrna_degradation(x, α)
end

function _protein_degradation(x::ReactionSystem, α)
    vars = filter(isdilutable, filter(isprotein, states(x, states(x))))
    return [Reaction(α, [v], nothing) for v in vars]
end

function protein_degradation(::Type{InputSpecies}, x::ReactionSystem, α)
    return _protein_degradation(x, α)
end

function protein_degradation(::Type{Monomer}, x::ReactionSystem, α)
    return _protein_degradation(x, α)
end

function protein_degradation(::Type{Dimer}, x::ReactionSystem, α)
    return _protein_degradation(x, α)
end


function randu0(::Type{ConstantSpecies}, x)
    return Dict(x.monomer => x.defaults[@nonamespace x.monomer])
end
randu0(::Type{InputSpecies}, x) = Dict(x.monomer => rand(0:16))
randu0(::Type{Monomer}, x) = Dict(x.rna => rand(0:16), x.monomer => rand(0:16))
randu0(::Type{Dimer}, x) = merge(randu0(Monomer, x), Dict(x.dimer => rand(0:4)))

function zerou0(::Type{ConstantSpecies}, x)
    return Dict(x.species => x.defaults[@nonamespace x.monomer])
end
zerou0(::Type{InputSpecies}, x) = Dict(x.monomer => 0)
zerou0(::Type{Monomer}, x) = Dict(x.rna => 0, x.monomer => 0)
zerou0(::Type{Dimer}, x) = merge(zerou0(Monomer, x), Dict(x.dimer => 0))

output(x::ReactionSystem) = output(component_type(x), x)
output(::Type{InputSpecies}, x::ReactionSystem) = x.monomer
output(::Type{Monomer}, x::ReactionSystem) = x.monomer
output(::Type{Dimer}, x::ReactionSystem) = x.dimer
