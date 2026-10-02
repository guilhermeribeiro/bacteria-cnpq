function PromoterRegion(transcription; name)
    @variables t
    @species promoter(t)      [description="RNAP recruiting region of DNA", dilute=false]
    @parameters λ=transcription [description="Transcription rate from this promoter region"]
    opts = Dict(:name => name, :connection_type => (PromoterRegion, ))
    return ReactionSystem(Reaction[], t, [promoter], [λ]; opts...)
end

function RegulatedPromoter(bound::ReactionSystem, unbound::ReactionSystem, ligand::ReactionSystem, binding, unbinding; name)
    R = GlobalScope.(states(ligand, ModelingToolkit.outputs(ligand)))[1]
    @variables t
    @parameters k₁=binding   [description="Binding rate of the ligand to the promoter region"]
    @parameters k₀=unbinding [description="Unbinding rate of the ligand from the promoter region"]
    binding   = Reaction(k₁, [unbound.promoter, R], [bound.promoter])
    unbinding = Reaction(k₀, [bound.promoter], [R, unbound.promoter])
    opts = Dict(
        :name => name,
        :systems => [unbound, bound],
        :connection_type => (RegulatedPromoter, ligand)
    )
    return ReactionSystem([binding, unbinding], t, [], [k₀, k₁]; opts...)
end

function RegulatedPromoter(bound::Real, unbound::Real, ligand::ReactionSystem, binding, unbinding; name)
    @named bound   = PromoterRegion(bound)
    @named unbound = PromoterRegion(unbound)
    RegulatedPromoter(bound, unbound, ligand, binding, unbinding; name=name)
end

function transcription(::Type{PromoterRegion}, ::Type{<:Species}, x, args...)
    rnas = filter(ismrna, reduce(vcat, states(y, states(y)) for y in args))
    if length(rnas) == 0
        a = nameof(x)
        bs = nameof.(args)
        @warn "$a is trying to transcribe $(bs) but there is no rna to transcribe"
        println("Clang")
    end
    return [Reaction(x.λ, [x.promoter], [x.promoter; rnas], [1], ones(Int, length(rnas) + 1))]
end

function transcription(::Type{PromoterRegion}, ::Type{RegulatedPromoter}, x, args...)
    return transcription(x, reduce(vcat, collect(component_args(y)) for y in args)...)
end

function transcription(::Type{RegulatedPromoter}, ::Type{RegulatedPromoter}, x, args...)
    return vcat(transcription(x.bound, args...), transcription(x.unbound, args...))
end

function transcription(::Type{RegulatedPromoter}, ::Type{<:Species}, x, args...)
    return vcat(transcription(x.bound, args...), transcription(x.unbound, args...))
end

randu0(::Type{PromoterRegion}, x::ReactionSystem) = Dict(x.promoter => 1)
function randu0(::Type{RegulatedPromoter}, x::ReactionSystem)
    i = rand(0:1)
    Dict(x.bound.promoter => i, x.unbound.promoter => 1 - i)
end

zerou0(::Type{PromoterRegion}, x::ReactionSystem) = Dict(x.promoter => 1)
function zerou0(::Type{RegulatedPromoter}, x::ReactionSystem)
    i = rand(0:1)
    Dict(x.bound.promoter => i, x.unbound.promoter => 1 - i)
end
