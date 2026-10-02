function transcription(A::Matrix{Bool}, systems)
    rxs = Reaction[]
    for i in 1:size(A, 1)
        idxs = findall(identity, A[i, :])
        if length(idxs) > 0
            append!(rxs, transcription(systems[i], systems[idxs]...))
        end
    end
    return rxs
end

function used_systems(A::Matrix{Bool}, systems)
    syss = ReactionSystem[]
    for i in 1:size(A, 1)
        idxs = findall(identity, A[i, :])
        if length(idxs) > 0
            for sys in systems[[i; idxs]]
                push!(syss, sys)
                append!(syss, collect(component_args(sys)))
            end
        end
    end
    return unique(syss)
end

function Circuit(G::DiGraph, systems; name)
    A = Matrix(Graphs.adjacency_matrix(G) .> 0)
    syss = used_systems(A, systems)
    @variables t
    @species N(t)=0
    @parameters μ α

    rxs = [
        transcription(A, systems);
        reduce(vcat, mrna_degradation(sys, α) for sys in syss);
        reduce(vcat, translation(sys) for sys in syss);
        [Reaction(μ, nothing, [N])]
    ]

    return ReactionSystem(
        rxs, t, [N], [μ, α];
        name=name, connection_type=(Circuit,), systems=syss
    )
end

function randu0(::Type{Circuit}, x::ReactionSystem)
    return reduce(merge, randu0.(x.systems), init=Dict())
end

function zerou0(::Type{Circuit}, x::ReactionSystem)
    return reduce(merge, zerou0.(x.systems), init=Dict())
end
