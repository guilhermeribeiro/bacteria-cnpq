function prune(graph, output::Int)
    A = adjacency_matrix(graph)
    for i ∈ vertices(graph)
        if !has_path(graph, i, output)
            A[i, :] .= false
            A[:, i] .= false
        end
    end
    return typeof(graph)(A)
end

function prune(graph, outputs::AbstractVector)
    result = typeof(graph)(nv(graph))
    for output in outputs
        result = union(result, prune(graph, output))
    end
    return result
end
