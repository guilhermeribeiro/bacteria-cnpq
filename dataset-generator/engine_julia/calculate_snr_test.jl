# ==============================================================================
# CALCULADOR MINIMO DE SNR PARA TEST_RUNS (DARTS)
# ==============================================================================
# Este script foi criado de forma isolada para interpretar os resultados dos
# testes (circuit_design.json e wiring_matrix.csv) e calcular o SNR simulando a
# rede construída através da biblioteca original (GeneticLogicGraph / ALIFE2023).
#
# COMO USAR:
# Rode no terminal passando a pasta do teste como argumento. Exemplo:
#   julia calculate_snr_test.jl "test_runs/2026-07-28_18-18-07"
#
# Caso nenhum argumento seja passado, ele rodará na pasta padrão do script.
# Os resultados (SNR e Média YFP) serão impressos na tela e salvos de forma
# fácil e automatizada no arquivo 'snr_results.json' na própria pasta do teste!
# ==============================================================================
# -*- coding: utf-8 -*-
ENV["GKSwstype"] = "100"
using Pkg

# Get directory paths
const SCRIPT_DIR = @__DIR__
const PROJ_DIR   = joinpath(SCRIPT_DIR, "ALIFE2023-master")
const GLG_DIR    = joinpath(SCRIPT_DIR, "GeneticLogicGraph.jl-master")
const MULTIMER_PATH = joinpath(SCRIPT_DIR, "custom_multimer.jl")

Pkg.activate(PROJ_DIR)
try
    using GeneticLogicGraph
catch
    Pkg.develop(PackageSpec(path=GLG_DIR))
    using GeneticLogicGraph
end

using GeneticLogicGraph, ALIFE2023, ALIFE2023.Inverter
using Graphs, JSON, ModelingToolkit, JumpProcesses, Statistics

include(MULTIMER_PATH)

# Load PROTEIN_LIBRARY
struct RealProteinParams
    name::Symbol
    ymin::Float64   
    ymax::Float64   
    Kd::Float64
    n::Float64
end

const PROTEIN_LIBRARY = [
    RealProteinParams(:SrpR, 0.0005, 3.000, 1.000, 2.0),
    RealProteinParams(:PhlF, 0.0005, 3.000, 1.000, 2.0),
    RealProteinParams(:AmeR, 0.0005, 3.000, 1.000, 2.0),
    RealProteinParams(:BetI, 0.0005, 3.000, 1.000, 2.0),
    RealProteinParams(:QacR, 0.0005, 3.000, 1.000, 2.0),
    RealProteinParams(:AmtR, 0.0005, 3.000, 1.000, 2.0),
    RealProteinParams(:Vazio, 0.0, 0.0, 1.0, 1.0)
]

function RealOperon(params::RealProteinParams, node_id::Int=0)
    suffix = node_id > 0 ? string("_n", node_id) : ""
    R = Dimer(1.0, 1.0, 1/60; name=Symbol(string(params.name), suffix))
    return RegulatedPromoter(params.ymin, params.ymax, R, 1.0 / params.Kd, 1.0; name=Symbol("p", string(params.name), suffix))
end

function get_components_from_json(json_path::String, N::Int)
    design = JSON.parsefile(json_path)
    slots = design["slots"]
    
    @named YFP = Monomer(1)
    pYFP = RegulatedPromoter(0.0, 0.0, YFP, 0.0, 0.0; name=:pYFP) 
    
    inputs = []
    # Slot 0 = pTac (ligado ao LacI)
    @named LacI = InputSpecies(1 * log(2) / 90)
    push!(inputs, RegulatedPromoter(0.0005, 3.0, LacI, 1.0, 1.0; name=:pTac))
    
    # Slot 1 = pTet (ligado ao TetR)
    @named TetR = InputSpecies(1 * log(2) / 90)
    push!(inputs, RegulatedPromoter(0.0005, 3.0, TetR, 1.0, 1.0; name=:pTet))
    
    internal_comps = []
    # Os nós intermediários começam do Slot 2 (índice 3 no Julia, pois N=5 vai de 1 a 5)
    for j in 3:N-1
        slot_winner = slots[j]["winner"]
        
        idx = findfirst(p -> string(p.name) == slot_winner, PROTEIN_LIBRARY)
        if !isnothing(idx) && PROTEIN_LIBRARY[idx].name != :Vazio
            push!(internal_comps, RealOperon(PROTEIN_LIBRARY[idx], j))
        else
            push!(internal_comps, RealOperon(RealProteinParams(Symbol("Node", j), 0.0, 0.0, 1.0, 1.0), j))
        end
    end
    
    return Tuple(vcat(inputs, internal_comps, [pYFP]))
end

const ENSEMBLE_SIZE = 32
const TARGET_GATE = "XOR"

function get_circuit_outputs(G::DiGraph, proteins)
    model = Circuit(G, proteins; name=:evaluated_model)
    problem = problem_from_model(model)
    
    in_lac = nothing
    in_tet = nothing
    output = nothing
    idx = nothing
    try
        in_lac = (@nonamespace model.LacI).λ
        in_tet = (@nonamespace model.TetR).λ
        output = (@nonamespace model.YFP).monomer
        idx = findfirst(isequal(output), states(problem.prob.f.sys))
    catch e
        # Se YFP (ou LacI/TetR) não estiver no modelo porque o circuito
        # foi completamente podado (sem conexoes pro YFP), ele vai cair aqui
        return Float64[], Float64[], Float64[], Float64[]
    end
    
    if isnothing(idx)
        return Float64[], Float64[], Float64[], Float64[]
    end
    
    low_level = 1 * log(2) / 90
    high_level = 100 * log(2) / 90
    
    # Define os 4 problemas
    prob_00 = change_input_levels(problem, [in_lac, in_tet], [low_level, low_level])
    prob_00 = remake(prob_00, tspan=(0.0, 1500.0))
    
    prob_01 = change_input_levels(problem, [in_lac, in_tet], [low_level, high_level])
    prob_01 = remake(prob_01, tspan=(0.0, 1500.0))
    
    prob_10 = change_input_levels(problem, [in_lac, in_tet], [high_level, low_level])
    prob_10 = remake(prob_10, tspan=(0.0, 1500.0))
    
    prob_11 = change_input_levels(problem, [in_lac, in_tet], [high_level, high_level])
    prob_11 = remake(prob_11, tspan=(0.0, 1500.0))
    
    s_00 = Vector{Float64}(undef, ENSEMBLE_SIZE)
    s_01 = Vector{Float64}(undef, ENSEMBLE_SIZE)
    s_10 = Vector{Float64}(undef, ENSEMBLE_SIZE)
    s_11 = Vector{Float64}(undef, ENSEMBLE_SIZE)
    
    Threads.@threads for k in 1:ENSEMBLE_SIZE
        sol = solve(prob_00, SSAStepper())
        s_00[k] = Float64(sol[end][idx])
        
        sol = solve(prob_01, SSAStepper())
        s_01[k] = Float64(sol[end][idx])
        
        sol = solve(prob_10, SSAStepper())
        s_10[k] = Float64(sol[end][idx])
        
        sol = solve(prob_11, SSAStepper())
        s_11[k] = Float64(sol[end][idx])
    end
    
    return s_00, s_01, s_10, s_11
end

function evaluate_test_run(run_dir::String)
    csv_file  = joinpath(run_dir, "wiring_matrix.csv")
    json_file = joinpath(run_dir, "circuit_design.json")
    
    if !isfile(csv_file) || !isfile(json_file)
        println("Erro: Arquivos não encontrados em ", run_dir)
        return 0.0
    end
    
    println("Lendo matriz de fiação: ", csv_file)
    lines = readlines(csv_file)
    N = length(lines) - 1
    W_float = zeros(Float64, N, N)
    for i in 2:length(lines)
        parts = split(lines[i], ',')
        for j in 2:length(parts)
            W_float[i-1, j-1] = parse(Float64, parts[j])
        end
    end
    
    println("Construindo circuito e extraindo proteínas (Lógica I/O)...")
    proteins = get_components_from_json(json_file, N)
    
    G = SimpleDiGraph(N)
    for i in 1:N
        for j in 1:N
            if W_float[i, j] > 0.01
                add_edge!(G, i, j)
            end
        end
    end
    
    println("Executando Simulação SSA (Julia) com $(nv(G)) nós e $(ne(G)) conexões...")
    
    if ne(G) == 0
        println("Aviso: O circuito foi completamente podado (0 conexões). Retornando SNR nulo.")
        return 0.0
    end
    
    s_00, s_01, s_10, s_11 = get_circuit_outputs(G, proteins)
    
    if isempty(s_00)
        println("Erro: Não foi possível calcular o SSA.")
        return 0.0
    end
    
    m00, m01, m10, m11 = mean(s_00), mean(s_01), mean(s_10), mean(s_11)
    sd00, sd01, sd10, sd11 = std(s_00), std(s_01), std(s_10), std(s_11)
    se00, se01, se10, se11 = sd00/sqrt(ENSEMBLE_SIZE), sd01/sqrt(ENSEMBLE_SIZE), sd10/sqrt(ENSEMBLE_SIZE), sd11/sqrt(ENSEMBLE_SIZE)
    
        # Lógica Dinâmica da Porta
    if TARGET_GATE == "XOR"
        s_high = vcat(s_01, s_10)
        s_low = vcat(s_00, s_11)
    elseif TARGET_GATE == "AND"
        s_high = s_11
        s_low = vcat(s_00, s_01, s_10)
    elseif TARGET_GATE == "OR"
        s_high = vcat(s_01, s_10, s_11)
        s_low = s_00
    elseif TARGET_GATE == "NAND"
        s_high = vcat(s_00, s_01, s_10)
        s_low = s_11
    elseif TARGET_GATE == "NOR"
        s_high = s_00
        s_low = vcat(s_01, s_10, s_11)
    elseif TARGET_GATE == "NOT"
        s_high = vcat(s_00, s_01)
        s_low  = vcat(s_10, s_11)
    else
        s_high = vcat(s_01, s_10)
        s_low = vcat(s_00, s_11)
    end
    
    high_mean = mean(s_high)
    low_mean = mean(s_low)
    
    signal = high_mean - low_mean
    noise_total = std(s_high) + std(s_low)
    snr_std = noise_total > 1e-6 ? (signal / noise_total) : 0.0
    snr_hiscock = mean(s_high .- s_low) / std(s_high .- s_low)
    snr_xor = max(snr_std, snr_hiscock)
    
    println("----------------------------------------")
    println("RESULTADOS DA SIMULAÇÃO (XOR):")
    println("  State 00: $(round(m00, digits=4)) ± $(round(se00, digits=4)) (SD: $(round(sd00, digits=4)))")
    println("  State 01: $(round(m01, digits=4)) ± $(round(se01, digits=4)) (SD: $(round(sd01, digits=4)))")
    println("  State 10: $(round(m10, digits=4)) ± $(round(se10, digits=4)) (SD: $(round(sd10, digits=4)))")
    println("  State 11: $(round(m11, digits=4)) ± $(round(se11, digits=4)) (SD: $(round(sd11, digits=4)))")
    println("  SNR (Standard):      $(round(snr_std, digits=4))")
    println("  SNR (Hiscock ALIFE): $(round(snr_hiscock, digits=4))")
    println("  SNR Final (XOR):     $(round(snr_xor, digits=4))")
    println("----------------------------------------")
    
    output_file = joinpath(run_dir, "snr_results.json")
    open(output_file, "w") do f
        JSON.print(f, Dict(
            "snr_xor" => snr_xor,
            "snr_standard" => snr_std,
            "snr_hiscock" => snr_hiscock,
            "mean_00" => m00, "se_00" => se00, "std_00" => sd00,
            "mean_01" => m01, "se_01" => se01, "std_01" => sd01,
            "mean_10" => m10, "se_10" => se10, "std_10" => sd10,
            "mean_11" => m11, "se_11" => se11, "std_11" => sd11,
            "samples_00" => s_00,
            "samples_01" => s_01,
            "samples_10" => s_10,
            "samples_11" => s_11
        ), 4)
    end
    println("✅ Resultados salvos em: ", output_file)
    
    return snr_xor
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_dir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "test_runs", "2026-07-28_18-18-07")
    evaluate_test_run(run_dir)
end

