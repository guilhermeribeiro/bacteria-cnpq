# -*- coding: utf-8 -*-
# evaluator_daemon.jl
# ==============================================================================
# Self-Contained Biological Gillespie SSA Evaluator Daemon
#
# Listens on STDIN for folder paths, evaluates circuits using JumpProcesses.jl,
# saves snr_results.json, and returns SNR_RESULT:<value> via STDOUT.
# ==============================================================================

ENV["GKSwstype"] = "100"

using Pkg
const JULIA_DIR = @__DIR__
const PROJ_DIR  = joinpath(JULIA_DIR, "ALIFE2023-master")
const GLG_DIR   = joinpath(JULIA_DIR, "GeneticLogicGraph.jl-master")
const MULTIMER_PATH = joinpath(JULIA_DIR, "custom_multimer.jl")

Pkg.activate(PROJ_DIR)

# Bulletproof check for GeneticLogicGraph:
try
    using GeneticLogicGraph
catch
    Pkg.develop(PackageSpec(path=GLG_DIR))
    using GeneticLogicGraph
end

using ALIFE2023, ALIFE2023.Inverter
using Graphs, JSON, ModelingToolkit, JumpProcesses, Statistics

include(MULTIMER_PATH)

const ENSEMBLE_SIZE = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 32
const TARGET_GATE   = length(ARGS) > 1 ? ARGS[2] : "XOR"

println(stderr, "[JULIA DAEMON] Inicializado com ENSEMBLE_SIZE=$ENSEMBLE_SIZE, GATE=$TARGET_GATE")

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
    @named LacI = InputSpecies(1 * log(2) / 90)
    push!(inputs, RegulatedPromoter(0.0005, 3.0, LacI, 1.0, 1.0; name=:pTac))
    
    @named TetR = InputSpecies(1 * log(2) / 90)
    push!(inputs, RegulatedPromoter(0.0005, 3.0, TetR, 1.0, 1.0; name=:pTet))
    
    internal_comps = []
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

function get_circuit_outputs(G::DiGraph, proteins)
    local model, problem
    in_lac = nothing
    in_tet = nothing
    output = nothing
    idx = nothing
    try
        model = Circuit(G, proteins; name=:evaluated_model)
        problem = problem_from_model(model)
        in_lac = (@nonamespace model.LacI).λ
        in_tet = (@nonamespace model.TetR).λ
        output = (@nonamespace model.YFP).monomer
        idx = findfirst(isequal(output), states(problem.prob.f.sys))
    catch e
        return Float64[], Float64[], Float64[], Float64[], "Erro na criacao do modelo Circuit: " * string(e)
    end
    
    if isnothing(idx)
        return Float64[], Float64[], Float64[], Float64[], "Especie YFP nao encontrada nos estados do problema"
    end
    
    low_level  = 1 * log(2) / 90
    high_level = 100 * log(2) / 90
    
    tspan_eval = 1000.0
    prob_00 = remake(change_input_levels(problem, [in_lac, in_tet], [low_level, low_level]), tspan=(0.0, tspan_eval))
    prob_01 = remake(change_input_levels(problem, [in_lac, in_tet], [low_level, high_level]), tspan=(0.0, tspan_eval))
    prob_10 = remake(change_input_levels(problem, [in_lac, in_tet], [high_level, low_level]), tspan=(0.0, tspan_eval))
    prob_11 = remake(change_input_levels(problem, [in_lac, in_tet], [high_level, high_level]), tspan=(0.0, tspan_eval))
    
    s_00 = zeros(Float64, ENSEMBLE_SIZE)
    s_01 = zeros(Float64, ENSEMBLE_SIZE)
    s_10 = zeros(Float64, ENSEMBLE_SIZE)
    s_11 = zeros(Float64, ENSEMBLE_SIZE)
    
    Threads.@threads for k in 1:ENSEMBLE_SIZE
        try
            sol00 = solve(prob_00, SSAStepper(); maxiters=250_000)
            s_00[k] = length(sol00) > 0 ? Float64(sol00[end][idx]) : 0.0
        catch e00
            s_00[k] = 0.0
        end
        try
            sol01 = solve(prob_01, SSAStepper(); maxiters=250_000)
            s_01[k] = length(sol01) > 0 ? Float64(sol01[end][idx]) : 0.0
        catch e01
            s_01[k] = 0.0
        end
        try
            sol10 = solve(prob_10, SSAStepper(); maxiters=250_000)
            s_10[k] = length(sol10) > 0 ? Float64(sol10[end][idx]) : 0.0
        catch e10
            s_10[k] = 0.0
        end
        try
            sol11 = solve(prob_11, SSAStepper(); maxiters=250_000)
            s_11[k] = length(sol11) > 0 ? Float64(sol11[end][idx]) : 0.0
        catch e11
            s_11[k] = 0.0
        end
    end
    
    return s_00, s_01, s_10, s_11, ""
end

function evaluate_test_run(run_dir::String)
    csv_file  = joinpath(run_dir, "wiring_matrix.csv")
    json_file = joinpath(run_dir, "circuit_design.json")
    
    if !isfile(csv_file) || !isfile(json_file)
        return 0.0, "Arquivos de circuito nao encontrados em $run_dir"
    end
    
    lines = readlines(csv_file)
    N = length(lines) - 1
    W_float = zeros(Float64, N, N)
    for i in 2:length(lines)
        parts = split(lines[i], ',')
        for j in 2:length(parts)
            W_float[i-1, j-1] = parse(Float64, parts[j])
        end
    end
    
    proteins = get_components_from_json(json_file, N)
    
    G = SimpleDiGraph(N)
    for i in 1:N
        for j in 1:N
            if W_float[i, j] > 0.01
                add_edge!(G, i, j)
            end
        end
    end
    
    if ne(G) == 0
        return 0.0, "Circuito podado (0 conexoes)"
    end
    
    s_00, s_01, s_10, s_11, err_out = get_circuit_outputs(G, proteins)
    
    if !isempty(err_out) || isempty(s_00) || isempty(s_01) || isempty(s_10) || isempty(s_11)
        return 0.0, isempty(err_out) ? "Falha na resolucao de especies ou construcao do modelo" : err_out
    end
    
    m00, m01, m10, m11 = mean(s_00), mean(s_01), mean(s_10), mean(s_11)
    sd00, sd01, sd10, sd11 = std(s_00), std(s_01), std(s_10), std(s_11)
    se00, se01, se10, se11 = sd00/sqrt(ENSEMBLE_SIZE), sd01/sqrt(ENSEMBLE_SIZE), sd10/sqrt(ENSEMBLE_SIZE), sd11/sqrt(ENSEMBLE_SIZE)
    
    if TARGET_GATE == "XOR"
        s_high = vcat(s_01, s_10)
        s_low  = vcat(s_00, s_11)
    elseif TARGET_GATE == "AND"
        s_high = s_11
        s_low  = vcat(s_00, s_01, s_10)
    elseif TARGET_GATE == "OR"
        s_high = vcat(s_01, s_10, s_11)
        s_low  = s_00
    elseif TARGET_GATE == "NAND"
        s_high = vcat(s_00, s_01, s_10)
        s_low  = s_11
    elseif TARGET_GATE == "NOR"
        s_high = s_00
        s_low  = vcat(s_01, s_10, s_11)
    elseif TARGET_GATE == "NOT"
        s_high = vcat(s_00, s_01)
        s_low  = vcat(s_10, s_11)
    else
        s_high = vcat(s_01, s_10)
        s_low  = vcat(s_00, s_11)
    end
    
    high_mean = mean(s_high)
    low_mean  = mean(s_low)
    signal = high_mean - low_mean
    noise_total = std(s_high) + std(s_low)
    snr_std = (noise_total > 1e-6 && !isnan(noise_total)) ? (signal / noise_total) : 0.0
    
    diff_sd = std(s_high .- s_low)
    snr_hiscock = (diff_sd > 1e-6 && !isnan(diff_sd)) ? (mean(s_high .- s_low) / diff_sd) : 0.0
    
    snr_final = max(0.0, max(isnan(snr_std) ? 0.0 : snr_std, isnan(snr_hiscock) ? 0.0 : snr_hiscock))
    if isnan(snr_final) || isinf(snr_final)
        snr_final = 0.0
    end
    
    clean_val(v) = (isnan(v) || isinf(v)) ? 0.0 : Float64(v)
    
    output_file = joinpath(run_dir, "snr_results.json")
    open(output_file, "w") do f
        JSON.print(f, Dict(
            "snr" => clean_val(snr_final),
            "snr_xor" => clean_val(snr_final),
            "snr_standard" => clean_val(snr_std),
            "snr_hiscock" => clean_val(snr_hiscock),
            "mean_00" => clean_val(m00), "se_00" => clean_val(se00), "std_00" => clean_val(sd00),
            "mean_01" => clean_val(m01), "se_01" => clean_val(se01), "std_01" => clean_val(sd01),
            "mean_10" => clean_val(m10), "se_10" => clean_val(se10), "std_10" => clean_val(sd10),
            "mean_11" => clean_val(m11), "se_11" => clean_val(se11), "std_11" => clean_val(sd11),
            "samples_00" => s_00,
            "samples_01" => s_01,
            "samples_10" => s_10,
            "samples_11" => s_11
        ), 4)
    end
    
    return clean_val(snr_final), ""
end

println(stdout, "JULIA_DAEMON_READY")
flush(stdout)

eval_counter = 0

while true
    line = readline(stdin)
    if isempty(line) || strip(line) == "EXIT"
        break
    end
    
    global eval_counter += 1
    run_dir = String(strip(line))
    try
        snr, err_msg = Base.invokelatest(evaluate_test_run, run_dir)
        if isempty(err_msg)
            println(stdout, "SNR_RESULT:" * string(snr))
        else
            println(stdout, "SNR_RESULT:" * string(snr) * ":MSG:" * err_msg)
        end
    catch e
        println(stdout, "ERROR:" * string(e))
    end
    flush(stdout)
    
    # Coleta de lixo rápida periódica para manter a memória RAM compacta
    if eval_counter % 25 == 0
        GC.gc(false)
    end
end
