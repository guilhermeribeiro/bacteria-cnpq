# -*- coding: utf-8 -*-
"""
evaluate_candidate.py
==============================================================================
Avaliador Estocástico Rápido de Candidatos de Circuitos Genéticos (Julia SSA).

Permite avaliar qualquer matriz de fiação (wiring_matrix) e lista de componentes
usando o mesmo motor biofísico de simulação estocástica de Gillespie (JumpProcesses.jl)
utilizado na geração do dataset oficial.

Pode ser usado como:
1. Módulo Python (import evaluate_circuit):
   from evaluate_candidate import evaluate_circuit
   snr, results = evaluate_circuit(matrix_W, slots, gate="XOR", ensemble_size=32)

2. Linha de Comando (CLI):
   python evaluate_candidate.py --demo
   python evaluate_candidate.py --file circuito.json --gate XOR
==============================================================================
"""
from __future__ import print_function, division
import os
import sys
import json
import csv
import time
import shutil
import tempfile
import argparse
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(CURRENT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from julia_daemon import JuliaDaemonClient

def detect_julia():
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, r"AppData\Local\Programs\Julia-1.8.5\bin\julia.exe"),
        os.path.join(home, r"AppData\Local\Programs\Julia\bin\julia.exe"),
        r"C:\Julia-1.8.5\bin\julia.exe",
        r"C:\Julia\bin\julia.exe",
        "/opt/julia-1.8.5/bin/julia",
        "/opt/julia/bin/julia",
        os.path.expanduser("~/julia-1.8.5/bin/julia"),
        "/usr/local/bin/julia",
        "/usr/bin/julia"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    w_jl = shutil.which("julia") if hasattr(shutil, "which") else None
    return w_jl if w_jl else "julia"

def evaluate_circuit(matrix_W, slots, gate="XOR", ensemble_size=32, julia_exe=None, timeout=120):
    """
    Avalia um circuito genético candidato contra o simulador estocástico de Gillespie em Julia.
    
    Parâmetros:
      - matrix_W: Matriz de adjacência (N x N) como lista 2D ou np.ndarray.
      - slots: Lista com os nomes dos genes (ex: ['pTac', 'pTet', 'BetI', 'AmtR', 'YFP'])
               ou lista de dicionários contendo a chave 'winner'.
      - gate: Porta lógica alvo ('XOR', 'NOT', 'AND', etc.). Padrão: 'XOR'.
      - ensemble_size: Número de trajetórias estocásticas independentes por condição (Padrão: 32).
      - julia_exe: Caminho para o executável do Julia (opcional, detecta automaticamente).
      - timeout: Tempo máximo em segundos para a simulação (Padrão: 120s).
      
    Retorna:
      - snr: Float com o Signal-to-Noise Ratio pareado de Hiscock.
      - results: Dicionário completo contendo médias, desvios e amostras brutas de cada estado.
    """
    W = np.asarray(matrix_W, dtype=np.float64)
    N = W.shape[0]
    if W.shape[0] != W.shape[1]:
        raise ValueError("matrix_W deve ser quadrada (N x N). Formato atual: {}".format(W.shape))
        
    formatted_slots = []
    for i, s in enumerate(slots):
        if isinstance(s, dict):
            winner = s.get("winner", "Vazio")
        else:
            winner = str(s)
        formatted_slots.append({"slot": i, "winner": winner})
        
    if len(formatted_slots) < N:
        while len(formatted_slots) < N:
            formatted_slots.append({"slot": len(formatted_slots), "winner": "Vazio"})
    elif len(formatted_slots) > N:
        formatted_slots = formatted_slots[:N]
        
    if not julia_exe:
        julia_exe = detect_julia()
        
    proj_dir = os.path.join(CURRENT_DIR, "engine_julia", "ALIFE2023-master")
    daemon_script = os.path.join(CURRENT_DIR, "engine_julia", "evaluator_daemon.jl")
    
    tmp_dir = tempfile.mkdtemp(prefix="candidate_eval_")
    try:
        csv_file = os.path.join(tmp_dir, "wiring_matrix.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Node"] + ["N{}".format(i+1) for i in range(N)]
            writer.writerow(header)
            for i in range(N):
                row = ["N{}".format(i+1)] + [float(W[i, j]) for j in range(N)]
                writer.writerow(row)
                
        json_file = os.path.join(tmp_dir, "circuit_design.json")
        with open(json_file, "w") as f:
            json.dump({"slots": formatted_slots, "gate": gate}, f, indent=4)
            
        daemon = JuliaDaemonClient(
            julia_exe=julia_exe,
            proj_dir=proj_dir,
            daemon_script=daemon_script,
            num_threads=4,
            ensemble_size=ensemble_size,
            gate=gate
        )
        
        snr_val, snr_data, elapsed, err_msg = daemon.evaluate(tmp_dir, timeout=timeout)
        daemon.close()
        
        if err_msg:
            print("[AVISO] Simulador retornou mensagem: {}".format(err_msg), file=sys.stderr)
            
        return snr_val, snr_data
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def run_demo():
    print("=" * 70)
    print("DEMO: Avaliação Biofísica Estocástica de Circuito Campeão XOR (Julia SSA)")
    print("=" * 70)
    
    # Circuito Campeão XOR (S=5: pTac, pTet, PhlF, AmtR, YFP)
    slots = ["pTac", "pTet", "PhlF", "AmtR", "YFP"]
    matrix_W = [
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    
    print("Genes:", slots)
    print("Matriz W (5x5):")
    for row in matrix_W:
        print(" ", row)
        
    print("\nIniciando simulação de trajetórias de Gillespie SSA (Ensemble=32)...")
    t0 = time.time()
    snr, res = evaluate_circuit(matrix_W, slots, gate="XOR", ensemble_size=32)
    elapsed = time.time() - t0
    
    print("\n" + "=" * 70)
    print("RESULTADO DA SIMULAÇÃO (Concluído em {:.2f}s):".format(elapsed))
    print("  SNR Pareado (Hiscock): {:.4f}".format(snr))
    if res:
        print("  Média Estado 00 (Baixo): {:.2f}".format(res.get("mean_00", 0.0)))
        print("  Média Estado 01 (Alto):  {:.2f}".format(res.get("mean_01", 0.0)))
        print("  Média Estado 10 (Alto):  {:.2f}".format(res.get("mean_10", 0.0)))
        print("  Média Estado 11 (Baixo): {:.2f}".format(res.get("mean_11", 0.0)))
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliador Estocástico Julia SSA para Circuitos Genéticos")
    parser.add_argument("--demo", action="store_true", help="Executa a avaliação de um circuito de demonstração")
    parser.add_argument("--file", type=str, default=None, help="Caminho para arquivo JSON contendo 'matrix_W' e 'slots'")
    parser.add_argument("--gate", type=str, default="XOR", choices=["XOR", "NOT", "AND", "OR"], help="Porta lógica alvo")
    parser.add_argument("--ensemble", type=int, default=32, help="Tamanho do ensemble estocástico SSA (default: 32)")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.file:
        if not os.path.exists(args.file):
            print("Erro: Arquivo não encontrado: {}".format(args.file), file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r") as f:
            data = json.load(f)
        w = data.get("matrix_W", data.get("wiring_matrix", []))
        s = data.get("slots", [])
        snr, res = evaluate_circuit(w, s, gate=args.gate, ensemble_size=args.ensemble)
        print("SNR {}: {:.4f}".format(args.gate, snr))
        if res:
            print("Médias de Estado: 00={:.2f}, 01={:.2f}, 10={:.2f}, 11={:.2f}".format(
                res.get("mean_00", 0.0), res.get("mean_01", 0.0), res.get("mean_10", 0.0), res.get("mean_11", 0.0)
            ))
    else:
        parser.print_help()
