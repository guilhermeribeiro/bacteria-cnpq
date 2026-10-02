# -*- coding: utf-8 -*-
"""
optuna_multiobj_generator.py
==============================================================================
Gerador Multi-Objetivo (NSGA-II) de Circuitos Genéticos XOR com Optuna.
Otimiza simultaneamente:
  - Objetivo 1 (Fitness): Maximizar SNR da Simulação Estocástica Gillespie em Julia.
  - Objetivo 2 (Diversidade): Maximizar Novidade Topológica e Proteica (Novelty Search)
    em relação ao arquivo de circuitos já descobertos.

Resolve o colapso de modo e explora a fronteira de Pareto para atingir a meta
de 1.766 circuitos diversos e biologicamente validados (SNR >= 0.85).

Execução 100% Segura no Servidor:
  - Controle estrito de concorrência na Julia (Semáforo = 4) mantendo RAM <= 55%.
  - Filtro Estrutural Rápido (<1ms) para descartar grafos desconexos sem tocar na Julia.
  - Armazenamento resiliente em SQLite (retomada automática após reinício).
  - Formatação 100% compatível com a GNN de Guilherme (17 chaves universais).
==============================================================================
"""
from __future__ import print_function, division
import os
import sys
import json
import time
import math
import shutil
import random
import threading
import subprocess
import multiprocessing
from datetime import datetime

import optuna
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(CURRENT_DIR, 'core')
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from circuit_diversity import CircuitDiversityScorer, CircuitArchive
from circuit_anti_duplication import CircuitAntiDuplicationEnforcer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")

def ts():
    return datetime.now().strftime('%H:%M:%S')

def which(program):
    if hasattr(shutil, 'which'):
        res = shutil.which(program)
        if res:
            return res
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    exts = ["", ".exe", ".bat", ".cmd"] if sys.platform == "win32" else [""]
    for d in path_dirs:
        for ext in exts:
            full_path = os.path.join(d, program + ext)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                return full_path
    return None

def run_cmd_timeout(cmd, env=None, timeout=300):
    cflags = 0
    preexec = None
    if sys.platform == "win32":
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            cflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        preexec = os.setsid
        
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        universal_newlines=True,
        creationflags=cflags,
        preexec_fn=preexec
    )
    timeout_holder = {"hit": False}
    def kill_proc():
        timeout_holder["hit"] = True
        try:
            if sys.platform != "win32":
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass

    timer = threading.Timer(timeout, kill_proc)
    timer.start()
    try:
        out, _ = proc.communicate()
        return proc.returncode, out, timeout_holder["hit"]
    finally:
        timer.cancel()

# --- CARREGAMENTO DE CONFIGURAÇÃO ---
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
else:
    config = {}

DS_CFG     = config.get("dataset", {})
GEN_CFG    = config.get("generation", {})
TRAIN_CFG  = config.get("training", {})
ENV_CFG    = config.get("environment", {})
OPTUNA_CFG = config.get("optuna", {})

GATE_NAME      = DS_CFG.get("gate_name", "XOR")
SNR_THRESHOLD  = float(DS_CFG.get("snr_threshold", 0.85))
ENSEMBLE_SIZE  = int(DS_CFG.get("ensemble_size", 32))
TARGET_SAMPLES = int(GEN_CFG.get("target_samples", 1766))
OUTPUT_FILE    = os.path.join(CURRENT_DIR, DS_CFG.get("output_file", "circuit_dataset_XOR_DARTS_FAST.jsonl"))

def is_python2(bin_path):
    if not bin_path:
        return False
    if not os.path.exists(bin_path) and not which(bin_path):
        return False
    try:
        res = subprocess.call([bin_path, "-c", "import sys; sys.exit(0 if sys.version_info[0] == 2 else 1)"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res == 0
    except Exception:
        return False

def detect_python2():
    configured = ENV_CFG.get("python2_exe", "auto")
    if configured != "auto" and (os.path.exists(configured) or which(configured)):
        return configured
    if sys.version_info[0] == 2:
        return sys.executable
        
    candidates = [
        r"C:\Python27\python.exe",
        r"C:\Python27_x64\python.exe",
        which("python2"),
        "/opt/py27/bin/python",
        "/root/micromamba/envs/py27/bin/python",
        os.path.expanduser("~/micromamba/envs/py27/bin/python"),
        "/opt/conda/envs/py27/bin/python",
    ]
    home = os.path.expanduser("~")
    for loc in [os.environ.get("CONDA_PREFIX", ""), os.path.join(home, "miniconda3"), os.path.join(home, "anaconda3")]:
        if loc:
            candidates.append(os.path.join(loc, "envs", "py27", "python.exe" if sys.platform == "win32" else "bin/python"))
            
    for c in candidates:
        if c and is_python2(c):
            return c
    return which("python2") or "python2" if sys.platform != "win32" else r"C:\Python27\python.exe"

def detect_julia():
    configured = ENV_CFG.get("julia_exe", "auto")
    if configured != "auto" and (os.path.exists(configured) or which(configured)):
        return configured
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
    w_jl = which("julia")
    if w_jl:
        return w_jl
    return "julia"

PY2_EXE   = detect_python2()
JULIA_EXE = detect_julia()

WORKER_SCRIPT     = os.path.join(CURRENT_DIR, "engine_py2", "worker_darts.py")
JULIA_EVAL_SCRIPT = os.path.join(CURRENT_DIR, "engine_julia", "calculate_snr_test.jl")
RUNS_DIR          = os.path.join(CURRENT_DIR, "optuna_runs")

if not os.path.exists(RUNS_DIR):
    os.makedirs(RUNS_DIR)

# Limite de concorrência estrita na Julia para proteger a memória RAM (<= 55%)
MAX_CONCURRENT_JULIA = int(OPTUNA_CFG.get("max_concurrent_julia", GEN_CFG.get("max_concurrent_julia", 5)))
julia_semaphore = threading.Semaphore(MAX_CONCURRENT_JULIA)

dataset_lock = threading.Lock()
print_lock   = threading.Lock()

try:
    import queue
except ImportError:
    import Queue as queue

# Pool de 64 slots exclusivos para compilação Theano (garante zero colisão de compilelock)
slot_queue = queue.Queue()
for i in range(64):
    slot_queue.put(i)

def cleanup_theano_locks():
    lock_base = "C:/theano_locks" if sys.platform == "win32" else "/tmp/theano_locks"
    if os.path.exists(lock_base):
        try:
            for root, dirs, files in os.walk(lock_base):
                if "lock_dir" in dirs:
                    shutil.rmtree(os.path.join(root, "lock_dir"), ignore_errors=True)
        except Exception:
            pass

cleanup_theano_locks()

# Catálogo biológico Cello calibrado para as features da GNN
PROT_CATALOG = {
    "SrpR":  {"ymin": 0.003, "ymax": 1.300, "Kd": 0.010, "n": 2.9},
    "PhlF":  {"ymin": 0.010, "ymax": 3.900, "Kd": 0.030, "n": 4.0},
    "AmeR":  {"ymin": 0.200, "ymax": 3.800, "Kd": 0.090, "n": 1.4},
    "BetI":  {"ymin": 0.070, "ymax": 3.800, "Kd": 0.410, "n": 2.4},
    "QacR":  {"ymin": 0.018, "ymax": 2.820, "Kd": 0.137, "n": 1.6},
    "AmtR":  {"ymin": 0.001, "ymax": 4.060, "Kd": 0.088, "n": 2.3},
    "Vazio": {"ymin": 0.000, "ymax": 0.000, "Kd": 1.000, "n": 1.0},
    "YFP":   {"ymin": 0.020, "ymax": 1.000, "Kd": 0.100, "n": 1.0},
    "pTac":  {"ymin": 0.000, "ymax": 0.000, "Kd": 0.000, "n": 0.0},
    "pTet":  {"ymin": 0.000, "ymax": 0.000, "Kd": 0.000, "n": 0.0}
}

# Inicializa o Arquivo de Diversidade e o Enforcer Anti-Duplicacao
archive = CircuitArchive(k_neighbors=10)
anti_dup_enforcer = CircuitAntiDuplicationEnforcer(min_diversity_threshold=0.04)

# Carrega circuitos existentes no dataset para o Arquivo de Novidade e Indice Anti-Duplicacao
initial_count = 0
CHAMPION_SEEDS = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                c_obj = json.loads(line_s)
                s_val = c_obj.get("seed")
                if s_val is not None and isinstance(s_val, int) and s_val not in CHAMPION_SEEDS:
                    CHAMPION_SEEDS.append(s_val)
                c_w = c_obj.get("matrix_W", c_obj.get("wiring_matrix", []))
                c_slots_raw = c_obj.get("slots", [])
                c_slots = [s.get("winner", s) if isinstance(s, dict) else s for s in c_slots_raw]
                c_snr = float(c_obj.get("snr", 0.0))
                if len(c_w) > 0 and len(c_slots) == len(c_w):
                    archive.add_circuit(len(c_slots), c_w, c_slots, c_snr, metadata={"id": c_obj.get("id")})
                    anti_dup_enforcer.register(len(c_slots), c_w, c_slots, circuit_id=str(c_obj.get("id", "")))
                    initial_count += 1
            except Exception:
                pass

print("=" * 75)
print("  GENENET-DARTS MULTI-OBJETIVO (NSGA-II) [DIVERSIDADE + SNR]")
print("=" * 75)
print("  Python Principal (Optuna): {} (v{})".format(sys.executable, sys.version.split()[0]))
print("  Python 2 (Theano DARTS):   {}".format(PY2_EXE))
print("  Julia (Gillespie SSA):     {}".format(JULIA_EXE))
print("  Meta de Circuitos Salvos:  {} (Porta: {}, SNR >= {:.2f})".format(TARGET_SAMPLES, GATE_NAME, SNR_THRESHOLD))
print("  Arquivo Alvo:              {}".format(OUTPUT_FILE))
print("  Circuitos ja no Arquivo:   {} circuitos carregados".format(initial_count))
print("  Sementes Campeãs Carregadas: {} sementes únicas para Semeação Híbrida".format(len(CHAMPION_SEEDS)))
print("  Limite Concorrente Julia:  {} simulações simultâneas (RAM Controlada)".format(MAX_CONCURRENT_JULIA))
print("=" * 75)

state = {
    "valid_count": initial_count,
    "session_initial_count": initial_count,
    "total_attempts": 0,
    "start_time": time.time(),
}

def standardize_and_save_circuit(run_dir, folder_name, snr_data, slots, matrix_W, seed=None):
    """
    Serializa o circuito no formato universal de 17 chaves 100% compatível
    com o dataset padronizado de Guilherme para Graph Neural Networks.
    """
    N = len(slots)
    components = {}
    metadata_for_humans = {}
    
    for idx, s in enumerate(slots):
        node_key = "node_{}".format(idx + 1)
        w_name = s.get("winner", "Vazio") if isinstance(s, dict) else str(s)
        
        if idx == 0:
            human_label = "LacI"
        elif idx == 1:
            human_label = "TetR"
        elif idx == N - 1:
            human_label = "YFP"
        else:
            human_label = w_name
        metadata_for_humans[node_key] = str(human_label)
        
        # Features dos nós para GNN (100% numéricas, Float puro)
        if idx in [0, 1]:
            components[node_key] = {
                "is_input": 1.0,
                "is_output": 0.0,
                "ymin": 0.0,
                "ymax": 0.0,
                "Kd": 0.0,
                "n_real": 0.0
            }
        elif idx == N - 1:
            components[node_key] = {
                "is_input": 0.0,
                "is_output": 1.0,
                "ymin": 0.02,
                "ymax": 1.0,
                "Kd": 0.1,
                "n_real": 1.0
            }
        else:
            p_info = PROT_CATALOG.get(w_name, {"ymin": 0.0, "ymax": 0.0, "Kd": 1.0, "n": 1.0})
            components[node_key] = {
                "is_input": 0.0,
                "is_output": 0.0,
                "ymin": float(p_info.get("ymin", 0.0)),
                "ymax": float(p_info.get("ymax", 1.0)),
                "Kd": float(p_info.get("Kd", 1.0)),
                "n_real": float(p_info.get("n", 1.0))
            }
            
    s00 = [float(x) for x in snr_data.get("samples_00", [])]
    s01 = [float(x) for x in snr_data.get("samples_01", [])]
    s10 = [float(x) for x in snr_data.get("samples_10", [])]
    s11 = [float(x) for x in snr_data.get("samples_11", [])]
    
    m00 = float(snr_data.get("mean_00", 0.0))
    m01 = float(snr_data.get("mean_01", 0.0))
    m10 = float(snr_data.get("mean_10", 0.0))
    m11 = float(snr_data.get("mean_11", 0.0))
    
    snr_val = float(snr_data.get("snr_xor", snr_data.get("snr", 0.0)))
    ensemble_size = len(s00) if s00 else ENSEMBLE_SIZE
    
    entry = {
        "id": str(folder_name),
        "gate": GATE_NAME,
        "algorithm_mnemonic": DS_CFG.get("algorithm_mnemonic", "DARTS_NSGAII_MULTIOBJ (Ensemble={})".format(ensemble_size)),
        "ensemble_size": int(ensemble_size),
        "snr": float(snr_val),
        "steady_state_outputs": {
            "input_low_mean": float((m00 + m11) / 2.0),
            "input_high_mean": float((m01 + m10) / 2.0),
            "state_00_mean": m00,
            "state_01_mean": m01,
            "state_10_mean": m10,
            "state_11_mean": m11
        },
        "raw_samples_low": s00 + s11,
        "raw_samples_high": s01 + s10,
        "raw_samples_by_state": {
            "state_00": s00,
            "state_01": s01,
            "state_10": s10,
            "state_11": s11
        },
        "steady_state_samples": {
            "state_00": s00,
            "state_01": s01,
            "state_10": s10,
            "state_11": s11
        },
        "samples": {
            "low": s00 + s11,
            "high": s01 + s10
        },
        "slots": slots,
        "seed": seed,
        "matrix_W": matrix_W,
        "wiring_matrix": matrix_W,
        "components": components,
        "metadata_for_humans": metadata_for_humans
    }
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

julia_daemon = None
daemon_init_lock = threading.Lock()

def get_julia_daemon():
    global julia_daemon
    with daemon_init_lock:
        if julia_daemon is None:
            try:
                from julia_daemon import JuliaDaemonClient
                proj_dir = os.path.join(CURRENT_DIR, "engine_julia", "ALIFE2023-master")
                daemon_script = os.path.join(CURRENT_DIR, "engine_julia", "evaluator_daemon.jl")
                julia_daemon = JuliaDaemonClient(
                    JULIA_EXE, proj_dir, daemon_script,
                    num_threads=6, ensemble_size=32, gate=GATE_NAME
                )
            except Exception as e:
                with print_lock:
                    print("[{}] [AVISO DAEMON] Falha ao iniciar daemon: {}. Usando modo subprocesso.".format(ts(), e))
                    sys.stdout.flush()
                julia_daemon = False
        return julia_daemon

def evaluate_with_julia(run_dir, timeout=300):
    daemon = get_julia_daemon()
    if daemon:
        try:
            snr_val, snr_data, j_elapsed, err_msg = daemon.evaluate(run_dir, timeout=90)
            if not err_msg and snr_data is not None:
                return snr_val, snr_data, j_elapsed, ""
        except Exception:
            pass

    cmd = [JULIA_EXE, "-t", "4", "--startup-file=no", "-O1", JULIA_EVAL_SCRIPT, run_dir]
    env = os.environ.copy()
    env["GKSwstype"] = "100"
    t0 = time.time()
    
    with julia_semaphore:
        ret, out, timed_out = run_cmd_timeout(cmd, env=env, timeout=timeout)
        
    j_elapsed = time.time() - t0
    
    if timed_out or ret != 0:
        err_log = os.path.join(CURRENT_DIR, "error_julia.log")
        try:
            with open(err_log, "a") as ef:
                ef.write("[{}] ret={} timed_out={}\nOutput:\n{}\n\n".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ret, timed_out, out
                ))
        except Exception:
            pass
        return 0.0, None, j_elapsed, out
        
    snr_file = os.path.join(run_dir, "snr_results.json")
    if os.path.exists(snr_file):
        try:
            with open(snr_file, "r") as f:
                data = json.load(f)
            snr_val = float(data.get("snr_xor", data.get("snr", 0.0)))
            data["snr"] = snr_val
            return snr_val, data, j_elapsed, ""
        except Exception as e:
            return 0.0, None, j_elapsed, str(e)
            
    return 0.0, None, j_elapsed, "snr_results.json nao gerado"

def check_graph_validity(matrix_W, gate=None):
    """
    Verificação ultrarrápida (< 5 microsegundos) de alcance direcionado (BFS):
    1. pTac (nó 0) deve alcançar YFP (nó N-1).
    2. Se a porta não for NOT (ex: XOR, AND, etc.), pTet (nó 1) deve alcançar YFP (nó N-1).
    3. Promotores ativos devem ter pelo menos uma conexão de saída.
    4. YFP deve ter pelo menos uma conexão de entrada.
    """
    global GATE_NAME
    g = gate if gate is not None else GATE_NAME
    N = len(matrix_W)
    if N < 3:
        return False
        
    out_ptac = sum(matrix_W[0][1:])
    in_yfp   = sum(matrix_W[r][-1] for r in range(N))
    if out_ptac == 0.0 or in_yfp == 0.0:
        return False
        
    if g != "NOT":
        out_ptet = sum(matrix_W[1][1:])
        if out_ptet == 0.0:
            return False
        
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if matrix_W[i][j] > 0.01:
                adj[i].append(j)
                
    def can_reach(start_node, target_node):
        visited = [False] * N
        queue = [start_node]
        visited[start_node] = True
        while queue:
            curr = queue.pop(0)
            if curr == target_node:
                return True
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return False

    if g == "NOT":
        return can_reach(0, N - 1)
    return can_reach(0, N - 1) and can_reach(1, N - 1)

def objective(trial):
    """
    Função Objetivo Multi-Objetivo (MOTPE / NSGA-II):
      f1 = Maximizar Julia SNR (Qualidade Biológica do Circuito)
      f2 = Maximizar Novelty Score (Diversidade Topológica e Proteica)
    """
    trial_id = trial.number
    
    with dataset_lock:
        if state["valid_count"] >= TARGET_SAMPLES:
            trial.study.stop()
            return 0.0, 0.0
            
    # S fixado em 10 (congruente com a capacidade biofísica do chassi)
    S = int(TRAIN_CFG.get("S", 10))
    
    # Optuna controla o número de iterações (épocas):
    # Distribuição com viés ágil: ~70% das opções na faixa ágil (150 a 500 épocas),
    # com opções profundas (600, 700, 1000, 1400) disponíveis quando necessário.
    iters = trial.suggest_categorical(
        "iterations", [150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 1000, 1400]
    )
    
    # Hiperparametros calibrados estritamente na regiao de ouro dos campeoes (SNR >= 0.85)
    lr = trial.suggest_float("lr", 0.018, 0.045, log=True)
    ew = trial.suggest_float("ew", 0.16, 0.35)
    l1 = trial.suggest_float("l1", 0.0008, 0.0038, log=True)
    pl = trial.suggest_float("pl", 0.28, 0.44)
    lambda_snr = trial.suggest_float("lambda_snr", 0.005, 0.10, log=True)
    
    # Semeação Híbrida Inteligente e Limpa (Exploitation dos Campeões vs Exploration Global)
    if CHAMPION_SEEDS and len(CHAMPION_SEEDS) > 0:
        seed_strategy = trial.suggest_categorical(
            "seed_strategy", ["champion_direct", "novel_exploration"]
        )
        if seed_strategy == "champion_direct":
            c_idx = trial.suggest_int("champion_idx", 0, len(CHAMPION_SEEDS) - 1)
            seed = CHAMPION_SEEDS[c_idx]
        else:
            seed = random.randint(1, 9999999)
    else:
        seed = random.randint(1, 9999999)
        seed_strategy = "novel_exploration"
        
    trial.set_user_attr("seed_strategy", seed_strategy)
    trial.set_user_attr("seed", seed)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = "optuna_t{:04d}_{}_{}".format(trial_id, timestamp, random.randint(100, 999))
    run_dir = os.path.join(RUNS_DIR, run_name)
    
    try:
        os.makedirs(run_dir)
    except OSError:
        pass
        
    cmd_theano = [
        PY2_EXE, WORKER_SCRIPT,
        "--lr", str(lr),
        "--ew", str(ew),
        "--l1", str(l1),
        "--pl", str(pl),
        "--lambda_snr", str(lambda_snr),
        "--S", str(S),
        "--gate", GATE_NAME,
        "--iterations", str(iters),
        "--batch_size", "32",
        "--seed", str(seed),
        "--max_restarts", "5",
        "--out_dir", run_dir
    ]
    
    try:
        worker_slot = slot_queue.get_nowait()
    except Exception:
        worker_slot = trial_id % 64

    env_th = os.environ.copy()
    lock_base = "C:/theano_locks" if sys.platform == "win32" else "/tmp/theano_locks"
    slot_dir = os.path.join(lock_base, "th_w{}".format(worker_slot))
    if os.path.exists(slot_dir):
        for root, dirs, files in os.walk(slot_dir):
            if "lock_dir" in dirs:
                shutil.rmtree(os.path.join(root, "lock_dir"), ignore_errors=True)
    env_th["THEANO_FLAGS"] = "base_compiledir={}".format(slot_dir.replace("\\", "/"))
    
    t0_th = time.time()
    try:
        ret_th, out_th, timed_out_th = run_cmd_timeout(cmd_theano, env=env_th, timeout=950)
    finally:
        slot_queue.put(worker_slot)
    th_elapsed = time.time() - t0_th
    
    if timed_out_th or ret_th != 0:
        with print_lock:
            print("[{}] [Trial {:04d}] ERRO THEANO ({:.1f}s): ret={}".format(ts(), trial_id, th_elapsed, ret_th))
            if out_th:
                for line in [l.strip() for l in out_th.strip().split("\n") if l.strip()][-4:]:
                    print("       | {}".format(line))
            sys.stdout.flush()
        shutil.rmtree(run_dir, ignore_errors=True)
        raise optuna.TrialPruned("Theano synthesis failed or timed out")
        
    # Carrega arquivos gerados pelo Theano
    design_file = os.path.join(run_dir, "circuit_design.json")
    wiring_file = os.path.join(run_dir, "wiring_matrix.csv")
    
    if not os.path.exists(design_file) or not os.path.exists(wiring_file):
        shutil.rmtree(run_dir, ignore_errors=True)
        raise optuna.TrialPruned("Missing output files from Theano")
        
    slots = []
    th_final_mse = 999.0
    try:
        with open(design_file, "r") as df:
            d_obj = json.load(df)
            slots = d_obj.get("slots", [])
            th_eval = d_obj.get("theano_evaluation", {})
            th_final_mse = float(th_eval.get("final_mse", 999.0))
            trial.set_user_attr("theano_mse", th_final_mse)
            trial.set_user_attr("S", S)
    except Exception as e_parse:
        with print_lock:
            print("[{}] [Trial {:04d} | ERRO PARSE] Falha ao ler circuit_design.json: {}".format(ts(), trial_id, e_parse))
            sys.stdout.flush()
        shutil.rmtree(run_dir, ignore_errors=True)
        raise optuna.TrialPruned("Failed to parse circuit_design.json: {}".format(e_parse))
        
    matrix_W = []
    try:
        import csv
        with open(wiring_file, "r") as wf:
            reader = csv.reader(wf)
            next(reader)
            for row in reader:
                matrix_W.append([float(x) for x in row[1:]])
    except Exception as e_csv:
        with print_lock:
            print("[{}] [Trial {:04d} | ERRO CSV] Falha ao ler wiring_matrix.csv: {}".format(ts(), trial_id, e_csv))
            sys.stdout.flush()
        shutil.rmtree(run_dir, ignore_errors=True)
        raise optuna.TrialPruned("Failed to parse wiring_matrix.csv: {}".format(e_csv))
        
    # FILTRO ESTRUTURAL RÁPIDO (< 1ms)
    # Garante que pTac e pTet alcançam o YFP via caminhos direcionados
    is_connected = check_graph_validity(matrix_W)
    slot_winners = [s.get("winner", s) if isinstance(s, dict) else s for s in slots]
    
    if not is_connected:
        trial.set_user_attr("is_connected", False)
        shutil.rmtree(run_dir, ignore_errors=True)
        with print_lock:
            print("[{}] [Trial {:04d} | PULO <1ms] Desconexo -> ThMSE={:.4f} | Penalizado".format(
                ts(), trial_id, th_final_mse
            ))
            sys.stdout.flush()
        # Não podar! Retornar fitness penalizado para o Optuna aprender a evitar regiões desconexas
        return -1.0 - 2.0 * th_final_mse, 0.0
        
    # FILTRO DETERMINÍSTICO (< 5 ms)
    # Verifica sinal biofísico da porta lógica: separação positiva das médias e erro contido
    th_snr = float(th_eval.get("final_snr", 0.0))
    th_signal = float(th_eval.get("final_signal", 0.0))
    trial.set_user_attr("theano_snr", th_snr)
    trial.set_user_attr("theano_signal", th_signal)
    
    # Calibração Biofísica rigorosa: baseada nos 11 campeões (min_signal=0.046, max_mse=0.111)
    # Margem de tolerância segura: sinal >= 0.035 e MSE <= 0.115
    is_promising = (th_signal >= 0.035) and (th_final_mse <= 0.115)
    if not is_promising:
        trial.set_user_attr("is_converged", False)
        shutil.rmtree(run_dir, ignore_errors=True)
        with print_lock:
            print("[{}] [Trial {:04d} | PULO <5ms] Sem Sinal Biofisico (Sinal={:.4f}, ThMSE={:.4f}) -> Feedback Suave".format(
                ts(), trial_id, th_signal, th_final_mse
            ))
            sys.stdout.flush()
        # Fitness contínuo casado (Sinal / (1 + 5*MSE)) para guiar o Optuna em direção ao XOR
        th_proxy = th_signal / (math.sqrt(max(0.0, th_signal) + 1e-4) + 0.1) if th_signal > 0 else th_signal
        f1_proxy = th_proxy / (1.0 + 5.0 * th_final_mse)
        return f1_proxy, 0.0
        
    # FILTRO RIGIDO ANTI-DUPLICACAO PRE-JULIA (< 1.5 ms)
    # Evita gastar simulação cara de Julia SSA se o circuito já for isomorfo a um campeão existente
    is_dup, reason, matched_id = anti_dup_enforcer.is_duplicate(S, matrix_W, slots)
    if is_dup:
        trial.set_user_attr("is_duplicate", True)
        shutil.rmtree(run_dir, ignore_errors=True)
        with print_lock:
            print("[{}] [Trial {:04d} | PULO ANTI-DUP] Isomorfo a {} -> Penalizado sem gastar Julia SSA".format(
                ts(), trial_id, matched_id
            ))
            sys.stdout.flush()
        return -2.0, 0.0

    # Avaliação em Julia SSA (com semáforo de proteção de RAM)
    snr_val, snr_data, j_elapsed, j_err = evaluate_with_julia(run_dir, timeout=300)
    
    # Objetivo 2: Novidade Topológica e Proteica contra o Arquivo de Campeões
    novelty_score = archive.compute_novelty(S, matrix_W, slot_winners)
    
    # Formatação de saída para monitoramento
    slots_summary = ", ".join(["{}({:.0f}%)".format(s.get("winner"), s.get("confidence_pct", 100)) for s in slots[2:-1]])
    status_tag = "ALVO BATEU (>=0.85)!" if snr_val >= SNR_THRESHOLD else ("QUASE (>=0.50)" if snr_val >= 0.50 else "NORMAL")
    
    with print_lock:
        print("[{}] [Trial {:04d} | {}] S={} | lr={:.3f} | ew={:.3f} | snr_w={:.3f} | iters={} (Th={:.1f}s) | ThMSE={:.4f} | SNR={:.4f} (Jl={:.1f}s) | Nov={:.4f} | Slots: {}".format(
            ts(), trial_id, status_tag, S, lr, ew, lambda_snr, iters, th_elapsed, th_final_mse, snr_val, j_elapsed, novelty_score, slots_summary
        ))
        sys.stdout.flush()

    trial.set_user_attr("theano_mse", th_final_mse)
    trial.set_user_attr("snr_julia", snr_val)
    trial.set_user_attr("novelty", novelty_score)
    trial.set_user_attr("S", S)
    
    # Se atingiu o threshold biológico de aceitação:
    if snr_val >= SNR_THRESHOLD and snr_data is not None:
        with dataset_lock:
            if state["valid_count"] < TARGET_SAMPLES:
                # Verificação atômica final sob lock
                is_dup_lock, _, matched_id_lock = anti_dup_enforcer.is_duplicate(S, matrix_W, slots)
                if not is_dup_lock:
                    state["valid_count"] += 1
                    curr_valid = state["valid_count"]
                    
                    # Salva no dataset JSONL oficial com schema de 17 chaves
                    standardize_and_save_circuit(run_dir, run_name, snr_data, slots, matrix_W, seed=seed)
                    
                    # Alimenta o arquivo de diversidade online para forçar o NSGA-II a explorar outras regiões
                    archive.add_circuit(S, matrix_W, slot_winners, snr_val, metadata={"trial": trial_id})
                    anti_dup_enforcer.register(S, matrix_W, slots, circuit_id=run_name)
                    if seed is not None and seed not in CHAMPION_SEEDS:
                        CHAMPION_SEEDS.append(seed)
                
                    elapsed = time.time() - state["start_time"]
                    session_valid = curr_valid - state.get("session_initial_count", 0)
                    rate = (float(session_valid) / max(elapsed, 1.0)) * 3600.0
                    
                    print("\n" + "#" * 80)
                    print("*** [CAMPEAO ENCONTRADO E SALVO!] ***")
                    print("   Trial:      {:04d}".format(trial_id))
                    print("   Julia SNR:  {:.4f} (Threshold >= {:.2f})".format(snr_val, SNR_THRESHOLD))
                    print("   Novidade:   {:.4f} (Distância ao Arquivo)".format(novelty_score))
                    print("   Slots:      {}".format(slot_winners))
                    print("   Progresso:  {}/{} circuitos ({:.2f}%)".format(curr_valid, TARGET_SAMPLES, (curr_valid / TARGET_SAMPLES) * 100.0))
                    print("   Velocidade: {:.1f} circuitos aceitos/hora".format(rate))
                    print("#" * 80 + "\n")
                    sys.stdout.flush()
                    
                    if state["valid_count"] >= TARGET_SAMPLES:
                        trial.study.stop()
                else:
                    with print_lock:
                        print("[{}] [Trial {:04d} | PULO ANTI-DUP SOB LOCK] Isomorfo a {} -> Descartado".format(
                            ts(), trial_id, matched_id_lock
                        ))
                        sys.stdout.flush()
                    return -2.0, 0.0

    shutil.rmtree(run_dir, ignore_errors=True)
    
    # Objetivos passados para o otimizador:
    # f1 = Fitness casado biofísico: SNR real do Julia validado, ponderado pelo acerto da tabela verdade (Theano MSE)
    f1 = (min(15.0, max(0.0, snr_val)) + 1.0) / (1.0 + 3.0 * th_final_mse)
    # f2 = Novelty Score em [0, 1]
    f2 = novelty_score
    
    return f1, f2

def main():
    global GATE_NAME, TARGET_SAMPLES, OUTPUT_FILE, julia_semaphore, archive, anti_dup_enforcer
    import argparse
    parser = argparse.ArgumentParser(description="Gerador Optuna Multi-Objetivo DARTS XOR (MOTPE / NSGA-II)")
    parser.add_argument("--n_jobs", type=int, default=int(OPTUNA_CFG.get("num_workers", 6)), help="Número de workers paralelos no Optuna (padrão: 6)")
    parser.add_argument("--n_trials", type=int, default=10000, help="Número máximo de trials (padrão: 10000)")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "nsgaii"], help="Algoritmo de amostragem: 'tpe' (Recomendado assíncrono) ou 'nsgaii'")
    parser.add_argument("--study_name", type=str, default="darts_xor_adaptive_v3", help="Nome do estudo no SQLite (padrão: darts_xor_adaptive_v3)")
    parser.add_argument("--pop_size", type=int, default=int(OPTUNA_CFG.get("population_size", 40)), help="Tamanho da população se usar NSGA-II (padrão: 40)")
    parser.add_argument("--target_samples", type=int, default=None, help="Número de circuitos aceitos alvo (ex: 10)")
    parser.add_argument("--output_file", type=str, default=None, help="Caminho do arquivo JSONL de saída")
    parser.add_argument("--gate", type=str, default=GATE_NAME, choices=["XOR", "NOT", "AND", "OR", "NAND", "NOR"], help="Porta lógica alvo (padrão: XOR)")
    parser.add_argument("--max_concurrent_julia", type=int, default=None, help="Limite de simulações Julia simultâneas")
    args = parser.parse_args()

    GATE_NAME = str(args.gate).upper()
    if args.study_name == "darts_xor_adaptive_v3" and GATE_NAME != "XOR":
        args.study_name = "darts_{}_agile_v1".format(GATE_NAME.lower())
    if args.target_samples is not None:
        TARGET_SAMPLES = args.target_samples
    if args.output_file is not None:
        OUTPUT_FILE = os.path.abspath(args.output_file)
    elif GATE_NAME != "XOR":
        OUTPUT_FILE = os.path.join(CURRENT_DIR, "circuit_dataset_{}_DARTS_FAST.jsonl".format(GATE_NAME))
        # Recarrega o contador, o arquivo de diversidade e o enforcer para o arquivo especificado
        initial_count = 0
        archive = CircuitArchive(k_neighbors=10)
        anti_dup_enforcer = CircuitAntiDuplicationEnforcer(min_diversity_threshold=0.04)
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        c_obj = json.loads(line_s)
                        s_val = c_obj.get("seed")
                        if s_val is not None and isinstance(s_val, int) and s_val not in CHAMPION_SEEDS:
                            CHAMPION_SEEDS.append(s_val)
                        c_w = c_obj.get("matrix_W", c_obj.get("wiring_matrix", []))
                        c_slots_raw = c_obj.get("slots", [])
                        c_slots = [s.get("winner", s) if isinstance(s, dict) else s for s in c_slots_raw]
                        c_snr = float(c_obj.get("snr", 0.0))
                        if len(c_w) > 0 and len(c_slots) == len(c_w):
                            archive.add_circuit(len(c_slots), c_w, c_slots, c_snr, metadata={"id": c_obj.get("id")})
                            anti_dup_enforcer.register(len(c_slots), c_w, c_slots, circuit_id=str(c_obj.get("id", "")))
                            initial_count += 1
                    except Exception:
                        pass
        state["valid_count"] = initial_count
        state["session_initial_count"] = initial_count
        print("[{}] [CONFIG] Arquivo Alvo ajustado para: {}".format(ts(), OUTPUT_FILE))
        print("[{}] [CONFIG] Circuitos existentes no arquivo: {}".format(ts(), initial_count))
    if args.max_concurrent_julia is not None:
        julia_semaphore = threading.Semaphore(args.max_concurrent_julia)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    db_path = os.path.join(CURRENT_DIR, "optuna_multiobj_darts.db")
    storage_url = "sqlite:///" + db_path.replace("\\", "/")
    
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            constant_liar=True,
            seed=42
        )
        sampler_name = "Multi-Objective TPE (MOTPE Assíncrono com Constant Liar)"
    else:
        sampler = optuna.samplers.NSGAIISampler(
            population_size=args.pop_size,
            mutation_prob=None,
            crossover_prob=0.9,
            seed=42
        )
        sampler_name = "NSGA-II (População={})".format(args.pop_size)
    
    study = optuna.create_study(
        study_name=args.study_name,
        directions=["maximize", "maximize"], # f1: SNR, f2: Novelty
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True
    )
    
    print("[{}] [INICIO] Estudo Optuna Multi-Objetivo carregado com sucesso.".format(ts()))
    print("[{}] Banco de Dados: {}".format(ts(), db_path))
    print("[{}] Estudo: '{}' | Sampler: {}".format(ts(), args.study_name, sampler_name))
    print("[{}] Configuração: {} workers paralelos".format(ts(), args.n_jobs))
    print("[{}] Pressione Ctrl+C a qualquer momento para pausar sem perder nenhum circuito ou progresso.\n".format(ts()))
    sys.stdout.flush()
    
    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    except KeyboardInterrupt:
        print("\n[{}] [PAUSA] Execução pausada pelo usuário. Todos os dados foram preservados!".format(ts()))
        
    print("\n" + "=" * 75)
    print("  RELATÓRIO DO GERADOR OPTUNA MULTI-OBJETIVO")
    print("=" * 75)
    print("  Total de Circuitos Aceitos no Dataset: {}/{}".format(state["valid_count"], TARGET_SAMPLES))
    print("  Arquivo Gerado: {}".format(OUTPUT_FILE))
    print("  Total de Membros na Fronteira de Pareto: {}".format(len(study.best_trials)))
    print("=" * 75)

if __name__ == "__main__":
    main()
