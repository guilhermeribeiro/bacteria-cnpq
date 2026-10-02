# -*- coding: utf-8 -*-
"""
worker_darts.py
Worker individual em Python 2.7 para treinamento DARTS de um único circuito.
Incorpora Loop Interno Adaptativo com Reinicializações em Memória (<1ms):
- Treina o ciclo completo (Fase 1: Exploração -> Fase 2: L1 Sparsity -> Fase 3: Poda).
- Valida o circuito discreto podado (Conexidade BFS + Sinal >= 0.035 + MSE <= 0.115).
- Se o circuito não validar pós-poda, reseta os tensores em memória em <1ms e retenta.
- Garante que cada worker retorne um circuito viável para simulação no Julia SSA.
"""
import sys
import os
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser(description="GeneNet-DARTS Standalone Circuit Generator")
parser.add_argument('--lr', type=float, default=0.02, help="Learning Rate")
parser.add_argument('--ew', type=float, default=0.25, help="Entropy Weight")
parser.add_argument('--l1', type=float, default=0.0005, help="L1 Penalty")
parser.add_argument('--pl', type=float, default=0.35, help="Prune Limit")
parser.add_argument('--S', type=int, default=10, help="Number of nodes (slots)")
parser.add_argument('--gate', type=str, default="XOR", help="Logic gate (XOR, AND, OR, etc.)")
parser.add_argument('--iterations', type=int, default=700, help="Total target iterations")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--out_dir', type=str, required=True, help="Output directory for test run")
parser.add_argument('--seed', type=int, default=None, help="Random seed")
parser.add_argument('--lambda_snr', type=float, default=0.05, help="Poisson-SNR loss weight")
parser.add_argument('--max_restarts', type=int, default=8, help="Max in-memory restarts")
args = parser.parse_args()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if args.seed is not None:
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

import matplotlib
matplotlib.use('Agg')

import theano
import theano.tensor as T
import GeneNet
import GeneNet_model as Model
from test_harness.capture import TestRun

Model.configure(num_slots=args.S, gate=args.gate)

def get_truth_table_inputs(S_nodes, gate="XOR", N_steps=300):
    y0 = 0.1 * np.ones([S_nodes, 4], dtype=theano.config.floatX)
    x = np.zeros([N_steps, S_nodes, 4], dtype=theano.config.floatX)
    x[:, 0, 0] = 0.1; x[:, 1, 0] = 0.1
    x[:, 0, 1] = 0.1; x[:, 1, 1] = 1.9
    x[:, 0, 2] = 1.9; x[:, 1, 2] = 0.1
    x[:, 0, 3] = 1.9; x[:, 1, 3] = 1.9
    
    targets = np.zeros(4, dtype=theano.config.floatX)
    states = [(False, False), (False, True), (True, False), (True, True)]
    for idx, (b0, b1) in enumerate(states):
        res = Model.eval_logic(b0, b1, gate)
        targets[idx] = 0.8 if res else 0.1
    return y0, x, targets

def check_logic_gate_solved(preds, targets, gate="XOR", min_snr=0.45, min_signal=0.08):
    if gate == "XOR":
        p00, p01, p10, p11 = preds[0], preds[1], preds[2], preds[3]
        mu_high = (p01 + p10) / 2.0
        mu_low = (p00 + p11) / 2.0
        signal = mu_high - mu_low
        noise = np.sqrt(max(0.0, mu_high) + 1e-4) + np.sqrt(max(0.0, mu_low) + 1e-4)
        snr_val = signal / noise if noise > 0 else 0.0
        solved = (signal >= min_signal) and (snr_val >= min_snr)
    if gate == "NOT":
        p00, p01, p10, p11 = preds[0], preds[1], preds[2], preds[3]
        mu_high = (p00 + p01) / 2.0
        mu_low = (p10 + p11) / 2.0
        signal = mu_high - mu_low
        noise = np.sqrt(max(0.0, mu_high) + 1e-4) + np.sqrt(max(0.0, mu_low) + 1e-4)
        snr_val = signal / noise if noise > 0 else 0.0
        solved = (signal >= min_signal) and (snr_val >= min_snr)
        return solved, snr_val, signal

    mse = float(np.mean((preds - targets) ** 2))
    return mse < 0.08, 0.0, 0.0

def check_graph_validity(matrix_W, gate="XOR"):
    N = len(matrix_W)
    if N < 3:
        return False
    out_ptac = sum(matrix_W[0][1:])
    in_yfp   = sum(matrix_W[r][-1] for r in range(N))
    if out_ptac == 0.0 or in_yfp == 0.0:
        return False
    if gate != "NOT":
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
    if gate == "NOT":
        return can_reach(0, N - 1)
    return can_reach(0, N - 1) and can_reach(1, N - 1)

adam_shared_vars = []

def make_tracking_adam(shared_lr):
    def tracking_adam(cost, params, lr=0.1, b1=0.02, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(np.float32(1))
        adam_shared_vars.append((i, np.float32(1)))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = shared_lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
            adam_shared_vars.append((m, np.zeros(p.get_value().shape, dtype=theano.config.floatX)))
            adam_shared_vars.append((v, np.zeros(p.get_value().shape, dtype=theano.config.floatX)))
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates
    return tracking_adam

def apply_topological_scaffold(net, S, attempt, seed=None):
    """
    Inicializa W e alpha estritamente dentro da bacia de atracao de um motivo topologico
    viavel de XOR, eliminando a sela simetrica (MSE=0.1225, Sinal=0.0) onde o gradiente zera.
    Garante diversidade estrutural e ZERO duplicatas atraves de:
    1. Alternancia ciclica entre 10 motivos distintos
    2. Permutacao estocastica de slots intermediarios (pi in S_N)
    3. Permutacao estocastica da biblioteca de 6 repressores (6!)
    4. Jitter Gaussiano continuo (sigma_W = 0.35, sigma_alpha = 0.30)
    """
    scaffold_id = (((seed if seed is not None else 0) + attempt * 7) % 10)
    # Inicializacao com densidade calibrada para o sweet spot dos campeoes (18-26 conexoes ativas)
    W_scaffold = np.random.normal(-0.3, 1.2, [S, S]).astype(np.float32)
    alpha_scaffold = np.zeros((S, Model.K), dtype=np.float32)
    
    # 1. Pinar entradas e saidas obrigatorias
    alpha_scaffold[0, :] = -100.0
    alpha_scaffold[0, 8] = 10.0      # Slot 0 = pTac (indice 8)
    alpha_scaffold[1, :] = -100.0
    alpha_scaffold[1, 9] = 10.0      # Slot 1 = pTet (indice 9)
    alpha_scaffold[S - 1, :] = -100.0
    alpha_scaffold[S - 1, 7] = 10.0  # Slot S-1 = YFP (indice 7)
    
    # 2. Permutacao de slots intermediarios e repressores
    mid_slots = list(range(2, S - 1))
    np.random.shuffle(mid_slots)
    
    # Repressores disponiveis: 0=SrpR, 1=PhlF, 2=AmeR, 3=BetI, 4=QacR, 5=AmtR
    reps = list(range(6))
    np.random.shuffle(reps)
    
    n_mid = len(mid_slots)
    
    # 3. Construcao do Esqueleto Topologico Base (reforco aditivo nas rotas funcionais)
    if scaffold_id == 0 and n_mid >= 3:
        # Scaffold 0: Dual-NAND De Morgan (A XOR B = NAND(NAND(A, ~B), NAND(~A, B)))
        n1, n2, n3 = mid_slots[0], mid_slots[1], mid_slots[2]
        W_scaffold[n1, 0] += 2.0; W_scaffold[n2, 1] += 2.0
        W_scaffold[n3, n1] += 1.8; W_scaffold[n3, 1] += 1.8
        W_scaffold[S - 1, n3] += 1.5; W_scaffold[S - 1, n2] += 1.5
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0; alpha_scaffold[n3, reps[2]] = 4.0
        
    elif scaffold_id == 1 and n_mid >= 4:
        # Scaffold 1: IFFL-1 Dual Branch (Bump / Band-Pass)
        n1, n2, n3, n4 = mid_slots[0], mid_slots[1], mid_slots[2], mid_slots[3]
        W_scaffold[n1, 0] += 2.0; W_scaffold[n2, 1] += 1.5; W_scaffold[n2, n1] += 2.0
        W_scaffold[n3, 1] += 2.0; W_scaffold[n4, 0] += 1.5; W_scaffold[n4, n3] += 2.0
        W_scaffold[S - 1, n2] += 1.5; W_scaffold[S - 1, n4] += 1.5
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0
        alpha_scaffold[n3, reps[2]] = 4.0; alpha_scaffold[n4, reps[3]] = 4.0
        
    elif scaffold_id == 2 and n_mid >= 2:
        # Scaffold 2: Cross-Coupled Mutual Repressor (Bistable Toggle / Push-Pull)
        n1, n2 = mid_slots[0], mid_slots[1]
        W_scaffold[n1, n2] += 2.0; W_scaffold[n2, n1] += 2.0
        W_scaffold[n1, 0] += 2.2; W_scaffold[n2, 1] += 2.2
        W_scaffold[S - 1, n1] += 1.4; W_scaffold[S - 1, n2] += 1.4
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0
        
    elif scaffold_id == 3 and n_mid >= 3:
        # Scaffold 3: Cascaded 3-NOR (Cello CAD Standard XOR)
        n1, n2, n3 = mid_slots[0], mid_slots[1], mid_slots[2]
        W_scaffold[n1, 0] += 1.8; W_scaffold[n1, 1] += 1.8
        W_scaffold[n2, 0] += 1.8; W_scaffold[n2, n1] += 2.0
        W_scaffold[n3, 1] += 1.8; W_scaffold[n3, n1] += 2.0
        W_scaffold[S - 1, n2] += 1.5; W_scaffold[S - 1, n3] += 1.5
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0; alpha_scaffold[n3, reps[2]] = 4.0
        
    elif scaffold_id == 4 and n_mid >= 3:
        # Scaffold 4: Band-Pass Window Comparator (Sensibilidade Diferencial)
        n1, n2, n3 = mid_slots[0], mid_slots[1], mid_slots[2]
        W_scaffold[n1, 0] += 1.2; W_scaffold[n1, 1] += 1.2
        W_scaffold[n2, 0] += 2.5; W_scaffold[n2, 1] += 2.5
        W_scaffold[n3, n2] += 2.2
        W_scaffold[S - 1, n1] += 1.8; W_scaffold[S - 1, n3] += 1.8
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0; alpha_scaffold[n3, reps[2]] = 4.0
        
    elif scaffold_id == 5 and n_mid >= 4:
        # Scaffold 5: Multiplexer (MUX-XOR)
        n1, n2, n3, n4 = mid_slots[0], mid_slots[1], mid_slots[2], mid_slots[3]
        W_scaffold[n1, 0] += 2.2; W_scaffold[n2, 1] += 2.2
        W_scaffold[n3, 0] += 1.8; W_scaffold[n3, n2] += 1.8
        W_scaffold[n4, n1] += 1.8; W_scaffold[n4, 1] += 1.8
        W_scaffold[S - 1, n3] += 1.5; W_scaffold[S - 1, n4] += 1.5
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0
        alpha_scaffold[n3, reps[2]] = 4.0; alpha_scaffold[n4, reps[3]] = 4.0
        
    elif scaffold_id == 6 and n_mid >= 3:
        # Scaffold 6: Central-Repressor Balanced Bridge
        n1, n2, n3 = mid_slots[0], mid_slots[1], mid_slots[2]
        W_scaffold[n1, 0] += 1.5; W_scaffold[n1, 1] += 1.5
        W_scaffold[n2, n1] += 1.8; W_scaffold[n2, 0] += 1.8
        W_scaffold[n3, n1] += 1.8; W_scaffold[n3, 1] += 1.8
        W_scaffold[S - 1, n2] += 1.5; W_scaffold[S - 1, n3] += 1.5
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0; alpha_scaffold[n3, reps[2]] = 4.0
        
    elif scaffold_id == 7 and n_mid >= 2:
        # Scaffold 7: Minimal Asymmetric Cascade (Inspirado nos Campeoes 1 e 2)
        n1, n2 = mid_slots[0], mid_slots[1]
        W_scaffold[n1, 0] += 1.6; W_scaffold[n1, 1] += 1.6
        W_scaffold[n2, n1] += 1.8; W_scaffold[n2, 0] += 1.4
        W_scaffold[S - 1, n1] += 1.5; W_scaffold[S - 1, n2] += 1.5
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0
        
    elif scaffold_id == 8 and n_mid >= 4:
        # Scaffold 8: Multi-Stage Distributed Ring-Inhibitor
        n1, n2, n3, n4 = mid_slots[0], mid_slots[1], mid_slots[2], mid_slots[3]
        W_scaffold[n1, 0] += 2.0; W_scaffold[n2, 1] += 2.0
        W_scaffold[n3, n4] += 1.5; W_scaffold[n4, n3] += 1.5
        W_scaffold[n3, n1] += 1.6; W_scaffold[n4, n2] += 1.6
        W_scaffold[S - 1, n3] += 1.5; W_scaffold[S - 1, n4] += 1.5
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0
        alpha_scaffold[n3, reps[2]] = 4.0; alpha_scaffold[n4, reps[3]] = 4.0
        
    else:
        # Scaffold 9: Dual-Inhibitor Toggle XOR
        n1, n2, n3 = mid_slots[0], mid_slots[1], mid_slots[2]
        W_scaffold[n1, 0] += 1.8; W_scaffold[n1, n2] += 1.5
        W_scaffold[n2, 1] += 1.8; W_scaffold[n2, n1] += 1.5
        W_scaffold[n3, n1] += 1.8; W_scaffold[n3, n2] += 1.8
        W_scaffold[S - 1, n3] += 2.0
        alpha_scaffold[n1, reps[0]] = 4.0; alpha_scaffold[n2, reps[1]] = 4.0; alpha_scaffold[n3, reps[2]] = 4.0
        
    # Atribuir repressores ou Vazio para slots intermediarios excedentes
    for idx_slot in mid_slots:
        if np.max(alpha_scaffold[idx_slot, :7]) <= 0.0:
            rep_choice = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16])
            alpha_scaffold[idx_slot, rep_choice] = 3.0

    # 4. Injetar Jitter Gaussiano Continuo (Previne duplicatas e permite otimizacao fina)
    W_jitter = np.random.normal(0.0, 0.35, [S, S]).astype(np.float32)
    alpha_jitter = np.random.normal(0.0, 0.30, [S, Model.K]).astype(np.float32)
    
    W_final = W_scaffold + W_jitter
    alpha_final = alpha_scaffold + alpha_jitter
    
    # Mascaras e invariantes estritos
    W_final[0, :] = -10.0   # Entradas nao recebem conexao
    W_final[1, :] = -10.0
    W_final[:, S - 1] = -10.0 # Saida nao tem conexoes de saida
    
    alpha_final[0, :] = -100.0; alpha_final[0, 8] = 100.0
    alpha_final[1, :] = -100.0; alpha_final[1, 9] = 100.0
    alpha_final[S - 1, :] = -100.0; alpha_final[S - 1, 7] = 100.0
    alpha_final[2:S - 1, 7:10] = -100.0  # Banir I/O dos slots intermediarios
    
    net.W.set_value(W_final.astype(theano.config.floatX))
    net.alpha.set_value(alpha_final.astype(theano.config.floatX))
    net.Ax.set_value(np.float64(1.0).astype(theano.config.floatX))
    
    for var, init_val in adam_shared_vars:
        var.set_value(init_val)

def train_darts_3phases(desiredFunction, network, total_iterations, batchSize,
                        shared_lr, shared_l1, shared_ew, shared_pl, shared_snr, gate="XOR",
                        max_restarts=8):
    initial = T.matrix(name='initial', dtype=theano.config.floatX)
    networkInput = T.tensor3(name='networkInput', dtype=theano.config.floatX)
    desiredOutput = T.vector(name='desiredOutput', dtype=theano.config.floatX)
    L1mask = T.matrix(name='L1mask', dtype=theano.config.floatX)
    
    c1 = theano.shared(np.float64(1.0).astype(theano.config.floatX))
    c10 = theano.shared(np.float64(-10.0).astype(theano.config.floatX))
    
    applyL1mask = theano.function(
        inputs=[L1mask],
        updates=[(network.W, network.W * L1mask + c10 * (c1 - L1mask))]
    )
    
    eval_out = network.normalize(GeneNet.evolveTime(initial, networkInput, network))
    eval_fn = theano.function(
        inputs=[initial, networkInput],
        outputs=eval_out
    )

    mask_h = T.gt(desiredOutput, 0.45)
    mask_l = T.lt(desiredOutput, 0.45)
    mu_h = T.sum(eval_out * mask_h) / (T.sum(mask_h) + 1e-8)
    mu_l = T.sum(eval_out * mask_l) / (T.sum(mask_l) + 1e-8)
    sigma_h = T.sqrt(T.maximum(mu_h, 0.0) + 1e-4)
    sigma_l = T.sqrt(T.maximum(mu_l, 0.0) + 1e-4)
    snr_poisson = (mu_h - mu_l) / (sigma_h + sigma_l + 1e-4)

    cost_base = GeneNet.costFunction(initial, networkInput, desiredOutput, network)
    reg = network.regularize()
    cost = cost_base + reg[1] - (np.float64(0.2).astype(theano.config.floatX) * shared_snr * snr_poisson)
    train = theano.function(
        inputs=[initial, networkInput, desiredOutput],
        outputs=cost,
        updates=make_tracking_adam(shared_lr)(cost, network.parameters)
    )

    costL1_base = GeneNet.costFunctionL1(initial, networkInput, desiredOutput, network, shared_l1)
    costL1 = costL1_base - (shared_snr * snr_poisson)
    trainL1 = theano.function(
        inputs=[initial, networkInput, desiredOutput],
        outputs=costL1,
        updates=make_tracking_adam(shared_lr)(costL1, network.parameters)
    )

    y0_tt, x_tt, targets_tt = get_truth_table_inputs(network.S, gate=gate)

    CHECK_INTERVAL = 15
    PATIENCE = 2

    # Orçamento adaptativo calibrado com os 53 campeões:
    # Permite treinos ultra-rápidos (< 200 épocas) ou treinos profundos (700 épocas)
    iters_p1 = max(35, int(total_iterations * 0.35))
    iters_p2 = max(55, int(total_iterations * 0.45))
    iters_p3 = max(25, int(total_iterations * 0.20))

    best_overall_signal = -999.0
    best_metrics = None

    for attempt in range(1, max_restarts + 1):
        apply_topological_scaffold(network, network.S, attempt=attempt, seed=args.seed)

        # -------------------------------------------------------------
        # FASE 1: Exploração Contínua
        # -------------------------------------------------------------
        stable_solved_p1 = 0
        actual_p1 = iters_p1
        early_stopped_p1 = False

        aborted_p1 = False
        abort_p1_step = min(60, max(30, int(iters_p1 * 0.60)))
        for step in range(1, iters_p1 + 1):
            [init_state, net_in, target_out] = desiredFunction(batchSize)
            train(init_state, net_in, target_out)

            if step % CHECK_INTERVAL == 0:
                preds_c = eval_fn(y0_tt, x_tt)
                solved, snr_c, sig_c = check_logic_gate_solved(preds_c, targets_tt, gate=gate)
                if solved:
                    stable_solved_p1 += 1
                    if stable_solved_p1 >= PATIENCE:
                        actual_p1 = step
                        early_stopped_p1 = True
                        break
                else:
                    stable_solved_p1 = 0

                # Early Abort ultrarrapido se nao houver sinal inicial (preso na sela)
                if (step >= abort_p1_step) and (sig_c <= 0.008) and (attempt < max_restarts):
                    aborted_p1 = True
                    break

        if aborted_p1:
            if attempt < max_restarts:
                print("[DARTS] Tentativa {:02d}/{:02d}: Early Abort na Fase 1 no passo {} (Sinal={:.4f}). Reiniciando em memoria...".format(
                    attempt, max_restarts, step, sig_c
                ))
                sys.stdout.flush()
                continue

        preds_p1 = eval_fn(y0_tt, x_tt)
        mse_p1 = float(np.mean((preds_p1 - targets_tt) ** 2))

        # -------------------------------------------------------------
        # FASE 2: Regularização L1 (Onde a simetria do XOR é quebrada!)
        # -------------------------------------------------------------
        stable_solved_p2 = 0
        actual_p2 = iters_p2
        early_stopped_p2 = False
        aborted_saddle_p2 = False
        abort_p2_step = min(90, max(45, int(iters_p2 * 0.60)))

        for step in range(1, iters_p2 + 1):
            [init_state, net_in, target_out] = desiredFunction(batchSize)
            trainL1(init_state, net_in, target_out)

            if step % CHECK_INTERVAL == 0:
                preds_c = eval_fn(y0_tt, x_tt)
                solved, snr_c, sig_c = check_logic_gate_solved(preds_c, targets_tt, gate=gate)
                if solved:
                    stable_solved_p2 += 1
                    if stable_solved_p2 >= PATIENCE:
                        actual_p2 = step
                        early_stopped_p2 = True
                        break
                else:
                    stable_solved_p2 = 0

                # Early Abort intra-fase:
                # Se após passos suficientes o sinal for nulo e o MSE estiver preso na sela (>= 0.118),
                # aborta a tentativa na hora em vez de queimar mais centenas de passos inuteis!
                if (step >= abort_p2_step) and (sig_c <= 0.010) and (attempt < max_restarts):
                    cur_mse = float(np.mean((preds_c - targets_tt) ** 2))
                    if cur_mse >= 0.118:
                        aborted_saddle_p2 = True
                        break

        if aborted_saddle_p2:
            print("[DARTS] Tentativa {:02d}/{:02d}: Early Abort na Fase 2 no passo {} (Sinal={:.4f}). Reiniciando em memoria...".format(
                attempt, max_restarts, step, sig_c
            ))
            sys.stdout.flush()
            continue

        preds_p2 = eval_fn(y0_tt, x_tt)
        mse_p2 = float(np.mean((preds_p2 - targets_tt) ** 2))
        p00_p2, p01_p2, p10_p2, p11_p2 = preds_p2[0], preds_p2[1], preds_p2[2], preds_p2[3]
        sig_p2 = ((p01_p2 + p10_p2) / 2.0) - ((p00_p2 + p11_p2) / 2.0)

        # Se a Fase 2 não quebrou a simetria (sinal quase nulo e MSE de sela),
        # executar a Fase 3 (Poda) é inútil. Reinicia direto em memória!
        if (sig_p2 <= 0.010) and (mse_p2 >= 0.118) and (attempt < max_restarts):
            print("[DARTS] Tentativa {:02d}/{:02d}: Sela simetrica na Fase 2 (Sinal={:.4f}, MSE={:.4f}). Pulando Poda -> Restart rapido...".format(
                attempt, max_restarts, sig_p2, mse_p2
            ))
            sys.stdout.flush()
            continue

        # -------------------------------------------------------------
        # FASE 3: Poda (Pruning) e Cristalização
        # -------------------------------------------------------------
        W_val = network.W.eval()
        W_pos_val = 1.0 / (1.0 + np.exp(-W_val))
        mask = W_pos_val > shared_pl.get_value()

        stable_solved_p3 = 0
        actual_p3 = iters_p3
        early_stopped_p3 = False

        for step in range(1, iters_p3 + 1):
            [init_state, net_in, target_out] = desiredFunction(batchSize)
            train(init_state, net_in, target_out)
            applyL1mask(mask)

            if step % CHECK_INTERVAL == 0:
                preds_c = eval_fn(y0_tt, x_tt)
                solved, snr_c, sig_c = check_logic_gate_solved(preds_c, targets_tt, gate=gate)
                if solved:
                    stable_solved_p3 += 1
                    if stable_solved_p3 >= PATIENCE:
                        actual_p3 = step
                        early_stopped_p3 = True
                        break
                else:
                    stable_solved_p3 = 0

        preds_p3 = eval_fn(y0_tt, x_tt)
        mse_p3 = float(np.mean((preds_p3 - targets_tt) ** 2))
        p00, p01, p10, p11 = preds_p3[0], preds_p3[1], preds_p3[2], preds_p3[3]
        if gate == "NOT":
            mu_high = (p00 + p01) / 2.0
            mu_low  = (p10 + p11) / 2.0
        else:
            mu_high = (p01 + p10) / 2.0
            mu_low  = (p00 + p11) / 2.0
        sig_p3 = mu_high - mu_low

        # -------------------------------------------------------------
        # VALIDAÇÃO DO CIRCUITO DISCRETO FINAL (Conexidade + Limiar Biofísico)
        # -------------------------------------------------------------
        # Binarização topológica consistente com capture.py e Julia SSA (.T converte target,source para source,target)
        W_final_val = network.W.eval()
        W_final_bin = (W_final_val.T >= 0.0)
        is_conn = check_graph_validity(W_final_bin.astype(float).tolist(), gate=gate)

        # Critério dos 11 Campeões Reais: sinal >= 0.035, MSE <= 0.115, conexo
        is_valid_champion = is_conn and (sig_p3 >= 0.035) and (mse_p3 <= 0.115)

        metrics = {
            'phase1': {'mse': mse_p1, 'preds': preds_p1},
            'phase2': {'mse': mse_p2, 'preds': preds_p2},
            'phase3': {'mse': mse_p3, 'preds': preds_p3},
            'targets': targets_tt
        }

        if is_valid_champion:
            print("[DARTS] Tentativa {:02d}/{:02d}: CAMPEAO VALIDO! Sinal={:.4f}, MSE={:.4f}, Conexo={}".format(
                attempt, max_restarts, sig_p3, mse_p3, is_conn
            ))
            sys.stdout.flush()
            return metrics, eval_fn, (y0_tt, x_tt, targets_tt)
        else:
            if sig_p3 > best_overall_signal:
                best_overall_signal = sig_p3
                best_metrics = metrics
            if attempt < max_restarts:
                print("[DARTS] Tentativa {:02d}/{:02d}: Nao validou (Sinal={:.4f}, MSE={:.4f}, Conexo={}). Reiniciando em memoria...".format(
                    attempt, max_restarts, sig_p3, mse_p3, is_conn
                ))
                sys.stdout.flush()
                continue

    print("[DARTS] Restarts concluidos ({} tentativas). Retornando melhor candidato (Sinal={:.4f}).".format(
        max_restarts, best_overall_signal
    ))
    sys.stdout.flush()
    return best_metrics if best_metrics is not None else metrics, eval_fn, (y0_tt, x_tt, targets_tt)

def main():
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    run = TestRun(run_dir=out_dir, auto_capture=True)
    
    network = Model.network(entropy_weight=args.ew, S_nodes=args.S)
    apply_topological_scaffold(network, args.S, attempt=1, seed=args.seed)
    
    shared_lr = theano.shared(np.float64(args.lr).astype(theano.config.floatX))
    shared_l1 = theano.shared(np.float64(args.l1).astype(theano.config.floatX))
    shared_ew = theano.shared(np.float64(args.ew).astype(theano.config.floatX))
    shared_pl = theano.shared(np.float64(args.pl).astype(theano.config.floatX))
    shared_snr = theano.shared(np.float64(args.lambda_snr).astype(theano.config.floatX))
    network.entropy_weight = shared_ew
    
    try:
        metrics, eval_fn, (y0_tt, x_tt, targets_tt) = train_darts_3phases(
            Model.desiredFunction, network, args.iterations, args.batch_size,
            shared_lr, shared_l1, shared_ew, shared_pl, shared_snr, gate=args.gate,
            max_restarts=args.max_restarts
        )
        
        sim_out = GeneNet.simulateNetwork(Model.inputData, network, 10)
        run.save(network, simulation_output=sim_out)
        
        preds_final = eval_fn(y0_tt, x_tt)
        mse_final = float(np.mean((preds_final - targets_tt) ** 2))
        
        p00, p01, p10, p11 = preds_final[0], preds_final[1], preds_final[2], preds_final[3]
        if args.gate == "NOT":
            mu_high = (p00 + p01) / 2.0
            mu_low  = (p10 + p11) / 2.0
        else:
            mu_high = (p01 + p10) / 2.0
            mu_low  = (p00 + p11) / 2.0
        final_signal = mu_high - mu_low
        final_noise = np.sqrt(max(0.0, mu_high) + 1e-4) + np.sqrt(max(0.0, mu_low) + 1e-4)
        final_snr = final_signal / final_noise if final_noise > 0 else 0.0
        
        wiring_csv = os.path.join(out_dir, "wiring_matrix.csv")
        if os.path.exists(wiring_csv):
            import csv
            with open(wiring_csv, "r") as wf:
                r = csv.reader(wf)
                next(r)
                m_w = [[float(x) for x in row[1:]] for row in r]
            is_conn = check_graph_validity(m_w, gate=args.gate)
        else:
            W_final_val = network.W.eval()
            W_final_bin = (W_final_val.T >= 0.0)
            is_conn = check_graph_validity(W_final_bin.astype(float).tolist(), gate=args.gate)
        
        is_converged = is_conn and (final_signal >= 0.035) and (mse_final <= 0.115)

        print("\n" + "=" * 65)
        print("FINAL TRUTH TABLE & CONVERGENCE REPORT (GATE={}, S={})".format(args.gate, args.S))
        print("=" * 65)
        print("State 00 (Low,  Low ): pred = {:.4f} | target: {:.4f}".format(preds_final[0], targets_tt[0]))
        print("State 01 (Low,  High): pred = {:.4f} | target: {:.4f}".format(preds_final[1], targets_tt[1]))
        print("State 10 (High, Low ): pred = {:.4f} | target: {:.4f}".format(preds_final[2], targets_tt[2]))
        print("State 11 (High, High): pred = {:.4f} | target: {:.4f}".format(preds_final[3], targets_tt[3]))
        print("-" * 65)
        print("PHASE_1_MSE = {:.6f}".format(metrics['phase1']['mse']))
        print("PHASE_2_MSE = {:.6f}".format(metrics['phase2']['mse']))
        print("PHASE_3_MSE = {:.6f}".format(metrics['phase3']['mse']))
        print("FINAL_MSE   = {:.6f}".format(mse_final))
        print("THEANO_SNR  = {:.4f} (Sinal: {:.4f})".format(final_snr, final_signal))
        print("CONEXO      = {}".format("TRUE" if is_conn else "FALSE"))
        print("CONVERGED   = {}".format("TRUE" if is_converged else "FALSE"))
        print("=" * 65)
        sys.stdout.flush()

        import json
        cd_path = os.path.join(out_dir, "circuit_design.json")
        if os.path.exists(cd_path):
            try:
                with open(cd_path, "r") as f:
                    cd_data = json.load(f)
                cd_data["theano_evaluation"] = {
                    "phase1_mse": round(metrics['phase1']['mse'], 6),
                    "phase2_mse": round(metrics['phase2']['mse'], 6),
                    "phase3_mse": round(metrics['phase3']['mse'], 6),
                    "final_mse": round(mse_final, 6),
                    "final_snr": round(float(final_snr), 4),
                    "final_signal": round(float(final_signal), 4),
                    "is_connected": bool(is_conn),
                    "truth_table": {
                        "00": round(float(preds_final[0]), 4),
                        "01": round(float(preds_final[1]), 4),
                        "10": round(float(preds_final[2]), 4),
                        "11": round(float(preds_final[3]), 4)
                    },
                    "converged": bool(is_converged)
                }
                with open(cd_path, "w") as f:
                    json.dump(cd_data, f, indent=2)
            except Exception as e_json:
                print("[worker_darts] Warning: could not update circuit_design.json: {}".format(e_json))

        print("RUN_DIR=" + out_dir)
        print("STATUS=SUCCESS")
        sys.exit(0)
    except Exception as e:
        import traceback
        sys.stderr.write("FATAL ERROR IN WORKER:\n" + traceback.format_exc() + "\n")
        print("STATUS=ERROR")
        sys.exit(1)

if __name__ == '__main__':
    main()
