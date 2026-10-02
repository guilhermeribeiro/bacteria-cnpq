# -*- coding: utf-8 -*-
"""
circuit_diversity.py
==============================================================================
Multi-Objective Diversity & Novelty Metric Engine for Genetic Circuits.
Engineered for Optuna 4.9.0 NSGA-II Multi-Objective Optimization.
Calculates topological and biochemical diversity distances in < 1 ms.
==============================================================================
"""
from __future__ import print_function, division
import numpy as np
import threading

class CircuitDiversityScorer(object):
    """
    Computes invariant topological and biochemical distances between genetic circuits
    of variable sizes S in [5, 11].
    """
    REPRESSORS = ['SrpR', 'PhlF', 'AmeR', 'BetI', 'QacR', 'AmtR']
    REP_TO_IDX = {r: i for i, r in enumerate(REPRESSORS)}
    NUM_REPS = len(REPRESSORS)
    MAX_S = 12  # Maximum slot count

    # Biophysical parameters from ALIFE2023 / GeneticLogicGraph: [log10(Kd), Hill n, log10(ymax/ymin)]
    BIO_PARAMS = {
        'SrpR': np.array([-2.000, 2.9, np.log10(1.300 / 0.003)]),
        'PhlF': np.array([-1.523, 4.0, np.log10(3.900 / 0.010)]),
        'AmeR': np.array([-1.046, 1.4, np.log10(3.800 / 0.200)]),
        'BetI': np.array([-0.387, 2.4, np.log10(3.800 / 0.070)]),
        'QacR': np.array([-0.863, 1.6, np.log10(2.820 / 0.018)]),
        'AmtR': np.array([-1.056, 2.3, np.log10(4.060 / 0.001)]),
    }
    BIO_MIN = np.array([-2.000, 1.4, 1.279], dtype=np.float32)
    BIO_MAX = np.array([-0.387, 4.0, 3.609], dtype=np.float32)

    def __init__(self, alpha=0.55, k_neighbors=10):
        self.alpha = float(alpha)
        self.k_neighbors = int(k_neighbors)

        # Precompute Null Baseline Circuit for cold-start evaluation
        null_slots = ['pTac', 'pTet'] + ['Vazio'] * (self.MAX_S - 3) + ['YFP']
        null_W = np.zeros((self.MAX_S, self.MAX_S), dtype=np.float32)
        self.null_features = self.extract_feature_vector(self.MAX_S, null_W, null_slots)

    def extract_feature_vector(self, S, W, slots):
        """
        Extracts fixed-dimensional topological and biochemical feature vectors
        from a candidate circuit of size S in [5, 11].
        """
        W = np.asarray(W, dtype=np.float32)
        if W.ndim != 2 or W.shape[0] != S or W.shape[1] != S:
            raise ValueError("W must be a 2D matrix of shape ({0}, {0}), got {1}".format(S, W.shape))

        # Normalizacao Automatica de Orientacao (Garante robustez W[src, tgt] vs W[tgt, src])
        if np.sum(W[0:2, :]) > 0:
            W_norm = W
        else:
            W_norm = W.T

        A_bin = (W_norm > 0.01).astype(np.float32)
        # Matriz de transmissao de sinal T onde T[origem, destino] = fluxo direcionado
        T = A_bin

        # -------------------------------------------------------------
        # 1. Topological Invariant Descriptors (19 dims, in [0, 1])
        # -------------------------------------------------------------
        f_size = (float(S) - 5.0) / 7.0  # S in [5, 12] -> [0, 1]
        
        admissible_edges = max(1, (S - 2) * S)
        f_density = float(np.sum(A_bin)) / float(admissible_edges)
        
        active_nodes = sum(
            1 for i in range(2, S - 1)
            if slots[i] != 'Vazio' and (np.sum(T[i, :]) + np.sum(T[:, i]) > 0)
        )
        f_active_ratio = float(active_nodes) / max(1.0, float(S - 3))
        
        fanout_0 = float(np.sum(T[0, 2:])) / max(1.0, float(S - 2))  # pTac
        fanout_1 = float(np.sum(T[1, 2:])) / max(1.0, float(S - 2))  # pTet
        input_asym = abs(fanout_0 - fanout_1)
        fanin_out = float(np.sum(T[:S-1, S-1])) / max(1.0, float(S - 1))  # YFP
        
        # Matrix powers for signal flow (1 to 4 hops)
        T2 = np.dot(T, T)
        T3 = np.dot(T2, T)
        T4 = np.dot(T3, T)
        
        flow_p1_0 = min(1.0, float(T[0, S - 1]))
        flow_p1_1 = min(1.0, float(T[1, S - 1]))
        flow_p2_0 = min(1.0, float(T2[0, S - 1]))
        flow_p2_1 = min(1.0, float(T2[1, S - 1]))
        flow_p3_0 = min(1.0, float(T3[0, S - 1]))
        flow_p3_1 = min(1.0, float(T3[1, S - 1]))
        flow_p4_0 = min(1.0, float(T4[0, S - 1]))
        flow_p4_1 = min(1.0, float(T4[1, S - 1]))
        
        toggle_cycles = float(np.trace(T2) - np.sum(np.diag(T)**2)) / max(1.0, float(S * (S - 1)))
        self_loops = float(np.trace(T)) / max(1.0, float(S))
        
        sv = np.linalg.svd(T / np.sqrt(max(1.0, float(S))), compute_uv=False)
        sv_top = np.zeros(4, dtype=np.float32)
        sv_top[:min(len(sv), 4)] = sv[:min(len(sv), 4)]
        
        topo_inv = np.array([
            f_size, f_density, f_active_ratio,
            fanout_0, fanout_1, input_asym, fanin_out,
            flow_p1_0, flow_p1_1, flow_p2_0, flow_p2_1,
            flow_p3_0, flow_p3_1, flow_p4_0, flow_p4_1,
            toggle_cycles, self_loops,
            sv_top[0], sv_top[1]
        ], dtype=np.float32)

        # -------------------------------------------------------------
        # 2. Role-Preserving Canonical Padded Edge Map (90 dims)
        # -------------------------------------------------------------
        int_nodes = list(range(2, S - 1))
        order_keys = [(slots[i], int(np.sum(T[:, i])), int(np.sum(T[i, :])), i) for i in int_nodes]
        order_keys.sort()
        sorted_int = [x[3] for x in order_keys]
        
        W_canon = np.zeros((self.MAX_S, self.MAX_S), dtype=np.float32)
        mapped_indices = [0, 1] + list(range(2, 2 + len(sorted_int))) + [self.MAX_S - 1]
        orig_indices = [0, 1] + sorted_int + [S - 1]
        
        for new_tgt, orig_tgt in zip(mapped_indices, orig_indices):
            for new_src, orig_src in zip(mapped_indices, orig_indices):
                W_canon[new_tgt, new_src] = A_bin[orig_tgt, orig_src]
                
        admissible_mask = np.ones((self.MAX_S, self.MAX_S), dtype=bool)
        admissible_mask[:2, :] = False
        admissible_mask[:, self.MAX_S - 1] = False
        canon_edges = W_canon[admissible_mask].flatten() # exactly 90 elements

        # -------------------------------------------------------------
        # 3. Biochemical Component Descriptors (16 dims)
        # -------------------------------------------------------------
        internal_slots = slots[2:S-1]
        n_int = max(1, len(internal_slots))
        
        rep_counts = np.zeros(self.NUM_REPS + 1, dtype=np.float32)
        for s in internal_slots:
            if s in self.REP_TO_IDX:
                rep_counts[self.REP_TO_IDX[s]] += 1.0
            else:
                rep_counts[self.NUM_REPS] += 1.0
        comp_dist = rep_counts / float(n_int)
        
        p_act = comp_dist[comp_dist > 0]
        rep_entropy = float(-np.sum(p_act * np.log(p_act)) / np.log(float(self.NUM_REPS + 1)))
        
        bio_sum = np.zeros(3, dtype=np.float32)
        active_count = np.sum(rep_counts[:self.NUM_REPS])
        if active_count > 0:
            for s in internal_slots:
                if s in self.BIO_PARAMS:
                    bio_sum += self.BIO_PARAMS[s]
            bio_mean = bio_sum / active_count
            comp_bio = np.clip((bio_mean - self.BIO_MIN) / (self.BIO_MAX - self.BIO_MIN), 0.0, 1.0).astype(np.float32)
        else:
            comp_bio = np.zeros(3, dtype=np.float32)
            
        comp_terminal = np.zeros(self.NUM_REPS, dtype=np.float32)
        for src in range(2, S - 1):
            if T[src, S - 1] > 0 and slots[src] in self.REP_TO_IDX:
                comp_terminal[self.REP_TO_IDX[slots[src]]] = 1.0
                
        wiring_entropy = float(f_density * (1.0 - f_density) * 4.0)
        self_entropy = 0.50 * rep_entropy + 0.50 * wiring_entropy

        return {
            'topo_inv': topo_inv,
            'canon_edges': canon_edges,
            'comp_dist': comp_dist,
            'comp_bio': comp_bio,
            'comp_terminal': comp_terminal,
            'self_entropy': float(self_entropy)
        }

    def compute_distance(self, f1, f2):
        """Pairwise composite distance D(c1, c2) in [0, 1]."""
        d_edges = np.mean(np.abs(f1['canon_edges'] - f2['canon_edges']))
        d_inv = np.mean(np.abs(f1['topo_inv'] - f2['topo_inv']))
        d_topo = 0.60 * d_edges + 0.40 * d_inv

        d_hist = 0.50 * np.sum(np.abs(f1['comp_dist'] - f2['comp_dist']))
        d_bio = np.mean(np.abs(f1['comp_bio'] - f2['comp_bio']))
        d_term = np.mean(np.abs(f1['comp_terminal'] - f2['comp_terminal']))
        d_comp = 0.50 * d_hist + 0.30 * d_bio + 0.20 * d_term

        D = self.alpha * d_topo + (1.0 - self.alpha) * d_comp
        return float(np.clip(D, 0.0, 1.0))


class CircuitArchive(object):
    """
    Thread-safe, high-speed circuit archive for online novelty search.
    Stores accepted circuits (SNR >= 0.85) and computes K-NN diversity in < 1 ms.
    """
    def __init__(self, scorer=None, k_neighbors=10):
        self.scorer = scorer if scorer is not None else CircuitDiversityScorer(k_neighbors=k_neighbors)
        self.k_neighbors = int(k_neighbors)
        self.circuits = []
        self._lock = threading.Lock()
        
        self._canon_edges_mat = None
        self._topo_inv_mat = None
        self._comp_dist_mat = None
        self._comp_bio_mat = None
        self._comp_term_mat = None
        self._count = 0

    def add_circuit(self, S, W, slots, snr, metadata=None):
        """Adds an accepted circuit (SNR >= 0.85) to the diversity archive."""
        W_arr = np.array(W, dtype=np.float32)
        feats = self.scorer.extract_feature_vector(S, W_arr, slots)
        record = {
            'S': S,
            'W': W_arr,
            'slots': slots,
            'snr': float(snr),
            'feats': feats,
            'metadata': metadata or {}
        }
        with self._lock:
            self.circuits.append(record)
            self._count += 1
            
            if self._canon_edges_mat is None:
                self._canon_edges_mat = np.array([feats['canon_edges']], dtype=np.float32)
                self._topo_inv_mat = np.array([feats['topo_inv']], dtype=np.float32)
                self._comp_dist_mat = np.array([feats['comp_dist']], dtype=np.float32)
                self._comp_bio_mat = np.array([feats['comp_bio']], dtype=np.float32)
                self._comp_term_mat = np.array([feats['comp_terminal']], dtype=np.float32)
            else:
                self._canon_edges_mat = np.vstack([self._canon_edges_mat, feats['canon_edges']])
                self._topo_inv_mat = np.vstack([self._topo_inv_mat, feats['topo_inv']])
                self._comp_dist_mat = np.vstack([self._comp_dist_mat, feats['comp_dist']])
                self._comp_bio_mat = np.vstack([self._comp_bio_mat, feats['comp_bio']])
                self._comp_term_mat = np.vstack([self._comp_term_mat, feats['comp_terminal']])

    def compute_novelty(self, S, W, slots):
        """
        Evaluates candidate novelty f2 in [0, 1] in < 1 ms against archive.
        """
        cand_feats = self.scorer.extract_feature_vector(S, W, slots)
        
        with self._lock:
            # Cold start handling when archive is empty
            if self._count == 0:
                dist_to_null = self.scorer.compute_distance(cand_feats, self.scorer.null_features)
                self_ent = cand_feats['self_entropy']
                novelty = 0.60 * dist_to_null + 0.40 * self_ent
                return float(np.clip(novelty, 0.0, 1.0))
                
            # Vectorized batch distance calculation
            d_edges = np.mean(np.abs(self._canon_edges_mat - cand_feats['canon_edges']), axis=1)
            d_inv = np.mean(np.abs(self._topo_inv_mat - cand_feats['topo_inv']), axis=1)
            d_topo = 0.60 * d_edges + 0.40 * d_inv
            
            d_hist = 0.50 * np.sum(np.abs(self._comp_dist_mat - cand_feats['comp_dist']), axis=1)
            d_bio = np.mean(np.abs(self._comp_bio_mat - cand_feats['comp_bio']), axis=1)
            d_term = np.mean(np.abs(self._comp_term_mat - cand_feats['comp_terminal']), axis=1)
            d_comp = 0.50 * d_hist + 0.30 * d_bio + 0.20 * d_term
            
            distances = self.scorer.alpha * d_topo + (1.0 - self.scorer.alpha) * d_comp
            
            k_eff = min(self.k_neighbors, self._count)
            if self._count <= self.k_neighbors:
                novelty = float(np.mean(distances))
            else:
                k_smallest = np.partition(distances, k_eff)[:k_eff]
                novelty = float(np.mean(k_smallest))
                
            return float(np.clip(novelty, 0.0, 1.0))

    def __len__(self):
        return self._count
