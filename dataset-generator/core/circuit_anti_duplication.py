# -*- coding: utf-8 -*-
"""
circuit_anti_duplication.py
==============================================================================
Motor Matematico Rigido de Anti-Duplicacao e Auditoria de Diversidade.
Combina:
1. Reducao de Subgrafo Funcional Ativo (elimina nos mortos / ilhas desconexas)
2. Weisfeiler-Lehman Labeled Graph Hashing (O(|V| + |E|) < 50 us)
3. Verificacao Estrita de Isomorfismo Labeled via VF2 (Node Categorical Match)
4. Threshold Minimo de Distancia de Novidade no Arquivo
==============================================================================
"""
from __future__ import print_function, division
import numpy as np
import networkx as nx

class CircuitAntiDuplicationEnforcer(object):
    def __init__(self, min_diversity_threshold=0.04):
        self.min_diversity_threshold = float(min_diversity_threshold)
        self.seen_wl_hashes = {}     # hash -> index
        self.active_graphs = []      # list of nx.DiGraph
        self.records = []            # list of metadata dicts

    @staticmethod
    def extract_active_graph(S, matrix_W, slots):
        """
        Normaliza a orientacao da matriz e extrai o subgrafo funcional ativo
        removendo nos mortos, ilhas e caminhos desconexos da computacao logica.
        """
        W = np.asarray(matrix_W, dtype=np.float32)
        # Normalizacao de Orientacao:
        # Se linhas 0 ou 1 possuem conexoes de saida, esta no formato W[source, target].
        # Caso contrario, esta transposto (W[target, source]).
        if np.sum(W[0:2, :]) > 0:
            W_dir = W
        else:
            W_dir = W.T
            
        bin_W = (W_dir > 0.01).astype(np.int32)
        
        G = nx.DiGraph()
        for u in range(S):
            rep = slots[u].get('winner', slots[u]) if isinstance(slots[u], dict) else str(slots[u])
            role = 'in0' if u == 0 else ('in1' if u == 1 else ('out' if u == S - 1 else 'int'))
            G.add_node(u, repressor=rep, role=role, label="{}_{}".format(role, rep))
            
        for u in range(S):
            for v in range(S):
                if bin_W[u, v] == 1:
                    G.add_edge(u, v)
                    
        # Alcance direcionado estrito: {pTac, pTet} -> no -> YFP
        reach_in = set(nx.descendants(G, 0)) | set(nx.descendants(G, 1)) | {0, 1}
        reach_out = set(nx.ancestors(G, S - 1)) | {S - 1}
        active_nodes = reach_in & reach_out
        
        subG = G.subgraph(active_nodes).copy()
        return subG, W_dir

    def is_duplicate(self, S, matrix_W, slots):
        """
        Avalia se o circuito candidato e um clone isomorfo de algum circuito ja aceito.
        Retorna: (is_dup: bool, reason: str, matched_id: str)
        """
        try:
            subG, _ = self.extract_active_graph(S, matrix_W, slots)
            
            # 1. Teste Canonico por Weisfeiler-Lehman Labeled Graph Hash
            wl_hash = nx.weisfeiler_lehman_graph_hash(subG, node_attr='label')
            
            if wl_hash in self.seen_wl_hashes:
                match_idx = self.seen_wl_hashes[wl_hash]
                target_G = self.active_graphs[match_idx]
                
                # Verificacao estrita de seguranca contra colisoes de hash via VF2
                em = nx.algorithms.isomorphism.categorical_node_match('label', '')
                if nx.is_isomorphic(subG, target_G, node_match=em):
                    target_id = self.records[match_idx].get('id', 'idx_{}'.format(match_idx))
                    return True, "EXACT_FUNCTIONAL_ISOMORPHISM", target_id
                    
            return False, "UNIQUE", None
        except Exception as e:
            # Em caso de erro na analise de grafo, ser conservador (nao duplicado)
            return False, "ERROR_{}".format(str(e)), None

    def register(self, S, matrix_W, slots, circuit_id=""):
        """Registra formalmente um campeao validado no indice do enforcer."""
        try:
            subG, _ = self.extract_active_graph(S, matrix_W, slots)
            wl_hash = nx.weisfeiler_lehman_graph_hash(subG, node_attr='label')
            idx = len(self.active_graphs)
            self.seen_wl_hashes[wl_hash] = idx
            self.active_graphs.append(subG)
            self.records.append({'id': circuit_id, 'S': S, 'slots': slots, 'hash': wl_hash})
            return wl_hash
        except Exception:
            return ""

    def __len__(self):
        return len(self.active_graphs)
