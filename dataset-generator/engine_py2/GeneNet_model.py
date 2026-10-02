# -*- coding: utf-8 -*-
"""
GeneNet_model.py
Modelo Dinâmico DARTS parametrizado para qualquer S (nós) e qualquer porta lógica.
Elimina qualquer necessidade de sobreescrita de arquivo em disco.
"""
import numpy as np
import theano.tensor as T
import theano

# 1. PARÂMETROS GLOBAIS DE SIMULAÇÃO (EULER)
dt = 0.01
N  = 300

# 2. BIBLIOTECA PADRÃO DE PROTEÍNAS (CALIBRADA COM CHASSI BIOFÍSICO CHAMPION)
PROT_NAMES = ['SrpR', 'PhlF', 'AmeR', 'BetI', 'QacR', 'AmtR', 'Vazio', 'YFP', 'pTac', 'pTet']
ymin_lib = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0, 0.0,    0.0005, 0.0005])
ymax_lib = np.array([3.000,  3.000,  3.000,  3.000,  3.000,  3.000,  0.0, 1.0,    3.000,  3.000])
Kd_lib   = np.array([1.000,  1.000,  1.000,  1.000,  1.000,  1.000,  1.0, 1.0,    1.000,  1.000])
n_lib    = np.array([2.000,  2.000,  2.000,  2.000,  2.000,  2.000,  1.0, 1.0,    1.000,  1.000])
is_repressor_lib = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

K = len(PROT_NAMES)
S = 7
GATE = "XOR"

def configure(num_slots=7, gate="XOR"):
    global S, GATE
    S = int(num_slots)
    GATE = str(gate).upper()

# 3. CLASSE DA REDE NEURAL DARTS
class network:
    def __init__(self, W=None, entropy_weight=0.1, S_nodes=None):
        global S
        self.S = S if S_nodes is None else int(S_nodes)
        
        if W is None:
            W = np.random.normal(0.0, 1.8, [self.S, self.S])
            
        self.W = theano.shared(value=W.astype(theano.config.floatX), name='W', borrow=True)
        
        alpha_init = np.random.normal(loc=0.0, scale=1.0, size=[self.S, K])
        # Slot 0 = pTac (índice 8)
        alpha_init[0, :] = -100.0
        alpha_init[0, 8] = 100.0
        # Slot 1 = pTet (índice 9)
        alpha_init[1, :] = -100.0
        alpha_init[1, 9] = 100.0
        # Slot S-1 = YFP (índice 7)
        alpha_init[self.S - 1, :] = -100.0
        alpha_init[self.S - 1, 7] = 100.0
        # Slots intermediários (2 a S-2): banir YFP (7), pTac (8), pTet (9)
        alpha_init[2:self.S - 1, 7:10] = -100.0
        
        self.alpha = theano.shared(value=alpha_init.astype(theano.config.floatX), name='alpha')
        self.Ax = theano.shared(value=np.float64(1.0).astype(theano.config.floatX), name='Ax', borrow=True)
        
        mask_val = np.ones((self.S, self.S), dtype=theano.config.floatX)
        mask_val[0, :] = 0.0
        mask_val[1, :] = 0.0
        mask_val[:, self.S - 1] = 0.0
        self.mask = theano.shared(value=mask_val, name='mask', borrow=True)
        
        self.parameters = [self.W, self.alpha, self.Ax]
        self.entropy_weight = entropy_weight
        
        self.y_min_t = theano.shared(ymin_lib.astype(theano.config.floatX))
        self.y_max_t = theano.shared(ymax_lib.astype(theano.config.floatX))
        self.Kd_t    = theano.shared(Kd_lib.astype(theano.config.floatX))
        self.n_t     = theano.shared(n_lib.astype(theano.config.floatX))
        self.is_repressor_t = theano.shared(is_repressor_lib.astype(theano.config.floatX))

    def networkFunction(self, y, x):
        probs = T.nnet.softmax(self.alpha)
        
        # Pinar entradas e saída
        probs = T.set_subtensor(probs[0, :], 0.0)
        probs = T.set_subtensor(probs[0, 8], 1.0)
        probs = T.set_subtensor(probs[1, :], 0.0)
        probs = T.set_subtensor(probs[1, 9], 1.0)
        probs = T.set_subtensor(probs[self.S - 1, :], 0.0)
        probs = T.set_subtensor(probs[self.S - 1, 7], 1.0)
        
        # Banir YFP, pTac, pTet dos slots intermediários
        probs = T.set_subtensor(probs[2:self.S - 1, 7:10], 0.0)
        sum_mid = T.sum(probs[2:self.S - 1, :7], axis=1, keepdims=True) + 1e-8
        probs = T.set_subtensor(probs[2:self.S - 1, :7], probs[2:self.S - 1, :7] / sum_mid)
        
        Kd_eff     = T.dot(probs, self.Kd_t)
        n_eff      = T.dot(probs, self.n_t)
        ymin_eff   = T.dot(probs, self.y_min_t)
        ymax_eff   = T.dot(probs, self.y_max_t)
        is_rep_eff = T.dot(probs, self.is_repressor_t)
        
        # Straight-Through Estimator (STE)
        W_sig = T.nnet.sigmoid(self.W)
        W_bin = T.round(W_sig)
        W_pos = (W_sig + theano.gradient.disconnected_grad(W_bin - W_sig)) * self.mask
        
        y_src = y
        Kd_src = Kd_eff.dimshuffle(0, 'x')
        n_src = n_eff.dimshuffle(0, 'x')
        ymin_src = ymin_eff.dimshuffle(0, 'x')
        ymax_src = ymax_eff.dimshuffle(0, 'x')
        is_rep_src = is_rep_eff.dimshuffle(0, 'x')
        
        repression_j = 1.0 / (1.0 + (y_src * is_rep_src / Kd_src) ** n_src)
        hill_j = ymin_src + (ymax_src - ymin_src) * repression_j
        
        total_production = T.dot(W_pos, hill_j)
        return total_production - y + x

    def normalize(self, y):
        return self.Ax * y

    def regularize(self):
        W_sig = T.nnet.sigmoid(self.W)
        W_bin = T.round(W_sig)
        W_pos = (W_sig + theano.gradient.disconnected_grad(W_bin - W_sig)) * self.mask
        l1_penalty = T.sum(T.abs_(W_pos))
        
        probs = T.nnet.softmax(self.alpha)
        probs = T.set_subtensor(probs[0, :], 0.0)
        probs = T.set_subtensor(probs[0, 8], 1.0)
        probs = T.set_subtensor(probs[1, :], 0.0)
        probs = T.set_subtensor(probs[1, 9], 1.0)
        probs = T.set_subtensor(probs[self.S - 1, :], 0.0)
        probs = T.set_subtensor(probs[self.S - 1, 7], 1.0)
        
        probs = T.set_subtensor(probs[2:self.S - 1, 7:10], 0.0)
        sum_mid = T.sum(probs[2:self.S - 1, :7], axis=1, keepdims=True) + 1e-8
        probs = T.set_subtensor(probs[2:self.S - 1, :7], probs[2:self.S - 1, :7] / sum_mid)
        
        entropy = -T.sum(probs * T.log(probs + 1e-8)) / self.S
        return (l1_penalty, self.entropy_weight * entropy)

# 4. FUNÇÕES DE DADOS E TABELAS VERDADE
def eval_logic(b0, b1, gate_name):
    if gate_name == "XOR":
        return b0 != b1
    elif gate_name == "AND":
        return b0 and b1
    elif gate_name == "OR":
        return b0 or b1
    elif gate_name == "NAND":
        return not (b0 and b1)
    elif gate_name == "NOR":
        return not (b0 or b1)
    elif gate_name == "NOT":
        return not b0
    return b0 != b1

def desiredFunction(B):
    global S, N, GATE
    K = max(1, B // 4)
    B_act = 4 * K
    y0 = 0.1 * np.ones([S, B_act], dtype=theano.config.floatX) * np.random.normal(loc=1.0, scale=0.001, size=[S, B_act])
    
    x0_base = np.array([0.1]*K + [0.1]*K + [1.9]*K + [1.9]*K, dtype=theano.config.floatX)
    x1_base = np.array([0.1]*K + [1.9]*K + [0.1]*K + [1.9]*K, dtype=theano.config.floatX)
    
    x = np.zeros([N, S, B_act], dtype=theano.config.floatX)
    x[:, 0, :] = x0_base.reshape(1, B_act) * np.random.normal(loc=1.0, scale=0.05, size=[N, B_act])
    x[:, 1, :] = x1_base.reshape(1, B_act) * np.random.normal(loc=1.0, scale=0.05, size=[N, B_act])
    
    y_ = np.zeros(B_act, dtype=theano.config.floatX)
    for i in range(B_act):
        b0 = (x0_base[i] > 1.0)
        b1 = (x1_base[i] > 1.0)
        y_[i] = 0.8 if eval_logic(b0, b1, GATE) else 0.1
            
    return [y0, x, y_]

def inputData(B):
    global S, N, GATE
    y0 = 0.1 * np.ones([S, B]).astype(theano.config.floatX) * np.random.normal(loc=1.0, scale=0.001, size=[S, B])
    
    x0_base = np.array([0.1, 0.1, 1.9, 1.9] * (B // 4 + 1))[:B].astype(theano.config.floatX)
    x1_base = np.array([0.1, 1.9, 0.1, 1.9] * (B // 4 + 1))[:B].astype(theano.config.floatX)
    
    x = np.zeros([N, S, B], dtype=theano.config.floatX)
    x[:, 0, :] = x0_base.reshape(1, B) * np.random.normal(loc=1.0, scale=0.1, size=[N, B])
    x[:, 1, :] = x1_base.reshape(1, B) * np.random.normal(loc=1.0, scale=0.1, size=[N, B])
    
    y_ = np.zeros(B, dtype=theano.config.floatX)
    for i in range(B):
        b0 = (x0_base[i] > 1.0)
        b1 = (x1_base[i] > 1.0)
        if eval_logic(b0, b1, GATE):
            y_[i] = 0.8
        else:
            y_[i] = 0.1
            
    return [y0, x, y_]
