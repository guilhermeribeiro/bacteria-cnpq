# Pipeline de Geração e Simulação de Circuitos (GeneNet-DARTS)

Pipeline de busca arquitetural contínua (DARTS) acoplado à validação estocástica (Gillespie SSA) para síntese e avaliação de portas lógicas genéticas (`XOR`, `NOT`, `AND`).

---

## Requisitos

- **Python 3.8+** (orquestrador e Optuna): `pip install optuna networkx numpy`
- **Python 2.7** com Theano (necessário para a relaxação contínua do GeneNet)
- **Julia 1.8+** (necessário para a simulação estocástica SSA via `JumpProcesses.jl`)

Os caminhos dos executáveis podem ser definidos manualmente em `config.json` ou detectados automaticamente pelo ambiente (`"auto"`).

---

## Como Usar

### 1. Geração de Datasets (`run_generator.py`)

Executa a busca híbrida (DARTS via Theano + simulação estocástica em Julia) e salva os circuitos no dataset JSONL.

O gerador suporta **dois modos de amostragem**:

#### A. Modo Multi-Espectro Estratificado (Recomendado para GNNs):
Coleta amostras em todo o espectro biofísico de SNR (evitando truncamento de dados e colapso de contraste nas IAs):
- **Falhas / Negativos / Ruído Alto ($SNR < 0.50$):** Cota de 1.500 amostras (ensina *anti-patterns* e falhas à IA).
- **Sub-ótimos / Transição ($0.50 \le SNR < 0.85$):** Cota de 1.500 amostras (restaura o gradiente de transição).
- **Funcionais Padrão ($0.85 \le SNR < 2.00$):** Cota de 4.500 amostras (amostras aprovadas com boa fidelidade).
- **Alta Fidelidade / Campeões ($SNR \ge 2.00$):** Cota de 2.500 amostras (circuitos de alto contraste).

```bash
# Geração Multi-Espectro com cotas balanceadas (Meta total: 10.000 circuitos)
python run_generator.py --gate XOR --target_samples 10000 --stratified_spectrum --n_jobs 8
```

#### B. Modo Clássico (Limiar Fixo):
Salva apenas os circuitos que atingem um limiar estrito:

```bash
# Salva apenas circuitos com SNR >= 0.85
python run_generator.py --gate XOR --target_samples 2000 --no_stratified_spectrum --snr_threshold 0.85 --n_jobs 8

# Porta NOT (Meta clássica: 12.000 circuitos)
python run_generator.py --gate NOT --target_samples 12000 --no_stratified_spectrum --snr_threshold 0.85 --n_jobs 8
```

O script salva o progresso de forma incremental em formato `.jsonl` e retoma automaticamente em caso de interrupção.

### 2. Avaliação de Circuitos Candidatos (`evaluate_candidate.py`)

Permite avaliar uma topologia candidata diretamente no simulador de Julia (Gillespie SSA, 32 réplicas) para obter o SNR pareado de Hiscock.

#### Linha de comando:
```bash
# Teste com circuito campeão
python evaluate_candidate.py --demo

# Teste a partir de arquivo JSON com 'matrix_W' e 'slots'
python evaluate_candidate.py --file circuito.json --gate XOR
```

#### Via Python:
```python
from evaluate_candidate import evaluate_circuit

# matrix_W: matriz de adjacência (N x N)
# slots: lista de genes (ex: ['pTac', 'pTet', 'PhlF', 'AmtR', 'YFP'])
snr, resultados = evaluate_circuit(matrix_W, slots, gate="XOR", ensemble_size=32)

print(f"SNR: {snr:.4f}")
```

---

## Estrutura do Módulo

- `run_generator.py`: Orquestrador de busca e geração de datasets.
- `evaluate_candidate.py`: Interface para avaliação estocástica de circuitos arbitrários.
- `config.json`: Parâmetros de execução (caminhos, workers, ensemble size e limiares).
- `core/`: Módulos de detecção de isomorfismo (VF2), métricas de novidade e cliente do daemon Julia.
- `engine_py2/`: Backend Theano para a relaxação contínua do DARTS.
- `engine_julia/`: Backend Julia para a simulação estocástica (Gillespie SSA com JumpProcesses.jl).
