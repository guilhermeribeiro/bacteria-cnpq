# Cello UCF - Gene Circuit Topology Dataset

Este dataset `circuit_dataset.jsonl`, é composto por **12.000 amostras** de topologias de circuitos genéticos simulados com alta precisão estocástica. O dataset foi gerado utilizando Algoritmos Genéticos Cartesianos (CGP) em Julia, buscando descobrir topologias viáveis de portas lógicas biológicas a partir da biblioteca real de proteínas do Cello UCF.
Ele foi projetado especificamente como combustível para treinar modelos preditivos de **Graph Neural Networks (GNN)**, atuando como um "modelo substituto" (surrogate model) para simulações caras de equações diferenciais/estocásticas.

---

## 📊 Estatísticas do Dataset

*   **Total de Registros:** 12.000 circuitos
*   **Tamanho dos Circuitos (Nós):**
    *   `N = 3`: 2.000 amostras
    *   `N = 4`: 3.391 amostras
    *   `N = 5`: 3.249 amostras
    *   `N = 6`: 3.360 amostras
*   **Limiar de Qualidade Física (SNR):** $\ge 0.85$ (Variando de 0.85 até 23.14)
*   **Tempo de Integração ($t$):** 500 (Garante o Steady State / Platô de resposta da proteína)
*   **Ensemble Estocástico:** 20 simulações do Algoritmo de Gillespie (SSAStepper) independentes por condição, por circuito.

---

## 🧬 Biologia e Simulação (Como os dados foram gerados)

Ao invés de apenas plugar equações determinísticas de ODE, a métrica de sobrevivência dos circuitos foi calculada estocasticamente.

1.  **Física do Sinal:** O circuito foi exposto a um input de indução baixo e um input alto. O algoritmo mediu a proteína repórter (YFP) em estado estacionário.
2.  **Ruído Estocástico (Ensemble de 20):** Como células bacterianas possuem ruído intrínseco de cópias moleculares, calcular a resposta biológica apenas 1 vez gera um fitness excessivamente ruidoso. Para resolver isso, **todo circuito gravado no dataset sofreu 20 simulações estocásticas independentes**.
3.  **Cálculo do SNR:** A média das 20 realizações foi utilizada para determinar a verdadeira janela diferencial do circuito. Somente circuitos que conseguiram entregar um *Signal-to-Noise Ratio* (SNR) robusto acima de 0.85 foram aprovados para o dataset final.
4.  **Comportamento (Lógica):** Devido à natureza diferencial da métrica de SNR ($Signal = High - Low$), todos os circuitos arquivados aprenderam a se comportar fisicamente como eficientes sensores diretos (**Portas YES / Buffers**).

---

## 🧬 Biblioteca de Proteínas (Cello UCF Mappings)

Esta tabela mapeia os índices numéricos utilizados na matriz de adjacência `matrix_W` para as respectivas portas lógicas e parâmetros físicos da biblioteca Cello UCF:

| Julia Dataset Index | Protein Name | UCF Gate Name | ymin | ymax | Kd | n |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | AmtR | A1_AmtR | 0.06 | 3.80 | 0.07 | 1.60 |
| 2 | BM3R1 | B1_BM3R1 | 0.004 | 0.50 | 0.04 | 3.40 |
| 3 | BM3R1 | B2_BM3R1 | 0.005 | 0.50 | 0.15 | 2.90 |
| 4 | BM3R1 | B3_BM3R1 | 0.01 | 0.80 | 0.26 | 3.40 |
| 5 | BetI | E1_BetI | 0.07 | 3.80 | 0.41 | 2.40 |
| 6 | AmeR | F1_AmeR | 0.20 | 3.80 | 0.09 | 1.40 |
| 7 | HlyIIR | H1_HlyIIR | 0.07 | 2.50 | 0.19 | 2.60 |
| 8 | IcaRA | I1_IcaRA | 0.08 | 2.20 | 0.10 | 1.40 |
| 9 | LitR | L1_LitR | 0.07 | 4.30 | 0.05 | 1.70 |
| 10 | LmrA | N1_LmrA | 0.20 | 2.20 | 0.18 | 2.10 |
| 11 | PhlF | P1_PhlF | 0.01 | 3.90 | 0.03 | 4.00 |
| 12 | PhlF | P2_PhlF | 0.02 | 4.10 | 0.13 | 3.90 |
| 13 | PhlF | P3_PhlF | 0.02 | 6.80 | 0.23 | 4.20 |
| 14 | QacR | Q1_QacR | 0.01 | 2.40 | 0.05 | 2.70 |
| 15 | QacR | Q2_QacR | 0.03 | 2.80 | 0.21 | 2.40 |
| 16 | PsrA | R1_PsrA | 0.20 | 5.90 | 0.19 | 1.80 |
| 17 | SrpR | S1_SrpR | 0.003 | 1.30 | 0.01 | 2.90 |
| 18 | SrpR | S2_SrpR | 0.003 | 2.10 | 0.04 | 2.60 |
| 19 | SrpR | S3_SrpR | 0.004 | 2.10 | 0.06 | 2.80 |
| 20 | SrpR | S4_SrpR | 0.007 | 2.10 | 0.10 | 2.80 |

---

## 📂 Estrutura do JSONL (Esquema de Dados)

Cada linha do arquivo `circuit_dataset.jsonl` é um objeto JSON independente com a seguinte estrutura:

```json
{
  "id": "38279134794832922_3",
  "gate": "NOT",
  "algorithm_mnemonic": "CGP Julia (Ensemble=20)",
  "ensemble_size": 20,
  "snr": 2.6694265772021075,
  "steady_state_outputs": {
    "input_low_mean": 74.55,
    "input_high_mean": 97.6
  },
  "raw_samples_low": [23.0, 37.0, 88.0, 33.0, 93.0, 97.0, 73.0, 86.0, 25.0, 45.0, 143.0, 59.0, 66.0, 86.0, 103.0, 77.0, 42.0, 132.0, 74.0, 109.0],
  "raw_samples_high": [83.0, 97.0, 55.0, 131.0, 95.0, 149.0, 84.0, 117.0, 72.0, 75.0, 38.0, 143.0, 118.0, 80.0, 45.0, 185.0, 117.0, 66.0, 134.0, 68.0],
  "matrix_W": [
    [0, 0, 4, 0],
    [0, 0, 0, 2],
    [0, 11, 0, 0],
    [0, 0, 0, 0]
  ],
  "components": {
    "node_1": {"is_output": 0.0, "is_input": 1.0, "ymin": 0.0, "ymax": 0.0, "n_real": 0.0, "Kd": 0.0},
    "node_2": {"is_output": 0.0, "is_input": 0.0, "ymin": 0.004, "ymax": 0.5, "n_real": 3.4, "Kd": 0.04},
    "node_3": {"is_output": 0.0, "is_input": 0.0, "ymin": 0.01, "ymax": 3.9, "n_real": 4.0, "Kd": 0.03},
    "node_4": {"is_output": 1.0, "is_input": 0.0, "ymin": 0.02, "ymax": 1.0, "n_real": 1.0, "Kd": 0.1}
  },
  "metadata_for_humans": {
    "node_1": "LacI",
    "node_2": "BM3R1",
    "node_3": "PhlF",
    "node_4": "YFP"
  }
}
```


### Dicionário de Chaves:
*   `matrix_W`: Matriz de adjacência (Topologia do Grafo). Ela dita tanto a conexão entre os nós quanto a biologia estrutural do circuito:
    *   **Direção do Fluxo:** Se existe um valor $V > 0$ na posição `W[i, j]`, significa que há uma conexão regulatória (aresta direcionada) **do nó `j` para o nó `i`** (Sinal flui de $j \rightarrow i$).
    *   **Identidade Biológica:** O valor inteiro $V$ na matriz não é apenas um booleano de conexão. Ele é o ID exato da proteína repressora (na biblioteca Cello UCF) que está atuando como o gate (nó receptor) daquela conexão.
    *   **Ordem dos Nós:** A primeira linha/coluna (`i=1`, `j=1`) é sempre o Input biológico (promotor indutível/LacI), e o último índice (`i=N`, `j=N`) é obrigatoriamente a saída do circuito (Output Repórter/YFP). 
*   `components`: Dicionário mapeando cada nodo biológico às suas características físicas da Equação de Hill (Parâmetros empíricos da proteína: produção basal $y_{min}$, produção máxima $y_{max}$, constante de dissociação $K_d$, e cooperatividade $n$). O nó de input puro (LacI) possui esses parâmetros modelados como zerados para atuar apenas como trigger externo.
*   `raw_samples_low` / `raw_samples_high`: Os 20 resultados exatos do SSA estocástico para que o modelo de IA possa, se desejado, aprender também a variância (ruído) além da média, reconstruindo distribuições inteiras do sistema sem precisar simular equações caras.
*   `snr`: O valor de aptidão (fitness escalar) otimizado pelo CGP. Usado primariamente para filtrar inviabilidade biológica e impor uma distância saudável entre os níveis lógicos (0 e 1).
