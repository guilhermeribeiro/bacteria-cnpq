# Cello UCF - Gene Circuit Topology Dataset

Este repositório contém o dataset `circuit_dataset.jsonl`, composto por **12.000 amostras** de topologias de circuitos genéticos simulados com alta precisão estocástica. O dataset foi gerado utilizando Algoritmos Genéticos Cartesianos (CGP) em Julia, buscando descobrir topologias viáveis de portas lógicas biológicas a partir da biblioteca real de proteínas do Cello UCF.

Este dataset foi projetado especificamente como combustível para treinar modelos preditivos de **Graph Neural Networks (GNN)**, atuando como um "modelo substituto" (surrogate model) para simulações caras de equações diferenciais/estocásticas.

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

## 📂 Estrutura do JSONL (Esquema de Dados)

Cada linha do arquivo `circuit_dataset.jsonl` é um objeto JSON independente com a seguinte estrutura:

```json
{
  "id": "1719717551465225272_1", 
  "gate": "NOT", 
  "algorithm_mnemonic": "CGP Julia (Ensemble=20)",
  "ensemble_size": 20,
  "snr": 2.81603,
  
  "steady_state_outputs": {
    "input_low_mean": 304.0,
    "input_high_mean": 353.1
  },
  
  "raw_samples_low": [360.0, 321.0, 130.0, ...], // Array com as 20 respostas estocásticas brutas (Low)
  "raw_samples_high": [345.0, 458.0, 137.0, ...], // Array com as 20 respostas estocásticas brutas (High)
  
  "matrix_W": [
    [0, 15, 0, 0],
    [0, 0, 7, 0],
    [0, 0, 0, 9],
    [0, 0, 0, 0]
  ],
  
  "components": {
    "node_1": {"is_input": 1.0, "is_output": 0.0, "ymin": 0.0, "ymax": 0.0, "Kd": 0.0, "n_real": 0.0},
    "node_2": {"is_input": 0.0, "is_output": 0.0, "ymin": 0.015, "ymax": 0.89, "Kd": 0.07, "n_real": 2.3},
    "node_3": {"is_input": 0.0, "is_output": 0.0, "ymin": 0.04, "ymax": 1.2, "Kd": 0.15, "n_real": 1.7},
    "node_4": {"is_input": 0.0, "is_output": 1.0, "ymin": 0.02, "ymax": 1.0, "Kd": 0.1, "n_real": 1.0}
  },
  
  "metadata_for_humans": {
    "node_1": "LacI",
    "node_2": "AmtR",
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
*   `raw_samples_low` / `raw_samples_high`: Os 20 resultados exatos do SSA estocástico para que o modelo de IA possa, se desejado, aprender também a variância (noise) além da média, reconstruindo distribuições inteiras do sistema sem precisar simular equações caras.
*   `snr`: O valor de aptidão (fitness escalar) otimizado pelo CGP. Usado primariamente para filtrar inviabilidade biológica e impor uma distância saudável entre os níveis lógicos (0 e 1).

---

## 🚀 Utilização em Graph Neural Networks (GNN)

Sugestão de uso para treinamento:
1.  **Nodes:** Podem ser codificados a partir dos parâmetros contínuos da Equação de Hill de cada porta em `components` + `is_input` / `is_output`.
2.  **Edges:** Construídas diretamente pela `matrix_W`. O valor inteiro de `matrix_W` pode ser utilizado com camadas de embedding, ou apenas como arestas booleanas para convoluções de grafo.
3.  **Target (Loss):** O modelo pode ser treinado via regressão (MSE) utilizando o target escalar das médias (`steady_state_outputs`), predição contínua do próprio `snr`, ou até predição distribucional sobre os arrays `raw_samples`.
