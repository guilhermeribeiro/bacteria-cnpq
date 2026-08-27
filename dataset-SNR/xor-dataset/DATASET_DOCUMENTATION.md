# GeneNet-DARTS Synthetic Biology Dataset
**Dataset Oficial - Circuito Logico XOR**

## Visao Geral
Este dataset contem topologias de circuitos geneticos sinteticos simulados e otimizados computacionalmente para reproduzir o comportamento de portas logicas (como **XOR**, **AND**, **OR**). O objetivo principal desta base de dados e servir como fundacao para o treinamento de modelos de **Machine Learning e Deep Learning em Grafos (GCNs)** capazes de prever o *Signal-to-Noise Ratio* (SNR) de uma topologia biologica sem a necessidade de rodar simulacoes mecanicistas custosas (ODEs/SSA).

* **Filtro de Qualidade:** Todos os circuitos listados possuem qualidade validada matematicamente (`SNR >= 0.8`). Circuitos com erro topologico, *timeouts* ou matrizes ausentes foram rigorosamente descartados.
* **Algoritmo Gerador:** DARTS Estocástico (Differentiable Architecture Search). O backend de busca arquitetural roda em Python 2 / Theano, enquanto a simulacao de avaliacao biologica (Fitness/SNR) roda nativamente em Julia (Gillespie SSA).

---

## Estrutura do Arquivo (JSON Lines)
O dataset e fornecido em formato `.jsonl` (ex: `circuit_dataset_XOR.jsonl`), onde cada linha corresponde a um circuito biologico completo.

Abaixo esta o **Schema** de cada objeto JSON:

```json
{
  "id": "2026-08-25_23-04-32",
  "gate": "XOR",
  "snr": 1.2543,
  "algorithm_mnemonic": "DARTS",
  "wiring_matrix": [ ... ],
  "slots": [ ... ],
  "steady_state_outputs": {
      "state_00_mean": 0.12,
      "state_01_mean": 85.4,
      ...
  },
  "steady_state_samples": {
      "state_00": [ ... 32 amostras ... ],
      "state_01": [ ... 32 amostras ... ],
      "state_10": [ ... 32 amostras ... ],
      "state_11": [ ... 32 amostras ... ]
  }
}
```

---

## Fluxo de Geracao de Novos Datasets

O pipeline de geracao e orquestrado por um script mestre (Python 3) que invoca o DARTS via Theano (Python 2) para encontrar a topologia, e envia o grafo para um Daemon de alta performance em Julia que realiza a extracao das matrizes e calculo do SNR. Nao e mais necessario pos-processamento.

### Gerando o Dataset (Passo Unico)
1. Edite o script `darts_orchestrator.py` (ou crie uma copia) alterando a logica alvo (ex: `target_gate = "AND"`).
2. Rode o script no seu servidor:
   ```bash
   python darts_orchestrator.py
   ```
3. O script iniciara a busca e, ao encontrar um circuito valido, automaticamente salvara o JSON final incluindo o SNR e as matrizes com as 32 amostras estocasticas de populacao (`steady_state_samples`).

**Nota sobre Resumo (Continue):**
Se o servidor for reiniciado, basta rodar o comando novamente. O `darts_orchestrator.py` ira contar quantas linhas o arquivo `.jsonl` alvo ja possui e retomara a geracao de onde parou.

## Dicionário de Dados (Data Dictionary)

Esta seção detalha rigorosamente a semântica, as convenções e a origem de cada atributo presente nos objetos `.jsonl` do dataset.

### Metadados e Identificação
* **`id`** (`String`): Identificador único do circuito. Convencionado como o *timestamp* da execução (ex: `"2026-08-25_23-04-32"`). Origina-se da nomenclatura da pasta de execução temporária.
* **`gate`** (`String`): A porta lógica alvo que o circuito tentou otimizar (ex: `"XOR"`, `"AND"`). Usado para categorização da base.
* **`algorithm_mnemonic`** (`String`): Sigla do pipeline gerador (ex: `"DARTS"`). Registra qual variante arquitetural foi utilizada para criar o dado.

### Topologia e Genética (O Grafo)
* **`wiring_matrix`** (`List[List[Float]]`): A matriz de adjacência contínua do circuito de tamanho $N \times N$. 
  * **Interpretação:** Representa os pesos estruturais (força de regulação) entre os nós. Linhas são origens (fontes), colunas são alvos. 
  * **Convenção:** Valores menores que `0.01` são considerados ruído do gradiente contínuo e são matematicamente podados (removidos) ao instanciar o grafo biológico discreto.
  * **Origem:** Produto direto da convergência dos pesos estruturais $\alpha$ do algoritmo DARTS (Differentiable Architecture Search).
* **`slots`** (`List[Dict]`): O mapeamento físico das proteínas (componentes biológicos) para cada nó da `wiring_matrix`. Contém $N$ objetos:
  * `slot` (`Int`): O índice posicional do nó. `0` e `1` são sempre os indutores (ex: `pTac`, `pTet`). O último slot (`N-1`) é sempre o repórter (ex: `YFP`). Os nós intermediários (`2` a `N-2`) são portas repressoras lógicas internas.
  * `winner` (`String`): O nome da proteína vencedora alocada no nó (ex: `"SrpR"`, `"PhlF"`, ou `"Vazio"` se o nó foi podado).
  * `probabilities` (`Dict[String, Float]`): A distribuição de probabilidade Softmax final das proteínas candidatas disputando aquele slot durante o treino contínuo.
  * `confidence_pct` (`Float`): Confiança da atribuição (`winner` vs. segundo colocado).
  * **Origem:** Relaxamento contínuo DARTS discretizado no final do pipeline pelo *Optuna*.

### Métricas de Avaliação e Ruído Biológico (Fitness)
* **`snr`** (`Float`): *Signal-to-Noise Ratio* (Relação Sinal-Ruído) calculada para a porta lógica alvo. 
  * **Interpretação:** É a principal "Nota de Fitness" do circuito. Valores $\ge 0.8$ indicam portas lógicas biologicamente viáveis e distinguíveis do ruído intrínseco. É calculado pela diferença das médias de sinal Alto/Baixo dividida pela soma dos desvios padrão.
  * **Origem:** Calculado pelo avaliador mecanicista em Julia (Hill Kinetics).
* **`steady_state_outputs`** (`Dict`): Contém a **Média** exata de expressão da proteína repórter (`YFP`) para cada estado de entrada (ex: `"state_00_mean"`, `"state_01_mean"`, etc.).
  * **Interpretação:** Serve como uma métrica rápida determinística para verificar se a tabela-verdade foi cumprida.
* **`steady_state_samples`** (`Dict`): O campo mais valioso e denso do dataset. Contém as 32 amostras estocásticas individuais para cada estado de entrada (`"state_00"`, `"state_01"`, `"state_10"`, `"state_11"`).
  * **Interpretação:** Cada lista possui **exatamente 32 números flutuantes**. Cada número representa a concentração de `YFP` no tempo $t = 500$ de **uma única célula estocástica** simulada isoladamente. É a representação pura do ruído biológico intrínseco do circuito.
  * **Origem:** Produto final e direto de 32 trajetórias independentes do *Gillespie Stochastic Simulation Algorithm (SSA)* rodando paralelamente nas *Threads* nativas do Julia via biblioteca `JumpProcesses`.



