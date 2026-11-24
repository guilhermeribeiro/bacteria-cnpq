# Regressão de SNR a partir do grafo XOR

Este diretório é independente do pipeline de meta-características. O script lê
diretamente `../xor-dataset/circuit_dataset_XOR.jsonl`, constrói um grafo dirigido
para cada linha e entrega os grafos à GNN.

Ele usa a mesma representação de `build_circuit_graph_dataset.py`:

- `matrix_W[i][j] > 0` representa uma aresta `j -> i`;
- os 7 atributos originais de nó são preservados;
- os 5 atributos originais de aresta são preservados;
- `snr` é somente o alvo da regressão;
- estatísticas globais, como densidade, clustering e centralidade, não são usadas.

O treinamento minimiza RMSE. A cada época são impressas MAE, RMSE, MSE e R²
para os conjuntos de treino e validação, sempre na escala original do SNR.

O split é estratificado para regressão. Os valores comuns são separados em dez
faixas de quantis e os SNRs raros (`SNR >= 2`, por padrão) formam uma faixa
própria. Cada faixa é distribuída proporcionalmente entre treino, validação e
teste, evitando que os casos extremos fiquem concentrados em um único conjunto.
Os parâmetros podem ser alterados com `--stratify-bins` e
`--outlier-threshold`.

Foi usada `GINEConv` porque ela realiza convolução no grafo usando todos os
atributos de aresta. Uma `GCNConv` convencional aceitaria somente um peso escalar
por aresta e descartaria parte da representação original.

## Execução

Na raiz do projeto:

```bash
python -m pip install -r xor_graph_regression/requirements.txt
python xor_graph_regression/train_xor_graph_regressor.py
```

Teste rápido:

```bash
python xor_graph_regression/train_xor_graph_regressor.py \
  --limit 100 --epochs 5 --output-dir xor_graph_regression/runs/teste
```

Os resultados incluem `best_model.pt`, `metrics.json`, `history.json` e
`test_predictions.csv`.
