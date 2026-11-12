# Análise dos resultados dos modelos de regressão

## 1. Resumo executivo

Foram avaliados 20 algoritmos de regressão em três cenários, com 30 divisões aleatórias de treino e teste por algoritmo:

1. dataset completo, com 12.000 amostras;
2. amostras com SNR menor que 5, totalizando 10.218 registros (85,15% do dataset);
3. amostras com SNR maior ou igual a 5, totalizando 1.782 registros (14,85% do dataset).

As 1.800 execuções esperadas foram concluídas sem erro. Os principais resultados são:

- **SVRRBF obteve o menor MAE nos três cenários**, mas não foi o melhor em RMSE ou R². Isso indica bom desempenho no erro absoluto típico, acompanhado de maior sensibilidade a erros grandes.
- No dataset completo, **MLPRegressor apresentou simultaneamente o menor RMSE (1,9841) e o maior R² (0,2567)**.
- Para SNR < 5, **StackingEnsemble obteve o menor RMSE (0,9408) e o maior R² (0,2596)**, mas sua vantagem sobre modelos mais rápidos foi pequena.
- Para SNR >= 5, **BayesianRidge obteve o menor RMSE (2,1691) e o maior R² (0,0043)**. Esse R² muito próximo de zero mostra que, nessa faixa, nenhum dos modelos avaliados explica de forma relevante a variação do alvo fora da amostra.
- Os rankings do dataset completo e de SNR < 5 são muito semelhantes. Já o cenário SNR >= 5 muda substancialmente a ordenação e favorece modelos lineares regularizados quando o critério é RMSE/R².
- **PolynomialRidge e GradientBoosting** oferecem relações atraentes entre desempenho e tempo no dataset completo e em SNR < 5. Em SNR >= 5, **BayesianRidge, PLSRegression e ElasticNet** são mais eficientes que os ensembles complexos.

Não existe, portanto, um único vencedor para todos os objetivos. A escolha depende de priorizar erro absoluto típico (MAE), erros grandes (RMSE), variância explicada (R²) ou custo computacional.

## 2. Fonte e preparação dos resultados

A análise foi recalculada diretamente a partir dos arquivos gerados por `regression_model_v2.py`:

- `regression_results_v2/resultados_execucoes.csv`: 1.800 linhas, uma para cada execução;
- `regression_results_v2/resultados_medios.csv`: 60 linhas, uma para cada combinação de cenário e algoritmo;
- `regression_results_v2/erros.csv`: nenhum erro registrado.

O total confere com o desenho experimental:

```text
20 algoritmos × 30 execuções × 3 cenários = 1.800 execuções
```

Em cada execução, o conjunto correspondente ao cenário foi dividido em 67% para treino e 33% para teste, usando o número da execução como `random_state`. Os valores apresentados neste relatório são médias das 30 execuções. Os desvios-padrão são populacionais, conforme o uso de `statistics.pstdev` no código.

## 3. Interpretação das métricas

- **MAE** representa o tamanho médio do erro em unidades de SNR. A métrica atribui peso linear aos erros e é menos afetada por poucos erros muito grandes.
- **RMSE** também está nas unidades de SNR, mas penaliza mais fortemente os erros grandes. Um modelo com MAE baixo e RMSE relativamente alto costuma acertar razoavelmente a maioria das amostras e falhar mais severamente em alguns casos.
- **MSE** é a versão quadrática do RMSE. Como ambos produzem a mesma ordenação em uma execução, as tabelas principais apresentam apenas RMSE, que é mais fácil de interpretar.
- **R²** mede o desempenho em relação à previsão constante da média do conjunto de teste. R² igual a 1 representa ajuste perfeito; R² igual a 0 equivale ao baseline da média; R² negativo indica desempenho pior que esse baseline.
- **DP** é o desvio-padrão populacional entre as 30 divisões treino/teste.
- **Tempo** é o tempo médio de ajuste e predição por execução, em segundos, no ambiente utilizado.

RMSE e R² refletem a mesma família de erro quadrático e apresentaram o mesmo ranking médio dos modelos em cada cenário. O R², contudo, acrescenta uma referência importante: informa se o erro quadrático realmente supera uma previsão ingênua baseada na média.

## 4. Dataset completo

### 4.1 Melhores modelos por MAE

| Algoritmo | MAE | DP MAE | RMSE | R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| SVRRBF | **1,2771** | 0,0145 | 2,0733 | 0,1885 | 3,0867 |
| LogTargetGradientBoosting | 1,2930 | 0,0136 | 2,0446 | 0,2108 | 0,6176 |
| NuSVR | 1,3220 | 0,0133 | 2,0461 | 0,2096 | 2,2094 |
| MLPRegressor | 1,3367 | 0,0209 | **1,9841** | **0,2567** | 1,6859 |
| HistGradientBoosting | 1,3370 | 0,0127 | 1,9971 | 0,2470 | 0,4476 |

### 4.2 Melhores modelos por RMSE e R²

| Algoritmo | MAE | RMSE | DP RMSE | R² | DP R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| MLPRegressor | 1,3367 | **1,9841** | 0,0321 | **0,2567** | 0,0100 | 1,6859 |
| StackingEnsemble | 1,3401 | 1,9904 | 0,0332 | 0,2521 | 0,0086 | 4,5824 |
| PolynomialRidge | 1,3554 | 1,9951 | **0,0309** | 0,2484 | 0,0095 | **0,0808** |
| HistGradientBoosting | 1,3370 | 1,9971 | 0,0317 | 0,2470 | 0,0095 | 0,4476 |
| GradientBoosting | 1,3518 | 1,9975 | 0,0335 | 0,2467 | 0,0092 | 0,5007 |

### 4.3 Discussão

O SVRRBF apresentou o menor MAE, com vantagem de 1,24% sobre o LogTargetGradientBoosting. Entretanto, seu RMSE foi 4,50% maior que o do MLPRegressor e seu R² foi 0,1885, contra 0,2567 do MLP. O contraste confirma que o SVRRBF reduz bem o erro absoluto típico, mas controla pior os erros de maior magnitude.

O MLPRegressor foi o modelo mais equilibrado sob o critério quadrático: teve o menor RMSE e o maior R², explicando em média 25,67% da variância observada nos conjuntos de teste. Sua vantagem sobre o StackingEnsemble foi pequena — 0,31% em RMSE —, enquanto o stacking foi aproximadamente 2,7 vezes mais lento.

PolynomialRidge merece destaque prático. Seu RMSE ficou apenas 0,56% acima do melhor resultado, seu R² alcançou 0,2484 e o tempo médio foi de 0,0808 segundo. Ele foi cerca de 21 vezes mais rápido que o MLPRegressor e 57 vezes mais rápido que o StackingEnsemble.

## 5. Cenário SNR < 5

### 5.1 Melhores modelos por MAE

| Algoritmo | MAE | DP MAE | RMSE | R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| SVRRBF | **0,7136** | 0,0091 | 0,9683 | 0,2157 | 2,2809 |
| LogTargetGradientBoosting | 0,7239 | 0,0087 | 0,9587 | 0,2312 | 0,4489 |
| NuSVR | 0,7281 | 0,0088 | 0,9512 | 0,2432 | 1,6262 |
| MLPRegressor | 0,7294 | 0,0128 | 0,9444 | 0,2540 | 1,3500 |
| HistGradientBoosting | 0,7297 | 0,0093 | 0,9447 | 0,2535 | 0,4644 |

### 5.2 Melhores modelos por RMSE e R²

| Algoritmo | MAE | RMSE | DP RMSE | R² | DP R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| StackingEnsemble | 0,7307 | **0,9408** | 0,0114 | **0,2596** | 0,0128 | 4,3547 |
| MLPRegressor | **0,7294** | 0,9444 | 0,0113 | 0,2540 | 0,0132 | 1,3500 |
| HistGradientBoosting | 0,7297 | 0,9447 | **0,0111** | 0,2535 | 0,0128 | 0,4644 |
| GradientBoosting | 0,7373 | 0,9468 | 0,0113 | 0,2501 | **0,0126** | **0,4378** |
| VotingEnsemble | 0,7347 | 0,9485 | 0,0113 | 0,2475 | 0,0136 | 0,7930 |

### 5.3 Discussão

Este cenário reproduziu de perto o comportamento do dataset completo. A correlação de Spearman entre os rankings dos dois cenários foi de 0,964 para MAE e 0,955 para RMSE/R². Essa semelhança é esperada porque SNR < 5 representa 85,15% das amostras e, portanto, influencia fortemente o resultado agregado.

SVRRBF voltou a apresentar o menor MAE, com vantagem de 1,45% sobre o segundo colocado. Mais uma vez, não liderou em RMSE/R²: seu R² de 0,2157 ficou abaixo dos valores de aproximadamente 0,25 a 0,26 alcançados pelos modelos mais fortes no critério quadrático.

StackingEnsemble apresentou o menor RMSE e o maior R², mas a vantagem de RMSE sobre o MLPRegressor foi de apenas 0,38%. HistGradientBoosting e GradientBoosting também ficaram próximos: o RMSE do GradientBoosting foi somente 0,64% maior que o do stacking, e seu tempo foi aproximadamente dez vezes menor. Assim, o pequeno ganho do stacking pode não compensar o custo computacional.

## 6. Cenário SNR >= 5

### 6.1 Melhores modelos por MAE

| Algoritmo | MAE | DP MAE | RMSE | R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| SVRRBF | **1,5533** | 0,0478 | 2,2413 | -0,0630 | 0,0788 |
| NuSVR | 1,5592 | 0,0421 | **2,1833** | **-0,0087** | 0,0595 |
| HuberRegressor | 1,5621 | 0,0456 | 2,2190 | -0,0419 | 0,0536 |
| LinearSVR | 1,5624 | 0,0499 | 2,2583 | -0,0791 | **0,0142** |
| LogTargetGradientBoosting | 1,5997 | 0,0464 | 2,2157 | -0,0391 | 0,0937 |

### 6.2 Melhores modelos por RMSE e R²

| Algoritmo | MAE | RMSE | DP RMSE | R² | DP R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| BayesianRidge | 1,6193 | **2,1691** | 0,0906 | **0,0043** | **0,0083** | 0,0084 |
| PLSRegression | **1,6185** | 2,1709 | 0,0899 | 0,0026 | 0,0123 | 0,0108 |
| ElasticNet | 1,6235 | 2,1719 | **0,0890** | 0,0016 | 0,0132 | 0,0117 |
| StackingEnsemble | 1,6238 | 2,1754 | 0,0918 | -0,0015 | 0,0102 | 1,5530 |
| Ridge | 1,6283 | 2,1804 | 0,0907 | -0,0063 | 0,0154 | **0,0076** |

### 6.3 Discussão

O cenário SNR >= 5 apresentou uma mudança clara de regime. A correlação do ranking de MAE com o dataset completo caiu para 0,507 e a correlação de RMSE/R² caiu para 0,200. Na comparação com SNR < 5, as correlações foram ainda menores: 0,376 para MAE e 0,174 para RMSE/R². Logo, as conclusões obtidas no conjunto completo não se transferem automaticamente para a faixa de maior SNR.

SVRRBF manteve o menor MAE, mas sua vantagem sobre NuSVR foi de apenas 0,38%. O NuSVR foi mais equilibrado: elevou o MAE em somente 0,0059 unidade, mas reduziu o RMSE de 2,2413 para 2,1833 e melhorou o R² de -0,0630 para -0,0087.

No critério quadrático, os modelos lineares regularizados dominaram. BayesianRidge, PLSRegression e ElasticNet ficaram praticamente empatados: a diferença de RMSE entre o primeiro e o terceiro foi de apenas 0,13%. O StackingEnsemble não trouxe ganho e foi aproximadamente 185 vezes mais lento que o BayesianRidge.

O resultado mais relevante, porém, é a magnitude do R². Apenas BayesianRidge (0,0043), PLSRegression (0,0026) e ElasticNet (0,0016) apresentaram R² médio positivo, e mesmo esses valores são praticamente zero. Os outros 17 modelos tiveram R² médio negativo. Portanto, com as características e configurações atuais, a previsão em SNR >= 5 é essencialmente equivalente — ou inferior — à previsão constante da média quando avaliada por erro quadrático.

Isso pode refletir uma combinação de fatores: apenas 1.782 amostras nessa faixa, menor relação sinal-informação nas 27 características utilizadas, mudança estrutural na relação entre atributos e alvo ou hiperparâmetros inadequados para esse subconjunto. O resultado não sustenta a conclusão de que uma relação linear forte foi aprendida; os modelos lineares apenas degradaram menos que os modelos mais flexíveis.

## 7. Discussão comparativa entre os três cenários

### 7.1 Síntese dos vencedores

| Cenário | Menor MAE | Melhor MAE | Menor RMSE | Melhor RMSE | Maior R² | Melhor R² |
|---|---|---:|---|---:|---|---:|
| Dataset completo | SVRRBF | 1,2771 | MLPRegressor | 1,9841 | MLPRegressor | 0,2567 |
| SNR < 5 | SVRRBF | 0,7136 | StackingEnsemble | 0,9408 | StackingEnsemble | 0,2596 |
| SNR >= 5 | SVRRBF | 1,5533 | BayesianRidge | 2,1691 | BayesianRidge | 0,0043 |

### 7.2 O dataset completo é dominado pela faixa SNR < 5

A faixa SNR < 5 contém 85,15% dos registros. Por isso, os rankings do dataset completo se aproximam muito mais desse cenário do que do cenário SNR >= 5. Esse efeito aparece tanto nas altas correlações de ranking quanto na repetição dos mesmos grupos de modelos entre os primeiros colocados.

Consequentemente, uma boa métrica agregada pode ocultar desempenho insuficiente na faixa minoritária. O MLPRegressor, por exemplo, apresenta R² de 0,2567 no conjunto completo e 0,2540 em SNR < 5, mas cai para -0,0442 em SNR >= 5. A avaliação estratificada é, portanto, indispensável.

### 7.3 Os erros absolutos não são diretamente comparáveis entre as faixas

MAE e RMSE são menores em SNR < 5, mas isso não prova, por si só, maior capacidade preditiva. Ao filtrar o próprio alvo, cada subconjunto passa a ter amplitude e variância diferentes. Além disso, os modelos são treinados sabendo implicitamente a faixa verdadeira à qual cada amostra pertence.

O R² ajuda a contextualizar essa diferença: os melhores valores são semelhantes entre o dataset completo (0,2567) e SNR < 5 (0,2596), embora MAE e RMSE sejam muito diferentes. Em SNR >= 5, o R² próximo de zero revela que o problema não é apenas uma mudança de escala dos erros; há uma perda efetiva de poder explicativo em relação ao baseline da média.

### 7.4 MAE e RMSE conduzem a decisões diferentes

SVRRBF lidera consistentemente em MAE, mas nunca lidera em RMSE ou R². Esse padrão indica uma distribuição de erros com bom desempenho central e uma cauda de erros maiores. Se todos os desvios tiverem custo aproximadamente linear, SVRRBF é adequado. Se erros grandes forem especialmente prejudiciais, MLPRegressor, StackingEnsemble, modelos de boosting ou modelos lineares regularizados — conforme o cenário — são escolhas mais apropriadas.

### 7.5 Complexidade não garante generalização

No dataset completo e em SNR < 5, métodos não lineares e ensembles ocupam as primeiras posições de RMSE/R². Em SNR >= 5, essa vantagem desaparece: BayesianRidge, PLSRegression e ElasticNet ficam no topo, enquanto ExtraTrees, KNNDistance e AdaBoost têm R² fortemente negativo.

O cenário de alta SNR combina menos dados com baixo sinal preditivo aparente. Nessas condições, a maior flexibilidade pode aumentar a variância e o sobreajuste. Ainda assim, como até os melhores modelos ficam próximos de R² = 0, a prioridade deve ser investigar dados, atributos e protocolo experimental antes de simplesmente aumentar a complexidade.

## 8. Ranking global

O ranking abaixo combina, com pesos iguais, a posição média de MAE e a posição média de RMSE nos três cenários. R² não recebeu um terceiro peso porque apresentou exatamente a mesma ordenação média do RMSE; incluí-lo duplicaria a influência do erro quadrático.

| Algoritmo | Ranking médio MAE | Ranking médio RMSE/R² | Ranking combinado | Tempo médio (s) |
|---|---:|---:|---:|---:|
| StackingEnsemble | 7,00 | 2,33 | **4,67** | 3,497 |
| MLPRegressor | 6,33 | 4,00 | 5,17 | 1,130 |
| NuSVR | 2,67 | 7,67 | 5,17 | 1,298 |
| LogTargetGradientBoosting | 3,00 | 8,00 | 5,50 | 0,387 |
| SVRRBF | **1,00** | 12,33 | 6,67 | 1,815 |
| HistGradientBoosting | 8,00 | 6,67 | 7,33 | 0,433 |
| GradientBoosting | 10,33 | 6,33 | 8,33 | 0,344 |
| PolynomialRidge | 10,67 | 6,67 | 8,67 | **0,063** |
| VotingEnsemble | 10,00 | 8,67 | 9,33 | 0,648 |
| BayesianRidge | 13,00 | 8,67 | 10,83 | 0,022 |

StackingEnsemble lidera esse resumo porque mantém bom RMSE/R² nos três cenários. O ranking, entretanto, deve ser interpretado com cautela: ele atribui a mesma importância aos cenários, apesar da diferença no número de amostras, e não representa a magnitude absoluta das diferenças. Também esconde que o R² do stacking é negativo em SNR >= 5 e que alternativas muito mais rápidas ficam praticamente empatadas em vários casos.

## 9. Recomendações

### 9.1 Se MAE for a métrica principal

- Usar **SVRRBF** nos três cenários.
- Para SNR >= 5, considerar **NuSVR**: seu MAE é apenas 0,38% maior e seu RMSE/R² é substancialmente melhor que o do SVRRBF.

### 9.2 Se RMSE ou R² forem as métricas principais

- Dataset completo: **MLPRegressor**.
- SNR < 5: **StackingEnsemble** se o custo computacional for aceitável; **HistGradientBoosting** ou **GradientBoosting** para uma solução mais eficiente.
- SNR >= 5: **BayesianRidge**, reconhecendo que o R² de 0,0043 representa ganho preditivo praticamente nulo sobre a média.

### 9.3 Se desempenho e velocidade tiverem importância semelhante

- Dataset completo: **PolynomialRidge** ou **HistGradientBoosting**.
- SNR < 5: **GradientBoosting**, **HistGradientBoosting** ou **PolynomialRidge**.
- SNR >= 5: **BayesianRidge**, **PLSRegression** ou **ElasticNet**.

### 9.4 Se for necessário um único modelo

- **NuSVR** é uma escolha equilibrada quando MAE tem prioridade, especialmente por degradar menos em SNR >= 5.
- **MLPRegressor** é competitivo no conjunto completo e em SNR < 5 quando RMSE/R² são prioritários, mas não generaliza bem para SNR >= 5.
- **LogTargetGradientBoosting** oferece bom compromisso entre MAE, RMSE e tempo nos dois cenários maiores.
- Nenhum modelo único resolve adequadamente o cenário SNR >= 5 segundo o R² observado.

## 10. Estabilidade

Os desvios-padrão foram baixos no dataset completo e em SNR < 5, indicando estabilidade diante das diferentes divisões treino/teste. Em SNR >= 5, a dispersão aumentou de forma clara. Entre os vencedores de RMSE, por exemplo, o DP passou de aproximadamente 0,03 no conjunto completo e 0,011 em SNR < 5 para aproximadamente 0,09 em SNR >= 5.

Essa maior variabilidade é compatível com o menor número de amostras: cada conjunto de teste contém aproximadamente 588 registros, contra cerca de 3.372 em SNR < 5. Alterações na composição do teste exercem, portanto, influência maior nas métricas.

No cenário SNR >= 5, BayesianRidge apresentou o menor DP de R² entre os cinco melhores modelos no critério quadrático (0,0083), enquanto ElasticNet apresentou o menor DP de RMSE (0,0890). As diferenças médias entre esses modelos são menores que a dispersão entre divisões, o que reforça a necessidade de evitar afirmações fortes de superioridade sem um teste pareado.

## 11. Limitações e cuidados de interpretação

1. **Os cenários têm tamanhos, amplitudes e variâncias do alvo diferentes.** MAE e RMSE não devem ser comparados diretamente entre as faixas como se representassem a mesma tarefa.
2. **O filtro dos cenários usa o próprio valor-alvo.** As análises SNR < 5 e SNR >= 5 são estratificações retrospectivas. Em produção, não é possível escolher o modelo da faixa sem conhecer previamente o SNR verdadeiro ou sem construir um primeiro estágio de classificação, sujeito aos seus próprios erros.
3. **O R² de SNR >= 5 é praticamente nulo.** Nessa faixa, selecionar o “melhor” regressor entre os avaliados não resolve a ausência de poder explicativo relevante.
4. **Não houve ajuste sistemático de hiperparâmetros.** O ranking representa as configurações testadas e pode mudar após otimização feita exclusivamente nos dados de treino.
5. **As 30 execuções são divisões aleatórias relacionadas.** Elas oferecem robustez descritiva, mas não substituem validação cruzada aninhada quando há seleção de hiperparâmetros.
6. **As diferenças entre os primeiros colocados são pequenas em vários casos.** Testes pareados sobre as mesmas divisões, intervalos de confiança das diferenças ou procedimentos como Wilcoxon/Friedman são recomendados antes de afirmar superioridade estatística.
7. **O tempo é específico do ambiente de execução.** Ele permite comparação relativa nesta máquina, mas não constitui estimativa universal.
8. **Não há baseline explícito entre os 20 modelos selecionados.** Embora o R² forneça comparação implícita com a média em cada teste, incluir `DummyRegressor` na tabela final tornaria o baseline visível também para MAE, RMSE e tempo.
9. **O particionamento é aleatório por linha.** Se houver múltiplas observações relacionadas ao mesmo circuito, deve-se considerar divisão agrupada por `circuit_id` para evitar que informações do mesmo circuito apareçam simultaneamente em treino e teste.

## 12. Próximos passos sugeridos

1. Investigar separadamente o cenário SNR >= 5, começando pela distribuição do alvo, correlações, qualidade das features e possíveis subgrupos.
2. Repetir a avaliação com `DummyRegressor` incluído no conjunto final e, se aplicável, com divisão agrupada por circuito.
3. Realizar ajuste de hiperparâmetros com validação interna, preservando um teste externo ou usando validação cruzada aninhada.
4. Calcular intervalos de confiança e testes pareados das diferenças entre os modelos mais próximos.
5. Analisar resíduos por faixa de SNR e por circuito para identificar heterocedasticidade, viés sistemático e outliers.
6. Se a aplicação exigir especialização por faixa, avaliar uma arquitetura de dois estágios e medir o impacto dos erros do classificador de faixa no desempenho final.

## 13. Conclusão

Os três cenários revelam dois regimes distintos. O dataset completo e a faixa SNR < 5 apresentam rankings muito semelhantes e permitem explicar aproximadamente 26% da variância com os melhores modelos. Neles, SVRRBF minimiza o MAE, enquanto MLPRegressor, StackingEnsemble e métodos de boosting controlam melhor o erro quadrático.

Em SNR >= 5, a ordenação muda e o poder preditivo praticamente desaparece: o melhor R² médio é apenas 0,0043. Modelos lineares regularizados são os menos frágeis e mais eficientes nessa faixa, mas não se deve interpretar sua liderança como evidência de boa predição.

Assim, as recomendações centrais são: **SVRRBF quando MAE for prioritário; MLPRegressor ou StackingEnsemble para RMSE/R² nos cenários completo e SNR < 5; BayesianRidge como opção mais eficiente em SNR >= 5, acompanhado da ressalva de R² quase nulo**. Antes de uso prático, a principal prioridade deve ser compreender e melhorar a previsibilidade da faixa SNR >= 5.
