# Análise dos resultados dos modelos de regressão

## 1. Resumo executivo

Foram avaliados **20 algoritmos de regressão** no cenário `dataset_completo`, com **1.500 amostras**, **27 features** e **30 execuções por algoritmo**. O log contém as **600 execuções esperadas**, todas com métricas e tempo informados. Não há outros cenários nesse arquivo.

- **PolynomialRidge foi o melhor modelo nas três métricas principais:** MAE de **0,060648**, RMSE de **0,077381** e R² de **0,459779**.
- **VotingEnsemble** ficou em segundo em MAE (0,061005), RMSE (0,077992) e R² (0,451389), mas levou em média 0,3957 s por execução, contra 0,0237 s do PolynomialRidge.
- **StackingEnsemble** ocupou a terceira posição nas mesmas métricas (MAE 0,061488; RMSE 0,078137; R² 0,449498), com tempo médio de 2,1087 s.
- **MLPRegressor** apresentou os piores MAE (0,071845), RMSE (0,092577) e R² (0,226282) entre as configurações avaliadas.
- A liderança do PolynomialRidge também representa um bom equilíbrio entre qualidade preditiva e custo computacional: foi um dos algoritmos mais rápidos.

Neste experimento, diferentemente do relatório de referência, **o mesmo algoritmo lidera em MAE, RMSE e R²**. As diferenças entre os primeiros colocados, entretanto, devem ser avaliadas com testes pareados antes de serem tratadas como estatisticamente significativas.

## 2. Fonte e preparação dos resultados

A análise foi construída a partir do arquivo de texto fornecido com o log das execuções. Os valores de desempenho médio são os informados nas linhas `Final (AVG 30)`; os desvios-padrão e tempos médios foram calculados a partir das 30 linhas `run` de cada modelo.

```text
20 algoritmos × 30 execuções × 1 cenário = 600 execuções
1.500 amostras | 27 features
Cenário: dataset_completo
```

Foram identificados 20 conjuntos completos de execuções numeradas de 0 a 29, sem ausência de métricas nas linhas analisadas. **O log não informa explicitamente** o percentual de treino/teste, o método de validação, os hiperparâmetros, a natureza do alvo nem sua unidade; por isso, esses detalhes não são assumidos neste relatório. Não há evidência de outros cenários nem de um arquivo separado de erros.

## 3. Interpretação das métricas

- **MAE** (*Mean Absolute Error*): média dos valores absolutos dos erros; menor é melhor.
- **MSE** (*Mean Squared Error*): média dos erros ao quadrado; menor é melhor.
- **RMSE** (*Root Mean Squared Error*): raiz quadrada do MSE; penaliza mais fortemente erros grandes; menor é melhor.
- **R²** (*coeficiente de determinação*): compara o desempenho quadrático ao baseline de predição da média no teste; maior é melhor, e valores negativos indicam desempenho inferior a esse baseline.
- **DP**: desvio-padrão populacional das 30 execuções, calculado com `statistics.pstdev`.
- **Tempo (s)**: média dos tempos exibidos por execução no log; inclui o que o programa mediu, cuja delimitação exata não está documentada no arquivo.

As tabelas priorizam MAE, RMSE e R². O MSE permanece disponível na tabela consolidada, para não perder informação do log original. As médias de RMSE e de R² mostram a mesma ordenação entre os 20 modelos neste conjunto de resultados.

## 4. Dataset completo

### 4.1 Melhores modelos por MAE

| Algoritmo | MAE | DP MAE | RMSE | R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| PolynomialRidge | 0,060648 | 0,001907 | 0,077381 | 0,459779 | 0,0237 |
| VotingEnsemble | 0,061005 | 0,001958 | 0,077992 | 0,451389 | 0,3957 |
| StackingEnsemble | 0,061488 | 0,002022 | 0,078137 | 0,449498 | 2,1087 |
| GradientBoosting | 0,061782 | 0,002028 | 0,078800 | 0,439867 | 0,1567 |
| LogTargetGradientBoosting | 0,061910 | 0,002040 | 0,078973 | 0,437434 | 0,1600 |

### 4.2 Melhores modelos por RMSE e R²

| Algoritmo | MAE | RMSE | DP RMSE | R² | DP R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| PolynomialRidge | 0,060648 | 0,077381 | 0,002352 | 0,459779 | 0,030304 | 0,0237 |
| VotingEnsemble | 0,061005 | 0,077992 | 0,002295 | 0,451389 | 0,026758 | 0,3957 |
| StackingEnsemble | 0,061488 | 0,078137 | 0,002309 | 0,449498 | 0,023846 | 2,1087 |
| GradientBoosting | 0,061782 | 0,078800 | 0,002395 | 0,439867 | 0,030133 | 0,1567 |
| LogTargetGradientBoosting | 0,061910 | 0,078973 | 0,002299 | 0,437434 | 0,028897 | 0,1600 |

### 4.3 Discussão

O PolynomialRidge obteve MAE 0,060648 e RMSE 0,077381. Comparado ao VotingEnsemble, sua vantagem relativa foi de **0,59% em MAE** e **0,78% em RMSE**, com R² médio de 0,459779 contra 0,451389.

O VotingEnsemble ficou próximo em erro, mas seu tempo médio foi **16,7 vezes** o do PolynomialRidge. O StackingEnsemble, também competitivo em precisão, demorou **89,1 vezes** mais. Portanto, o aumento de complexidade não trouxe ganho nas métricas médias deste experimento.

Os modelos GradientBoosting e LogTargetGradientBoosting também tiveram resultados próximos dos líderes, ainda que inferiores. Já o MLPRegressor ficou na última posição nas três métricas, o que descreve **esta configuração e este conjunto de dados**, não uma inferioridade geral das redes neurais.

## 5. Comparação dos algoritmos e custo computacional

### 5.1 Modelos mais rápidos

| Algoritmo | Tempo médio (s) | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Ridge | 0,0100 | 0,063006 | 0,079842 | 0,424975 |
| ElasticNet | 0,0100 | 0,066059 | 0,082854 | 0,381182 |
| BayesianRidge | 0,0100 | 0,063225 | 0,079960 | 0,423348 |
| PLSRegression | 0,0107 | 0,069732 | 0,086777 | 0,321077 |
| KNNDistance | 0,0170 | 0,066060 | 0,084537 | 0,355499 |
| PolynomialRidge | 0,0237 | 0,060648 | 0,077381 | 0,459779 |
| SVRRBF | 0,0257 | 0,063490 | 0,080363 | 0,417749 |
| HuberRegressor | 0,0577 | 0,063007 | 0,080147 | 0,420494 |

**Ridge, ElasticNet e BayesianRidge** aparecem entre os métodos de menor tempo, mas nenhum supera o PolynomialRidge em erro ou R². O modelo líder também tem execução curta, reduzindo o incentivo a adotar ensembles mais caros apenas por ganhos marginais — que, neste caso, nem se observaram nas médias.

### 5.2 Comparação direta dos principais candidatos

| Critério | PolynomialRidge | VotingEnsemble | StackingEnsemble | GradientBoosting |
|---|---:|---:|---:|---:|
| MAE | 0,060648 | 0,061005 | 0,061488 | 0,061782 |
| RMSE | 0,077381 | 0,077992 | 0,078137 | 0,078800 |
| R2 | 0,459779 | 0,451389 | 0,449498 | 0,439867 |
| Tempo médio (s) | 0,0237 | 0,3957 | 2,1087 | 0,1567 |

## 6. Ranking geral por métrica

### 6.1 Ranking completo por RMSE (crescente)

| Posição | Algoritmo | MAE | RMSE | R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| 1 | PolynomialRidge | 0,060648 | 0,077381 | 0,459779 | 0,0237 |
| 2 | VotingEnsemble | 0,061005 | 0,077992 | 0,451389 | 0,3957 |
| 3 | StackingEnsemble | 0,061488 | 0,078137 | 0,449498 | 2,1087 |
| 4 | GradientBoosting | 0,061782 | 0,078800 | 0,439867 | 0,1567 |
| 5 | LogTargetGradientBoosting | 0,061910 | 0,078973 | 0,437434 | 0,1600 |
| 6 | NuSVR | 0,062271 | 0,079425 | 0,430764 | 0,0627 |
| 7 | RandomForest | 0,062221 | 0,079581 | 0,428795 | 0,2813 |
| 8 | BaggingDecisionTree | 0,062399 | 0,079790 | 0,425785 | 0,4680 |
| 9 | Ridge | 0,063006 | 0,079842 | 0,424975 | 0,0100 |
| 10 | BayesianRidge | 0,063225 | 0,079960 | 0,423348 | 0,0100 |
| 11 | HuberRegressor | 0,063007 | 0,080147 | 0,420494 | 0,0577 |
| 12 | LinearSVR | 0,063053 | 0,080269 | 0,418745 | 0,0933 |
| 13 | SVRRBF | 0,063490 | 0,080363 | 0,417749 | 0,0257 |
| 14 | ExtraTrees | 0,063260 | 0,081082 | 0,407008 | 0,2067 |
| 15 | HistGradientBoosting | 0,063396 | 0,081323 | 0,403263 | 0,5000 |
| 16 | ElasticNet | 0,066059 | 0,082854 | 0,381182 | 0,0100 |
| 17 | KNNDistance | 0,066060 | 0,084537 | 0,355499 | 0,0170 |
| 18 | AdaBoost | 0,068558 | 0,085095 | 0,347055 | 0,0727 |
| 19 | PLSRegression | 0,069732 | 0,086777 | 0,321077 | 0,0107 |
| 20 | MLPRegressor | 0,071845 | 0,092577 | 0,226282 | 0,3800 |

### 6.2 Ranking combinado de MAE e RMSE

O ranking combinado utiliza pesos iguais para a posição em MAE e a posição em RMSE. O R² não recebe um terceiro peso por reproduzir o ranking de RMSE neste experimento. Em caso de empate, os modelos compartilham a mesma pontuação combinada (a ordem de apresentação não indica diferença real).

| Algoritmo | Rank MAE | Rank RMSE/R² | Média dos ranks | Tempo (s) |
|---|---:|---:|---:|---:|
| PolynomialRidge | 1 | 1 | 1,00 | 0,0237 |
| VotingEnsemble | 2 | 2 | 2,00 | 0,3957 |
| StackingEnsemble | 3 | 3 | 3,00 | 2,1087 |
| GradientBoosting | 4 | 4 | 4,00 | 0,1567 |
| LogTargetGradientBoosting | 5 | 5 | 5,00 | 0,1600 |
| RandomForest | 6 | 7 | 6,50 | 0,2813 |
| NuSVR | 7 | 6 | 6,50 | 0,0627 |
| BaggingDecisionTree | 8 | 8 | 8,00 | 0,4680 |
| Ridge | 9 | 9 | 9,00 | 0,0100 |
| HuberRegressor | 10 | 11 | 10,50 | 0,0577 |
| BayesianRidge | 12 | 10 | 11,00 | 0,0100 |
| LinearSVR | 11 | 12 | 11,50 | 0,0933 |
| ExtraTrees | 13 | 14 | 13,50 | 0,2067 |
| SVRRBF | 15 | 13 | 14,00 | 0,0257 |
| HistGradientBoosting | 14 | 15 | 14,50 | 0,5000 |
| ElasticNet | 16 | 16 | 16,00 | 0,0100 |
| KNNDistance | 17 | 17 | 17,00 | 0,0170 |
| AdaBoost | 18 | 18 | 18,00 | 0,0727 |
| PLSRegression | 19 | 19 | 19,00 | 0,0107 |
| MLPRegressor | 20 | 20 | 20,00 | 0,3800 |

A classificação combinada reforça a liderança do PolynomialRidge. Como há somente um cenário, esse ranking não permite inferir robustez entre diferentes faixas ou distribuições do alvo.

## 7. Estabilidade das 30 execuções

### 7.1 Dispersão entre os principais modelos

| Algoritmo | DP MAE | DP RMSE | DP R² | DP tempo (s) |
|---|---:|---:|---:|---:|
| PolynomialRidge | 0,001907 | 0,002352 | 0,030304 | 0,0091 |
| VotingEnsemble | 0,001958 | 0,002295 | 0,026758 | 0,0204 |
| StackingEnsemble | 0,002022 | 0,002309 | 0,023846 | 0,0330 |
| GradientBoosting | 0,002028 | 0,002395 | 0,030133 | 0,0070 |
| LogTargetGradientBoosting | 0,002040 | 0,002299 | 0,028897 | 0,0058 |
| NuSVR | 0,001666 | 0,002141 | 0,031645 | 0,0044 |
| RandomForest | 0,002034 | 0,002428 | 0,028932 | 0,0399 |
| BaggingDecisionTree | 0,002015 | 0,002433 | 0,029101 | 0,0218 |

O desvio-padrão mede a variação observada entre execuções, não a incerteza estatística da diferença entre dois modelos. Sem conhecer o protocolo exato de divisão dos dados e sem um teste pareado, pequenas vantagens nas médias não justificam uma conclusão de superioridade estatística.

### 7.2 Variações e melhores execuções observadas

| Algoritmo | Menor MAE | Menor RMSE | Maior R² | Faixa de R² |
|---|---:|---:|---:|---:|
| PolynomialRidge | 0,056396 | 0,072503 | 0,525076 | 0,4005 a 0,5251 |
| VotingEnsemble | 0,057097 | 0,072917 | 0,517290 | 0,3899 a 0,5173 |
| StackingEnsemble | 0,057564 | 0,073644 | 0,507617 | 0,3941 a 0,5076 |
| GradientBoosting | 0,057521 | 0,073974 | 0,489364 | 0,3652 a 0,4894 |
| LogTargetGradientBoosting | 0,058107 | 0,074591 | 0,481436 | 0,3628 a 0,4814 |

Esses extremos representam execuções individuais e **não** devem substituir a comparação pelos resultados médios ou por validação externa.

## 8. Análise interpretativa dos resultados

### 8.1 A complexidade não garantiu o melhor desempenho

O PolynomialRidge superou os ensembles VotingEnsemble e StackingEnsemble nas médias de todas as métricas principais e apresentou menor tempo de execução. Uma possível interpretação é que a transformação polinomial associada à regularização capturou bem relações relevantes deste conjunto. Contudo, o log não traz diagnóstico de atributos nem detalhes dos hiperparâmetros; essa hipótese precisaria ser examinada diretamente.

### 8.2 O melhor R² ainda indica espaço para melhorias

O maior R² médio observado foi **0,4598**, equivalente a aproximadamente **45,98%** da variância do alvo explicada, no sentido da métrica de teste. Isso demonstra ganho em relação à predição pela média, mas também sinaliza que parcela relevante da variação não é capturada pelo modelo.

### 8.3 Diferenças pequenas no topo exigem cautela

A diferença entre os dois menores RMSE médios foi de somente **0,000611**, ou cerca de **0,78%** em relação ao VotingEnsemble. Para concluir que esse ganho é robusto, seria preciso comparar o erro dos modelos nas mesmas amostras de teste ou nas mesmas partições.

### 8.4 Desempenho fraco de alguns métodos

MLPRegressor, PLSRegression e AdaBoost ficaram nas posições inferiores do ranking por RMSE. Esse resultado pode depender de pré-processamento, escala, configuração, tamanho amostral e relação entre features e alvo. Não é possível isolar as causas usando apenas as métricas agregadas do log.

## 9. Recomendações

### 9.1 Se MAE for prioritário

- Preferir **PolynomialRidge**, que apresentou o menor MAE médio (0,060648).
- **VotingEnsemble** (0,061005) e **StackingEnsemble** (0,061488) são alternativas próximas em erro, porém mais custosas.

### 9.2 Se RMSE ou R² forem prioritários

- Preferir **PolynomialRidge**, com menor RMSE (0,077381) e maior R² (0,459779).
- **VotingEnsemble** e **StackingEnsemble** podem ser mantidos como comparadores em testes de robustez e validação externa.

### 9.3 Se desempenho e velocidade tiverem importância semelhante

- **PolynomialRidge** representa o melhor compromisso observado, sem sacrificar nenhuma das métricas principais.
- **Ridge** e **BayesianRidge** são baselines lineares rápidos e úteis para verificar o ganho adicional da transformação polinomial.

### 9.4 Se for necessário escolher um único modelo

A recomendação com base **estritamente neste log** é **PolynomialRidge**. A escolha final deve, contudo, ser confirmada em dados externos ou avaliação adequadamente isolada da seleção de hiperparâmetros.

## 10. Limitações e cuidados de interpretação

1. **Há somente um cenário no log.** Não é possível reproduzir a comparação entre `SNR < 5` e `SNR >= 5` do relatório-modelo, nem falar em desempenho por faixa.
2. **A escala e unidade do alvo não foram informadas.** MAE e RMSE são apresentados nas unidades numéricas do alvo; não se pode afirmar que estejam em unidades de SNR.
3. **O protocolo treino/teste não foi documentado.** Não se deve assumir divisão 67/33, uso de `random_state` ou divisão por circuito apenas porque apareciam no relatório anterior.
4. **Os hiperparâmetros e etapas de pré-processamento não constam no log.** Os rankings descrevem as configurações efetivamente executadas, sem garantir que cada família tenha sido otimizada.
5. **Não foi informado um baseline explícito**, como `DummyRegressor`, para comparar MAE/RMSE e custo de execução diretamente.
6. **As 30 execuções podem ser estatisticamente dependentes.** Desvios-padrão descritivos não substituem avaliação de diferenças pareadas nem validação cruzada aninhada.
7. **O custo computacional é específico do ambiente e da medição adotada**, não sendo transferível diretamente a outras máquinas.
8. **Não há resultados por subgrupo, distribuição dos resíduos ou intervalos de confiança**. Esses elementos são importantes para verificar estabilidade e riscos de erros extremos.
9. **O histórico de falhas não está documentado em separado.** O log contém 600 execuções completas, mas não prova a ausência de erros em etapas não registradas.

## 11. Próximos passos sugeridos

1. Confirmar o protocolo de treino/teste, a origem dos 27 atributos e a escala do alvo.
2. Incluir `DummyRegressor` e registrar as métricas do baseline com o mesmo protocolo.
3. Executar validação externa ou validação cruzada aninhada, evitando usar o teste para ajuste de hiperparâmetros.
4. Aplicar comparação pareada entre PolynomialRidge, VotingEnsemble e StackingEnsemble nas mesmas divisões, com intervalos de confiança para as diferenças.
5. Inspecionar resíduos, outliers, relação entre predições e valores reais, e resultados por faixa relevante do alvo.
6. Avaliar a sensibilidade do PolynomialRidge ao grau polinomial, regularização, padronização e composição das features.
7. Se houver identificadores de grupos ou circuitos, investigar necessidade de separação agrupada para reduzir riscos de vazamento.

## 12. Tabela consolidada dos 20 modelos

Médias reportadas nas linhas `Final (AVG 30)` e tempo médio calculado a partir das 30 execuções. Ordenação por RMSE crescente.

| Algoritmo | MAE | MSE | RMSE | R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| PolynomialRidge | 0,060648 | 0,005993 | 0,077381 | 0,459779 | 0,0237 |
| VotingEnsemble | 0,061005 | 0,006088 | 0,077992 | 0,451389 | 0,3957 |
| StackingEnsemble | 0,061488 | 0,006111 | 0,078137 | 0,449498 | 2,1087 |
| GradientBoosting | 0,061782 | 0,006215 | 0,078800 | 0,439867 | 0,1567 |
| LogTargetGradientBoosting | 0,061910 | 0,006242 | 0,078973 | 0,437434 | 0,1600 |
| NuSVR | 0,062271 | 0,006313 | 0,079425 | 0,430764 | 0,0627 |
| RandomForest | 0,062221 | 0,006339 | 0,079581 | 0,428795 | 0,2813 |
| BaggingDecisionTree | 0,062399 | 0,006372 | 0,079790 | 0,425785 | 0,4680 |
| Ridge | 0,063006 | 0,006380 | 0,079842 | 0,424975 | 0,0100 |
| BayesianRidge | 0,063225 | 0,006399 | 0,079960 | 0,423348 | 0,0100 |
| HuberRegressor | 0,063007 | 0,006429 | 0,080147 | 0,420494 | 0,0577 |
| LinearSVR | 0,063053 | 0,006449 | 0,080269 | 0,418745 | 0,0933 |
| SVRRBF | 0,063490 | 0,006464 | 0,080363 | 0,417749 | 0,0257 |
| ExtraTrees | 0,063260 | 0,006579 | 0,081082 | 0,407008 | 0,2067 |
| HistGradientBoosting | 0,063396 | 0,006618 | 0,081323 | 0,403263 | 0,5000 |
| ElasticNet | 0,066059 | 0,006870 | 0,082854 | 0,381182 | 0,0100 |
| KNNDistance | 0,066060 | 0,007152 | 0,084537 | 0,355499 | 0,0170 |
| AdaBoost | 0,068558 | 0,007246 | 0,085095 | 0,347055 | 0,0727 |
| PLSRegression | 0,069732 | 0,007535 | 0,086777 | 0,321077 | 0,0107 |
| MLPRegressor | 0,071845 | 0,008584 | 0,092577 | 0,226282 | 0,3800 |

## 13. Conclusão

O experimento com 1.500 amostras, 27 features, 20 modelos e 30 execuções por modelo aponta **PolynomialRidge como o candidato mais forte**: liderou simultaneamente em MAE, RMSE e R² e apresentou tempo médio de aproximadamente 0,024 segundo. VotingEnsemble e StackingEnsemble chegaram perto em desempenho, mas com custos computacionais substancialmente superiores.

O maior R² médio de 0,459779 indica desempenho preditivo relevante em comparação ao baseline implícito da média, embora ainda exista espaço para melhoria. A conclusão prática é priorizar **PolynomialRidge para validação adicional**, sem transformar diferenças pequenas nas médias em alegações de superioridade estatística. Como o material possui apenas o cenário completo, qualquer análise estratificada exigirá novos resultados.
