# Análise dos resultados dos modelos de regressão — 2.000 amostras

## 1. Resumo executivo

Foram avaliados **20 algoritmos de regressão** no cenário `dataset_completo`, utilizando **2.000 amostras**, **27 features** e **30 execuções por algoritmo**. O log contém as **600 execuções previstas**, com MAE, MSE, RMSE, R² e tempo registrados para cada execução.

- **GradientBoosting apresentou o menor MAE médio (0,127197)**, seguido por LogTargetGradientBoosting (0,128408) e VotingEnsemble (0,130496).
- **VotingEnsemble apresentou o menor RMSE médio (0,230137)**, praticamente empatado com ExtraTrees (0,230599), que foi aproximadamente 2,17 vezes mais rápido.
- **ExtraTrees apresentou o maior R² médio (0,669316)**, seguido por StackingEnsemble (0,629194 aproximadamente) e VotingEnsemble (0,618783).
- **As classificações por RMSE e R² não coincidem**: GradientBoosting é 3º em RMSE, mas apenas 10º em R², sobretudo pela influência de uma execução com R² de -5,018574.
- **A estabilidade merece atenção**: entre as 600 execuções, 48 tiveram R² negativo; alguns algoritmos exibiram dispersão elevada, especialmente no R².

Não existe um vencedor universal: **GradientBoosting** é indicado quando o erro absoluto médio é prioritário; **VotingEnsemble** quando o RMSE deve ser minimizado; **ExtraTrees** quando se privilegia o R² médio com custo computacional moderado. Essas recomendações são descritivas e não estabelecem superioridade estatística.

## 2. Fonte e preparação dos resultados

O relatório foi elaborado diretamente a partir do log textual fornecido, referente à execução de `regression_model_v2.py` sobre `graph_statistics.csv` no diretório `artifacts_256_merged`. Não foram utilizados dados brutos, predições individuais ou arquivos CSV de resultados.

```text
20 algoritmos × 30 execuções × 1 cenário = 600 execuções
2.000 amostras | 27 features
Cenário: dataset_completo
Execuções numeradas de 0 a 29 para cada algoritmo
Process finished with exit code 0
```

As médias de MAE, MSE, RMSE e R² foram recalculadas a partir das linhas `run` e conferidas com os valores `Final (AVG 30)` (sujeitos ao arredondamento de seis casas). Os desvios-padrão são **populacionais** (`statistics.pstdev`) e os tempos são médias dos valores apresentados em segundos, originalmente arredondados a centésimos. O log **não especifica explicitamente** proporção treino/teste, hiperparâmetros, forma de pré-processamento, estratégia de particionamento, unidade do alvo ou estrutura de dependência entre linhas; essas informações não são presumidas.

## 3. Interpretação das métricas

- **MAE**: erro absoluto médio; valores menores indicam previsões tipicamente mais próximas do alvo.
- **MSE**: erro quadrático médio, sensível a erros extremos.
- **RMSE**: raiz quadrada do MSE; expressa o erro quadrático na escala do alvo, enfatizando grandes desvios.
- **R²**: coeficiente de determinação relativo à previsão constante da média do conjunto de teste; 1 indica ajuste perfeito, 0 corresponde à referência de média e valores negativos indicam resultado inferior a essa referência.
- **DP**: desvio-padrão populacional das 30 execuções, uma medida descritiva da dispersão entre rodadas.
- **Tempo**: duração média registrada em cada linha do log; é dependente do computador e da implementação.

**Atenção:** menor RMSE médio não implica necessariamente maior R² médio. O R² de cada execução é normalizado pela variância do alvo no conjunto de teste. Se essa variância muda entre divisões, a média dos R² pode ordenar modelos de modo distinto da média dos RMSE.

## 4. Dataset completo

### 4.1 Melhores modelos por MAE

| Algoritmo | MAE | DP MAE | RMSE | R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| GradientBoosting | 0,127197 | 0,00822 | 0,236044 | 0,510061 | 0,1923 |
| LogTargetGradientBoosting | 0,128408 | 0,00896 | 0,256357 | 0,531057 | 0,1990 |
| VotingEnsemble | 0,130496 | 0,00838 | 0,230137 | 0,618783 | 0,4577 |
| AdaBoost | 0,135319 | 0,00983 | 0,273349 | 0,518203 | 0,0517 |
| RandomForest | 0,135411 | 0,00940 | 0,253036 | 0,536229 | 0,3120 |

### 4.2 Melhores modelos por RMSE

| Algoritmo | MAE | RMSE | DP RMSE | R² | DP R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| VotingEnsemble | 0,130496 | 0,230137 | 0,07394 | 0,618783 | 0,58694 | 0,4577 |
| ExtraTrees | 0,138519 | 0,230599 | 0,06472 | 0,669316 | 0,28510 | 0,2107 |
| GradientBoosting | 0,127197 | 0,236044 | 0,08235 | 0,510061 | 1,04508 | 0,1923 |
| StackingEnsemble | 0,143576 | 0,241708 | 0,07302 | 0,629193 | 0,35297 | 2,4570 |
| BaggingDecisionTree | 0,135498 | 0,252436 | 0,07977 | 0,540720 | 0,73379 | 0,6730 |

### 4.3 Melhores modelos por R²

| Algoritmo | R² | DP R² | RMSE | MAE | R² negativos |
|---|---:|---:|---:|---:|---:|
| ExtraTrees | 0,669316 | 0,28510 | 0,230599 | 0,138519 | 1/30 |
| StackingEnsemble | 0,629193 | 0,35297 | 0,241708 | 0,143576 | 1/30 |
| VotingEnsemble | 0,618783 | 0,58694 | 0,230137 | 0,130496 | 1/30 |
| PolynomialRidge | 0,562742 | 0,32871 | 0,266997 | 0,153179 | 2/30 |
| BaggingDecisionTree | 0,540720 | 0,73379 | 0,252436 | 0,135498 | 1/30 |

### 4.4 Discussão dos líderes

O **GradientBoosting** alcançou MAE de 0,127197, com vantagem de 0,95% em relação ao segundo colocado por essa métrica. Porém, seu RMSE médio foi 0,236044 e seu R² médio apenas 0,510061, prejudicado por ao menos uma rodada atípica.

O **VotingEnsemble** registrou RMSE de 0,230137, somente 0,20% menor que o valor de ExtraTrees, diferença muito pequena para justificar conclusões fortes sem avaliação pareada. O ExtraTrees, entretanto, levou em média 0,2107 s, contra 0,4577 s para VotingEnsemble.

O **ExtraTrees** também exibiu o maior R² médio (0,669316), mas seu DP de R² foi 0,2851, sinal de que o desempenho ainda depende sensivelmente da divisão utilizada. O StackingEnsemble teve R² médio de 0,629193, consumindo em média 2,4570 s por rodada.

## 5. Comparação entre MAE, RMSE e R²

### 5.1 As métricas escolhem vencedores diferentes

| Métrica | Melhor modelo | Valor médio | Interpretação |
|---|---|---:|---|
| MAE (menor) | GradientBoosting | 0,127197 | Menor erro absoluto típico |
| RMSE (menor) | VotingEnsemble | 0,230137 | Melhor controle quadrático médio |
| R² (maior) | ExtraTrees | 0,669316 | Melhor desempenho relativo médio à variância do teste |

O **GradientBoosting** lidera MAE, mas está em 10º lugar em R², enquanto o **ExtraTrees** é 7º em MAE e 1º em R². O **SVRRBF**, por exemplo, apresenta MAE intermediário (10º lugar), porém o penúltimo RMSE (19º lugar), evidenciando que o erro absoluto e os erros quadráticos privilegiam comportamentos diferentes.

### 5.2 Uma execução atípica altera muito o R² médio

Na execução `run: 8`, o GradientBoosting registrou **R² = -5,018574**, apesar de MAE = 0,141005 e RMSE = 0,380156. Na mesma rodada, RandomForest apresentou R² = -3,353411, BaggingDecisionTree = -3,313496 e VotingEnsemble = -2,445679. Esses valores sugerem que o conjunto de teste dessa rodada pode ter variância do alvo especialmente pequena em relação aos erros cometidos. **Essa é uma hipótese**, pois o log não traz a variância do alvo nem as predições para confirmá-la.

Como essa rodada afeta vários modelos, vale examiná-la antes de concluir que as diferenças de R² médio refletem apenas qualidade do algoritmo. Além de médias, podem ser úteis medianas, quantis e inspeção pareada das métricas por `run`.

## 6. Eficiência computacional

Os tempos medidos na execução revelam diferenças relevantes. A tabela a seguir reúne opções com perfil distinto de custo e desempenho.

| Algoritmo | Tempo (s) | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| KNNDistance | 0,0170 | 0,146556 | 0,322571 | 0,453071 |
| PolynomialRidge | 0,0273 | 0,153179 | 0,266997 | 0,562742 |
| AdaBoost | 0,0517 | 0,135319 | 0,273349 | 0,518203 |
| GradientBoosting | 0,1923 | 0,127197 | 0,236044 | 0,510061 |
| ExtraTrees | 0,2107 | 0,138519 | 0,230599 | 0,669316 |
| VotingEnsemble | 0,4577 | 0,130496 | 0,230137 | 0,618783 |
| StackingEnsemble | 2,4570 | 0,143576 | 0,241708 | 0,629193 |

O **KNNDistance** foi o mais rápido no log (0,0170 s), mas ficou em 11º em RMSE e R². O **PolynomialRidge** teve tempo de 0,0273 s e ficou em 4º por R², embora em 14º por MAE. **ExtraTrees** apresentou uma combinação mais forte de RMSE, R² e velocidade: foi cerca de 2,17 vezes mais rápido que VotingEnsemble e cerca de 11,7 vezes mais rápido que StackingEnsemble.

## 7. Ranking global

Como síntese descritiva, o ranking abaixo calcula a média simples das posições de **MAE, RMSE e R²**, atribuindo peso igual aos três indicadores. **Há dupla ênfase na família quadrática** porque RMSE e R² são relacionados; por isso, esse ranking não deve substituir a seleção baseada na métrica prioritária da aplicação.

| Algoritmo | Rank MAE | Rank RMSE | Rank R² | Média ranks | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| VotingEnsemble | 3 | 1 | 3 | 2,33 | 0,4577 |
| ExtraTrees | 7 | 2 | 1 | 3,33 | 0,2107 |
| GradientBoosting | 1 | 3 | 10 | 4,67 | 0,1923 |
| StackingEnsemble | 8 | 4 | 2 | 4,67 | 2,4570 |
| BaggingDecisionTree | 6 | 5 | 5 | 5,33 | 0,6730 |
| LogTargetGradientBoosting | 2 | 7 | 7 | 5,33 | 0,1990 |
| RandomForest | 5 | 6 | 6 | 5,67 | 0,3120 |
| AdaBoost | 4 | 9 | 9 | 7,33 | 0,0517 |
| PolynomialRidge | 14 | 8 | 4 | 8,67 | 0,0273 |
| MLPRegressor | 16 | 10 | 8 | 11,33 | 0,2863 |

Um ranking alternativo com peso igual apenas a MAE e RMSE pode ser apropriado quando se deseja evitar dupla contagem de critérios quadráticos. Como as escalas dos indicadores diferem, rankings por posição também não capturam o tamanho das diferenças entre modelos.

## 8. Estabilidade das execuções

### 8.1 Dispersão entre divisões

| Algoritmo | MAE | DP MAE | RMSE | DP RMSE | R² | DP R² | R² neg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| GradientBoosting | 0,127197 | 0,00822 | 0,236044 | 0,08235 | 0,510061 | 1,04508 | 1/30 |
| LogTargetGradientBoosting | 0,128408 | 0,00896 | 0,256357 | 0,07994 | 0,531057 | 0,67099 | 1/30 |
| VotingEnsemble | 0,130496 | 0,00838 | 0,230137 | 0,07394 | 0,618783 | 0,58694 | 1/30 |
| ExtraTrees | 0,138519 | 0,00893 | 0,230599 | 0,06472 | 0,669316 | 0,28510 | 1/30 |
| StackingEnsemble | 0,143576 | 0,01188 | 0,241708 | 0,07302 | 0,629193 | 0,35297 | 1/30 |
| PolynomialRidge | 0,153179 | 0,00698 | 0,266997 | 0,07142 | 0,562742 | 0,32871 | 2/30 |
| SVRRBF | 0,144912 | 0,01544 | 0,425046 | 0,13146 | 0,103272 | 0,03236 | 0/30 |
| PLSRegression | 0,198519 | 0,00914 | 0,412823 | 0,08306 | -0,010038 | 0,64777 | 7/30 |

Os desvios-padrão de RMSE são relativamente elevados em diversos algoritmos em comparação com suas médias. O R² é ainda mais sensível a determinadas execuções: o GradientBoosting teve DP de R² superior a 1, ao passo que ExtraTrees apresentou DP de aproximadamente 0,285. Isso recomenda evitar a avaliação apenas pela média.

### 8.2 Frequência de R² negativo

Foram observados **48 valores negativos de R² nas 600 execuções**. Essas ocorrências não significam que a execução falhou tecnicamente; indicam desempenho inferior ao baseline constante da média naquela divisão específica. SVRRBF e NuSVR não tiveram R² negativo, mas mantiveram R² médio próximo de 0,10 e RMSE médio alto: estabilidade em um patamar fraco não significa boa predição.

## 9. Recomendações

### 9.1 Quando MAE for a métrica principal

- **GradientBoosting** é a primeira opção pelos resultados médios, seguido por **LogTargetGradientBoosting**. A diferença pequena entre eles deve ser conferida com análise pareada.
- **AdaBoost** pode ser considerado quando o tempo é muito restrito, aceitando MAE maior.

### 9.2 Quando RMSE for a métrica principal

- **VotingEnsemble** foi o melhor em média, mas **ExtraTrees** ficou muito próximo e teve menor tempo; ambos devem ser considerados candidatos prioritários.
- **GradientBoosting** é alternativa competitiva em RMSE, mas sua instabilidade em R² deve ser investigada.

### 9.3 Quando R² for a métrica principal

- **ExtraTrees** apresentou o maior R² médio, embora ainda com oscilações entre rodadas.
- **StackingEnsemble** e **VotingEnsemble** também apresentaram R² médio elevado; o custo adicional do stacking precisa ser justificado pela aplicação.

### 9.4 Quando equilíbrio entre precisão e tempo for importante

- **ExtraTrees** é um candidato particularmente forte: segundo melhor RMSE, maior R² e tempo inferior ao do VotingEnsemble.
- **GradientBoosting** é atraente pelo menor MAE e tempo reduzido, com a ressalva da rodada atípica.
- **PolynomialRidge** merece consideração em ambientes de latência baixa, mas não foi líder em erro neste conjunto.

## 10. Limitações e cuidados de interpretação

1. **Existe apenas um cenário no log.** Não é possível comparar faixas de SNR ou extrair conclusões sobre generalização entre cenários.
2. **A divisão de dados não está descrita no log.** Não se deve inferir proporções de treino/teste ou validação externa sem consultar o código.
3. **Não há acesso ao alvo real nem às predições.** Não é possível verificar causas específicas de outliers, heterocedasticidade, distribuição de resíduos ou a variância do teste por rodada.
4. **Os modelos podem compartilhar as mesmas divisões.** A presença de índices `run` iguais sugere comparações pareadas possíveis, mas a igualdade exata dos conjuntos de treino e teste deve ser confirmada no script.
5. **Médias não demonstram superioridade estatística.** Diferenças pequenas em RMSE, como a de VotingEnsemble versus ExtraTrees, precisam de intervalos de confiança ou testes pareados adequados.
6. **O R² médio é sensível a rodadas negativas extremas.** Compará-lo isoladamente pode ocultar comportamentos diferentes de MAE e RMSE.
7. **Hiperparâmetros e pré-processamento não são verificáveis a partir do log.** Os resultados refletem as configurações executadas, não o desempenho máximo potencial de cada família de modelos.
8. **Tempos foram exibidos com precisão limitada.** Diferenças muito pequenas de tempo entre modelos rápidos não devem ser superinterpretadas.
9. **Não há baseline explícito de erro no relatório.** Incluir um `DummyRegressor` facilitaria a comparação direta de MAE, RMSE e tempo.
10. **Possíveis dependências entre amostras precisam ser avaliadas.** Se houver medições ligadas ao mesmo circuito, grupos de origem ou estrutura temporal, uma separação aleatória por linha pode superestimar a generalização.

## 11. Próximos passos sugeridos

1. Verificar a **execução 8**, conferindo dispersão do alvo, amostras presentes, erros extremos e possíveis particularidades do teste.
2. Reavaliar **GradientBoosting, ExtraTrees e VotingEnsemble** em divisões idênticas e calcular diferenças pareadas de MAE, RMSE e R², com intervalos de confiança.
3. Complementar médias e DPs com **mediana, percentis, mínimo/máximo** e gráficos de distribuição por algoritmo.
4. Adicionar **DummyRegressor** e uma validação externa ou cruzada adequada ao desenho do dataset.
5. Testar ajuste de hiperparâmetros com procedimento de validação interna, mantendo isolamento do conjunto de teste.
6. Inspecionar resíduos, outliers e importância das 27 características, especialmente nos modelos líderes.
7. Confirmar no código se o particionamento deve considerar agrupamento por circuito ou outra unidade experimental.

## 12. Tabela consolidada dos 20 modelos

A tabela abaixo contém as médias recalculadas a partir das 30 execuções, ordenadas por MAE.

| Algoritmo | MAE | MSE | RMSE | R² | Tempo (s) |
|---|---:|---:|---:|---:|---:|
| GradientBoosting | 0,127197 | 0,062499 | 0,236044 | 0,510061 | 0,1923 |
| LogTargetGradientBoosting | 0,128408 | 0,072109 | 0,256357 | 0,531057 | 0,1990 |
| VotingEnsemble | 0,130496 | 0,058431 | 0,230137 | 0,618783 | 0,4577 |
| AdaBoost | 0,135319 | 0,085743 | 0,273349 | 0,518203 | 0,0517 |
| RandomForest | 0,135411 | 0,070346 | 0,253036 | 0,536229 | 0,3120 |
| BaggingDecisionTree | 0,135498 | 0,070087 | 0,252436 | 0,540720 | 0,6730 |
| ExtraTrees | 0,138519 | 0,057364 | 0,230599 | 0,669316 | 0,2107 |
| StackingEnsemble | 0,143576 | 0,063755 | 0,241708 | 0,629193 | 2,4570 |
| LinearSVR | 0,144027 | 0,196280 | 0,422608 | 0,112552 | 0,0847 |
| SVRRBF | 0,144912 | 0,197945 | 0,425046 | 0,103272 | 0,0640 |
| NuSVR | 0,145157 | 0,198244 | 0,425382 | 0,101828 | 0,0890 |
| HuberRegressor | 0,146014 | 0,166946 | 0,387471 | 0,203968 | 0,1647 |
| KNNDistance | 0,146556 | 0,118308 | 0,322571 | 0,453071 | 0,0170 |
| PolynomialRidge | 0,153179 | 0,076389 | 0,266997 | 0,562742 | 0,0273 |
| HistGradientBoosting | 0,157825 | 0,129623 | 0,347537 | 0,303859 | 0,4487 |
| MLPRegressor | 0,160964 | 0,087036 | 0,284318 | 0,520714 | 0,2863 |
| BayesianRidge | 0,166678 | 0,123440 | 0,340523 | 0,285912 | 0,0100 |
| Ridge | 0,175777 | 0,134617 | 0,356698 | 0,246563 | 0,0100 |
| ElasticNet | 0,188744 | 0,166306 | 0,399054 | 0,053738 | 0,0103 |
| PLSRegression | 0,198519 | 0,177321 | 0,412823 | -0,010038 | 0,0107 |

## 13. Conclusão

O experimento com 2.000 amostras revelou um compromisso entre **erro absoluto**, **controle dos erros maiores** e **capacidade explicativa relativa ao teste**. **GradientBoosting** obteve o menor MAE; **VotingEnsemble**, o menor RMSE; e **ExtraTrees**, o maior R² médio. Como a diferença de RMSE entre VotingEnsemble e ExtraTrees é muito pequena e este último possui menor tempo médio, **ExtraTrees aparece como opção prática promissora quando RMSE, R² e custo devem ser considerados em conjunto**.

A presença de R² negativos e de uma rodada particularmente influente impede afirmar que o líder é universalmente superior. A prioridade analítica deve ser **inspecionar a execução 8 e validar as diferenças entre os primeiros colocados de forma pareada**, antes de escolher o modelo definitivo.

---
*Nota metodológica: os tempos e métricas foram extraídos do log fornecido; não representam testes reproduzidos neste relatório. Valores formatados com vírgula decimal para leitura em português.*
