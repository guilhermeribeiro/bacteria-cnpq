# Pipeline do `dataset_by_circuit_from_protein.py`

Este script monta um dataset tabular a partir de duas fontes principais:

1. registros de portas/circuitos do repositório Cello-UCF;
2. sequências e anotações proteicas consultadas no UniProt.

O objetivo final é gerar uma tabela em que cada linha representa uma instância de porta ou fragmento de circuito associado a uma proteína reguladora, com variáveis do circuito, descritores físico-químicos da proteína e uma métrica alvo (`target_metric`) quando ela pode ser obtida.

## Entrada principal

Por padrão, o script usa as proteínas:

```text
SrpR, PhlF, AmeR, BetI, QacR, AmtR, LmrA
```

Essas proteínas podem ser alteradas com o argumento `--proteins`.

Exemplo usando a pasta de saída já adotada neste projeto:

```bash
python dataset_by_circuit_from_protein.py --output-dir outputs
```

Se `--ucf-dir` não for informado, o script baixa automaticamente o repositório Cello-UCF do GitHub, usando por padrão o branch `develop`. Para usar uma cópia local do Cello-UCF:

```bash
python dataset_by_circuit_from_protein.py --output-dir outputs --ucf-dir caminho/para/Cello-UCF
```

## Etapas da pipeline

### 1. Localização dos arquivos UCF

A função `build_dataset()` recebe a lista de proteínas, a pasta de saída e a origem dos arquivos Cello-UCF.

Se uma pasta UCF local não for passada por `--ucf-dir`, o script chama `download_cello_ucf()`, baixa o ZIP do repositório Cello-UCF e extrai os arquivos em uma pasta de cache dentro do diretório de saída.

Depois disso, `find_json_files()` procura todos os arquivos `.json` dentro da pasta Cello-UCF.

### 2. Extração de registros de portas ou fragmentos de circuito

Para cada proteína, `extract_ucf_gate_records_for_protein()` percorre todos os objetos JSON encontrados nos arquivos UCF.

Um objeto é mantido quando:

- contém o nome da proteína pesquisada;
- parece ser relacionado a porta, resposta, regulador ou parâmetros de função de resposta;
- ou possui parâmetros reconhecidos, como `ymax`, `ymin`, `K`, `n`, `SNR` ou `on_off_ratio`.

Para cada registro aceito, o script tenta inferir:

- `query_protein`: proteína pesquisada;
- `ucf_json_file`: arquivo JSON de origem;
- `ucf_object_index`: posição do objeto dentro da varredura do JSON;
- `ucf_record_type`: tipo/coleção do registro, quando disponível;
- `gate_or_fragment_name`: nome da porta ou fragmento;
- parâmetros UCF prefixados com `ucf_param_`, por exemplo `ucf_param_ymax`, `ucf_param_ymin`, `ucf_param_K` e `ucf_param_n`.

Essa etapa gera:

```text
outputs/ucf_gate_records.csv
```

### 3. Definição da variável alvo

A função `derive_target()` define `target_metric` seguindo esta prioridade:

1. usa `SNR`, se existir explicitamente no JSON;
2. usa `on_off_ratio`, se existir explicitamente no JSON;
3. se houver `ymax` e `ymin`, calcula `on_off_ratio = ymax / ymin` quando `ymin > 0`;
4. se `ymin` não for positivo, usa `dynamic_range = ymax - ymin`;
5. usa `dynamic_range`, se existir explicitamente no JSON;
6. se nada disso existir, marca a origem como `not_found`.

Quando o alvo é calculado a partir de `ymax` e `ymin`, o script também pode preencher:

- `derived_dynamic_range`;
- `derived_log10_on_off_ratio`.

Linhas sem `target_metric` continuam em `dataset_full.csv`, mas são removidas de `dataset_with_target.csv`, `X.csv` e `y.csv`.

### 4. Consulta ao UniProt e descritores proteicos

Para cada proteína, `query_uniprot()` consulta a API REST do UniProt usando a busca:

```text
(nome_da_proteina) AND (bacteria)
```

O termo `bacteria` pode ser alterado por `--organism-query`.

O script guarda informações como:

- `uniprot_accession`;
- `uniprot_id`;
- `uniprot_entry_type`;
- `uniprot_protein_name`;
- `uniprot_gene_names`;
- `uniprot_organism`;
- `uniprot_length`;
- `uniprot_sequence`;
- `uniprot_function`.

Depois, `physicochemical_features()` usa a sequência retornada para calcular descritores com `Bio.SeqUtils.ProtParam.ProteinAnalysis`, incluindo:

- tamanho da sequência;
- massa molecular;
- aromaticidade;
- índice de instabilidade;
- ponto isoelétrico;
- GRAVY;
- carga em pH 7;
- fração de cada aminoácido.

Essa etapa gera:

```text
outputs/protein_features.csv
```

### 5. Junção dos dados

O script junta os registros UCF com os descritores proteicos usando a coluna `query_protein`.

Depois cria `instance_id`, combinando proteína, nome da porta/fragmento e índice da linha.

Essa tabela completa é salva em:

```text
outputs/dataset_full.csv
```

Em seguida, o script remove as linhas sem `target_metric` e salva:

```text
outputs/dataset_with_target.csv
```

### 6. Separação em `X` e `y`

A função `make_xy()` separa:

- `y`: a coluna `target_metric`;
- `X`: as variáveis explicativas.

Algumas colunas são removidas de `X` por serem identificadores, texto longo ou vazamento direto do alvo:

- `instance_id`;
- `query_protein`;
- `ucf_json_file`;
- `ucf_object_index`;
- `target_metric`;
- `target_metric_name`;
- `target_metric_source`;
- `uniprot_sequence`;
- `uniprot_function`.

Colunas categóricas restantes são transformadas com `pd.get_dummies()`. Valores ausentes são preenchidos pela mediana das colunas numéricas e, se ainda restarem ausentes, por zero.

Arquivos finais:

```text
outputs/X.csv
outputs/y.csv
```

## Arquivos gerados

Ao final, a pipeline gera:

```text
ucf_gate_records.csv      Registros extraídos dos JSONs UCF.
protein_features.csv      Dados do UniProt e descritores físico-químicos.
dataset_full.csv          Junção completa, incluindo linhas sem alvo.
dataset_with_target.csv   Apenas linhas com target_metric disponível.
X.csv                     Matriz de atributos para modelagem.
y.csv                     Vetor alvo para modelagem.
```



## Possíveis limitações

- A busca no UniProt escolhe automaticamente o melhor resultado por uma heurística simples. Isso deve ser revisado manualmente.
- A extração dos JSONs UCF é tolerante e varre objetos aninhados. Por isso, pode capturar registros redundantes ou registros que precisam de inspeção manual.
- O alvo pode representar métricas diferentes (`SNR`, `on_off_ratio` ou `dynamic_range`). Antes de treinar modelos, é importante verificar a distribuição de `target_metric_name`.
- Resultados podem mudar se o Cello-UCF ou o UniProt forem atualizados.


## Link do chatgpt

- Abixo o link da conversa/validação que foi feita com o ChatGPT:
  - https://chatgpt.com/share/6a345a42-a028-83e9-bab0-dfd2f6897185

