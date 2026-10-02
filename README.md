## Projeto CEC Cells 

Repositório utilizado para compartilhamento de análises, dados, pipelines e artigos do Projeto CEC Cells CNPq.


## Organização do Repositório

```text
├── analysis           <- Códigos de experimentos, análises e testes preliminares;
├── data               <- Fontes de dados brutos do projeto (genomas, tabelas BV-BRC);
├── dataset-build      <- Montagem de conjuntos de dados e extração de características de proteínas;
├── dataset-SNR        <- Datasets gerados de circuitos lógicos (NOT, XOR) e documentação de schemas;
│   ├── circuit_dataset.jsonl        <- Base de referência da porta NOT;
│   └── xor-dataset/                 <- Base oficial da porta XOR (2.000 circuitos, SNR >= 0.85);
├── dataset-generator  <- Pipeline de busca contínua (DARTS) e simulador estocástico (Julia SSA);
│   ├── run_generator.py             <- Orquestrador Optuna + DARTS para geração massiva de datasets;
│   ├── evaluate_candidate.py        <- Avaliador direto de candidatos contra o simulador de Julia;
│   ├── engine_py2/                  <- Otimizador contínuo Theano (Hill relaxation);
│   └── engine_julia/                <- Motor estocástico de Gillespie (JumpProcesses.jl);
├── papers             <- Artigos e publicações relacionados ao projeto;
└── README.md          <- Informações gerais do repositório.
```