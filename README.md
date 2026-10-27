# Olist

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Brazilian e-commerce public dataset.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         module_olist and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── module_olist   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_olist a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Treinamento e validação

O pipeline de modelagem usa uma divisão temporal: os 20% de pedidos mais
recentes ficam reservados para o teste final. Nos 80% anteriores, cada modelo
é avaliado com `TimeSeriesSplit` em cinco folds. O imputador e o encoder ficam
dentro do `Pipeline`, portanto são ajustados novamente em cada fold.

O threshold de classificação é escolhido a partir das previsões out-of-fold
da validação cruzada. O conjunto de teste é usado somente uma vez para a
avaliação final, sem reotimização do threshold.

Para executar o fluxo completo:

```bash
make train
```

O projeto compara Gradient Boosting, XGBoost e LightGBM e registra ROC-AUC,
precision, recall e F1 para a validação cruzada e para o teste temporal. O
melhor modelo é selecionado pelo maior F1 calculado após a aplicação do
threshold calibrado na validação cruzada.

O melhor pipeline é salvo em `models/best_model.joblib`, e seu threshold e
suas configurações ficam em `models/best_model_metadata.json`. Para gerar
previsões depois do treinamento:

```bash
python -m module_olist.modeling.predict
```

As previsões são gravadas em `data/processed/predictions.csv`.

Para gerar explicações SHAP do XGBoost salvo:

```bash
python -m module_olist.explain
```

Os gráficos são salvos em `reports/figures/shap_feature_importance.png` e
`reports/figures/shap_beeswarm.png`.

--------



## Notebooks das aulas

1. `notebooks/01-data-prof.ipynb`: qualidade de orders, profiling e contrato de dados.
2. `notebooks/02-analise-exploratoria.ipynb`: cinco hipóteses do material do professor.
3. `notebooks/03-active-learning.ipynb`: consultas por incerteza com anotador simulado.

Os dois primeiros adaptam os notebooks fornecidos pelo professor. O exemplo de
active learning usa dados sintéticos porque não há exemplo dessa etapa entre os
arquivos disponibilizados. O teste fica reservado e não participa das consultas.
Execute `uv sync --locked` e `uv run jupyter lab` na raiz do projeto.
Os CSVs Olist devem estar em `data/raw`; os relatórios HTML são regenerados.

A avaliação mantém a separação temporal, métricas de precisão, recall, F1,
ROC-AUC, PR-AUC e acurácia. O SHAP gera gráficos global, beeswarm e waterfall
em `reports/figures`. As interfaces e nomes dos arquivos podem diferir do
material da aula, preservando as etapas e o threshold escolhido na validação.
