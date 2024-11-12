# Modelo preditivo dos anos de vida perdidos por morte prematura para os municípios brasileiros de médio e grande porte utilizando aprendizagem de máquina

Projeto de avaliação de modelos preditivos dos anos de vida perdidos por morte prematura para os municípios brasileiros de médio e grande porte, utilizando aprendizagem de máquina.

Ativar venv: venv\Scripts\activate

Gerar requirements: pip freeze > requirements.txt

Necessário a pasta keys com o arquivo json gerado no GCP

```
machine_learning_time_series_yll
├── keys
│   ├── nome_projeto_gcp.json
├── pipeline
│   ├── data
│   │   ├── processed
│   │   ├── raw
│   ├── docs
│   │   ├── ibge_populacao.csv
│   │   ├── modelo_relacional.jpg
│   │   ├── modelo_relacional.mdj
│   ├── src
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── connect.py
│   │   ├── create.py
│   │   ├── extract.py
│   │   ├── load.py
│   │   ├── transform.py
│   ├── main.py
├── predicao
│   ├── an_exp.ipynb
│   ├── exploratory_analysis.ipynb
│   ├── modeling.ipynb
│   ├── nb_etl.ipynb
├── .gitignore
├── config.yaml
├── README.md
├── requirements.txt
```