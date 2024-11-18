# Modelo preditivo dos anos de vida perdidos por morte prematura para os municípios brasileiros de médio porte utilizando aprendizagem de máquina

Projeto de avaliação de modelos preditivos dos anos de vida perdidos por morte prematura para os municípios brasileiros de médio porte, utilizando aprendizagem de máquina.

O projeto é composto por duas fazes, uma com atividades de Enganharia de Dados e outra puramente de Ciência de Dados.

Como resultados são obtidas as previsões dos índices de taxa média do YLL (anos de vida perdidos por morte prematura).

#

Ativar o ambiente virtual:
```
venv\Scripts\activate
```
Gerar requirements:
```
pip freeze > requirements.txt
```
Observação: É necessária a pasta keys com o arquivo json gerado no GCP.

Abaixo é apresentada a estrutura de tópicos do projeto:

```
machine_learning_time_series_yll
├── keys
│   ├── nome_projeto_gcp.json
├── models
│   ├── lstm_model.h5
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
│   ├── arima_e_sarima_model.ipynb
│   ├── arima_e_sarima_model.py
│   ├── exploratory_analysis.ipynb
│   ├── exploratory_analysis.py
│   ├── lstm_model.ipynb
│   ├── lstm_model.py
│   ├── predicao_yll.ipynb
│   ├── predicao_yll.py
│   ├── prophet_model.ipynb
│   ├── prophet_model.py
│   ├── query.sql
│   ├── xgboost_model.ipynb
│   ├── xgboost_model.py
├── .gitignore
├── config.yaml
├── README.md
├── requirements.txt
```

