from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Realizar conexão com o GCP
credentials = service_account.Credentials.from_service_account_file("keys/ml-na-saude-ed1fc3c1a83e.json")
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Query para consulta dos dados agrupados por taxa média por quadrimestre
consulta_sql = """
select
	tx.quadrimestre,
	avg(tx.taxa_yll) as taxa_media_yll
from (
	with yll_quadrimestral as (
		select
			case
				when extract(month from y.dt_obito) between 1 and 4 then date(extract(year from y.dt_obito), 4, 30)
				when extract(month from y.dt_obito) between 5 and 8 then date(extract(year from y.dt_obito), 8, 31)
				when extract(month from y.dt_obito) between 9 and 12 then date(extract(year from y.dt_obito), 12, 31)
				end as quadrimestre,
			m.nm_municipio,
			p.populacao,
			sum(y.yll) as soma_yll
		from `ml-na-saude.yll_por_obito.yll` y
		join `ml-na-saude.yll_por_obito.populacao` p on y.cd_mun_res = p.cd_municipio and y.ano_obito = p.ano
		join `ml-na-saude.yll_por_obito.municipio` m on p.cd_municipio = m.cd_municipio
		where p.porte = 'Médio Porte'
		group by 1,2,3
	)
	select
		quadrimestre,
		nm_municipio,
		soma_yll,
		populacao,
		soma_yll / populacao * 1000 as taxa_yll
	from yll_quadrimestral
	group by 1,2,3,4
) tx
group by 1
order by 1
"""

# Ignorar avisos e gerar dataframe
warnings.simplefilter("ignore")
df = client.query(consulta_sql).to_dataframe()
# Copiar dataframe para manipular dados
time_series = df.copy()
# Transformando o quadrimestre em data
time_series['quadrimestre'] = pd.to_datetime(time_series['quadrimestre'])
# # Filtrar dados até final de 2019
time_series = time_series[time_series['quadrimestre'] <= '2019-12-31']
# Setando o quadrimestre como índice da tabela
time_series = time_series.set_index('quadrimestre')

# Normaliza os Dados
normalizer = MinMaxScaler(feature_range=(0, 1))
train_data = normalizer.fit_transform(time_series.values)

# Cria os Arrays No Formato Certo
window_size = 3
x = []
y = []
for i in range(window_size, len(train_data)):
    x.append(train_data[i - window_size:i, 0])
    y.append(train_data[i, 0])
x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Treina com todos os dados
def build_model_for_future(hp=None):
    regressor = Sequential()
    regressor.add(LSTM(units=800, return_sequences=True, input_shape=(x.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=300, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=100, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=200))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return regressor

# Inicializa o modelo
model = build_model_for_future()

# Callback para Early Stopping
early_stopping = EarlyStopping(
    monitor='loss',
    patience=100,
    restore_best_weights=True
)

# Treina o modelo com todos os dados disponíveis
history = model.fit(
    x, y,
    batch_size=16,
    epochs=1000,
    callbacks=[early_stopping],
    validation_split=0.1
)

# Previsão Futura
future_steps = 9
last_window = train_data[-window_size:]
predictions = []

for _ in range(future_steps):
    last_window_reshaped = np.reshape(last_window, (1, last_window.shape[0], 1))
    prediction = model.predict(last_window_reshaped)
    predictions.append(prediction[0, 0])
    last_window = np.append(last_window[1:], prediction, axis=0)

# Revertendo a Normalização
future_predictions = normalizer.inverse_transform(np.array(predictions).reshape(-1, 1))

# Gerando Datas para os Próximos 9 Quadrimestres
last_date = time_series.index[-1]
future_dates = [last_date + pd.DateOffset(months=4 * i) for i in range(1, future_steps + 1)]

# Criação do DataFrame com as Previsões Futuras
future_results = pd.DataFrame({
    'Data': future_dates,
    'Previsao YLL': future_predictions.flatten()
})
future_results.set_index('Data', inplace=True)

# Exibindo as Previsões Futuras
print(future_results)

# Plotando as Previsões Futuras
plt.figure(figsize=(12, 6))
# plt.plot(time_series.index, time_series.values, color='red', label='Real')
plt.plot(time_series.index, time_series.values, color='red', label='Real')
plt.plot(future_results.index, future_results['Previsao YLL'], color='blue', label='Previsão Futura', linestyle='dashed')
plt.title('Previsão Futura da Taxa Média de YLL')
plt.ylabel('Taxa Média YLL')
plt.xlabel('Ano')
plt.legend()
plt.show()

# Salvar o modelo LSTM treinado em formato .h5
model.save('models/lstm_model.h5')
print("Modelo salvo como 'lstm_model.h5'")
