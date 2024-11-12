from google.cloud import bigquery
import os


def load_data(tables_dfs,client,dataset_fonte):
    # Load data into gcp
    print('--------------------------------------------------------------------------')
    print('Carregando dados no GCP...')
    print('--------------------------------------------------------------------------')
    for tabela, df in tables_dfs.items():
        table_ref = client.dataset(dataset_fonte.dataset_id).table(tabela.table_id)
        job_config = bigquery.LoadJobConfig()
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        print(f'Dados carregados na tabela {tabela}.')

    print('--------------------------------------------------------------------------')

def download_data(tables_dfs,datafolder_processed):
    print('--------------------------------------------------------------------------')
    print(f'Baixando os dados em .CSV para a pasta {datafolder_processed}...')
    print('--------------------------------------------------------------------------')
    # Create the directory for the data if it does not already exist
    if not os.path.exists(datafolder_processed):
        os.makedirs(datafolder_processed)
    # Dowload data
    for table, df in tables_dfs.items():
        df.to_csv(f'{datafolder_processed}/{table.table_id}.csv', index=False)
        print(f'Os dados foram baixados para o arquivo {table.table_id}.csv...')
    print('--------------------------------------------------------------------------')
    print('##########################################################################')
    print('#                           Execução Finalizada                          #')
    print('##########################################################################')

