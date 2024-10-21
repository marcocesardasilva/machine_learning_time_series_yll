from google.oauth2 import service_account
from google.cloud import bigquery
import os


def connect_to_gcp(file_key):
    # Create connection to GCP
    print("##########################################################################")
    print("#                     Iniciando execução do programa                     #")
    print("##########################################################################")
    print("--------------------------------------------------------------------------")
    print("Criando conexão com o GCP...")
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, file_key)
        credentials = service_account.Credentials.from_service_account_file(file_path)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        print(f"Conexão realizada com sucesso com o projeto {credentials.project_id}.")
        print("--------------------------------------------------------------------------")
    except Exception:
        print(f"Não foi possível efetivar a conexão com o GCP.")
        print("--------------------------------------------------------------------------")
    return client

def create_dataset(client,dataset_name):
    # Create the dataset if it does not already exist
    print("--------------------------------------------------------------------------")
    print("Verificando a existência do dataset no GCP...")
    dataset_fonte = client.dataset(dataset_name)
    try:
        client.get_dataset(dataset_fonte)
        print(f"O conjunto de dados {dataset_fonte} já existe no GCP.")
        print("--------------------------------------------------------------------------")
    except Exception:
        print(f"Dataset {dataset_fonte} não foi encontrado no GCP, criando o dataset...")
        client.create_dataset(dataset_fonte)
        print(f"O conjunto de dados {dataset_fonte} foi criado no GCP com sucesso.")
        print("--------------------------------------------------------------------------")
    return dataset_fonte

def create_table(client,dataset_fonte):
    # Create tables if they do not already exist
    table_yll = dataset_fonte.table("yll_por_obito")
    # schema attributes
    schema_yll = [
        bigquery.SchemaField("ano_obito", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("quad_obito", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("dt_obito", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("dt_nasc", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("idade", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("yll", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("cid10", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cd_mun_res", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("populacao", "INTEGER", mode="REQUIRED")
    ]

    print("--------------------------------------------------------------------------")
    print("Verificando a existência das tabelas no GCP...")
    try:
        client.get_table(table_yll, timeout=30)
        print(f"A tabela {table_yll} já existe!")
        print("--------------------------------------------------------------------------")
    except:
        print(f"Tabela {table_yll} não encontrada! Criando tabela {table_yll}...")
        client.create_table(bigquery.Table(table_yll, schema=schema_yll))
        print(f"A tabela {table_yll} foi criada.")
        print("--------------------------------------------------------------------------")

    return table_yll

def load_data(tables_dfs,client,dataset_fonte):
    # Load data into gcp
    print("--------------------------------------------------------------------------")
    print("Carregando dados no GCP...")
    for tabela, df in tables_dfs.items():
        table_ref = client.dataset(dataset_fonte.dataset_id).table(tabela.table_id)
        job_config = bigquery.LoadJobConfig()
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        print(f"Dados carregados na tabela {tabela}.")

    print("--------------------------------------------------------------------------")
    print("##########################################################################")
    print("#                         Dados carregados no GCP                        #")
    print("##########################################################################")

