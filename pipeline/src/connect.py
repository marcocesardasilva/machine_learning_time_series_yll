from google.oauth2 import service_account
from google.cloud import bigquery


def connect_to_gcp(file_key):
    # Create connection to GCP
    print('##########################################################################')
    print('#                     Iniciando execução do programa                     #')
    print('##########################################################################')
    print('--------------------------------------------------------------------------')
    print('Criando conexão com o GCP...')
    print('--------------------------------------------------------------------------')
    try:
        credentials = service_account.Credentials.from_service_account_file(file_key)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        print(f'Conexão realizada com sucesso com o projeto {credentials.project_id}.')
        print('--------------------------------------------------------------------------')
    except Exception:
        print(f'Não foi possível efetivar a conexão com o GCP.')
        print('--------------------------------------------------------------------------')
    return client

