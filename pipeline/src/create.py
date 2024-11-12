from google.cloud import bigquery


def create_dataset(client,dataset_name):
    # Create the dataset if it does not already exist
    print('--------------------------------------------------------------------------')
    print('Verificando a existência do dataset...')
    dataset_fonte = client.dataset(dataset_name)
    try:
        client.get_dataset(dataset_fonte)
        print(f'Conjunto de dados {dataset_fonte} já existe.')
        print('--------------------------------------------------------------------------')
    except Exception:
        print(f'Dataset {dataset_fonte} não encontrado, criando dataset...')
        client.create_dataset(dataset_fonte)
        print(f'Conjunto de dados {dataset_fonte} criado com sucesso.')
        print('--------------------------------------------------------------------------')
    return dataset_fonte

def create_tables(client,dataset_fonte):
    # Create tables if they do not already exist
    print('--------------------------------------------------------------------------')
    print('Verificando a existência das tabelas...')

    table_population = dataset_fonte.table('populacao')
    # Schema attributes
    schema_population = [
        bigquery.SchemaField('cd_municipio', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('ano', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('populacao', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('porte', 'STRING', mode='REQUIRED')
    ]

    try:
        client.get_table(table_population, timeout=30)
        print(f'A tabela {table_population} já existe!')
    except:
        print(f'Tabela {table_population} não encontrada! Criando tabela...')
        client.create_table(bigquery.Table(table_population, schema=schema_population))
        print(f'A tabela {table_population} foi criada.')

    table_municipality = dataset_fonte.table('municipio')
    # schema attributes
    schema_municipality = [
        bigquery.SchemaField('cd_municipio', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('nm_municipio', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('cd_uf', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('sl_uf', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('cd_regiao', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('nm_regiao', 'STRING', mode='REQUIRED')
    ]

    try:
        client.get_table(table_municipality, timeout=30)
        print(f'A tabela {table_municipality} já existe!')
    except:
        print(f'Tabela {table_municipality} não encontrada! Criando tabela...')
        client.create_table(bigquery.Table(table_municipality, schema=schema_municipality))
        print(f'A tabela {table_municipality} foi criada.')

    table_yll = dataset_fonte.table('yll')
    # schema attributes
    schema_yll = [
        bigquery.SchemaField('ano_obito', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('quadrimestre_obto', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('dt_obito', 'DATETIME', mode='REQUIRED'),
        bigquery.SchemaField('dt_nasc', 'DATETIME', mode='REQUIRED'),
        bigquery.SchemaField('idade', 'FLOAT', mode='REQUIRED'),
        bigquery.SchemaField('yll', 'FLOAT', mode='REQUIRED'),
        bigquery.SchemaField('cid10', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('cd_mun_res', 'STRING', mode='REQUIRED')
    ]

    try:
        client.get_table(table_yll, timeout=30)
        print(f'A tabela {table_yll} já existe!')
    except:
        print(f'Tabela {table_yll} não encontrada. Criando tabela...')
        client.create_table(bigquery.Table(table_yll, schema=schema_yll))
        print(f'A tabela {table_yll} foi criada.')

    print('--------------------------------------------------------------------------')

    return table_yll, table_population, table_municipality

