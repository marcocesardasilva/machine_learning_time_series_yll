import os
import pandas as pd
# import basedosdados as bd


def calculate_life_expectancy(age):
    # Calculate life expectancy
    if age < 0.08:
        return 89.99
    elif 0.08 <= age < 1:
        return 89.55
    elif 1 <= age < 5:
        return 89.07
    elif 5 <= age < 10:
        return 82.58
    elif 10 <= age < 15:
        return 77.58
    elif 15 <= age < 20:
        return 72.60
    elif 20 <= age < 25:
        return 67.62
    elif 25 <= age < 30:
        return 62.66
    elif 30 <= age < 35:
        return 57.71
    elif 35 <= age < 40:
        return 52.76
    elif 40 <= age < 45:
        return 47.83
    elif 45 <= age < 50:
        return 42.94
    elif 50 <= age < 55:
        return 38.10
    elif 55 <= age < 60:
        return 33.33
    elif 60 <= age < 65:
        return 28.66
    elif 65 <= age < 70:
        return 24.12
    elif 70 <= age < 75:
        return 19.76
    elif 75 <= age < 80:
        return 15.65
    elif 80 <= age < 85:
        return 11.69
    else:
        return 7.05

def create_df(data_folder_raw):
    # Select the folder where the raw files are located
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, data_folder_raw)
    # List of ICD-10 codes belonging to ICSAPS
    cod_icsaps = ["A37","A36","A33","A34","A35","B26","B06","B05","A95","B16","G000","A170","A19","A150","A151","A152","A153","A160",
                  "A161","A162","A154","A155","A156","A157","A158","A159","A163","A164","A165","A166","A167","A168","A169","A171","A172",
                  "A173","A174","A175","A176","A177","A178","A179","A18","I00","I01","I02","A51","A52","A53","B50","B51","B52","B53",
                  "B54","B77","E86","A00","A01","A02","A03","A04","A05","A06","A07","A08","A09","D50","E40","E41","E42","E43","E44","E45",
                  "E46","E50","E51","E52","E53","E54","E55","E56","E57","E58","E59","E60","E61","E62","E63","E64","H66","J00","J01","J02",
                  "J03","J06","J31","J13","J14","J153","J154","J158","J159","J181","J45","J46","J20","J21","J40","J41","J42","J43","J47",
                  "J44","I10","I11","I20","I50","J81","I63","I64","I65","I66","I67","I69","G45","G46","E100","E101","E110","E111","E120",
                  "E121","E130","E131","E140","E141","E102","E103","E104","E105","E106","E107","E108","E112","E113","E114","E115","E116",
                  "E117","E118","E122","E123","E124","E125","E126","E127","E128","E132","E133","E134","E135","E136","E137","E138","E142",
                  "E143","E144","E145","E146","E147","E148","E109","E119","E129","E139","E149","G40","G41","N10","N11","N12","N30","N34",
                  "N390","A46","L01","L02","L03","L04","L08","N70","N71","N72","N73","N75","N76","K25","K26","K27","K28","K920","K921",
                  "K922","O23","A50","P350"]
    # List with the dataframes already processed
    dfs = []

    print("--------------------------------------------------------------------------")
    print("Carregando os dados dos arquivos extraídos, tratando e concatenando...")
    # Generate the dataframe
    for file in os.listdir(file_path):
        if file.endswith('.csv'):
            # Read CSV file with Pandas
            df = pd.read_csv(os.path.join(file_path, file), delimiter=';', encoding='ISO-8859-1', low_memory=False)
            # Analyze whether the ICD-10 code belongs to ICSAPs
            df['icsaps'] = df['CAUSABAS'].apply(lambda x: 'Sim' if x in cod_icsaps else 'Não')
            # Keep only data that is ICSAPs
            df = df[df['icsaps'] == 'Sim']
            # Perform transformation of birth and death dates
            df['dt_obito'] = pd.to_datetime(df['DTOBITO'], format='%d%m%Y', errors='coerce')
            df['dt_nasc'] = pd.to_datetime(df['DTNASC'], format='%d%m%Y', errors='coerce')
            # Delete null data for date of birth and date of death
            df = df.dropna(subset=['dt_nasc'])
            df = df.dropna(subset=['dt_obito'])
            # Create the age column
            df['idade'] = ((df['dt_obito'] - df['dt_nasc']).dt.days / 365.25).round(2)
            # Keep only data with valid ages
            df = df[df['idade'] >= 0]
            # Create column yll
            df['yll'] = df.apply(lambda row: calculate_life_expectancy(row['idade']), axis=1)
            # Create the columns ano_obito and quad_obito
            df['ano_obito'] = df['dt_obito'].dt.year.astype(float).astype(pd.Int64Dtype()).astype(str).where(df['dt_obito'].notna())
            df['quad_obito'] = pd.cut(df['dt_obito'].dt.month, bins=[1, 5, 9, 13], labels=[1, 2, 3], right=False)
            # Extract the first 6 digits from the CODMUNRES column
            df['cd_mun_res'] = df['CODMUNRES'].astype(str).str.slice(stop=6)
            # Renomear colunas
            df = df.rename(columns={'CAUSABAS':'cid10'})
            # Select desired columns
            df = df[['ano_obito','quad_obito','dt_obito','dt_nasc','idade','yll','cid10','cd_mun_res']]
            # Add the dataframe to the list of dataframes
            dfs.append(df)

    # concatena os dataframes em um único dataframe final
    df_group = pd.concat(dfs, ignore_index=True)
    # Baixa a base de população por município
    df_download = bd.read_table(dataset_id='br_ms_populacao',table_id='municipio',billing_project_id="ml-na-saude")
    # Agrupa apenas as colunas desejadas
    df_populacao = df_download.groupby(['ano', 'id_municipio'])['populacao'].sum().reset_index()
    # Transformar id_municipio para apenas 6 digitos
    df_populacao['id_municipio'] = df_populacao['id_municipio'].astype(str).str[:6]
    # Alterar tipo de dados do ano
    df_populacao['ano'] = df_populacao['ano'].astype(str)
    # Merge os dataframes com base nas condições especificadas
    df_yll = df_group.merge(df_populacao, how='left', left_on=['ano_obito', 'cd_mun_res'], right_on=['ano', 'id_municipio'])
    # Drop das colunas desnecessárias após a junção
    df_yll.drop(['ano', 'id_municipio'], axis=1, inplace=True)

    return df_yll

