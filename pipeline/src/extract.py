import os
from urllib import request


def download_files(data_folder_raw, starting_year, final_year):
    # Create the directory for the data if it does not already exist
    if not os.path.exists(data_folder_raw):
        os.makedirs(data_folder_raw)
    
    print('--------------------------------------------------------------------------')
    print('Extraindo os dados...')
    print('--------------------------------------------------------------------------')

    # Download the files for the desired years
    for year in range(starting_year, final_year):
        year_to_download = str(year)

        print('--------------------------------------------------------------------------')
        print(f'Proximo ano a carregar: {year_to_download}')

        # Set the link and files
        if year <= 2020:
            file_mortality = f'Mortalidade_Geral_{year_to_download}.csv'
            url_mortality = f'https://diaad.s3.sa-east-1.amazonaws.com/sim/{file_mortality}'
        elif year == 2021:
            file_mortality = f'Mortalidade_Geral_{year_to_download}.csv'
            url_mortality = f'https://S3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/{file_mortality}'
        elif year >= 2022:
            reduced_year = year_to_download[-2:]
            file_mortality = f'DO{reduced_year}OPEN.csv'
            url_mortality = f'https://S3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/{file_mortality}'

        # Check if the file was downloaded
        if(os.path.exists(f'{data_folder_raw}/{file_mortality}')):
            print(f'Arquivo {file_mortality} já foi baixado')
        else:
        # Try to download the file
            try:
                print(f'Baixando o arquivo {file_mortality}')
                request.urlretrieve(f'{url_mortality}', f'{data_folder_raw}/{file_mortality}')
                # Check if the file was downloaded
                if(os.path.exists(f'{data_folder_raw}/{file_mortality}')):
                    print(f'Arquivo {file_mortality} baixado')
                else:
                    print('--------------------------------------------------------------------------')
                    print('Não foi possível baixar o arquivo. Execução finalizada!')
                    print('--------------------------------------------------------------------------')
            except:
                print(f'Arquivos de {year_to_download} ainda não disponibilizado')
    
    print('--------------------------------------------------------------------------')
    print(f'Extraindo os arquivos de população anual por município...')

    # Download file with IBGE codes for municipalities
    file_ibge_codes = 'municipios.csv'
    url_ibge_codes = f'https://www.gov.br/receitafederal/dados/{file_ibge_codes}'

    # Check if the file was downloaded
    if(os.path.exists(f'{data_folder_raw}/{file_ibge_codes}')):
        print(f'Arquivo {file_ibge_codes} já foi baixado')
    else:
    # Try to download the file
        try:
            print(f'Baixando o arquivo {file_ibge_codes}')
            request.urlretrieve(f'{url_ibge_codes}', f'{data_folder_raw}/{file_ibge_codes}')
            # Check if the file was downloaded
            if(os.path.exists(f'{data_folder_raw}/{file_ibge_codes}')):
                print(f'Arquivo {file_ibge_codes} baixado')
            else:
                print('--------------------------------------------------------------------------')
                print('Não foi possível baixar o arquivo. Execução finalizada!')
                print('--------------------------------------------------------------------------')
        except:
            print(f'Arquivos {file_ibge_codes} não disponibilizado')

    print('--------------------------------------------------------------------------')
    print('Todos os arquivos disponíveis foram baixados!')
    print('--------------------------------------------------------------------------')

