import os
from urllib import request


def download_files(data_folder_raw, starting_year, final_year):
    # Create the directory for the data if it does not already exist
    if not os.path.exists(data_folder_raw):
        os.makedirs(data_folder_raw)

    # Download the files for the desired years
    for year in range(starting_year, final_year):
        year_to_download = str(year)

        print("--------------------------------------------------------------------------")
        print(f'Proximo ano a carregar: {year_to_download}')
        print("--------------------------------------------------------------------------")

        # Set the link and files
        file = f'Mortalidade_Geral_{year_to_download}.csv'
        link = f'https://diaad.s3.sa-east-1.amazonaws.com/sim/{file}'

        # Check if the file was downloaded
        if(os.path.exists(f'{data_folder_raw}/{file}')):
            print(f'Arquivo {file} já foi baixado')
        else:
        # Try to download the file
            try:
                print(f'Baixando o arquivo {file}')
                request.urlretrieve(f'{link}', f'{data_folder_raw}/{file}')
                # Check if the file was downloaded
                if(os.path.exists(f'{data_folder_raw}/{file}')):
                    print(f'Arquivo {file} baixado')
                else:
                    print("--------------------------------------------------------------------------")
                    print('Não foi possível baixar o arquivo. Execução finalizada!')
                    print("--------------------------------------------------------------------------")
            except:
                print(f'Arquivos de {year_to_download} ainda não disponibilizado')
                print("--------------------------------------------------------------------------")
                print('Todos os arquivos disponíveis foram baixados!')
                print("--------------------------------------------------------------------------")

