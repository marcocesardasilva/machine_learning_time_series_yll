from src.config import *
from src.load import *
from src.extract import *
from src.transform import *


def main():

    # ACCESS SETTINGS
    file_key, dataset, data_folder_raw, data_folder_processed, starting_year, final_year = config()

    # CONNECT TO GCP
    client = connect_to_gcp(file_key)

    # CREATE DATASET
    dataset_fonte = create_dataset(client,dataset)

    # CREATE TABLES
    table_yll = create_table(client,dataset_fonte)

    # EXTRACT
    download_files(data_folder_raw, starting_year, final_year)

    # # TRANSFORM
    # df_yll = create_df(data_folder_raw)

    # # LOAD
    # tables_dfs = {table_yll:df_yll} 
    # load_data(tables_dfs,client,dataset_fonte)

if __name__ == "__main__":
    main()

