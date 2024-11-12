from pathlib import Path
from src.config import *
from src.connect import *
from src.create import *
from src.extract import *
from src.transform import *
from src.load import *

def main():

    # IDENTIFY BASE DIRECTORY
    base_dir = Path(__file__).resolve().parent.parent

    # ACCESS TO CONFIGURATION VARIABLES
    file_key, dataset, datafolder_raw, datafolder_processed, starting_year, final_year = config(base_dir)

    # CONNECT TO GCP
    client = connect_to_gcp(file_key)

    # CREATE DATASET
    dataset_fonte = create_dataset(client,dataset)

    # CREATE TABLES
    table_yll, table_population, table_municipality = create_tables(client,dataset_fonte)

    # EXTRACT
    download_files(datafolder_raw, starting_year, final_year)

    # TRANSFORM    
    population_df = create_population_df()
    municipality_df = create_municipality_df(datafolder_raw)
    yll_df = create_yll_df(datafolder_raw)

    # LOAD
    tables_dfs = {table_population:population_df,table_municipality:municipality_df,table_yll:yll_df} 
    load_data(tables_dfs,client,dataset_fonte)
    download_data(tables_dfs,datafolder_processed)

if __name__ == "__main__":
    main()

