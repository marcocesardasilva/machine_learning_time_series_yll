import yaml


def config():

    # Load settings file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Access settings values
    file_key = config['variables']['file_key']
    dataset = config['variables']['dataset']
    data_folder_raw = config['variables']['data_folder_raw']
    data_folder_processed = config['variables']['data_folder_processed']
    starting_year = config['variables']['starting_year']
    final_year = config['variables']['final_year']

    return file_key, dataset, data_folder_raw, data_folder_processed, starting_year, final_year

