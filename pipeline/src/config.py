import yaml


def config(base_dir):

    # Load settings file
    with open(base_dir/'config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Access to configuration variables
    file_key = base_dir / config['variables']['file_key']
    dataset = config['variables']['dataset']
    datafolder_raw = config['variables']['data_folder_raw']
    datafolder_processed = config['variables']['data_folder_processed']
    starting_year = config['variables']['starting_year']
    final_year = config['variables']['final_year']

    return file_key, dataset, datafolder_raw, datafolder_processed, starting_year, final_year

