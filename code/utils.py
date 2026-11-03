import pickle
import numpy as np
import pandas as pd
import json

def load_dataset_desc(dataset_name: str) -> str:
    """
    Get the description of a dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
    
    Returns:
        str: Description of the dataset.
    """

    # read json file: datasets/dataset_name/desc.json
    with open(f'{dataset_name}/desc.json', 'r') as f:
        desc = json.load(f)
    return desc

def load_dataset_data(dataset_name: str) -> np.ndarray:
    """
    Load data from a .dat file (memmap) via numpy.

    Args:
        dataset_name (str): Path to the .dat file.

    Returns:
        np.ndarray: Loaded data.
    """

    shape = load_dataset_desc(dataset_name)['shape']
    dat_file_path = f'{dataset_name}/data.dat'
    data = np.memmap(dat_file_path, mode='r', dtype=np.float32, shape=tuple(shape)).copy()
    return data

def load_pkl(pickle_file: str) -> object:
    """
    Load data from a pickle file.

    Args:
        pickle_file (str): Path to the pickle file.

    Returns:
        object: Loaded object from the pickle file.
    """

    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'Unable to load data from {pickle_file}: {e}')
        raise
    # import ipdb; ipdb.set_trace()
    # pickle_data is a tuple, (adj_mx)
    return pickle_data

def load_and_preprocess_pems_data(data_file_path):
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    data = np.load(data_file_path)['data']
    data = data[..., 0]
    print(f'Raw time series shape: {data.shape}')
    return data

def load_and_preprocess_data(data_file_path):
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    df = pd.read_hdf(data_file_path)
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., 0]
    print(f'Raw time series shape: {data.shape}')
    return data, df