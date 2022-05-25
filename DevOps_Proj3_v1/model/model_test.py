import os
from pathlib import Path
import logging
import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import sys
sys.path.append('.')
from  data_processing.process_data import entire_data_processing  

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


def data_folder_path(subfolder, file):
    """[summary]
    Args:
        subfolder ([str]): [subfolder name]
        file ([str]): [file name]
    Returns:
        [str]: [file complete path]
    """
    root = os.path.dirname(os.getcwd())
    return os.path.join(root, "Udacity-Machine-Learning-DevOps-Engineer/DevOps_Proj3_v1", subfolder, file)


DATA_PATH = data_folder_path('data', 'census_cleaned.csv')
MODEL_PATH = data_folder_path('model','best_clf.pkl')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(name='data')
def data():
    """
    Fixture will be used by the unit tests.
    """
    yield pd.read_csv(DATA_PATH)


def test_load_data(data):
    
    """ Check the data received """

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0]>0
    assert data.shape[1]>0


def test_model():

    """ Check model type """

    model = joblib.load(MODEL_PATH)
    assert str(type(model)) == "<class 'lightgbm.sklearn.LGBMClassifier'>"


def test_process_data(data):

    """ data shape and label content """
    
    np_data = entire_data_processing(data, True)
    
    y = np_data[:, -1]
    X = np_data[:, :-1]
    assert X.shape == (20312, 9)
    assert X.shape[0] == len(y)
    assert {0,1} == set(y)
    
    
    
    
    
    
    
    
    
    
