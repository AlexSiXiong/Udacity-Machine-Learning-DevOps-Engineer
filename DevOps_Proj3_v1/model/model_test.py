from pathlib import Path
import logging
import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


DATA_PATH = '../data/census_cleaned.csv'
MODEL_PATH = './best_clf.pkl'

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

    """ Test the data split """

    train, _ = train_test_split(data, test_size=0.20)
    
    assert int(len(data)*0.8) == len(train)
    
