import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    """
    Tests GET. Status code and if it is returning what is expected
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hi": "This app predicts wether income exceeds $50K/yr through user input info"}


def test_post_more_than_50(client):
    """
    Tests POST for a prediction less than 50k.
    Status code and if the prediction is the expected one
    """
    response = client.post("/inference", json={
        "age": 37,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 80,
        "native_country": "United-States"
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": ">=50K"}


def test_post_less_than_50(client):
    """
    Tests POST for a prediction more than 50k.
    Status code and if the prediction is the expected one
    """
    response = client.post("/inference", json={
        "age": 28,
        "workclass": "Private",
        "fnlgt": 183175,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
