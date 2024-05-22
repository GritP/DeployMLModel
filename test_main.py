from fastapi.testclient import TestClient
import json
from main import app


client = TestClient(app)

sample_0 = {'age': 34,
            'workclass': "Pivate",
            'fnlgt': 148291,
            'education': "HS-grad",
            'education_num': 9,
            'marital_status': "Married-civ-spouse",
            'occupation': "Tech-support",
            'relationship': "Wife",
            'race': "White",
            'sex': "Female",
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 32,
            'native_country': "United-States"
            }

sample_1 = {'age': 52,
            'workclass': "Self-emp-not-inc",
            'fnlgt': 334273,
            'education': "Bachelors",
            'education_num': 13,
            'marital_status': "Married-civ-spouse",
            'occupation': "Prof-specialty",
            'relationship': "Husband",
            'race': "White",
            'sex': "Male",
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 60,
            'native_country': "United-States"
            }


def test_get():
    r = client.get("/")
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "Welcome to the inference API for census data"


def test_post_data_0():
    data = sample_0
    r = client.post("/inference/", data=json.dumps(data))
    print(r.json())
    assert r.status_code == 200
    assert r.json() == [0]


def test_post_data_1():
    data = sample_1
    r = client.post("/inference/", data=json.dumps(data))
    print(r.json())
    assert r.status_code == 200
    assert r.json() == [1]
