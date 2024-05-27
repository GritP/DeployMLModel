import pytest

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.metrics import f1_score

from ml.data import process_data
from ml.model import train_model, inference


# Fixtures
@pytest.fixture(scope="module")
def data():
    df = pd.read_csv("./data/census.csv")
    return df


@pytest.fixture(scope="module")
def cat_features():
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"]
    return cat_features


@pytest.fixture(scope="module")
def train_dataset(data, cat_features):
    train, test = train_test_split(data,
                                   test_size=0.20,
                                   stratify=data['salary']
                                   )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=cat_features,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train


# Test functions
def test_dataset(data):
    assert data.shape[0] > 10000, f"Dataset has only {data.shape[0]} rows."
    assert data.shape[1] == 15, f"Dataset has {data.shape[1]} columns."


def test_process_data(data, cat_features):
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
        )

    assert X_train.shape[0] == y_train.shape[0], \
        "X_train and y_train do not have the same length."
#    assert X_train.shape[1] == 108, \
#        f"X_train has {X_train.shape[1]} instead of 108 columns."
    assert isinstance(encoder, OneHotEncoder), \
        "Returned encoder is not of the expected type."
    assert isinstance(lb, LabelBinarizer), \
        "Returned binarizer is not of the expected type."


def test_train_model(train_dataset):
    X_train, y_train = train_dataset
    model = train_model(X_train, y_train)
    assert isinstance(model, HistGradientBoostingClassifier), \
        "Returned model is not of the expected type."


def test_saved_model():
    with open('./model/model.pkl', 'rb') as f:
        model = pickle.load(f)
        f.close()
    assert isinstance(model, HistGradientBoostingClassifier), \
        "Saved model is not of the expected type."


def test_inference(train_dataset):
    with open('./model/model.pkl', 'rb') as f:
        model = pickle.load(f)
        f.close()
    X_train, y_train = train_dataset
    y_pred = inference(model, X_train)
    assert y_pred.shape == y_train.shape, \
        "Predictions have not the right shape"
    assert f1_score(y_train, y_pred) >= 0, \
        "F1 on training set is negative"
    assert f1_score(y_train, y_pred) <= 1, \
        "F1 on training set is above 1"


def test_saved_encoder():
    with open('./model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
        f.close()
    assert isinstance(encoder, OneHotEncoder), \
        "Saved encoder is not of the expected type."


def test_saved_binarizer():
    with open('./model/lb.pkl', 'rb') as f:
        lb = pickle.load(f)
        f.close()
    assert isinstance(lb, LabelBinarizer), \
        "Saved binarizer is not of the expected type."
