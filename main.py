# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import pickle

from ml.data import process_data
from ml.model import inference

# define input data
class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    # add example as described on
    # https://fastapi.tiangolo.com/tutorial/schema-extra-example/
    model_config = {
        "json_schema_extra": {
            "examples": [
                {'age': 52,
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
            ]
        }
    }


# Instantiate app
app = FastAPI(
    title="Inference API",
    description="An API to perform inference on the provided data.",
    version="1.0.0",
)

# suggested by reviewer to reduce latency
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, binarizer
    model = pickle.load(open("./model/model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    binarizer = pickle.load(open("./model/lb.pkl", "rb"))

# Define welcome get method
@app.get("/")
async def welcome():
    return "Welcome to the inference API for census data"


# Define predict method to perform inference on posted data
@app.post("/inference/")
async def predict(data: Data):
    sample = {'age': data.age,
              'workclass': data.workclass,
              'fnlgt': data.fnlgt,
              'education': data.education,
              'education-num': data.education_num,
              'marital-status': data.marital_status,
              'occupation': data.occupation,
              'relationship': data.relationship,
              'race': data.race,
              'sex': data.sex,
              'capital-gain': data.capital_gain,
              'capital-loss': data.capital_loss,
              'hours-per-week': data.hours_per_week,
              'native-country': data.native_country,
              }

    # prepare the sample for inference as a dataframe
    sample_df = pd.DataFrame(sample, index=[0])

    # apply transformation to sample data
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

    # process posted data
    sample, _, _, _ = process_data(
                                sample_df,
                                categorical_features=cat_features,
                                training=False,
                                encoder=encoder,
                                lb=binarizer
                                )

    # perform inference
    y_pred = inference(model, sample)

    return y_pred.tolist()
