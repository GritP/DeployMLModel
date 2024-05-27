# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field

import pandas as pd
import pickle

from ml.data import process_data
from ml.model import inference


# define input data as suggested by reviewer
def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class Data(BaseModel):
    age: int = Field(..., example=45)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=2334)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Wife")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=60)
    native_country: str = Field(..., example="Cuba")

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

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True


# Instantiate app
app = FastAPI(
    title="Inference API",
    description="An API to perform inference on the provided data.",
    version="1.0.0",
)

# suggested by reviewer to reduce latency but then pytest fails
# @app.on_event("startup")
# async def startup_event():
#    global model, encoder, binarizer
#    model = pickle.load(open("./model/model.pkl", "rb"))
#    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
#    binarizer = pickle.load(open("./model/binarizer.pkl", "rb"))


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

    # load model, encoder, and binarizer
    model = pickle.load(open("./model/model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    binarizer = pickle.load(open("./model/lb.pkl", "rb"))

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
