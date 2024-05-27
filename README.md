# Deploying a ML Model to a Cloud Platform with FastAPI
Final project of the Udacity course "Deploying a Scalable ML Pipeline in Production" which is part of the ML DevOps Engineer Nanodegree. 

The project is based on 1994 Census Bureau data from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). The aim is to predict whether an individual's yearly income exceeded $50,000 in 1994 based on the following demographic and socio-economic information:
- sex
- race
- marital status
- age
- native-country	
- education	
- relationship status
- occupation
- hours-per-week
- workclass	
- capital-gain	
- capital-loss 
Details about the model used, data preprocessing, and model training can be found in the model card. The training can be repeated for census.csv stored in the /data folder by running
```
python train_model.py
```

An API for model inference is created which accepts sample data in json format and returns 0 (yearly income <= $50,000) or 1 (yearly income > $50,000) as classified by the model. To call the API the Python script api_requests.py can be used which also includes a sample request. The API can be run locally using
```
uvicorn main:app
```
For the sake of this project the API was temporarily deployed to Render but suspended to avoid unnecessary costs.

Model training as well as the API script are tested automatically employing pytest on push to GitHub and also flake8 is run to ensure proper code style.
