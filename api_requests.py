import requests
import json

# URL = "https://deploymlmodel.onrender.com/"
URL = "http://127.0.0.1:8000/"

# test get method and collect response
response = requests.get(URL)

print("response status code:", response.status_code)
print("response content:", response.json())


# some random sample to perform inference on
sample = {'age': 52,
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

data = json.dumps(sample)

# post to API and collect response
response = requests.post(URL+"inference?",
                         data=data)

print("response status code:", response.status_code)
print("response content:", response.json())
