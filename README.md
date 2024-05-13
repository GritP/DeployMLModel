# Deploying a ML Model to a Cloud Platform with FastAPI
Final project of the Udacity course "Deploying a Scalable ML Pipeline in Production" which is part of the ML DevOps Engineer Nanodegree.

## Instructions
### Instructions Summary
Review the rubric(opens in a new tab) requirements to understand what is being graded before starting.

Working in a command line environment is recommended for ease of use with Git and DVC. If on Windows, WSL1 or 2 is recommended.

### Repositories
- Create a directory for the project and initialize Git.
- As you work on the code, continually commit changes. Trained models you want to keep must be committed to GitHub.
- Connect your local Git repository to GitHub.
### GitHub Actions
- Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
- Make sure you set up the GitHub Action to have the same version of Python as you used in development.
### Data
- Download census.csv from the data folder in the starter repository.
- Information on the dataset can be found here(opens in a new tab).
- This data is messy, try to open it in pandas and see what you get.
- To clean it, use your favorite text editor to remove all spaces.
### Model
- Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
- Write unit tests for at least 3 functions in the model code.
- Write a function that outputs the performance of the model on slices of the data.
- Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
- Write a model card using the provided template.
### API Creation
- Create a RESTful API using FastAPI this must implement:
GET on the root giving a welcome message.
POST that does model inference.
- Type hinting must be used.
- Use a Pydantic model to ingest the body from POST. This model should contain an example.
Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
- Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).
- Run sanity check for your test cases:
Run python sanitycheck.py. This script is located inside the starter directory in the starter code.
The script will scan the test cases written for the GET() and POST() APIs and generate a report.
The report will list any problems it detects with your test cases. Fix the problems and run the sanitycheck.py script again.
The script uses heuristics to detect common problems and can sometimes overlook a problem or raise a false alarm. You should still check your implementation against the project rubric to be absolutely sure your submission will meet the requirements.
### API Deployment to a Cloud Application Platform
- Create a Cloud Application Platform account, such as Heroku or Render. For the following steps, you can either use the web GUI or download the CLI.
Note: starting Nov 28, 2022, Heroku has announced the removal of the free-tier account, which will be replaced by a low-cost subscription plan. (ref: Removal of Heroku Free Product Plans FAQ(opens in a new tab))
- Create a new app and have it deployed from your GitHub repository.
- Enable automatic deployments that only deploy if your continuous integration passes.
Hint: think about how paths will differ in your local environment vs. on Heroku.
Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
- Write a script that uses the requests module to do one POST on your live API.
