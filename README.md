# Pret à Dépenser 

## Description

The goal of this project was to build and train a prediction model that would allow the company "Prêt à dépenser" to determine the probabily that a given client would be able or not to repay their loan. 

The data provided allowed us to train and test different models. 

Based on the results of the different model, the aim was then to develop an API that would predict a client's score using the best model. 


## Steps 

### Training
We first started by cleaning the data in order to train different mmodels. 

The results were logged on a MLFlow registry. For each model, we logged the model itself, the best parameters identified by a GridSearchCV, along with the different metrics. Using SHAP, we were also able to calculate the feature importance for each model and store the generated graph as an artifact. 

The registry is accessible by running the command `mlflow server --host 127.0.0.1 --port 8080 `

### API

Once we trained a model that was able to make acceptable predictions based on a in house business score, we were able to develop an API that would leverage that model. 

We created the endpoint `POST /predict` that takes a `clientId` as a parameter and returns a response containing the prediction. 

The API documentation is accessible [here.](https://pret-a-depenser.azurewebsites.net/docs#/)

### Deployment

Once the API was up and running, we could then deploy it using Azure. 

We started by creating a docker container, build an image which we uploaded on an Azure Container Registry and then created the Web App that would deploy the container image. 

We then configured Github Actions to run these steps everytime a new commit is pushed to the repository, in order to automatically create an updated image and deploy the new version. 

Github Actions was also configured to run the API unit and integration tests before the build and deploy steps in order to make sure the app works ass expected before being pushed to production. 

## Project Folder Structure

A single github repository was used for the entire project, not just the api. The project was organized as follows: 

- The modeling part of the project is located in the notebooks folder. 

- The api is located in the api folder and is independent from the modeling part of the projet. This is the only folder that is used by docker and pushed in production. 

`project-root/`
- `notebooks/`
  - `notebook_modelisation` - model training and testing
  - `notebook_test_api` - notebook to test production api
  - `notebook_evidently` - notebook to create the data drift report
  - `data_drift_report` - html page containing the data drift report
  - `data/` 
    - `csv` - data used by modeling notebook
- `api/`
  - `requirements.txt` - api package requirements
  - `main.py` - main application
  - `data/` - data used by api
  - `tests/`
    - `tests.py` - unit and integration tests for api
  - `model/`
    - `model.pkl` - best model saved on mlflow
- `README.md`
