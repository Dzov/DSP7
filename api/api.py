from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import os
import pandas as pd
import numpy as np

api = FastAPI()


class BodyItem(BaseModel):
    clientId: int


class ResponseItem(BaseModel):
    prediction: int
    probability: float
    predictionThreshold: float
    clientId: int


def get_model():
    lgbm = open('api/model.pkl', 'rb')
    return pickle.load(lgbm)


def get_client_information(client_id):
    data_path = os.path.join('api', 'client_data.csv')
    data = pd.read_csv(data_path)
    data.drop(columns='Unnamed: 0', inplace=True)

    client_info = data[data.SK_ID_CURR == int(client_id)]
    if client_info.empty:
        raise ValueError(f"Client ID {client_id} not found in the dataset.")

    return client_info.values


@api.post('/predict', response_model=ResponseItem)
async def predict_loan_eligibility(item: BodyItem):
    client_info = get_client_information(item.clientId)
    proba = get_model().predict_proba(client_info)[:, 1]
    prediction = (proba > 0.1).astype(int)
    response_data = {
        "clientId": item.clientId,
        "probability": np.round(proba, 2).tolist(),
        "predictionThreshold": 0.1,
        "prediction": prediction.tolist(),
    }
    return JSONResponse(content=response_data)