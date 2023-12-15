from fastapi import FastAPI, HTTPException
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
    lgbm = open('model.pkl', 'rb')
    return pickle.load(lgbm)


def get_client_information(client_id):
    data_path = os.path.join('data', 'sample_client_data.csv')
    data = pd.read_csv(data_path)
    data.drop(columns='Unnamed: 0', inplace=True)

    client_info = data[data.SK_ID_CURR == int(client_id)]
    if client_info.empty:
        raise HTTPException(
            status_code=404, detail=f"Client ID {client_id} not found.")

    return client_info.values


def predict_loan_eligibility(client_id: int):
    try:
        client_info = get_client_information(client_id)
    except HTTPException as e:
        raise

    proba = get_model().predict_proba(client_info)[:, 1]
    prediction = (proba > 0.1).astype(int)
    return {
        "clientId": client_id,
        "probability": np.round(proba, 2).tolist(),
        "predictionThreshold": 0.1,
        "prediction": prediction.tolist(),
    }

@api.get('/')
async def home():
    return {'message': "Prêt à Dépenser"}

@api.post('/predict', response_model=ResponseItem)
async def predict(item: BodyItem):
    response_data = predict_loan_eligibility(item.clientId)
    return JSONResponse(content=response_data)