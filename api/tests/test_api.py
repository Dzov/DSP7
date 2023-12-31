import pandas as pd
import os
import json
import numpy as np
from fastapi.testclient import TestClient
from .. import main

client = TestClient(main.api)

data = pd.read_csv(os.path.join('data', 'sample_client_data.csv'))
data.drop(columns='Unnamed: 0', inplace=True)

def get_client_id():
    return int(data.loc[0, 'SK_ID_CURR'])

def test_home():
    response = client.get('/')
    assert response.status_code == 200

def test_post_predict():
    client_id = get_client_id()
    response = client.post(
        "/predict/",
        json={"clientId": client_id},
    )
    assert response.status_code == 200
    assert response.json().get('clientId') == client_id

def test_client_not_found():
    response = client.post(
        "/predict/",
        json={"clientId": -1},
    )
    assert response.status_code == 404

def test_client_bad_request():
    clientId = get_client_id()
    response = client.post(
        "/predict/",
        json={"clientdId": clientId},
    )
    assert response.status_code == 422


def test_get_client_information():
    client_id = get_client_id()
    actual_client_info = main.get_client_information(client_id)
    expected_client_info = data[data.SK_ID_CURR == int(client_id)]
    assert np.array_equal(actual_client_info,  expected_client_info.values)
