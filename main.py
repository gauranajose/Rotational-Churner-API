from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orig_churn = pd.read_csv('./data/churned_user_data.csv')
orig_new = pd.read_csv('./data/new_user_data.csv')

comp_churn = pd.read_csv('./data/sim_churned.csv')
comp_new = pd.read_csv('./data/sim_new.csv')


orig_churn['mobile_number'] = orig_churn['mobile_number'].astype(str)
orig_new['mobile_number'] = orig_new['mobile_number'].astype(str)

comp_churn['mobile_number'] = orig_churn['mobile_number'].astype(str)
comp_new['mobile_number'] = orig_new['mobile_number'].astype(str)


def convert_to_mdlist(df, user):
    res = df[df['mobile_number'] == user].copy()
    res = res.drop('mobile_number', axis=1)
    return np.array(res.values.tolist())


def euclidean_distance(doc_a, doc_b):
    return np.sqrt(((doc_a - doc_b) ** 2).sum(-1))


@app.get("/")
def home():
    return {"message": "Hello World"}


@app.get('/users')
def get_all_mobile_users():
    return {key: val for key, val in zip(range(len(orig_new)), orig_new['mobile_number'].unique())}


@app.get('/similarity/{mobile_number}')
async def get_similarity(mobile_number: str):
    if mobile_number not in orig_new['mobile_number'].unique():
        return {'error': 'user not found'}

    distances = {}
    user_mat = convert_to_mdlist(comp_new, mobile_number)
    for c_user in orig_churn['mobile_number'].unique():
        churn_mat = convert_to_mdlist(comp_churn, c_user)
        res = euclidean_distance(user_mat, churn_mat)
        distances[c_user] = res.sum() / len(res)

    closest_user = min(distances, key=distances.get)

    data_new = orig_new[orig_new['mobile_number']
                        == mobile_number].to_json(orient="records")
    data_churned = orig_churn[orig_churn['mobile_number']
                              == closest_user].to_json(orient="records")

    parsed_data_churn = json.loads(data_churned)
    parsed_data_new = json.loads(data_new)

    return {'user': mobile_number, 'user_data':  parsed_data_new, 'closest_user': closest_user, 'closest_user_data': parsed_data_churn, 'distance': distances[closest_user]}
