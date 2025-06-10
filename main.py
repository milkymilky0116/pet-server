from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch

from models.ranknet_unidirectional import load_model as load_uni_model
from models.ranknet_bidirectional import load_model as load_bi_model
from utils.preprocess import prepare_vectors
from utils.evaluator import evaluate_all
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_excel("data/dog_dummy_data_1200_with_gps_final.xlsx")


class DogInput(BaseModel):
    name: str
    age_month: float
    weight: float
    color: str
    personality: str
    region: str
    vaccinated: str
    preference_age_range: str
    preference_weight_range: str
    preference_color: str
    preference_personality: str
    preference_region: str
    preference_vaccine: str
    lat: float
    lon: float


@app.post("/recommend")
def recommend(input: DogInput, model_version: str = "bi"):
    user_row = pd.DataFrame([input.dict()])
    user_vectors, dog_vectors, gps, ids, names, personalities = prepare_vectors(
        user_row, df, mode=model_version
    )
    model = load_bi_model() if model_version == "bi" else load_uni_model()
    with torch.no_grad():
        scores = model(torch.tensor(dog_vectors, dtype=torch.float32)).squeeze().numpy()
    top_idx = np.argsort(scores)[::-1][:10]
    results = []
    for idx in top_idx:
        results.append(
            {
                "id": str(ids[idx]),
                "name": names[idx],
                "score": float(scores[idx]),
                "lat": gps[idx][0],
                "lon": gps[idx][1],
                "personality": personalities[idx],
                "image": f"{ids[idx]}.png",
            }
        )
    return {"recommendations": results}


@app.get("/admin/evaluate")
def evaluate_model(version: str = "bi"):
    result = evaluate_all(df, version)
    return {"version": version, "evaluation": result}


@app.get("/")
def root():
    return {"status": "OK", "message": "애견 추천 API 작동 중"}
