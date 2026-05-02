from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent.parent / "model" / "artifact.pkl"))


class DiamondInput(BaseModel):
    carat: float = Field(..., gt=0)
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float = Field(..., gt=0)
    y: float = Field(..., gt=0)
    z: float = Field(..., gt=0)

'''
Загружает модель и карты категорий из файла artifact.pkl
:return: dict - модель и карты категорий
'''
def load_artifact():
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


artifact = None
app = FastAPI(title="Diamond price API", version="1.0.0")


@app.on_event("startup")
def startup():
    global artifact
    artifact = load_artifact()

'''
Проверяет, что модель загружена
:return: dict - статус ok
'''
@app.get("/health")
def health():
    return {"status": "ok"}


'''
Предсказывает цену бриллианта
:param body: DiamondInput - данные бриллианта
:return: dict - предсказанная цена
:raises: HTTPException если модель не загружена или данные некорректны
'''
@app.post("/predict")
def predict(body: DiamondInput):
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    maps: dict = artifact["maps"]
    model = artifact["model"]
    feature_columns: list[str] = artifact["feature_columns"]

    for field, m in [("cut", maps["cut"]), ("clarity", maps["clarity"]), ("color", maps["color"])]:
        val = getattr(body, field)
        if val not in m:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown {field}={val!r}. Allowed: {sorted(m.keys())}",
            )

    volume = body.x * body.y * body.z
    if volume == 0:
        raise HTTPException(status_code=400, detail="volume (x*y*z) must be non-zero")

    row = {
        "carat": body.carat,
        "cut": maps["cut"][body.cut],
        "color": maps["color"][body.color],
        "clarity": maps["clarity"][body.clarity],
        "depth": body.depth,
        "table": body.table,
        "x": body.x,
        "y": body.y,
        "z": body.z,
        "volume": volume,
    }
    X = pd.DataFrame([row])[feature_columns]
    price = float(model.predict(X)[0])
    return {"price": price}
