import time
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict

MODEL_PATH = "models/model.txt"
FB_PATH = "models/feature_builder.pkl"

RAW_COLS = [
    "srch_id", "date_time", "site_id", "visitor_location_country_id", "visitor_hist_starrating",
    "visitor_hist_adr_usd", "prop_country_id", "prop_id", "prop_starrating", "prop_review_score",
    "prop_brand_bool", "prop_location_score1", "prop_location_score2", "prop_log_historical_price",
    "price_usd", "promotion_flag", "srch_destination_id", "srch_length_of_stay", "srch_booking_window",
    "srch_adults_count", "srch_children_count", "srch_room_count", "srch_saturday_night_bool",
    "srch_query_affinity_score", "orig_destination_distance",
] + [f"comp{i}_{s}" for i in range(1, 9) for s in ("rate", "inv", "rate_percent_diff")]
PLACEHOLDER_COLS = ["position", "random_bool", "click_bool", "gross_bookings_usd", "booking_bool"]
NUM_COLS = [c for c in RAW_COLS if c != "date_time"]

state = {}

REQUESTS = Counter("predict_requests_total", "Requêtes /predict", ["status"])
LATENCY = Histogram("predict_latency_seconds", "Latence de /predict",
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
LIST_SIZE = Histogram("predict_list_size", "Nombre d'hôtels par requête",
                      buckets=(1, 5, 10, 20, 30, 50, 100, 200))
SCORES = Histogram("predict_score", "Scores prédits",
                   buckets=(-2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2))
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = lgb.Booster(model_file=MODEL_PATH)
    state["fb"] = joblib.load(FB_PATH)
    yield
    state.clear()


app = FastAPI(title="Expedia Hotel Ranking API", lifespan=lifespan)


class Hotel(BaseModel):
    model_config = ConfigDict(extra="allow")
    prop_id: int
    price_usd: float
    date_time: datetime
    srch_id: int = 0


class SearchRequest(BaseModel):
    hotels: list[Hotel]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(req: SearchRequest):
    start = time.perf_counter()
    if not req.hotels:
        REQUESTS.labels("422").inc()
        raise HTTPException(422, "La liste d'hôtels est vide")

    df = pd.DataFrame([h.model_dump() for h in req.hotels])
    df["srch_id"] = 0
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.reindex(columns=RAW_COLS)

    # Forcer les colonnes en nombres AVANT le FeatureBuilder
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")

    for c in PLACEHOLDER_COLS:
        df[c] = 0

    feats = state["fb"].transform(df)

    # Forcer le type float APRÈS le FeatureBuilder
    X = feats[state["model"].feature_name()].apply(pd.to_numeric, errors="coerce").astype(float)
    df["score"] = state["model"].predict(X)

    ranked = df.sort_values("score", ascending=False).reset_index(drop=True)

    LIST_SIZE.observe(len(df))
    for s in df["score"]:
        SCORES.observe(float(s))
    REQUESTS.labels("200").inc()
    LATENCY.observe(time.perf_counter() - start)

    return {"ranking": [
        {"rank": i + 1, "prop_id": int(r.prop_id), "score": round(float(r.score), 5)}
        for i, r in ranked.iterrows()
    ]}