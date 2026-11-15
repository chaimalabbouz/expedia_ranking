import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

N = 20
model = lgb.Booster(model_file="models/model.txt")
fb = joblib.load("models/feature_builder.pkl")

df = pd.read_parquet("data/splits/test.parquet")
ids = df["srch_id"].drop_duplicates().head(N).tolist()
diffs = []

for sid in ids:
    srch = df[df["srch_id"] == sid]

    # Score "offline" (comme evaluate.py)
    feats = fb.transform(srch)
    offline = pd.Series(model.predict(feats[model.feature_name()]), index=srch["prop_id"].values)

    # Score "API"
    payload_df = srch.drop(columns=["position", "random_bool", "click_bool",
                                    "gross_bookings_usd", "booking_bool"])
    payload = {"hotels": json.loads(payload_df.to_json(orient="records", date_format="iso"))}
    r = requests.post("http://127.0.0.1:8000/predict", json=payload)
    api = pd.Series({x["prop_id"]: x["score"] for x in r.json()["ranking"]})

    diffs.append(np.abs(offline.reindex(api.index) - api).max())

print(f"Écart max sur {N} recherches : {max(diffs):.6f}")
print("OK" if max(diffs) < 1e-3 else "ATTENTION : décalage entre API et entraînement")