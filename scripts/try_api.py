import json

import pandas as pd
import requests

df = pd.read_parquet("data/splits/test.parquet")
srch = df[df["srch_id"] == df["srch_id"].iloc[0]]
payload_df = srch.drop(columns=["position", "random_bool", "click_bool",
                                "gross_bookings_usd", "booking_bool"])
payload = {"hotels": json.loads(payload_df.to_json(orient="records", date_format="iso"))}

r = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(r.status_code)
print(json.dumps(r.json()["ranking"][:5], indent=2))
print("Hôtels réservés/cliqués :", srch.loc[srch["click_bool"] == 1, "prop_id"].tolist())