import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

COLS = [
    "price_usd", "prop_starrating", "prop_review_score", "prop_location_score1",
    "prop_location_score2", "prop_log_historical_price", "promotion_flag",
    "srch_length_of_stay", "srch_booking_window", "srch_adults_count",
    "srch_children_count", "srch_room_count", "orig_destination_distance",
]
N = 50_000
OUT = Path("reports")


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLS)
    return df.sample(min(N, len(df)), random_state=42).astype(float).reset_index(drop=True)


def run(ref: pd.DataFrame, cur: pd.DataFrame, html_name: str) -> dict:
    result = Report([DataDriftPreset()]).run(cur, ref)
    result.save_html(str(OUT / html_name))
    return result.dict()


def drift_share(d: dict):
    for m in d["metrics"]:
        if str(m.get("metric_name", "")).startswith("DriftedColumnsCount"):
            return m["value"]["share"]
    return None


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ref = load("data/splits/train.parquet")
    prod = load("data/production/production.parquet")

    real = run(ref, prod, "drift_production.html")

    sim_df = prod.copy()
    sim_df["price_usd"] = sim_df["price_usd"] + 1.5 * ref["price_usd"].std()
    sim = run(ref, sim_df, "drift_simulated.html")

    summary = {"drift_share_production": drift_share(real), "drift_share_simulated": drift_share(sim)}
    json.dump(summary, open(OUT / "drift_summary.json", "w"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()