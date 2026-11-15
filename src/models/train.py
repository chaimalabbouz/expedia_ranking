import json
import subprocess
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml

from src.features.build_features import FeatureBuilder

SPLITS = Path("data/splits")
TARGET, GROUP = "relevance", "srch_id"
EXCLUDE = [TARGET, GROUP, "date_time"]

P = yaml.safe_load(open("params.yaml"))
SEED, NDCG_K = P["train"]["seed"], P["train"]["ndcg_k"]
RANDOM_WEIGHT = P["train"]["random_weight"]
NUM_ROUNDS, EARLY_STOP = P["train"]["num_rounds"], P["train"]["early_stop"]
PARAMS = {**P["lgbm"], "metric": "ndcg", "eval_at": [NDCG_K], "seed": SEED}


def load(name: str, fb: FeatureBuilder, train: bool = False):
    raw = pd.read_parquet(SPLITS / f"{name}.parquet")
    # random_bool lu AVANT que FeatureBuilder ne la supprime (jamais une feature)
    is_random = raw.groupby(GROUP)["random_bool"].max()

    df = fb.fit_transform(raw) if train else fb.transform(raw)
    df = df.sort_values(GROUP, kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c not in EXCLUDE]
    group_sizes = df.groupby(GROUP, sort=False).size().to_numpy()

    weight = np.where(df[GROUP].map(is_random) == 1, RANDOM_WEIGHT, 1.0)
    return df[features], df[TARGET], group_sizes, weight, features


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    fb = FeatureBuilder(**P["features"])
    X_tr, y_tr, g_tr, w_tr, features = load("train", fb, train=True)
    X_va, y_va, g_va, _, _ = load("val", fb)

    dtrain = lgb.Dataset(X_tr, y_tr, group=g_tr, weight=w_tr)  # poids seulement sur le train
    dval = lgb.Dataset(X_va, y_va, group=g_va, reference=dtrain)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("expedia-ranking")
    with mlflow.start_run(run_name="v0_baseline_ndcg38"):
        mlflow.log_params({**PARAMS, "num_rounds": NUM_ROUNDS, "n_features": len(features),
                           "random_weight": RANDOM_WEIGHT})
        mlflow.set_tags({"git_commit": git_commit(), "feature_set": "base+temporal+relative+prop+missing"})
        mlflow.log_dict({"features": features}, "features.json")

        model = lgb.train(
            PARAMS, dtrain, NUM_ROUNDS, valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(25)],
        )

        ndcg = model.best_score["val"][f"ndcg@{NDCG_K}"]
        mlflow.log_metric(f"val_ndcg_at_{NDCG_K}", ndcg)
        mlflow.log_metric("best_iteration", model.best_iteration)

        imp = pd.Series(model.feature_importance("gain"), index=features).sort_values(ascending=False)
        mlflow.log_dict(imp.round(1).to_dict(), "feature_importance_gain.json")

        print(f"\nNDCG@{NDCG_K} (val) : {ndcg:.4f} | best_iteration : {model.best_iteration}")
        print("Top 10 features (gain) :\n", imp.head(10).round(0).to_string())

        # --- Sorties pour DVC (nouveau) ---
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        model.save_model("models/model.txt")
        joblib.dump(fb, "models/feature_builder.pkl")
        json.dump({f"val_ndcg_at_{NDCG_K}": ndcg, "best_iteration": model.best_iteration},
                  open("reports/metrics.json", "w"), indent=2)
        mlflow.log_artifact("models/model.txt")


if __name__ == "__main__":
    main()