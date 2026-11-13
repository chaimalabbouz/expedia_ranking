import subprocess
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.features.build_features import FeatureBuilder

SPLITS = Path("data/splits")
TARGET, GROUP = "relevance", "srch_id"
EXCLUDE = [TARGET, GROUP, "date_time"]
SEED = 42
NDCG_K = 5

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [NDCG_K],
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "seed": SEED,
    "verbose": -1,
}
NUM_ROUNDS = 300
EARLY_STOP = 30


def load(name: str, fb: FeatureBuilder):
    df = fb.transform(pd.read_parquet(SPLITS / f"{name}.parquet"))
    df = df.sort_values(GROUP, kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c not in EXCLUDE]
    group_sizes = df.groupby(GROUP, sort=False).size().to_numpy()
    return df[features], df[TARGET], group_sizes, features


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    fb = FeatureBuilder().fit(pd.read_parquet(SPLITS / "train.parquet"))
    X_tr, y_tr, g_tr, features = load("train", fb)
    X_va, y_va, g_va, _ = load("val", fb)

    dtrain = lgb.Dataset(X_tr, y_tr, group=g_tr)
    dval = lgb.Dataset(X_va, y_va, group=g_va, reference=dtrain)

    mlflow.set_experiment("expedia-ranking")
    with mlflow.start_run(run_name="v1_relative"):
        mlflow.log_params({**PARAMS, "num_rounds": NUM_ROUNDS, "n_features": len(features)})
        mlflow.set_tags({"git_commit": git_commit(), "feature_set": "base+temporal+relative"})
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


if __name__ == "__main__":
    main()