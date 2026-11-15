import json
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml

GROUP = "srch_id"
P = yaml.safe_load(open("params.yaml"))
KS = P["evaluate"]["ks"]  # [5, 38]


def _dcg_per_search(srch_id: pd.Series, order_by: np.ndarray, rel: np.ndarray, k: int) -> pd.Series:
    """DCG@k par recherche, gain linéaire (officiel ICDM), tri stable."""
    d = pd.DataFrame({"s": srch_id.to_numpy(), "o": order_by, "r": rel})
    d = d.sort_values(["s", "o"], ascending=[True, False], kind="stable")
    rank = d.groupby("s").cumcount() + 1
    d["g"] = np.where(rank <= k, d["r"] / np.log2(rank + 1), 0.0)
    return d.groupby("s")["g"].sum()


def ndcg_per_search(srch_id, score, rel, k) -> pd.Series:
    dcg = _dcg_per_search(srch_id, score, rel, k)
    idcg = _dcg_per_search(srch_id, rel, rel, k)
    out = dcg / idcg.replace(0, np.nan)
    return out.dropna()  # recherches sans clic ni réservation exclues


def main() -> None:
    model = lgb.Booster(model_file="models/model.txt")
    fb = joblib.load("models/feature_builder.pkl")

    raw = pd.read_parquet("data/splits/test.parquet")
    seg = raw.groupby(GROUP).agg(is_random=("random_bool", "max"), list_size=("prop_id", "size"))

    df = fb.transform(raw).sort_values(GROUP, kind="stable").reset_index(drop=True)
    X = df[model.feature_name()]
    df["score"] = model.predict(X)

    metrics, per_search = {}, {}
    for k in KS:
        s = ndcg_per_search(df[GROUP], df["score"].to_numpy(), df["relevance"].to_numpy(), k)
        per_search[k] = s
        metrics[f"test_ndcg_at_{k}"] = round(float(s.mean()), 5)

    # Métriques par segment (NDCG@5)
    s5 = per_search[5].to_frame("ndcg").join(seg)
    s5["size_bin"] = pd.cut(s5["list_size"], [0, 10, 20, 30, 100], labels=["<=10", "11-20", "21-30", ">30"])
    metrics["segments_ndcg_at_5"] = {
        "random_display": s5.groupby("is_random")["ndcg"].mean().round(5).rename(
            {0: "non_random", 1: "random"}).to_dict(),
        "list_size": s5.groupby("size_bin", observed=True)["ndcg"].mean().round(5).astype(float)
                      .rename(str).to_dict(),
    }

    # Importance SHAP (contributions natives LightGBM, sur un échantillon)
    sample = X.sample(min(20000, len(X)), random_state=42)
    contrib = model.predict(sample, pred_contrib=True)[:, :-1]
    shap_imp = pd.Series(np.abs(contrib).mean(axis=0), index=sample.columns).sort_values(ascending=False)

    Path("reports").mkdir(exist_ok=True)
    json.dump(metrics, open("reports/test_metrics.json", "w"), indent=2)
    json.dump(shap_imp.round(5).to_dict(), open("reports/shap_importance.json", "w"), indent=2)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("expedia-ranking")
    with mlflow.start_run(run_name="evaluate_test"):
        mlflow.log_metrics({k: v for k, v in metrics.items() if k.startswith("test_ndcg")})
        mlflow.log_artifact("reports/test_metrics.json")
        mlflow.log_artifact("reports/shap_importance.json")

    print(json.dumps(metrics, indent=2))
    print("Top 10 SHAP :\n", shap_imp.head(10).round(4).to_string())


if __name__ == "__main__":
    main()