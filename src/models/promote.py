import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

NAME = "expedia-ranker"
ALIAS = "champion"
METRIC = "test_ndcg_at_5"


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("expedia-ranking")
    client = MlflowClient()

    new_score = json.load(open("reports/test_metrics.json"))[METRIC]
    model = lgb.Booster(model_file="models/model.txt")

    # 1. Enregistrer une nouvelle version du modèle
    with mlflow.start_run(run_name="register_model"):
        mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=NAME)
        mlflow.log_artifact("models/feature_builder.pkl", artifact_path="model")
        mlflow.log_metric(METRIC, new_score)

    version = max(int(v.version) for v in client.search_model_versions(f"name='{NAME}'"))
    client.set_model_version_tag(NAME, str(version), METRIC, str(new_score))

    # 2. Comparer au champion actuel
    try:
        champ = client.get_model_version_by_alias(NAME, ALIAS)
        champ_score = float(champ.tags.get(METRIC, -1))
    except MlflowException:
        champ, champ_score = None, -1.0

    # 3. Promouvoir seulement si strictement meilleur
    promoted = new_score > champ_score
    if promoted:
        client.set_registered_model_alias(NAME, ALIAS, str(version))

    result = {
        "new_version": version,
        "new_score": new_score,
        "previous_champion_score": None if champ is None else champ_score,
        "promoted": promoted,
    }
    Path("reports").mkdir(exist_ok=True)
    json.dump(result, open("reports/promotion.json", "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()