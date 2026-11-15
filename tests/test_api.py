import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main


class FakeFB:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"f1": df["price_usd"].to_numpy()}, index=df.index)


class FakeModel:
    def feature_name(self):
        return ["f1"]

    def predict(self, X):
        return -X["f1"].to_numpy()  # moins cher = meilleur score


@pytest.fixture
def client():
    # TestClient sans "with" : le lifespan (chargement des vrais fichiers) n'est pas exécuté
    main.state["model"] = FakeModel()
    main.state["fb"] = FakeFB()
    yield TestClient(main.app)
    main.state.clear()


def hotel(prop_id: int, price: float) -> dict:
    return {"prop_id": prop_id, "price_usd": price, "date_time": "2013-01-01T10:00:00"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_ranks_by_score(client):
    r = client.post("/predict", json={"hotels": [hotel(1, 200.0), hotel(2, 50.0), hotel(3, 100.0)]})
    assert r.status_code == 200
    ranking = r.json()["ranking"]
    assert [x["prop_id"] for x in ranking] == [2, 3, 1]
    assert [x["rank"] for x in ranking] == [1, 2, 3]


def test_predict_empty_list_rejected(client):
    r = client.post("/predict", json={"hotels": []})
    assert r.status_code == 422


def test_predict_missing_field_rejected(client):
    r = client.post("/predict", json={"hotels": [{"prop_id": 1}]})
    assert r.status_code == 422