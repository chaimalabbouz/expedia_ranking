# Expedia Hotel Ranking: Learning-to-Rank MLOps Pipeline

End-to-end MLOps project built on the [Personalize Expedia Hotel Searches (ICDM 2013)](https://www.kaggle.com/c/expedia-personalized-sort) dataset. Given the hotels shown in a search, the model ranks them to maximize the chance of a click or booking (metric: **NDCG**).

The focus is the **system around the model**: reproducible pipeline, model registry with a quality gate, containerized serving, CI, and monitoring.

## Results

| Metric (held-out test set) | Value |
|---|---|
| NDCG@5 (linear gain, ICDM definition) | **0.5** |
| Train-serving score gap (API vs offline, 20 searches) | < 1e-5 |
| API latency p95 (local Docker, simulated traffic) | ~80 ms |
| Data drift, train vs production (13 raw features) | 15 % of features |

NDCG@5 by segment: non-random display 0.429, random display 0.345; lists of at most 10 hotels 0.621, lists of more than 30 hotels 0.337.

## Architecture

```text
offline.parquet (Nov-Mar)
   -> validate (Pandera)
   -> split (chronological, grouped by search)
   -> train (LightGBM LambdaRank)
   -> evaluate (NDCG@5, segments, SHAP)
   -> promote (MLflow Registry, champion alias)

train + production.parquet (Apr-Jun)
   -> monitor (Evidently drift report)

trained model + FeatureBuilder
   -> FastAPI service (Docker)
   -> /metrics -> Prometheus -> Grafana
```

The first five stages are a DVC pipeline (`dvc.yaml`) driven by `params.yaml`. Changing one parameter only reruns the affected stages.

## Key design decisions

- **Chronological split, grouped by search.** Train (Nov to mid-Feb, 160K searches), validation and test (about 34K searches each). A search never spans two splits (checked by a unit test).
- **Leakage-safe features.** Relative price, stars and location within a search, temporal and missing-value features, and per-hotel click and booking rates computed **out-of-fold** on train and applied from train statistics on validation and test. Columns such as `position`, `click_bool` and `booking_bool` are never features.
- **Target.** Relevance 5 for a booking, 1 for a click, 0 otherwise. Searches with random display (no position bias) are supported through a configurable sample weight.
- **Evaluation.** The test set is used once, in `evaluate`. Searches with no click or booking are excluded from NDCG (their ideal DCG is 0).
- **Quality-gated releases.** `promote` registers every trained model in the MLflow Model Registry, and gives it the `champion` alias only if its test NDCG@5 is strictly better than the current champion.
- **Training-serving consistency.** The fitted `FeatureBuilder` is saved with the model and reused by the API. `scripts/check_skew.py` compares API scores to offline scores.

## Serving

FastAPI service (`src/api/main.py`):

- `POST /predict`: takes the hotels of one search (raw columns), returns them ranked with scores. Input validated with Pydantic.
- `GET /health`: model loaded check.
- `GET /metrics`: Prometheus metrics (request count by status, latency histogram, list size, score distribution).

## Monitoring

- **Data drift (Evidently).** Reference = train, current = held-out production set (April to June). Result: 2 of 13 features drifted (15 %), below the 50 % global alert threshold. A sensitivity check with an artificial price shift raised this to 3 of 13, so the detector reacts to real changes. Note: with default settings, a +30 % price shift was **not** detected (normalized Wasserstein distance below the 0.1 threshold), which is worth knowing when tuning thresholds.
- **Service health (Prometheus + Grafana).** `docker compose up` starts the API, Prometheus and Grafana. Traffic is **simulated** by replaying production searches (`scripts/send_traffic.py`, optional price shift to move the score distribution).

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) on every push:

1. `ruff` lint
2. `pytest` (split leakage, data schema, API contract)
3. Docker build, and push to GitHub Container Registry on `main`

## Tech stack

Python, LightGBM, pandas, scikit-learn, SHAP, DVC, MLflow, Pandera, FastAPI, Pydantic, Docker, Docker Compose, GitHub Actions, GHCR, pytest, ruff, Evidently, Prometheus, Grafana

## Project structure

```
├── dvc.yaml, params.yaml      # pipeline and hyperparameters
├── src/
│   ├── data/                  # validate, split
│   ├── features/              # feature engineering (fit/transform)
│   ├── models/                # train, evaluate, promote
│   ├── api/                   # FastAPI service
│   └── monitoring/            # drift report (Evidently)
├── scripts/                   # API checks, traffic simulation
├── tests/                     # pytest suite
├── monitoring/prometheus.yml
├── docker-compose.yml, Dockerfile
└── .github/workflows/ci.yml
```

## Getting started

The dataset is not included (size and Kaggle terms). Download it from Kaggle, then build `data/offline/offline.parquet` (November to March) and `data/production/production.parquet` (April to June).

```bash
pip install -r requirements-dev.txt
pip install dvc mlflow joblib shap

dvc repro                                  # validate -> split -> train -> evaluate -> promote
mlflow ui --backend-store-uri sqlite:///mlflow.db

docker compose up --build                  # API :8000, Prometheus :9090, Grafana :3000
python scripts/send_traffic.py             # simulated traffic
python scripts/send_traffic.py 1.5         # same, with a price shift
```

Run the tests with `pytest -v` and the linter with `ruff check .`.

