import pandas as pd
from src.features.base import add_target, clean
from src.features.temporal import add_time_features


def build(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_features(clean(add_target(df)))


if __name__ == "__main__":
    train = pd.read_parquet("data/splits/train.parquet")
    out = build(train)
    print("Avant :", train.shape, "| Après :", out.shape)
    print(out[["month", "day_of_week", "hour"]].describe().loc[["min", "max"]])