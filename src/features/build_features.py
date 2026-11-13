import pandas as pd
from src.features.base import add_target, clean
from src.features.temporal import add_time_features
from src.features.relative import add_relative_features


def build(df: pd.DataFrame, relative: bool = True) -> pd.DataFrame:
    df = add_time_features(clean(add_target(df)))
    if relative:
        df = add_relative_features(df)
    return df


if __name__ == "__main__":
    train = pd.read_parquet("data/splits/train.parquet")
    out = build(train)
    print("Avant :", train.shape, "| Après :", out.shape)
    new = ["log_price", "price_vs_median", "price_rank", "price_per_night",
           "loc2_rank", "loc1_rank", "loc2_vs_mean", "star_vs_mean", "review_vs_mean"]
    print(out[new].describe().loc[["count", "min", "max"]].T)