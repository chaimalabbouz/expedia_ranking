import numpy as np
import pandas as pd

GROUP = "srch_id"


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compare chaque hôtel aux autres hôtels de la même recherche."""
    df = df.copy()
    g = df.groupby(GROUP)

    # Prix
    df["log_price"] = np.log1p(df["price_usd"])
    df["price_vs_median"] = df["price_usd"] - g["price_usd"].transform("median")
    df["price_rank"] = g["price_usd"].rank(method="average")
    df["price_per_night"] = df["price_usd"] / df["srch_length_of_stay"].clip(lower=1)

    # Localisation (rang 1 = meilleur score)
    df["loc2_rank"] = g["prop_location_score2"].rank(ascending=False, method="average")
    df["loc1_rank"] = g["prop_location_score1"].rank(ascending=False, method="average")
    df["loc2_vs_mean"] = df["prop_location_score2"] - g["prop_location_score2"].transform("mean")

    # Étoiles et avis
    df["star_vs_mean"] = df["prop_starrating"] - g["prop_starrating"].transform("mean")
    df["review_vs_mean"] = df["prop_review_score"] - g["prop_review_score"].transform("mean")

    return df