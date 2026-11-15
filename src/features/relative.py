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

    # Taux historiques par hôtel (OOF sur le train), comparés aux autres hôtels de la recherche
    for col in ["prop_click_rate", "prop_book_rate"]:
        if col in df.columns:
            df[f"{col}_rank"] = g[col].rank(ascending=False, method="average")
            df[f"{col}_vs_mean"] = df[col] - g[col].transform("mean")
            df[f"{col}_vs_max"] = df[col] - g[col].transform("max")

    return df