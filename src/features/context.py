import numpy as np
import pandas as pd

GROUP = "srch_id"


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(GROUP)

    # Nombre de personnes et prix par personne
    guests = (df["srch_adults_count"] + df["srch_children_count"]).clip(lower=1)
    nights = df["srch_length_of_stay"].clip(lower=1)
    df["total_guests"] = guests
    df["price_per_person"] = df["price_usd"] / guests
    df["price_per_night_per_person"] = df["price_usd"] / (nights * guests)

    # Prix actuel vs prix historique (0 = pas d'historique, donc NaN)
    hist = df["prop_log_historical_price"].where(df["prop_log_historical_price"] > 0)
    df["price_vs_hist"] = np.log1p(df["price_usd"]) - hist

    # Taille de la recherche et position relative du prix (0 à 1)
    df["n_props"] = g[GROUP].transform("size")
    df["price_rank_norm"] = g["price_usd"].rank(pct=True)
    return df