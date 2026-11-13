import pandas as pd

COMP = range(1, 9)


def add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rate = df[[f"comp{i}_rate" for i in COMP]]
    inv = df[[f"comp{i}_inv" for i in COMP]]

    # Indicateurs de valeur manquante
    df["review_is_null"] = df["prop_review_score"].isna().astype(int)
    df["distance_is_null"] = df["orig_destination_distance"].isna().astype(int)
    df["hist_is_null"] = df["visitor_hist_starrating"].isna().astype(int)

    # Résumé des 8 concurrents (rate = 1 si Expedia est moins cher, -1 si plus cher)
    df["comp_rate_sum"] = rate.sum(axis=1)
    df["n_exp_cheaper"] = (rate == 1).sum(axis=1)
    df["n_exp_pricier"] = (rate == -1).sum(axis=1)
    df["n_comp_no_avail"] = (inv == 1).sum(axis=1)
    df["n_comp_known"] = rate.notna().sum(axis=1)
    return df