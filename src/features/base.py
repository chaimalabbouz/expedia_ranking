# Création de la cible et suppression des colonnes de leakage
import pandas as pd

LEAKAGE_COLS = ["click_bool", "booking_bool", "gross_bookings_usd", "position", "random_bool"]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """relevance = 5 si réservé, 1 si cliqué seul, 0 sinon."""
    df = df.copy()
    df["relevance"] = 0
    df.loc[df["click_bool"] == 1, "relevance"] = 1
    df.loc[df["booking_bool"] == 1, "relevance"] = 5
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les colonnes interdites (fuite de données / biais d'affichage)."""
    return df.drop(columns=LEAKAGE_COLS)