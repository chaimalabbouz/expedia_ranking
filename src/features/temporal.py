#pour faire les colones de month , day et hour par ce que le model de ml ne peut pas comprendre la strucutre de date
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date_time"].dt.month
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["hour"] = df["date_time"].dt.hour
    return df