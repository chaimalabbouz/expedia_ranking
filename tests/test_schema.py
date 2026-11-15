import numpy as np
import pandas as pd
import pandera.errors as pae
import pytest

from src.data.validate import validate


def good_df() -> pd.DataFrame:
    return pd.DataFrame({
        "srch_id": np.array([1, 1, 2], dtype="int64"),
        "date_time": pd.to_datetime(["2013-01-01", "2013-01-01", "2013-01-02"]).astype("datetime64[us]"),
        "prop_id": np.array([10, 11, 10], dtype="int64"),
        "prop_starrating": np.array([3, 4, 0], dtype="int64"),
        "price_usd": np.array([100.0, 150.0, 80.0]),
        "click_bool": np.array([0, 1, 0], dtype="int64"),
        "booking_bool": np.array([0, 0, 0], dtype="int64"),
        "random_bool": np.array([0, 0, 1], dtype="int64"),
    })


def test_valid_data_passes():
    validate(good_df())


def test_negative_price_rejected():
    df = good_df()
    df.loc[0, "price_usd"] = -5.0
    with pytest.raises(pae.SchemaErrors):
        validate(df)


def test_booking_without_click_rejected():
    df = good_df()
    df.loc[0, "booking_bool"] = 1  # réservé mais pas cliqué : impossible
    with pytest.raises(pae.SchemaErrors):
        validate(df)