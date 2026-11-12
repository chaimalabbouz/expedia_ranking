import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

schema = DataFrameSchema(
    {
        "srch_id": Column(int, nullable=False),
        "date_time": Column("datetime64[us]", nullable=False),
        "prop_id": Column(int, nullable=False),
        "prop_starrating": Column(int, Check.in_range(0, 5)),
        "price_usd": Column(float, Check.ge(0), nullable=False),
        "click_bool": Column(int, Check.isin([0, 1])),
        "booking_bool": Column(int, Check.isin([0, 1])),
        "random_bool": Column(int, Check.isin([0, 1])),
    },
    checks=[
        Check(
            lambda df: (df["booking_bool"] <= df["click_bool"]).all(),
            error="booking_bool > click_bool",
        ),
    ],
    strict=False,  # les autres colonnes ne sont pas vérifiées pour l'instant
    coerce=False,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Lève une erreur si les données ne respectent pas le schéma."""
    return schema.validate(df, lazy=True)


if __name__ == "__main__":
    df = pd.read_parquet("data/offline/offline.parquet")
    validate(df)
    print("Validation OK :", df.shape)