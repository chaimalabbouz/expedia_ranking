import pandas as pd


class DestAggregates:
    """Statistiques par destination, calculées sur le train (pas de cible, donc pas d'OOF)."""

    def fit(self, train: pd.DataFrame) -> "DestAggregates":
        self.table_ = train.groupby("srch_destination_id").agg(
            dest_n=("srch_id", "size"),
            dest_price_median=("price_usd", "median"),
            dest_star_mean=("prop_starrating", "mean"),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        t = self.table_.reindex(df["srch_destination_id"]).set_axis(df.index)
        out = pd.DataFrame(index=df.index)
        out["dest_n"] = t["dest_n"].fillna(0)
        out["dest_price_median"] = t["dest_price_median"]
        out["dest_star_mean"] = t["dest_star_mean"]
        out["price_vs_dest"] = df["price_usd"] - t["dest_price_median"]
        out["star_vs_dest"] = df["prop_starrating"] - t["dest_star_mean"]
        return out