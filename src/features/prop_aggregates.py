import pandas as pd
from sklearn.model_selection import GroupKFold

M = 10  # lissage : évite les taux extrêmes pour les hôtels rares
NEW_COLS = ["prop_n", "prop_click_rate", "prop_book_rate",
            "prop_price_mean", "price_vs_prop_mean"]


class PropAggregates:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    @staticmethod
    def _stats(df: pd.DataFrame) -> dict:
        t = df.groupby("prop_id").agg(
            n=("srch_id", "size"),
            clicks=("click_bool", "sum"),
            books=("booking_bool", "sum"),
            price_mean=("price_usd", "mean"),
        )
        return {"table": t,
                "prior_click": df["click_bool"].mean(),
                "prior_book": df["booking_bool"].mean()}

    @staticmethod
    def _apply(df: pd.DataFrame, s: dict) -> pd.DataFrame:
        t = s["table"].reindex(df["prop_id"]).set_axis(df.index)
        n = t["n"].fillna(0)
        out = pd.DataFrame(index=df.index)
        out["prop_n"] = n
        out["prop_click_rate"] = (t["clicks"].fillna(0) + s["prior_click"] * M) / (n + M)
        out["prop_book_rate"] = (t["books"].fillna(0) + s["prior_book"] * M) / (n + M)
        out["prop_price_mean"] = t["price_mean"]
        out["price_vs_prop_mean"] = df["price_usd"] - t["price_mean"]
        return out

    def fit(self, train: pd.DataFrame) -> "PropAggregates":
        self.stats_ = self._stats(train)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pour val et test : utilise les stats du train."""
        return self._apply(df, self.stats_)

    def oof_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """Pour le train : chaque ligne utilise des stats qui ne contiennent pas sa recherche."""
        parts = []
        for fit_idx, app_idx in GroupKFold(self.n_splits).split(train, groups=train["srch_id"]):
            s = self._stats(train.iloc[fit_idx])
            parts.append(self._apply(train.iloc[app_idx], s))
        return pd.concat(parts).loc[train.index]