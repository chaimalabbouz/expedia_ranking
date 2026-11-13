import pandas as pd
from src.features.base import add_target, clean
from src.features.temporal import add_time_features
from src.features.relative import add_relative_features
from src.features.missing import add_missing_features
from src.features.context import add_context_features
from src.features.prop_aggregates import PropAggregates
from src.features.dest_aggregates import DestAggregates


class FeatureBuilder:
    def __init__(self, relative: bool = True, prop: bool = True,
                 dest: bool = False, context: bool = False):
        self.relative = relative
        self.prop = prop
        self.dest = dest
        self.context = context

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self.prop_ = PropAggregates().fit(train) if self.prop else None
        self.dest_ = DestAggregates().fit(train) if self.dest else None
        return self

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """Pour le train uniquement (OOF pour les stats par hôtel)."""
        self.fit(train)
        extra = self.prop_.oof_transform(train) if self.prop else None
        return self._build(train, extra)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pour val et test."""
        extra = self.prop_.transform(df) if self.prop else None
        return self._build(df, extra)

    def _build(self, df: pd.DataFrame, extra) -> pd.DataFrame:
        if extra is not None:
            df = df.join(extra)  # avant clean(), car il faut click_bool/booking_bool
        if self.dest:
            df = df.join(self.dest_.transform(df))
        df = add_time_features(clean(add_target(df)))
        df = add_missing_features(df)
        if self.context:
            df = add_context_features(df)
        if self.relative:
            df = add_relative_features(df)
        return df