import pandas as pd
from src.features.base import add_target, clean
from src.features.temporal import add_time_features
from src.features.relative import add_relative_features


class FeatureBuilder:
    def __init__(self, relative: bool = True):
        self.relative = relative

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        # Étape 1 : rien à apprendre pour l'instant.
        # Plus tard : statistiques par hôtel calculées ici, sur le train seulement.
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_time_features(clean(add_target(df)))
        if self.relative:
            df = add_relative_features(df)
        return df


if __name__ == "__main__":
    train = pd.read_parquet("data/splits/train.parquet")
    out = FeatureBuilder().fit(train).transform(train)
    print("Avant :", train.shape, "| Après :", out.shape)