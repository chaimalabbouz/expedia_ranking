from pathlib import Path
import pandas as pd

IN_PATH = Path("data/offline/offline.parquet")
OUT_DIR = Path("data/splits")
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


def split_offline() -> None:
    df = pd.read_parquet(IN_PATH)

    # Date de chaque recherche = première date de ses lignes
    search_date = df.groupby("srch_id")["date_time"].transform("min")
    s = df.groupby("srch_id")["date_time"].min()

    d_train = s.quantile(TRAIN_RATIO)
    d_val = s.quantile(TRAIN_RATIO + VAL_RATIO)

    train = df[search_date < d_train]
    val = df[(search_date >= d_train) & (search_date < d_val)]
    test = df[search_date >= d_val]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUT_DIR / "val.parquet", index=False)
    test.to_parquet(OUT_DIR / "test.parquet", index=False)

    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{name:5}: {len(part)} lignes | {part['srch_id'].nunique()} recherches | "
              f"{part['date_time'].min()} -> {part['date_time'].max()}")

    total = len(train) + len(val) + len(test)
    print("Total lignes :", total, "| offline :", len(df), "| OK :", total == len(df))
    ids = [set(p["srch_id"]) for p in (train, val, test)]
    print("Recherches communes :", len(ids[0] & ids[1]) + len(ids[0] & ids[2]) + len(ids[1] & ids[2]))


if __name__ == "__main__":
    split_offline()