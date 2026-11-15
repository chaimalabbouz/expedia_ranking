import pandas as pd

from src.data import split


def make_df(n_searches: int = 200, rows: int = 5) -> pd.DataFrame:
    base = pd.Timestamp("2013-01-01")
    recs = []
    for s in range(n_searches):
        start = base + pd.Timedelta(hours=s)
        for j in range(rows):
            recs.append({"srch_id": s, "date_time": start + pd.Timedelta(seconds=j), "prop_id": j})
    return pd.DataFrame(recs)


def test_split_no_leak_and_chronological(tmp_path, monkeypatch):
    in_path = tmp_path / "offline.parquet"
    out_dir = tmp_path / "splits"
    df = make_df()
    df.to_parquet(in_path, index=False)
    monkeypatch.setattr(split, "IN_PATH", in_path)
    monkeypatch.setattr(split, "OUT_DIR", out_dir)

    split.split_offline()

    tr, va, te = (pd.read_parquet(out_dir / f"{n}.parquet") for n in ("train", "val", "test"))

    # aucune ligne perdue
    assert len(tr) + len(va) + len(te) == len(df)
    # une recherche n'apparaît que dans un seul split
    s_tr, s_va, s_te = set(tr["srch_id"]), set(va["srch_id"]), set(te["srch_id"])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    # ordre chronologique
    assert tr["date_time"].max() < va["date_time"].min()
    assert va["date_time"].max() < te["date_time"].min()
    # les trois splits ne sont pas vides
    assert min(len(tr), len(va), len(te)) > 0