"""Data loading, the validation gate, feature construction and time-ordered splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
REQUIRED = ["Time", "Amount"] + V_COLS
FEATURES = V_COLS + ["log_amount", "hour"]
MAX_AMOUNT = 1e7  # anything above is treated as an implausible value


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate(df: pd.DataFrame, with_label: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation gate. Returns (clean, quarantined).

    Bad rows are never scored; they are kept with a human-readable reason.
    Raises ValueError if the schema itself is wrong (missing columns).
    """
    missing = [c for c in REQUIRED + (["Class"] if with_label else []) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask, text):
        nonlocal reason
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        reason = reason.where(~mask, reason + (text + "; "))

    num = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    flag(num.isna().any(axis=1), "missing or non-numeric feature")
    flag(~np.isfinite(num.fillna(0)).all(axis=1), "non-finite value")
    flag(num["Amount"] < 0, "negative amount")
    flag(num["Amount"] > MAX_AMOUNT, "implausibly large amount")
    flag(num["Time"] < 0, "negative time")
    if with_label:
        flag(~df["Class"].isin([0, 1]), "label not in {0,1}")

    bad = reason != ""
    clean = df.loc[~bad].copy()
    clean[REQUIRED] = num.loc[~bad]
    quarantined = df.loc[bad].copy()
    quarantined["reason"] = reason[bad].str.rstrip("; ")
    return clean.reset_index(drop=True), quarantined.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model input matrix. Raw Amount stays out of the model (log-scaled instead) but remains
    in the frame because the cost model needs it."""
    X = df[V_COLS].copy()
    X["log_amount"] = np.log1p(df["Amount"].astype(float))
    X["hour"] = (df["Time"].astype(float) / 3600.0) % 24
    return X[FEATURES]


def time_split(df: pd.DataFrame, fracs=(0.6, 0.1, 0.1, 0.2)) -> dict:
    """Contiguous, time-ordered split: train / calib / tune / test.

    calib fits the probability calibrator, tune picks the decision thresholds,
    test is used once for reporting. No random shuffling (avoids future leakage).
    """
    assert abs(sum(fracs) - 1) < 1e-9
    d = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(d)
    cuts = np.cumsum([int(round(f * n)) for f in fracs[:-1]])
    parts = np.split(np.arange(n), cuts)
    return {k: d.iloc[ix].reset_index(drop=True) for k, ix in zip(["train", "calib", "tune", "test"], parts)}


def make_synthetic(n: int = 60000, fraud_rate: float = 0.004, seed: int = 0) -> pd.DataFrame:
    """Synthetic stand-in with the same schema as the ULB data. For testing only."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=12, n_redundant=4,
        weights=[1 - fraud_rate, fraud_rate], class_sep=1.2, flip_y=0.002,
        n_clusters_per_class=2, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=V_COLS)
    df["Time"] = np.sort(rng.uniform(0, 172800, n))
    df["Amount"] = np.round(rng.lognormal(3.2, 1.4, n) * np.where(y == 1, 1.8, 1.0), 2)
    df["Class"] = y
    return df
