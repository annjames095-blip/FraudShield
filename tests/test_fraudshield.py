import numpy as np
import pandas as pd
import pytest

from fraudshield.costs import APPROVE, BLOCK, DEFER, CostModel
from fraudshield.data import add_features, make_synthetic, time_split, validate, FEATURES
from fraudshield.decision import decide, evaluate_decisions, tune_thresholds
from fraudshield.models import ece, fit_calibrator, fit_models, score


@pytest.fixture(scope="module")
def df():
    return make_synthetic(8000, 0.01, seed=1)


# ---------- validation gate ----------
def test_validate_accepts_clean_data(df):
    clean, bad = validate(df)
    assert len(clean) == len(df) and bad.empty


def test_validate_quarantines_bad_rows_with_reason(df):
    d = df.copy()
    d.loc[0, "Amount"] = -5
    d.loc[1, "V3"] = np.nan
    d.loc[2, "Amount"] = 1e9
    d.loc[3, "Class"] = 7
    clean, bad = validate(d)
    assert len(bad) == 4 and len(clean) == len(df) - 4
    assert bad["reason"].str.len().gt(0).all()
    assert "negative amount" in bad["reason"].iloc[0]


def test_validate_rejects_wrong_schema(df):
    with pytest.raises(ValueError):
        validate(df.drop(columns=["V5"]))


def test_validate_unlabelled_batch(df):
    clean, bad = validate(df.drop(columns=["Class"]), with_label=False)
    assert len(clean) == len(df)


# ---------- time split ----------
def test_time_split_is_ordered_and_disjoint(df):
    p = time_split(df)
    assert sum(len(v) for v in p.values()) == len(df)
    assert p["train"].Time.max() <= p["calib"].Time.min()
    assert p["calib"].Time.max() <= p["tune"].Time.min()
    assert p["tune"].Time.max() <= p["test"].Time.min()


# ---------- cost model (hand-worked) ----------
def test_cost_hand_worked_example():
    c = CostModel(false_block_cost=10, review_cost=2, reviewer_accuracy=0.9)
    y = np.array([1, 0, 1, 0, 1])
    amt = np.array([100.0, 50.0, 200.0, 30.0, 400.0])
    dec = np.array([APPROVE, BLOCK, BLOCK, APPROVE, DEFER])
    # missed fraud 100 + false block 10 + correct block 0 + correct approve 0 + defer(2 + 0.1*400)
    assert c.total(dec, y, amt) == pytest.approx(100 + 10 + 0 + 0 + 42)


# ---------- decision rule and thresholds ----------
def test_decide_partitions_cases():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    assert list(decide(p, 0.1, 0.8)) == [APPROVE, DEFER, DEFER, BLOCK]


def test_thresholds_respect_review_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    p = np.clip(0.02 + 0.6 * y * rng.random(5000) + 0.1 * rng.random(5000), 0, 1)
    amt = rng.lognormal(3, 1, 5000)
    for budget in (0.0, 0.01, 0.05):
        t = tune_thresholds(p, y, amt, CostModel(), budget, allow_defer=budget > 0)
        d = decide(p, t.t_low, t.t_high)
        assert (d == DEFER).mean() <= budget + 1e-9
        assert t.t_low <= t.t_high


def test_deferral_never_costs_more_than_no_deferral_on_tuning_set():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.03).astype(int)
    p = np.clip(0.03 + 0.5 * y * rng.random(4000) + 0.2 * rng.random(4000), 0, 1)
    amt = rng.lognormal(3, 1, 4000)
    c = CostModel()
    with_def = tune_thresholds(p, y, amt, c, 0.05, True).tune_cost
    without = tune_thresholds(p, y, amt, c, 0.05, False).tune_cost
    assert with_def <= without + 1e-9


def test_evaluate_decisions_silent_miss_rate():
    y = np.array([1, 1, 1, 0])
    dec = np.array([APPROVE, BLOCK, DEFER, APPROVE])
    m = evaluate_decisions(dec, y, np.ones(4), CostModel())
    assert m["silent_miss_rate"] == pytest.approx(1 / 3)
    assert m["defer_rate"] == pytest.approx(0.25)


# ---------- models and reproducibility ----------
def test_model_pipeline_runs_and_is_reproducible(df):
    parts = time_split(df)
    Xtr, ytr = add_features(parts["train"]), parts["train"].Class.to_numpy()
    Xte = add_features(parts["test"])
    assert list(Xtr.columns) == FEATURES
    a = score(fit_models(Xtr, ytr, seed=3)["LightGBM"], Xte)
    b = score(fit_models(Xtr, ytr, seed=3)["LightGBM"], Xte)
    assert np.allclose(a, b)
    assert ((a >= 0) & (a <= 1)).all()


def test_calibrator_outputs_valid_probabilities_and_is_monotone():
    rng = np.random.default_rng(2)
    s = rng.random(2000)
    y = (rng.random(2000) < s * 0.3).astype(int)
    cal = fit_calibrator(s, y)
    grid = np.linspace(0, 1, 50)
    out = cal.predict(grid)
    assert (out >= 0).all() and (out <= 1).all() and (np.diff(out) >= -1e-12).all()
    assert 0 <= ece(out, (rng.random(50) < out).astype(int)) <= 1
