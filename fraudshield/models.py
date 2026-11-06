
"""
Machine learning models and probability calibration for FraudShield.
"""

from __future__ import annotations

import numpy as np

from lightgbm import LGBMClassifier

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class AnomalyScorer:
    """
    Isolation Forest anomaly detector.

    Higher scores indicate more unusual observations.
    These scores are NOT calibrated fraud probabilities.
    """

    def __init__(self, seed: int = 0):
        self.model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=seed,
            n_jobs=-1
        )

    def fit(self, X, y=None):
        self.model.fit(X)

        train_scores = -self.model.score_samples(X)

        self.minimum = float(train_scores.min())
        self.maximum = float(train_scores.max())

        return self

    def score(self, X):
        raw_scores = -self.model.score_samples(X)

        scaled = (
            (raw_scores - self.minimum)
            / (self.maximum - self.minimum + 1e-12)
        )

        return np.clip(scaled, 0, 1)


def _lightgbm(seed: int) -> LGBMClassifier:
    """Create an unweighted LightGBM classifier."""

    return LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=-1,
        n_jobs=4
    )


def fit_models(X, y, seed: int = 0) -> dict:
    """
    Train the supervised and unsupervised model candidates.

    All supervised models use the same training split.
    """

    y = np.asarray(y)

    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both classes.")

    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain fraud and legitimate transactions.")

    models = {
        "LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed
            )
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1
        ),

        "LightGBM": _lightgbm(seed),

        "LightGBM-weighted": LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=min(negative_count / positive_count, 200),
            random_state=seed,
            verbosity=-1,
            n_jobs=4
        )
    }

    for name, model in models.items():
        model.fit(X, y)

    models["IsolationForest"] = AnomalyScorer(seed).fit(X)

    return models


def score(model, X) -> np.ndarray:
    """
    Return a score between 0 and 1.

    For supervised classifiers this is the predicted
    probability of the positive class.

    For Isolation Forest this is a scaled anomaly score.
    """

    if isinstance(model, AnomalyScorer):
        return model.score(X)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must provide predict_proba or be an AnomalyScorer.")

    probabilities = model.predict_proba(X)

    classes = model.classes_

    positive_index = np.where(classes == 1)[0]

    if len(positive_index) != 1:
        raise ValueError("Model does not contain a positive class.")

    return probabilities[:, positive_index[0]]


class SigmoidCalibrator:
    """
    Platt-style probability calibration.

    Logistic regression is fitted to the log-odds of raw
    model scores using a separate calibration dataset.
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        )

    def fit(self, raw_scores, y):
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        y = np.asarray(y).reshape(-1)

        if len(raw_scores) != len(y):
            raise ValueError("Scores and labels must have equal lengths.")

        if len(np.unique(y)) != 2:
            raise ValueError("Calibration data must contain both classes.")

        clipped = np.clip(raw_scores, 1e-7, 1 - 1e-7)

        log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)

        self.model.fit(log_odds, y)

        return self

    def predict(self, raw_scores):
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)

        clipped = np.clip(raw_scores, 1e-7, 1 - 1e-7)

        log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)

        return self.model.predict_proba(log_odds)[:, 1]


def fit_calibrator(raw_scores, y, method="sigmoid"):
    """
    Fit a probability calibrator.

    Supported methods:
        sigmoid: Platt-style calibration (default).
        isotonic: Isotonic regression.
    """

    raw_scores = np.asarray(raw_scores, dtype=float)
    y = np.asarray(y)

    if method == "sigmoid":
        return SigmoidCalibrator().fit(raw_scores, y)

    if method == "isotonic":
        calibrator = IsotonicRegression(
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0
        )

        return calibrator.fit(raw_scores, y)

    raise ValueError("Calibration method must be 'sigmoid' or 'isotonic'.")


def ece(p, y, bins: int = 10) -> float:
    """
    Expected Calibration Error using equal-frequency bins.
    """

    p = np.asarray(p, dtype=float)
    y = np.asarray(y)

    if len(p) != len(y) or len(p) == 0:
        raise ValueError("Scores and labels must be non-empty and equal in length.")

    if bins < 1:
        raise ValueError("Number of bins must be positive.")

    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1.")

    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, min(bins, len(p)))

    error = 0.0

    for chunk in chunks:
        if len(chunk) == 0:
            continue

        predicted = p[chunk].mean()
        observed = y[chunk].mean()

        error += len(chunk) / len(p) * abs(predicted - observed)

    return float(error)
