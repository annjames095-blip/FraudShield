
"""
FraudShield experimental pipeline.

Run:
    python run_experiments.py --data data/creditcard.csv --seeds 5

The pipeline:
1. Validates the dataset.
2. Creates chronological train/calibration/tuning/test splits.
3. Trains candidate models.
4. Selects a model using tuning-set PR-AUC.
5. Calibrates the selected model.
6. Tunes cost-sensitive decisions.
7. Evaluates on the untouched test split.
8. Performs a human-review budget sweep.
9. Saves results, figures and model artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score
)

from fraudshield.costs import CostModel
from fraudshield.data import (
    FEATURES,
    add_features,
    load_csv,
    make_synthetic,
    time_split,
    validate
)
from fraudshield.decision import (
    decide,
    expected_cost_decisions,
    expected_decision_costs,
    evaluate_decisions,
    tune_thresholds
)
from fraudshield.models import (
    ece,
    fit_calibrator,
    fit_models,
    score
)


REVIEW_BUDGET = 0.02
N_BOOT = 500

CANDIDATES = [
    "LogReg",
    "RandomForest",
    "LightGBM"
]

REVIEW_BUDGETS = [
    0.0,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.05
]


def single_threshold_decisions(probabilities, threshold):
    """Make approve/block decisions without human review."""

    probabilities = np.asarray(probabilities, dtype=float)

    return np.where(probabilities >= threshold, 1, 0)


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=0):
    """Calculate a bootstrap confidence interval for a mean."""

    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    estimates = []

    for _ in range(n_boot):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        estimates.append(sample.mean())

    return [
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5))
    ]


def bootstrap_total_ci(values, n_boot=N_BOOT, seed=0):
    """Calculate a bootstrap confidence interval for a total."""

    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    estimates = []

    for _ in range(n_boot):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        estimates.append(sample.sum())

    return [
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5))
    ]


def safe_roc_auc(y, probabilities):
    """Return ROC-AUC when both classes are present."""

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        roc_auc_score(y, probabilities)
    )


def run_seed(seed, parts, cost):
    """Train and evaluate one experiment seed."""

    X = {
        name: add_features(frame)
        for name, frame in parts.items()
    }

    y = {
        name: frame["Class"].to_numpy()
        for name, frame in parts.items()
    }

    amount = {
        name: frame["Amount"].to_numpy()
        for name, frame in parts.items()
    }

    # Train candidate models.
    models = fit_models(
        X["train"],
        y["train"],
        seed
    )

    # Generate predictions for each split.
    predictions = {
        name: {
            split: score(model, X[split])
            for split in ("calib", "tune", "test")
        }
        for name, model in models.items()
    }

    # Select the supervised model using tuning-set PR-AUC.
    selection_scores = {
        name: float(
            average_precision_score(
                y["tune"],
                predictions[name]["tune"]
            )
        )
        for name in CANDIDATES
    }

    selected_name = max(
        selection_scores,
        key=selection_scores.get
    )

    selected_model = models[selected_name]

    # Fit calibration on the separate calibration split.
    calibrator = fit_calibrator(
        predictions[selected_name]["calib"],
        y["calib"],
        method="sigmoid"
    )

    calibrated = {
        split: calibrator.predict(
            predictions[selected_name][split]
        )
        for split in ("calib", "tune", "test")
    }

    rows = []

    def add_result(name, probabilities, decisions, extra=None):

        result = {
            "method": name,
            "seed": seed,
            "pr_auc": float(
                average_precision_score(
                    y["test"],
                    probabilities
                )
            ),
            "roc_auc": safe_roc_auc(
                y["test"],
                probabilities
            )
        }

        result.update(
            evaluate_decisions(
                decisions,
                y["test"],
                amount["test"],
                cost
            )
        )

        result.update(extra or {})

        rows.append(result)

    # Evaluate candidate models using cost-tuned thresholds.
    for name in CANDIDATES + [
        "LightGBM-weighted",
        "IsolationForest"
    ]:

        thresholds = tune_thresholds(
            predictions[name]["tune"],
            y["tune"],
            amount["tune"],
            cost,
            allow_defer=False
        )

        decisions = single_threshold_decisions(
            predictions[name]["test"],
            thresholds.t_low
        )

        add_result(
            f"{name} @cost-tuned",
            predictions[name]["test"],
            decisions,
            {
                "threshold": thresholds.t_low
            }
        )

    # Evaluate calibrated model without human review.
    calibrated_no_review = tune_thresholds(
        calibrated["tune"],
        y["tune"],
        amount["tune"],
        cost,
        allow_defer=False
    )

    decisions_no_review = single_threshold_decisions(
        calibrated["test"],
        calibrated_no_review.t_low
    )

    add_result(
        "Calibrated model (no deferral)",
        calibrated["test"],
        decisions_no_review,
        {
            "selected_model": selected_name,
            "threshold": calibrated_no_review.t_low
        }
    )

    # Calculate the no-review baseline cost for paired comparison.
    no_review_case_costs = cost.case_costs(
        decisions_no_review,
        y["test"],
        amount["test"]
    )

    # Evaluate FraudShield with human review.
    thresholds = tune_thresholds(
        calibrated["tune"],
        y["tune"],
        amount["tune"],
        cost,
        review_budget=REVIEW_BUDGET,
        allow_defer=True
    )

    decisions = decide(
        calibrated["test"],
        thresholds.t_low,
        thresholds.t_high
    )

    add_result(
        "FraudShield (full)",
        calibrated["test"],
        decisions,
        {
            "selected_model": selected_name,
            "t_low": thresholds.t_low,
            "t_high": thresholds.t_high,
            "tune_defer_rate": thresholds.defer_rate
        }
    )
        # Expected-cost-based FraudShield policy.
    expected_decisions = expected_cost_decisions(
        calibrated["test"],
        amount["test"],
        cost
    )

    expected_case_costs = cost.case_costs(
        expected_decisions,
        y["test"],
        amount["test"]
    )

    expected_cost_savings = (
        no_review_case_costs - expected_case_costs
    )

    baseline_total_cost = float(no_review_case_costs.sum())

    expected_savings_total = float(expected_cost_savings.sum())

    expected_savings_percent = (
        100.0 * expected_savings_total / baseline_total_cost
        if baseline_total_cost > 0
        else 0.0
    )

    add_result(
        "FraudShield (expected-cost)",
        calibrated["test"],
        expected_decisions,
        {
            "selected_model": selected_name,
            "policy": "expected-cost",
            "cost_savings_vs_no_review": expected_savings_total,
            "cost_savings_pct_vs_no_review": expected_savings_percent
        }
    )
    # Calculate calibration metrics on the test set.
    calibration_results = []

    for name in CANDIDATES:

        probabilities = predictions[name]["test"]

        calibration_results.append({
            "model": name,
            "seed": seed,
            "brier": float(
                brier_score_loss(
                    y["test"],
                    probabilities
                )
            ),
            "ece": ece(
                probabilities,
                y["test"]
            )
        })

    calibration_results.append({
        "model": f"{selected_name} + sigmoid",
        "seed": seed,
        "brier": float(
            brier_score_loss(
                y["test"],
                calibrated["test"]
            )
        ),
        "ece": ece(
            calibrated["test"],
            y["test"]
        )
    })

    return {
        "rows": rows,
        "calibration": calibration_results,
        "models": models,
        "selected_name": selected_name,
        "selected_model": selected_model,
        "calibrator": calibrator,
        "thresholds": thresholds,
        "calibrated": calibrated,
        "predictions": predictions,
        "labels": y,
        "amounts": amount,
        "features": X,
        "parts": parts,
        "decisions": decisions,
        "expected_decisions": expected_decisions
    }
    


def run_deferral_sweep(run, cost):
    """
    Tune review policies on the tuning split.

    Evaluate every policy on the untouched test split.
    """

    sweep_results = []

    for budget in REVIEW_BUDGETS:

        thresholds = tune_thresholds(
            run["calibrated"]["tune"],
            run["labels"]["tune"],
            run["amounts"]["tune"],
            cost,
            review_budget=budget,
            allow_defer=budget > 0
        )

        if budget == 0:

            decisions = single_threshold_decisions(
                run["calibrated"]["test"],
                thresholds.t_low
            )

        else:

            decisions = decide(
                run["calibrated"]["test"],
                thresholds.t_low,
                thresholds.t_high
            )

        metrics = evaluate_decisions(
            decisions,
            run["labels"]["test"],
            run["amounts"]["test"],
            cost
        )

        row = {
            "review_budget": budget,
            "t_low": thresholds.t_low,
            "t_high": thresholds.t_high,
            "tuning_defer_rate": thresholds.defer_rate,
            **metrics
        }

        sweep_results.append(row)

    return pd.DataFrame(sweep_results)


def calculate_summary(results):
    """
    Create mean and standard deviation tables.

    Only numeric columns are included.
    Text columns such as method, selected_model and policy
    are automatically excluded.
    """

    # Select only numeric columns.
    metrics = (
        results
        .select_dtypes(include="number")
        .columns
        .drop("seed", errors="ignore")
    )

    # Group results by experiment method.
    summary = results.groupby(
        "method",
        sort=False
    )[metrics].agg(["mean", "std"])

    # Flatten the multi-level column names.
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    return summary

def create_figures(summary, run, cost):
    """Create the main evaluation plots."""

    os.makedirs(
        "results/figures",
        exist_ok=True
    )

    y_test = run["labels"]["test"]

    # --------------------------------------------------
    # 1. Precision-recall curves
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    for name in CANDIDATES:

        probabilities = run["predictions"][name]["test"]

        precision, recall, _ = precision_recall_curve(
            y_test,
            probabilities
        )

        ap = average_precision_score(
            y_test,
            probabilities
        )

        ax.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.3f})"
        )

    probabilities = run["calibrated"]["test"]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    ap = average_precision_score(
        y_test,
        probabilities
    )

    ax.plot(
        recall,
        precision,
        label=f"FraudShield score (AP={ap:.3f})"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        "results/figures/pr_curves.png",
        dpi=200
    )

    plt.close(fig)

    # --------------------------------------------------
    # 2. Cost comparison
    # --------------------------------------------------

    cost_column = "total_cost_mean"

    if cost_column in summary.columns:

        fig, ax = plt.subplots(figsize=(8, 5))

        selected = summary.sort_values(
            cost_column
        )

        ax.barh(
            selected.index,
            selected[cost_column]
        )

        ax.set_xlabel("Total expected cost")
        ax.set_title("Cost Comparison Across Methods")

        fig.tight_layout()

        fig.savefig(
            "results/figures/cost_bars.png",
            dpi=200
        )

        plt.close(fig)

    # --------------------------------------------------
    # 3. Calibration plot
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(5, 5))

    for name in CANDIDATES + ["selected"]:

        if name == "selected":

            probabilities = run["calibrated"]["test"]
            label = "Selected model + sigmoid"

        else:

            probabilities = run["predictions"][name]["test"]
            label = name

        order = np.argsort(probabilities)

        chunks = np.array_split(
            order,
            10
        )

        mean_predicted = [
            probabilities[c].mean()
            for c in chunks
            if len(c)
        ]

        observed = [
            y_test[c].mean()
            for c in chunks
            if len(c)
        ]

        ax.plot(
            mean_predicted,
            observed,
            marker="o",
            label=label
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--"
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Probability Calibration")
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        "results/figures/calibration.png",
        dpi=200
    )

    plt.close(fig)

    # --------------------------------------------------
    # 4. Review workload vs missed fraud
    # --------------------------------------------------

    sweep = run_deferral_sweep(
        run,
        cost
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        sweep["defer_rate"] * 100,
        sweep["silent_miss_rate"] * 100,
        marker="o"
    )

    ax.set_xlabel("Transactions sent for review (%)")
    ax.set_ylabel("Silent fraud miss rate (%)")
    ax.set_title("Review Workload vs Missed Fraud")

    fig.tight_layout()

    fig.savefig(
        "results/figures/deferral_tradeoff.png",
        dpi=200
    )

    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/creditcard.csv"
    )

    parser.add_argument(
        "--synthetic",
        action="store_true"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if args.seeds < 1:
        raise ValueError(
            "At least one experiment seed is required."
        )

    start = time.time()

    os.makedirs(
        "results/figures",
        exist_ok=True
    )

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    # --------------------------------------------------
    # Load and validate dataset
    # --------------------------------------------------

    if args.synthetic:

        df = make_synthetic(
            n=60000,
            fraud_rate=0.004,
            seed=0
        )

        print(
            "WARNING: Synthetic data. "
            "Do not report these results."
        )

    else:

        df = load_csv(args.data)

    clean, quarantined = validate(df)

    print("Dataset rows:", len(df))
    print("Valid rows:", len(clean))
    print("Quarantined rows:", len(quarantined))

    duplicate_count = int(
        clean.duplicated().sum()
    )

    print(
        "Exact duplicate rows:",
        duplicate_count
    )

    # --------------------------------------------------
    # Chronological split
    # --------------------------------------------------

    parts = time_split(clean)

    split_info = {
        name: {
            "rows": int(len(frame)),
            "frauds": int(frame["Class"].sum())
        }
        for name, frame in parts.items()
    }

    print(
        "Chronological split:",
        split_info
    )

    cost = CostModel()

    # --------------------------------------------------
    # Run experiments
    # --------------------------------------------------

    all_rows = []
    all_calibration = []
    runs = []

    for seed in range(args.seeds):

        print(
            f"\nTraining experiment seed {seed}..."
        )

        run = run_seed(
            seed,
            parts,
            cost
        )

        all_rows.extend(
            run["rows"]
        )

        all_calibration.extend(
            run["calibration"]
        )

        runs.append(run)

        print(
            "Selected model:",
            run["selected_name"]
        )

    # --------------------------------------------------
    # Save model evaluation results
    # --------------------------------------------------

    results = pd.DataFrame(all_rows)

    results.to_csv(
        "results/per_seed.csv",
        index=False
    )

    summary = calculate_summary(results)

    summary.to_csv(
        "results/summary.csv"
    )

    calibration_df = pd.DataFrame(
        all_calibration
    )

    calibration_summary = calibration_df.groupby(
        "model"
    )[["brier", "ece"]].agg(["mean", "std"])

    calibration_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in calibration_summary.columns
    ]

    calibration_summary.to_csv(
        "results/calibration.csv"
    )

    # --------------------------------------------------
    # Seed-zero detailed evaluation
    # --------------------------------------------------

    first = runs[0]

    y_test = first["labels"]["test"]
    amount_test = first["amounts"]["test"]

    # Cost of FraudShield with human review.
    fraudshield_costs = cost.case_costs(
        first["decisions"],
        y_test,
        amount_test
    )

    # Cost of calibrated model without human review.
    no_review_thresholds = tune_thresholds(
        first["calibrated"]["tune"],
        first["labels"]["tune"],
        first["amounts"]["tune"],
        cost,
        allow_defer=False
    )

    no_review_decisions = single_threshold_decisions(
        first["calibrated"]["test"],
        no_review_thresholds.t_low
    )

    no_review_costs = cost.case_costs(
        no_review_decisions,
        y_test,
        amount_test
    )

    # Expected-cost policy evaluation for seed 0.
    expected_decisions = first["expected_decisions"]

    expected_metrics = evaluate_decisions(
        expected_decisions,
        y_test,
        amount_test,
        cost
    )

    expected_costs = cost.case_costs(
        expected_decisions,
        y_test,
        amount_test
    )

    expected_policy_savings = (
        no_review_costs - expected_costs
    )

    # Paired savings: positive means FraudShield is cheaper.
    paired_savings = (
        no_review_costs - fraudshield_costs
    )

    savings_ci = bootstrap_total_ci(
        paired_savings,
        seed=0
    )

    mean_savings_ci = bootstrap_mean_ci(
        paired_savings,
        seed=0
    )

    # --------------------------------------------------
    # PR-AUC confidence interval
    # --------------------------------------------------

    ap_values = []

    rng = np.random.default_rng(0)

    for _ in range(200):

        indices = rng.integers(
            0,
            len(y_test),
            len(y_test)
        )

        if len(np.unique(y_test[indices])) < 2:
            continue

        ap_values.append(
            average_precision_score(
                y_test[indices],
                first["calibrated"]["test"][indices]
            )
        )

    ap_ci = [
        float(np.percentile(ap_values, 2.5)),
        float(np.percentile(ap_values, 97.5))
    ]

    # --------------------------------------------------
    # Human-review budget sweep
    # --------------------------------------------------

    sweep_df = run_deferral_sweep(
        first,
        cost
    )

    sweep_df.to_csv(
        "results/deferral_sweep.csv",
        index=False
    )

    print("\nDEFERRAL BUDGET SWEEP")

    print(
        sweep_df[
            [
                "review_budget",
                "defer_rate",
                "n_deferred",
                "fraud_deferred_rate",
                "silent_miss_rate",
                "total_cost"
            ]
        ].to_string(index=False)
    )

    baseline_cost = float(
        no_review_costs.sum()
    )

    # Aggregate expected-cost policy results across every experiment seed.
    expected_policy_rows = results[
        results["method"] == "FraudShield (expected-cost)"
    ].copy()

    expected_policy_seed_summary = (
        expected_policy_rows[
            [
                "seed",
                "total_cost",
                "cost_savings_vs_no_review",
                "cost_savings_pct_vs_no_review",
                "defer_rate",
                "silent_miss_rate",
                "fraud_blocked_rate",
                "fraud_deferred_rate"
            ]
        ]
        .sort_values("seed")
    )

    expected_policy_seed_summary.to_csv(
        "results/expected_cost_per_seed.csv",
        index=False
    )

    expected_policy_aggregate = (
        expected_policy_rows[
            [
                "total_cost",
                "cost_savings_vs_no_review",
                "cost_savings_pct_vs_no_review",
                "defer_rate",
                "silent_miss_rate",
                "fraud_blocked_rate",
                "fraud_deferred_rate"
            ]
        ]
        .agg(["mean", "std"])
    )

    print("\\nEXPECTED-COST POLICY ACROSS SEEDS")
    print(expected_policy_seed_summary.round(4).to_string(index=False))
    print("\\nEXPECTED-COST POLICY MEAN AND STANDARD DEVIATION")
    print(expected_policy_aggregate.round(4).to_string())

    # --------------------------------------------------
    # Save model artifact for dashboard
    # --------------------------------------------------

    model_artifact = {
        "model": first["selected_model"],
        "calibrator": first["calibrator"],
        "selected_model_name": first["selected_name"],
        "t_low": first["thresholds"].t_low,
        "t_high": first["thresholds"].t_high,
        "features": FEATURES,
        "cost": cost.__dict__,
        "review_budget": REVIEW_BUDGET
    }

    joblib.dump(
        model_artifact,
        "artifacts/fraudshield.joblib"
    )

    # --------------------------------------------------
    # Save test sample for dashboard demo
    # --------------------------------------------------

    test = parts["test"]

    fraud_rows = test[
        test["Class"] == 1
    ]

    legitimate_rows = test[
        test["Class"] == 0
    ].sample(
        min(
            3000,
            int((test["Class"] == 0).sum())
        ),
        random_state=0
    )

    test_sample = pd.concat(
        [fraud_rows, legitimate_rows]
    ).sort_values("Time")

    test_sample.to_csv(
        "artifacts/test_sample.csv",
        index=False
    )

    # --------------------------------------------------
    # Save evaluation arrays
    # --------------------------------------------------

    np.savez_compressed(
        "artifacts/eval.npz",
        p_tune=first["calibrated"]["tune"],
        y_tune=first["labels"]["tune"],
        a_tune=first["amounts"]["tune"],
        p_test=first["calibrated"]["test"],
        y_test=y_test,
        a_test=amount_test
    )

    # --------------------------------------------------
    # SHAP feature importance
    # --------------------------------------------------

    shap_info = {}

    if first["selected_name"] in (
        "RandomForest",
        "LightGBM"
    ):

        try:

            import shap

            X_sample = first["features"]["test"].sample(
                min(
                    2000,
                    len(first["features"]["test"])
                ),
                random_state=0
            )

            explainer = shap.TreeExplainer(
                first["selected_model"]
            )

            shap_values = explainer.shap_values(
                X_sample
            )

            if isinstance(shap_values, list):

                shap_values = shap_values[1]

            elif np.asarray(shap_values).ndim == 3:

                shap_values = np.asarray(
                    shap_values
                )[:, :, 1]

            importance = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=X_sample.columns
            ).sort_values(
                ascending=False
            )

            importance.to_csv(
                "results/shap_feature_importance.csv"
            )

            shap_info = {
                "top10": list(
                    importance.head(10).index
                )
            }

        except Exception as error:

            print(
                "SHAP analysis skipped:",
                error
            )

    # --------------------------------------------------
    # Create figures
    # --------------------------------------------------

    create_figures(
        summary,
        first,
        cost
    )

    # --------------------------------------------------
    # Save headline results
    # --------------------------------------------------

    headline = {
        "synthetic": bool(args.synthetic),
        "seeds": args.seeds,

        "selected_model_seed0": first["selected_name"],

        "selection_method": "Highest validation PR-AUC",

        "split": split_info,

        "rows_clean": int(len(clean)),
        "rows_quarantined": int(len(quarantined)),
        "exact_duplicate_rows": duplicate_count,

        "review_budget": REVIEW_BUDGET,

        "cost_model": cost.__dict__,

        "thresholds_seed0": {
            "t_low": first["thresholds"].t_low,
            "t_high": first["thresholds"].t_high,
            "tuning_defer_rate": first["thresholds"].defer_rate
        },

        "baseline_method": "Calibrated model (no deferral)",

        "baseline_cost_seed0": baseline_cost,

        "fraudshield_cost_seed0": float(
            fraudshield_costs.sum()
        ),

        "cost_saving_seed0": float(
            paired_savings.sum()
        ),

        "cost_saving_ci95_seed0": savings_ci,

        "mean_cost_saving_per_transaction_ci95_seed0": mean_savings_ci,

        "fraudshield_pr_auc_ci95_seed0": ap_ci,
        
        "expected_cost_policy_seed0": expected_metrics,

        "expected_cost_policy_total_seed0": float(
            expected_costs.sum()
        ),

        "expected_cost_policy_savings_seed0": float(
            expected_policy_savings.sum()
        ),

        "expected_cost_policy_across_seeds": {
            "total_cost_mean": float(expected_policy_aggregate.loc["mean", "total_cost"]),
            "total_cost_std": float(expected_policy_aggregate.loc["std", "total_cost"]),
            "cost_savings_mean": float(expected_policy_aggregate.loc["mean", "cost_savings_vs_no_review"]),
            "cost_savings_std": float(expected_policy_aggregate.loc["std", "cost_savings_vs_no_review"]),
            "cost_savings_percent_mean": float(expected_policy_aggregate.loc["mean", "cost_savings_pct_vs_no_review"]),
            "defer_rate_mean": float(expected_policy_aggregate.loc["mean", "defer_rate"]),
            "silent_miss_rate_mean": float(expected_policy_aggregate.loc["mean", "silent_miss_rate"]),
            "fraud_blocked_rate_mean": float(expected_policy_aggregate.loc["mean", "fraud_blocked_rate"]),
            "fraud_deferred_rate_mean": float(expected_policy_aggregate.loc["mean", "fraud_deferred_rate"])
        },

        "shap": shap_info,

        "runtime_s": round(
            time.time() - start,
            1
        )
    }

    with open(
        "results/headline.json",
        "w"
    ) as file:

        json.dump(
            headline,
            file,
            indent=2
        )

    # --------------------------------------------------
    # Print final summary
    # --------------------------------------------------

    pd.set_option(
        "display.width",
        200
    )

    pd.set_option(
        "display.max_columns",
        20
    )

    print("\nEXPERIMENT SUMMARY")

    print(
        summary[
            [
                "pr_auc_mean",
                "total_cost_mean",
                "silent_miss_rate_mean",
                "defer_rate_mean"
            ]
        ].round(4)
    )

    print("\nHEADLINE RESULTS")

    print(
        json.dumps(
            headline,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
