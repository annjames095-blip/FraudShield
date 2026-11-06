"""FraudShield cost sensitivity analysis.

Loads the fixed calibrated test predictions saved by run_experiments.py and
re-evaluates the expected-cost policy under alternative assumed costs.
No model is retrained and no thresholds are tuned on the test set.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from fraudshield.costs import APPROVE, BLOCK, CostModel
from fraudshield.decision import expected_cost_decisions, evaluate_decisions


FALSE_BLOCK_COSTS = [5.0, 10.0, 20.0, 50.0]
REVIEW_COSTS = [1.0, 2.0, 5.0, 10.0]
REVIEWER_ACCURACIES = [0.90, 0.95, 0.99]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FraudShield cost sensitivity analysis."
    )
    parser.add_argument(
        "--input",
        default="artifacts/eval.npz",
        help="Path to the evaluation arrays saved by run_experiments.py",
    )
    parser.add_argument(
        "--output",
        default="results/cost_sensitivity.csv",
        help="Where to save the sensitivity results CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path}. Run run_experiments.py first."
        )

    with np.load(input_path) as data:
        required = {"p_test", "y_test", "a_test"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"{input_path} is missing required arrays: {sorted(missing)}"
            )
        p = np.asarray(data["p_test"], dtype=float)
        y = np.asarray(data["y_test"], dtype=int)
        amount = np.asarray(data["a_test"], dtype=float)

    if not (len(p) == len(y) == len(amount)):
        raise ValueError("Test probabilities, labels and amounts have different lengths.")

    rows = []

    for false_block_cost, review_cost, reviewer_accuracy in product(
        FALSE_BLOCK_COSTS, REVIEW_COSTS, REVIEWER_ACCURACIES
    ):
        cost = CostModel(
            false_block_cost=false_block_cost,
            review_cost=review_cost,
            reviewer_accuracy=reviewer_accuracy,
        )

        # Expected-cost policy: uses predicted probabilities and amounts only.
        policy_decisions = expected_cost_decisions(p, amount, cost)
        policy_metrics = evaluate_decisions(policy_decisions, y, amount, cost)

        # Comparator: choose the cheaper of approve/block, without human review.
        approve_expected_cost = p * amount
        block_expected_cost = (1.0 - p) * false_block_cost
        baseline_decisions = np.where(
            approve_expected_cost <= block_expected_cost, APPROVE, BLOCK
        )
        baseline_metrics = evaluate_decisions(
            baseline_decisions, y, amount, cost
        )

        baseline_total = baseline_metrics["total_cost"]
        policy_total = policy_metrics["total_cost"]
        savings = baseline_total - policy_total
        savings_pct = (
            100.0 * savings / baseline_total if baseline_total > 0 else 0.0
        )

        rows.append(
            {
                "false_block_cost": false_block_cost,
                "review_cost": review_cost,
                "reviewer_accuracy": reviewer_accuracy,
                "baseline_no_review_cost": baseline_total,
                "expected_policy_cost": policy_total,
                "cost_savings": savings,
                "cost_savings_pct": savings_pct,
                "defer_rate": policy_metrics["defer_rate"],
                "n_deferred": policy_metrics["n_deferred"],
                "silent_miss_rate": policy_metrics["silent_miss_rate"],
                "fraud_blocked_rate": policy_metrics["fraud_blocked_rate"],
                "fraud_deferred_rate": policy_metrics["fraud_deferred_rate"],
                "auto_block_precision": policy_metrics["auto_block_precision"],
                "n_false_blocks": policy_metrics["n_false_blocks"],
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["cost_savings_pct", "cost_savings"], ascending=False
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nCOST SENSITIVITY RESULTS")
    print(f"Test transactions: {len(p):,}")
    print(f"Scenarios evaluated: {len(results)}")
    print(f"Saved: {output_path}")
    print("\nTop 10 scenarios by modelled percentage cost saving:")
    print(results.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBOTTOM 5 SCENARIOS BY MODELLED PERCENTAGE COST SAVING")
    print(
        results.tail(5)
        .sort_values("cost_savings_pct")
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nINTERPRETATION")
    print(
        "These are test-set results under hypothetical cost assumptions, "
        "not measured real-world financial savings. The same fixed test set "
        "and calibrated predictions are reused for every scenario."
    )


if __name__ == "__main__":
    main()
