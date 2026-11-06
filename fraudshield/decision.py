
"""
FraudShield decision engine.

Policies:
1. Threshold-based triage.
2. Expected-cost triage.

The tuning split is used for threshold selection.
The test split must never be used to select thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .costs import APPROVE, BLOCK, DEFER, CostModel


@dataclass(frozen=True)
class Thresholds:
    """Stores thresholds selected during model tuning."""

    t_low: float
    t_high: float
    tune_cost: float
    defer_rate: float


def _validate_inputs(p, y=None, amount=None):
    """Validate prediction scores and optional inputs."""

    p = np.asarray(p, dtype=float).reshape(-1)

    if len(p) == 0:
        raise ValueError("Prediction scores cannot be empty.")

    if not np.all(np.isfinite(p)):
        raise ValueError("Prediction scores contain invalid values.")

    if np.any((p < 0) | (p > 1)):
        raise ValueError("Prediction scores must be between 0 and 1.")

    if y is not None:
        y = np.asarray(y).reshape(-1)

        if len(y) != len(p):
            raise ValueError("Scores and labels must have equal lengths.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Labels must contain only 0 and 1.")

    if amount is not None:
        amount = np.asarray(amount, dtype=float).reshape(-1)

        if len(amount) != len(p):
            raise ValueError("Scores and amounts must have equal lengths.")

        if not np.all(np.isfinite(amount)) or np.any(amount < 0):
            raise ValueError(
                "Transaction amounts must be finite and non-negative."
            )

    return p, y, amount


def decide(p, t_low: float, t_high: float) -> np.ndarray:
    """
    Threshold-based three-way decision.

    p < t_low:
        Approve.

    t_low <= p < t_high:
        Defer.

    p >= t_high:
        Block.
    """

    p, _, _ = _validate_inputs(p)

    if not (0 <= t_low <= t_high <= 1):
        raise ValueError(
            "Thresholds must satisfy 0 <= t_low <= t_high <= 1."
        )

    decisions = np.full(len(p), DEFER, dtype=int)

    decisions[p < t_low] = APPROVE
    decisions[p >= t_high] = BLOCK

    return decisions


def expected_cost_decisions(
    p,
    amount,
    cost: CostModel
) -> np.ndarray:
    """
    Expected-cost decision policy.

    Calculates the expected cost of approving, blocking,
    and sending each transaction for human review.

    Selects the action with the lowest expected cost.

    This policy uses predicted probabilities and transaction
    amounts, not the true fraud labels.
    """

    p, _, amount = _validate_inputs(
        p,
        amount=amount
    )

    if not isinstance(cost, CostModel):
        raise TypeError("cost must be a CostModel instance.")

    false_block_cost = cost.false_block_cost
    review_cost = cost.review_cost
    reviewer_error = 1.0 - cost.reviewer_accuracy

    # Expected cost if the transaction is approved.
    approve_cost = p * amount

    # Expected cost if the transaction is blocked.
    block_cost = (1.0 - p) * false_block_cost

    # Expected cost if a human reviewer handles it.
    review_costs = (
        review_cost
        + reviewer_error * (
            p * amount
            + (1.0 - p) * false_block_cost
        )
    )

    # Columns represent approve, block and review.
    all_costs = np.column_stack([
        approve_cost,
        block_cost,
        review_costs
    ])

    # np.argmin chooses the lowest-cost action.
    # Column indices match APPROVE, BLOCK and DEFER.
    decisions = np.argmin(
        all_costs,
        axis=1
    )

    return decisions.astype(int)


def expected_decision_costs(p, amount, cost: CostModel):
    """Return the three expected costs for inspection."""

    p, _, amount = _validate_inputs(
        p,
        amount=amount
    )

    if not isinstance(cost, CostModel):
        raise TypeError("cost must be a CostModel instance.")

    approve_cost = p * amount

    block_cost = (
        (1.0 - p) * cost.false_block_cost
    )

    review_costs = (
        cost.review_cost
        + (1.0 - cost.reviewer_accuracy)
        * (
            p * amount
            + (1.0 - p) * cost.false_block_cost
        )
    )

    return {
        "approve": approve_cost,
        "block": block_cost,
        "defer": review_costs
    }


def candidate_grid(p=None) -> np.ndarray:
    """Generate possible probability thresholds."""

    fixed_grid = np.concatenate([
        np.array([0.0, 1.0]),
        np.logspace(-5, -0.001, 70),
        np.linspace(0.01, 0.99, 80)
    ])

    if p is not None:

        p = np.asarray(p, dtype=float)

        data_grid = np.quantile(
            p,
            np.linspace(0, 1, 101)
        )

        fixed_grid = np.concatenate([
            fixed_grid,
            data_grid
        ])

    return np.unique(
        np.clip(fixed_grid, 0, 1)
    )


def tune_thresholds(
    p,
    y,
    amount,
    cost: CostModel,
    review_budget: float = 0.02,
    allow_defer: bool = True
) -> Thresholds:
    """
    Select threshold-based decisions by minimising
    empirical cost on the tuning dataset.

    Review budget is a maximum, not a target.
    """

    p, y, amount = _validate_inputs(
        p,
        y,
        amount
    )

    if not isinstance(cost, CostModel):
        raise TypeError("cost must be a CostModel instance.")

    if not 0 <= review_budget <= 1:
        raise ValueError(
            "Review budget must be between 0 and 1."
        )

    n = len(p)

    cost_approve, cost_block, cost_defer = (
        cost.per_decision_costs(y, amount)
    )

    order = np.argsort(
        p,
        kind="stable"
    )

    sorted_p = p[order]

    def prefix_sum(values):
        return np.concatenate([
            np.array([0.0]),
            np.cumsum(values[order])
        ])

    prefix_approve = prefix_sum(cost_approve)
    prefix_block = prefix_sum(cost_block)
    prefix_defer = prefix_sum(cost_defer)

    grid = candidate_grid(sorted_p)

    positions = np.searchsorted(
        sorted_p,
        grid,
        side="left"
    )

    best_result = None

    for i, t_low in enumerate(grid):

        if allow_defer:
            possible_highs = range(i, len(grid))
        else:
            possible_highs = [i]

        for j in possible_highs:

            t_high = grid[j]

            low_position = positions[i]
            high_position = positions[j]

            deferred_count = (
                high_position - low_position
            )

            defer_rate = deferred_count / n

            if defer_rate > review_budget + 1e-12:
                continue

            approve_total = prefix_approve[low_position]

            defer_total = (
                prefix_defer[high_position]
                - prefix_defer[low_position]
            )

            block_total = (
                prefix_block[n]
                - prefix_block[high_position]
            )

            total_cost = (
                approve_total
                + defer_total
                + block_total
            )

            result_key = (
                total_cost,
                defer_rate
            )

            if (
                best_result is None
                or result_key < best_result[0]
            ):

                best_result = (
                    result_key,
                    float(t_low),
                    float(t_high)
                )

    if best_result is None:
        raise RuntimeError(
            "No feasible threshold combination was found."
        )

    (best_cost, best_defer_rate), best_low, best_high = (
        best_result
    )

    return Thresholds(
        t_low=best_low,
        t_high=best_high,
        tune_cost=float(best_cost),
        defer_rate=float(best_defer_rate)
    )


def evaluate_decisions(
    decision,
    y,
    amount,
    cost: CostModel
) -> dict:
    """Evaluate decisions using realised labelled outcomes."""

    decision = np.asarray(decision).reshape(-1)
    y = np.asarray(y).reshape(-1)
    amount = np.asarray(amount, dtype=float).reshape(-1)

    if not (
        len(decision) == len(y) == len(amount)
    ):
        raise ValueError(
            "Decision, label and amount arrays must have equal lengths."
        )

    if len(y) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if not np.all(
        np.isin(decision, [APPROVE, BLOCK, DEFER])
    ):
        raise ValueError("Invalid decision value.")

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1.")

    if (
        not np.all(np.isfinite(amount))
        or np.any(amount < 0)
    ):
        raise ValueError(
            "Amounts must be finite and non-negative."
        )

    fraud = y == 1
    legitimate = y == 0

    approved = decision == APPROVE
    blocked = decision == BLOCK
    deferred = decision == DEFER

    total_frauds = int(fraud.sum())
    total_legitimate = int(legitimate.sum())

    fraud_denominator = max(total_frauds, 1)
    block_denominator = max(int(blocked.sum()), 1)

    return {
        "total_cost": cost.total(
            decision,
            y,
            amount
        ),

        "silent_miss_rate": float(
            (fraud & approved).sum()
            / fraud_denominator
        ),

        "fraud_blocked_rate": float(
            (fraud & blocked).sum()
            / fraud_denominator
        ),

        "fraud_deferred_rate": float(
            (fraud & deferred).sum()
            / fraud_denominator
        ),

        "auto_block_precision": float(
            (fraud & blocked).sum()
            / block_denominator
        ) if blocked.any() else float("nan"),

        "defer_rate": float(deferred.mean()),

        "n_deferred": int(deferred.sum()),

        "n_approved": int(approved.sum()),

        "n_blocked": int(blocked.sum()),

        "n_false_blocks": int(
            (legitimate & blocked).sum()
        ),

        "n_frauds": total_frauds,

        "n_legitimate": total_legitimate
    }
