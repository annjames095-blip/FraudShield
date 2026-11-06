"""Cost model. All numbers are ASSUMPTIONS and are adjustable in the dashboard."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

APPROVE, BLOCK, DEFER = 0, 1, 2
NAMES = {APPROVE: "approve", BLOCK: "block", DEFER: "defer to review"}


@dataclass(frozen=True)
class CostModel:
    false_block_cost: float = 10.0   # handling cost / friction of blocking a legitimate transaction
    review_cost: float = 2.0         # analyst time per reviewed transaction
    reviewer_accuracy: float = 0.95  # probability a human reviewer decides correctly

    def case_costs(self, decision, y, amount) -> np.ndarray:
        """Expected cost of each case under its decision.

        approve: a missed fraud costs its amount.
        block:   a false block costs false_block_cost.
        defer:   review_cost, plus the expected cost of the reviewer being wrong.
        """
        decision, y, amount = np.asarray(decision), np.asarray(y), np.asarray(amount, float)
        fraud = y == 1
        c_app = np.where(fraud, amount, 0.0)
        c_blk = np.where(fraud, 0.0, self.false_block_cost)
        err = 1.0 - self.reviewer_accuracy
        c_def = self.review_cost + err * np.where(fraud, amount, self.false_block_cost)
        return np.select([decision == APPROVE, decision == BLOCK], [c_app, c_blk], default=c_def)

    def total(self, decision, y, amount) -> float:
        return float(self.case_costs(decision, y, amount).sum())

    def per_decision_costs(self, y, amount):
        """Cost of each case under each of the three decisions (used by threshold tuning)."""
        y = np.asarray(y)
        f = lambda d: self.case_costs(np.full(len(y), d), y, amount)
        return f(APPROVE), f(BLOCK), f(DEFER)
