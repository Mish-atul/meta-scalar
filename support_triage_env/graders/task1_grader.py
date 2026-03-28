from __future__ import annotations

from typing import Optional

from support_triage_env.models import Category, GroundTruthRecord, Priority


class Task1Grader:
    """Grader for Task 1 with deterministic partial credit."""

    PRIORITY_RANKS = {Priority.P1: 1, Priority.P2: 2, Priority.P3: 3, Priority.P4: 4}
    CATEGORY_ADJ = {
        tuple(sorted((Category.BILLING.value, Category.ACCOUNT.value))),
        tuple(sorted((Category.TECHNICAL.value, Category.GENERAL.value))),
        tuple(sorted((Category.REFUND.value, Category.BILLING.value))),
    }

    def score(
        self,
        submitted_category: Optional[Category],
        submitted_priority: Optional[Priority],
        ground_truth: GroundTruthRecord,
    ) -> float:
        score = 0.0

        if submitted_category == ground_truth.category:
            score += 0.60
        elif submitted_category and self._are_adjacent(submitted_category, ground_truth.category):
            score += 0.20

        if submitted_priority == ground_truth.priority:
            score += 0.40
        elif submitted_priority is not None:
            diff = abs(self.PRIORITY_RANKS[submitted_priority] - self.PRIORITY_RANKS[ground_truth.priority])
            if diff == 1:
                score += 0.15

        return max(0.0, min(1.0, score))

    def _are_adjacent(self, cat_a: Category, cat_b: Category) -> bool:
        return tuple(sorted((cat_a.value, cat_b.value))) in self.CATEGORY_ADJ
