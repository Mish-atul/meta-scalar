from __future__ import annotations

from typing import Dict, List

from support_triage_env.models import GroundTruthRecord, Priority, TriageRecord


class Task2Grader:
    """Batch triage grader with completion ratio penalty."""

    RANKS = {Priority.P1: 1, Priority.P2: 2, Priority.P3: 3, Priority.P4: 4}

    def score(
        self,
        triage_records: Dict[str, TriageRecord],
        ground_truth: Dict[str, GroundTruthRecord],
        inbox_emails: List[str],
    ) -> float:
        if not inbox_emails:
            return 0.0

        per_email = []
        for email_id in inbox_emails:
            record = triage_records.get(email_id)
            truth = ground_truth[email_id]
            s = 0.0

            if record is None:
                per_email.append(0.0)
                continue

            if record.category == truth.category:
                s += 0.33
            elif record.category and self._adjacent_category(record.category.value, truth.category.value):
                s += 0.10

            if record.priority == truth.priority:
                s += 0.33
            elif record.priority is not None:
                diff = abs(self.RANKS[record.priority] - self.RANKS[truth.priority])
                if diff == 1:
                    s += 0.10

            if record.team == truth.team:
                s += 0.34

            per_email.append(s)

        raw_mean = sum(per_email) / len(inbox_emails)
        completion_ratio = len([email for email in inbox_emails if email in triage_records]) / len(inbox_emails)
        return max(0.0, min(1.0, raw_mean * completion_ratio))

    def _adjacent_category(self, cat_a: str, cat_b: str) -> bool:
        pair = tuple(sorted((cat_a, cat_b)))
        return pair in {
            ("account", "billing"),
            ("billing", "refund"),
            ("general", "technical"),
        }
