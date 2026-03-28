from __future__ import annotations

from typing import Dict, Iterable

from support_triage_env.graders.common import (
    ClassificationMetrics,
    DuplicateGrader,
    EscalationGrader,
    PriorityMetrics,
    ResponseGrader,
    RoutingMetrics,
)
from support_triage_env.models import GroundTruthRecord, Priority, SenderTier, TriageRecord


class Task3Grader:
    """Composite grader for full pipeline task with penalties."""

    def __init__(self) -> None:
        self.response_grader = ResponseGrader()

    def score(
        self,
        triage_records: Dict[str, TriageRecord],
        ground_truth: Dict[str, GroundTruthRecord],
        inbox_emails: Iterable[str],
        sender_tiers: Dict[str, SenderTier],
    ) -> float:
        email_ids = list(inbox_emails)

        classification_f1 = ClassificationMetrics.f1_score(triage_records, ground_truth, email_ids)
        priority_weighted = PriorityMetrics.weighted_score(triage_records, ground_truth, email_ids)
        routing_accuracy = RoutingMetrics.accuracy(triage_records, ground_truth, email_ids)
        response_quality = self._mean_response_quality(triage_records, ground_truth, email_ids)
        escalation_recall = EscalationGrader.recall(triage_records, ground_truth, email_ids)
        duplicate_precision = DuplicateGrader.precision(triage_records, ground_truth, email_ids)

        score = (
            0.20 * classification_f1
            + 0.20 * priority_weighted
            + 0.20 * routing_accuracy
            + 0.20 * response_quality
            + 0.10 * escalation_recall
            + 0.10 * duplicate_precision
        )

        score -= 0.15 * self._count_p1_without_response(triage_records, ground_truth, email_ids)
        score -= 0.20 * self._count_vip_downgraded(triage_records, sender_tiers, email_ids)
        score -= 0.10 * self._count_incomplete(triage_records, email_ids)
        score -= 0.05 * self._count_missed_escalations(triage_records, ground_truth, email_ids)
        score -= 0.05 * self._count_missed_duplicates(triage_records, ground_truth, email_ids)

        return max(0.0, min(1.0, score))

    def _mean_response_quality(
        self,
        records: Dict[str, TriageRecord],
        truth: Dict[str, GroundTruthRecord],
        email_ids: list[str],
    ) -> float:
        scores = []
        for email_id in email_ids:
            gt = truth[email_id]
            if gt.priority not in {Priority.P1, Priority.P2}:
                continue
            response_text = records.get(email_id).response_text if records.get(email_id) else None
            scores.append(self.response_grader.quality_score(response_text, gt))
        if not scores:
            return 1.0
        return sum(scores) / len(scores)

    def _count_p1_without_response(
        self,
        records: Dict[str, TriageRecord],
        truth: Dict[str, GroundTruthRecord],
        email_ids: list[str],
    ) -> int:
        return sum(
            1
            for email_id in email_ids
            if truth[email_id].priority == Priority.P1
            and not (records.get(email_id) and records[email_id].response_text)
        )

    def _count_vip_downgraded(
        self,
        records: Dict[str, TriageRecord],
        sender_tiers: Dict[str, SenderTier],
        email_ids: list[str],
    ) -> int:
        return sum(
            1
            for email_id in email_ids
            if sender_tiers.get(email_id) == SenderTier.VIP
            and records.get(email_id)
            and records[email_id].priority in {Priority.P3, Priority.P4}
        )

    def _count_incomplete(self, records: Dict[str, TriageRecord], email_ids: list[str]) -> int:
        count = 0
        for email_id in email_ids:
            rec = records.get(email_id)
            if not rec:
                count += 1
                continue
            if rec.category is None or rec.priority is None or rec.team is None:
                count += 1
        return count

    def _count_missed_escalations(
        self,
        records: Dict[str, TriageRecord],
        truth: Dict[str, GroundTruthRecord],
        email_ids: list[str],
    ) -> int:
        return sum(
            1
            for email_id in email_ids
            if truth[email_id].requires_escalation
            and not (records.get(email_id) and records[email_id].escalated)
        )

    def _count_missed_duplicates(
        self,
        records: Dict[str, TriageRecord],
        truth: Dict[str, GroundTruthRecord],
        email_ids: list[str],
    ) -> int:
        return sum(
            1
            for email_id in email_ids
            if truth[email_id].is_duplicate
            and not (records.get(email_id) and records[email_id].marked_duplicate)
        )
