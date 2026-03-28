from __future__ import annotations

import re
from statistics import mean
from typing import Dict, Iterable, List

from support_triage_env.models import Category, GroundTruthRecord, Priority, TriageRecord


class ClassificationMetrics:
    """Deterministic classification metrics for category labels."""

    @staticmethod
    def f1_score(records: Dict[str, TriageRecord], truth: Dict[str, GroundTruthRecord], email_ids: Iterable[str]) -> float:
        ids = list(email_ids)
        if not ids:
            return 0.0

        classes = {Category.BILLING, Category.TECHNICAL, Category.ACCOUNT, Category.REFUND, Category.ABUSE, Category.GENERAL}
        f1_values: List[float] = []

        for cat in classes:
            tp = fp = fn = 0
            for email_id in ids:
                pred = records.get(email_id).category if records.get(email_id) else None
                gold = truth[email_id].category
                if pred == cat and gold == cat:
                    tp += 1
                elif pred == cat and gold != cat:
                    fp += 1
                elif pred != cat and gold == cat:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            f1_values.append(f1)

        return max(0.0, min(1.0, mean(f1_values)))


class PriorityMetrics:
    """Deterministic priority scoring with adjacent partial credit."""

    RANKS = {Priority.P1: 1, Priority.P2: 2, Priority.P3: 3, Priority.P4: 4}

    @classmethod
    def weighted_score(
        cls,
        records: Dict[str, TriageRecord],
        truth: Dict[str, GroundTruthRecord],
        email_ids: Iterable[str],
    ) -> float:
        scores: List[float] = []
        for email_id in email_ids:
            record = records.get(email_id)
            if not record or not record.priority:
                scores.append(0.0)
                continue
            if record.priority == truth[email_id].priority:
                scores.append(1.0)
                continue
            diff = abs(cls.RANKS[record.priority] - cls.RANKS[truth[email_id].priority])
            scores.append(0.3 if diff == 1 else 0.0)
        return max(0.0, min(1.0, mean(scores) if scores else 0.0))


class RoutingMetrics:
    """Exact routing accuracy metric."""

    @staticmethod
    def accuracy(records: Dict[str, TriageRecord], truth: Dict[str, GroundTruthRecord], email_ids: Iterable[str]) -> float:
        ids = list(email_ids)
        if not ids:
            return 0.0
        correct = 0
        for email_id in ids:
            record = records.get(email_id)
            if record and record.team == truth[email_id].team:
                correct += 1
        return correct / len(ids)


class ResponseGrader:
    """Deterministic response scoring from text patterns and category fit."""

    UNPROFESSIONAL = ["wtf", "stupid", "idiot", "deal with it", "not my problem"]
    GREETING = re.compile(r"\b(hello|hi|dear|thank you for contacting)\b", re.IGNORECASE)
    CLOSING = re.compile(r"\b(regards|sincerely|thank you)\b", re.IGNORECASE)
    CATEGORY_KEYWORDS = {
        Category.BILLING: ["charge", "invoice", "billing", "payment", "refund"],
        Category.TECHNICAL: ["issue", "error", "bug", "fix", "troubleshoot"],
        Category.ACCOUNT: ["account", "login", "access", "password", "settings"],
        Category.REFUND: ["refund", "return", "cancel", "credit"],
        Category.ABUSE: ["report", "policy", "abuse", "review", "safety"],
        Category.GENERAL: ["feedback", "question", "information", "request"],
    }

    def quality_score(self, response_text: str | None, ground_truth: GroundTruthRecord) -> float:
        if not response_text:
            return 0.0

        text = response_text.strip().lower()
        words = text.split()
        if len(words) < 8:
            return 0.0

        relevance = self._score_relevance(text, ground_truth.category)
        correctness = 1.0 if any(k in text for k in ["review", "investigate", "check", "resolve", "update"]) else 0.4
        actionability = 1.0 if any(k in text for k in ["next", "within", "hour", "day", "follow up"]) else 0.4
        completeness = 1.0 if self.GREETING.search(text) and self.CLOSING.search(text) else 0.4
        tone = 0.0 if any(bad in text for bad in self.UNPROFESSIONAL) else 1.0
        clarity = 1.0 if len(words) <= 220 else 0.7

        score = (
            0.25 * relevance
            + 0.20 * correctness
            + 0.20 * actionability
            + 0.15 * completeness
            + 0.10 * tone
            + 0.10 * clarity
        )
        return max(0.0, min(1.0, score))

    def _score_relevance(self, text: str, category: Category) -> float:
        keywords = self.CATEGORY_KEYWORDS[category]
        overlap = sum(1 for k in keywords if k in text)
        if overlap >= 3:
            return 1.0
        if overlap == 2:
            return 0.75
        if overlap == 1:
            return 0.5
        return 0.2


class EscalationGrader:
    """Recall-oriented escalation metric."""

    @staticmethod
    def recall(records: Dict[str, TriageRecord], truth: Dict[str, GroundTruthRecord], email_ids: Iterable[str]) -> float:
        required = [email_id for email_id in email_ids if truth[email_id].requires_escalation]
        if not required:
            return 1.0
        hit = sum(1 for email_id in required if records.get(email_id) and records[email_id].escalated)
        return hit / len(required)


class DuplicateGrader:
    """Precision metric for duplicate marking."""

    @staticmethod
    def precision(records: Dict[str, TriageRecord], truth: Dict[str, GroundTruthRecord], email_ids: Iterable[str]) -> float:
        predicted = [email_id for email_id in email_ids if records.get(email_id) and records[email_id].marked_duplicate]
        if not predicted:
            return 1.0
        correct = sum(1 for email_id in predicted if truth[email_id].is_duplicate)
        return correct / len(predicted)
