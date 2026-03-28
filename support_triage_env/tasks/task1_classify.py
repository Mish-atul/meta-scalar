from __future__ import annotations

from typing import Dict, List

from support_triage_env.graders.task1_grader import Task1Grader
from support_triage_env.models import GroundTruthRecord, SenderTier, TriageRecord
from support_triage_env.tasks.base_task import BaseTask, TaskConfig


class Task1(BaseTask):
    """Single email classification task."""

    def __init__(self) -> None:
        self.config = TaskConfig(
            task_id=1,
            max_steps=3,
            required_actions=["classify", "set_priority"],
            optional_actions=["request_info"],
            forbidden_actions=["draft_response", "escalate"],
            hint_visible=True,
        )
        self.grader = Task1Grader()

    def get_hint(self) -> str | None:
        return "Task 1: classify and prioritize the single email before submit_triage."

    def grade(
        self,
        triage_records: Dict[str, TriageRecord],
        ground_truth: Dict[str, GroundTruthRecord],
        inbox_emails: List[str],
        sender_tiers: Dict[str, SenderTier],
    ) -> float:
        if not inbox_emails:
            return 0.0
        email_id = inbox_emails[0]
        rec = triage_records.get(email_id)
        truth = ground_truth[email_id]
        return self.grader.score(rec.category if rec else None, rec.priority if rec else None, truth)
