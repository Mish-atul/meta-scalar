from __future__ import annotations

from typing import Dict, List

from support_triage_env.graders.task3_grader import Task3Grader
from support_triage_env.models import GroundTruthRecord, SenderTier, TriageRecord
from support_triage_env.tasks.base_task import BaseTask, TaskConfig


class Task3(BaseTask):
    """Full triage pipeline task."""

    def __init__(self) -> None:
        self.config = TaskConfig(
            task_id=3,
            max_steps=40,
            required_actions=[
                "classify",
                "set_priority",
                "route",
                "draft_response",
                "mark_duplicate",
                "escalate",
            ],
            optional_actions=["request_info"],
            forbidden_actions=[],
            hint_visible=False,
        )
        self.grader = Task3Grader()

    def grade(
        self,
        triage_records: Dict[str, TriageRecord],
        ground_truth: Dict[str, GroundTruthRecord],
        inbox_emails: List[str],
        sender_tiers: Dict[str, SenderTier],
    ) -> float:
        return self.grader.score(triage_records, ground_truth, inbox_emails, sender_tiers)
