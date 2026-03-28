from __future__ import annotations

from typing import Dict, List

from support_triage_env.graders.task2_grader import Task2Grader
from support_triage_env.models import GroundTruthRecord, SenderTier, TriageRecord
from support_triage_env.tasks.base_task import BaseTask, TaskConfig


class Task2(BaseTask):
    """Batch triage task with priority and routing."""

    def __init__(self) -> None:
        self.config = TaskConfig(
            task_id=2,
            max_steps=15,
            required_actions=["classify", "set_priority", "route"],
            optional_actions=["request_info", "mark_duplicate"],
            forbidden_actions=["draft_response"],
            hint_visible=False,
        )
        self.grader = Task2Grader()

    def grade(
        self,
        triage_records: Dict[str, TriageRecord],
        ground_truth: Dict[str, GroundTruthRecord],
        inbox_emails: List[str],
        sender_tiers: Dict[str, SenderTier],
    ) -> float:
        return self.grader.score(triage_records, ground_truth, inbox_emails)
