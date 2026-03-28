from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from support_triage_env.models import GroundTruthRecord, SenderTier, TriageRecord


@dataclass(frozen=True)
class TaskConfig:
    """Static configuration for a task."""

    task_id: int
    max_steps: int
    required_actions: List[str]
    optional_actions: List[str]
    forbidden_actions: List[str]
    hint_visible: bool


class BaseTask:
    """Abstract task interface used by environment engine."""

    config: TaskConfig

    def grade(
        self,
        triage_records: Dict[str, TriageRecord],
        ground_truth: Dict[str, GroundTruthRecord],
        inbox_emails: List[str],
        sender_tiers: Dict[str, SenderTier],
    ) -> float:
        raise NotImplementedError

    def get_hint(self) -> str | None:
        return None
