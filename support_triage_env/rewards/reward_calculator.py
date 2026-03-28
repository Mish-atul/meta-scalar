from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

from support_triage_env.graders.common import ResponseGrader
from support_triage_env.models import ActionType, GroundTruthRecord, Priority, SenderTier, TriageAction, TriageRecord


@dataclass
class RewardResult:
    value: float
    reason: str
    components: Dict[str, float]


class RewardCalculator:
    """Step-level reward calculator with deterministic rules."""

    def __init__(self) -> None:
        self.response_grader = ResponseGrader()
        self._request_info_rewarded: Set[str] = set()

    def compute_step_reward(
        self,
        task_id: int,
        action: TriageAction,
        record: TriageRecord | None,
        ground_truth: GroundTruthRecord | None,
        sender_tier: SenderTier | None,
        is_repeat_action: bool,
        is_invalid_action: bool,
        is_ambiguous: bool,
        is_redundant_request_info: bool,
    ) -> RewardResult:
        if is_invalid_action:
            return RewardResult(-0.05, "invalid_action", {"invalid": -0.05})
        if is_repeat_action:
            return RewardResult(-0.05, "repeat_action", {"repeat": -0.05})

        if action.action_type == ActionType.CLASSIFY:
            if ground_truth and action.category == ground_truth.category:
                reward = 0.35 if ground_truth.difficulty_level >= 2 else 0.25
                return RewardResult(reward, "correct_classification", {"classification": reward})
            return RewardResult(0.0, "incorrect_classification", {"classification": 0.0})

        if action.action_type == ActionType.SET_PRIORITY:
            if ground_truth and action.priority == ground_truth.priority:
                return RewardResult(0.20, "correct_priority", {"priority": 0.20})
            # VIP mis-prioritization penalty applies only in Task 2 and Task 3.
            if task_id in {2, 3} and sender_tier == SenderTier.VIP and action.priority in {Priority.P3, Priority.P4}:
                return RewardResult(-0.20, "vip_priority_downgrade", {"vip_penalty": -0.20})
            return RewardResult(0.0, "incorrect_priority", {"priority": 0.0})

        if action.action_type == ActionType.ROUTE:
            if ground_truth and action.team == ground_truth.team:
                return RewardResult(0.18, "correct_routing", {"routing": 0.18})
            return RewardResult(0.0, "incorrect_routing", {"routing": 0.0})

        if action.action_type == ActionType.DRAFT_RESPONSE:
            score = self.response_grader.quality_score(action.response_text, ground_truth) if ground_truth else 0.0
            reward = 0.10 + (0.10 * score)
            return RewardResult(reward, "response_quality", {"response": reward})

        if action.action_type == ActionType.ESCALATE:
            if ground_truth and ground_truth.requires_escalation:
                return RewardResult(0.15, "correct_escalation", {"escalation": 0.15})
            return RewardResult(0.0, "unneeded_escalation", {"escalation": 0.0})

        if action.action_type == ActionType.MARK_DUPLICATE:
            if ground_truth and ground_truth.is_duplicate and action.original_email_id == ground_truth.original_email_id:
                return RewardResult(0.20, "correct_duplicate", {"duplicate": 0.20})
            return RewardResult(0.0, "incorrect_duplicate", {"duplicate": 0.0})

        if action.action_type == ActionType.REQUEST_INFO:
            if action.email_id and action.email_id in self._request_info_rewarded:
                return RewardResult(-0.10, "request_info_overused", {"request_info": -0.10})
            if is_redundant_request_info:
                return RewardResult(-0.05, "request_info_unnecessary", {"request_info": -0.05})
            if is_ambiguous:
                if action.email_id:
                    self._request_info_rewarded.add(action.email_id)
                return RewardResult(0.05, "request_info_helpful", {"request_info": 0.05})
            return RewardResult(0.0, "request_info_optional", {"request_info": 0.0})

        if action.action_type == ActionType.SUBMIT_TRIAGE:
            return RewardResult(0.0, "submit_action", {})

        return RewardResult(0.0, "no_reward_rule", {})

    def completion_adjustment(
        self,
        fully_processed: bool,
        unprocessed_count: int,
        coverage_ratio: float,
    ) -> RewardResult:
        components: Dict[str, float] = {}
        value = 0.0

        if fully_processed:
            value += 0.25
            components["completion_bonus"] = 0.25

        if unprocessed_count > 0:
            penalty = min(0.5, 0.05 * unprocessed_count + (0.05 if coverage_ratio < 0.5 else 0.0))
            value -= penalty
            components["early_submit_penalty"] = -penalty

        return RewardResult(value, "completion_adjustment", components)
