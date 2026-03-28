from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from support_triage_env.data.generate_dataset import main as generate_dataset_main
from support_triage_env.models import (
    ActionType,
    Category,
    EmailObject,
    EmailStatus,
    EmailSummary,
    GroundTruthRecord,
    Priority,
    SenderTier,
    Team,
    TriageAction,
    TriageObservation,
    TriageRecord,
    TriageState,
)
from support_triage_env.rewards.reward_calculator import RewardCalculator
from support_triage_env.tasks import Task1, Task2, Task3


class SupportTriageEnvironment:
    """Stateful environment implementing reset, step, and state."""

    def __init__(self) -> None:
        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        self._ensure_dataset()
        self.emails, self.ground_truth = self._load_data()
        self.reward_calculator = RewardCalculator()
        self.tasks = {1: Task1(), 2: Task2(), 3: Task3()}

        self.current_task_id = 1
        self.current_task = self.tasks[1]
        self.inbox_ids: List[str] = []
        self.email_by_id: Dict[str, EmailObject] = {}
        self.sender_tiers: Dict[str, SenderTier] = {}
        self.records: Dict[str, TriageRecord] = {}
        self.current_email_index = 0
        self.done = False
        self.state_obj = TriageState(task_id=1)
        self.team_queue_depths = {
            Team.BILLING.value: 5,
            Team.TECH.value: 7,
            Team.ACCOUNT.value: 4,
            Team.ESCALATIONS.value: 3,
            Team.GENERAL.value: 6,
        }

    def _ensure_dataset(self) -> None:
        emails_file = self.data_dir / "emails.json"
        truth_file = self.data_dir / "ground_truth.json"
        if emails_file.exists() and truth_file.exists():
            return
        generate_dataset_main()

    def _load_data(self) -> Tuple[List[EmailObject], Dict[str, GroundTruthRecord]]:
        emails_json = json.loads((self.data_dir / "emails.json").read_text(encoding="utf-8"))
        truth_json = json.loads((self.data_dir / "ground_truth.json").read_text(encoding="utf-8"))
        emails = [EmailObject.model_validate(item) for item in emails_json]
        truth = {email_id: GroundTruthRecord.model_validate(item) for email_id, item in truth_json.items()}
        return emails, truth

    def reset(self, task_id: int = 1, seed: int = 42) -> TriageObservation:
        rng = random.Random(seed)
        if task_id not in self.tasks:
            raise ValueError(f"Unsupported task_id: {task_id}")

        self.current_task_id = task_id
        self.current_task = self.tasks[task_id]
        self.done = False

        if task_id == 1:
            pool = [e for e in self.emails if self.ground_truth[e.email_id].difficulty_level == 1]
            inbox = rng.sample(pool, k=1)
        elif task_id == 2:
            vip_pool = [e for e in self.emails if e.sender_tier == SenderTier.VIP]
            edge_pool = [e for e in self.emails if self.ground_truth[e.email_id].difficulty_level >= 2]
            rest_pool = [e for e in self.emails if e not in vip_pool]
            inbox = [rng.choice(vip_pool), rng.choice(edge_pool)] + rng.sample(rest_pool, k=3)
        else:
            inbox = self._sample_task3_inbox(rng)

        self.inbox_ids = [email.email_id for email in inbox]
        self.email_by_id = {email.email_id: email for email in inbox}
        self.sender_tiers = {email.email_id: email.sender_tier for email in inbox}
        self.records = {email.email_id: TriageRecord(email_id=email.email_id) for email in inbox}
        self.current_email_index = 0
        self.state_obj = TriageState(
            episode_id=str(uuid4()),
            task_id=task_id,
            max_steps=self.current_task.config.max_steps,
            emails_in_inbox=len(self.inbox_ids),
            emails_processed=0,
            cumulative_reward=0.0,
            is_done=False,
        )

        return self._build_observation(last_result="Episode reset", action_valid=True)

    def _sample_task3_inbox(self, rng: random.Random) -> List[EmailObject]:
        selected: Dict[str, EmailObject] = {}

        def pick_one(candidates: List[EmailObject]) -> EmailObject:
            choices = [c for c in candidates if c.email_id not in selected]
            return rng.choice(choices)

        p1_candidates = [e for e in self.emails if self.ground_truth[e.email_id].priority == Priority.P1]
        vip_candidates = [e for e in self.emails if e.sender_tier == SenderTier.VIP]
        sla_candidates = [e for e in self.emails if self.ground_truth[e.email_id].sla_breach_risk]
        abuse_candidates = [e for e in self.emails if self.ground_truth[e.email_id].category == Category.ABUSE]
        ambiguous_candidates = [e for e in self.emails if self.ground_truth[e.email_id].difficulty_level >= 2]

        first = pick_one(p1_candidates)
        selected[first.email_id] = first

        second = pick_one(vip_candidates)
        selected[second.email_id] = second

        third = pick_one(sla_candidates)
        selected[third.email_id] = third

        fourth = pick_one(abuse_candidates)
        selected[fourth.email_id] = fourth

        # Duplicate pair: one duplicate and its original.
        duplicate_candidates = [e for e in self.emails if self.ground_truth[e.email_id].is_duplicate]
        duplicate = pick_one(duplicate_candidates)
        selected[duplicate.email_id] = duplicate
        original_id = self.ground_truth[duplicate.email_id].original_email_id
        if original_id and original_id in {e.email_id for e in self.emails}:
            original_obj = next(e for e in self.emails if e.email_id == original_id)
            selected[original_obj.email_id] = original_obj

        for _ in range(2):
            amb = pick_one(ambiguous_candidates)
            selected[amb.email_id] = amb

        size = rng.randint(8, 10)
        if len(selected) < size:
            hard_pool = [e for e in self.emails if self.ground_truth[e.email_id].difficulty_level >= 2]
            while len(selected) < size:
                nxt = pick_one(hard_pool)
                selected[nxt.email_id] = nxt

        inbox = list(selected.values())
        rng.shuffle(inbox)
        return inbox

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, Dict[str, Any]]:
        if self.done:
            obs = self._build_observation(last_result="Episode already completed", action_valid=False)
            return obs, -0.05, True, {"error": "episode_done"}

        self.state_obj.step_count += 1
        info: Dict[str, Any] = {}

        if action.action_type == ActionType.SUBMIT_TRIAGE:
            reward, info = self._handle_submit()
            self.state_obj.cumulative_reward += reward
            self.done = True
            self.state_obj.is_done = True
            obs = self._build_observation(last_result="Triage submitted", action_valid=True)
            return obs, reward, True, info

        is_invalid, message = self._validate_action(action)
        record = self.records.get(action.email_id) if action.email_id else None
        gt = self.ground_truth.get(action.email_id) if action.email_id else None
        sender_tier = self.sender_tiers.get(action.email_id) if action.email_id else None

        is_repeat = False
        if record and action.action_type.value in record.actions_taken:
            is_repeat = True

        is_ambiguous = bool(gt and gt.difficulty_level >= 2)
        is_redundant_request = bool(record and record.category and record.priority and record.team)

        result = self.reward_calculator.compute_step_reward(
            task_id=self.current_task_id,
            action=action,
            record=record,
            ground_truth=gt,
            sender_tier=sender_tier,
            is_repeat_action=is_repeat,
            is_invalid_action=is_invalid,
            is_ambiguous=is_ambiguous,
            is_redundant_request_info=is_redundant_request,
        )

        if not is_invalid and record:
            self._apply_action(record, action)
            message = f"action_applied:{action.action_type.value}"

        self.state_obj.cumulative_reward += result.value
        self.state_obj.emails_processed = self._count_processed()

        if self.state_obj.step_count >= self.state_obj.max_steps:
            final_score = self.current_task.grade(self.records, self._task_truth(), self.inbox_ids, self.sender_tiers)
            processed = self._count_processed()
            total = len(self.inbox_ids)
            coverage_ratio = processed / total if total else 0.0
            unprocessed = total - processed
            completion = self.reward_calculator.completion_adjustment(
                fully_processed=(unprocessed == 0),
                unprocessed_count=unprocessed,
                coverage_ratio=coverage_ratio,
            )
            result.value += final_score + completion.value
            self.state_obj.cumulative_reward += final_score + completion.value
            self.done = True
            self.state_obj.is_done = True
            info.update(
                {
                    "final_score": final_score,
                    "completion_adjustment": completion.components,
                    "coverage_ratio": coverage_ratio,
                    "processed": processed,
                    "unprocessed": unprocessed,
                    "auto_terminated": True,
                }
            )

        obs = self._build_observation(last_result=message or result.reason, action_valid=not is_invalid)
        merged_info = {
            "reason": result.reason,
            "components": result.components,
            "task_id": self.current_task_id,
        }
        merged_info.update(info)
        return obs, result.value, self.done, merged_info

    def state(self) -> TriageState:
        return self.state_obj

    def _validate_action(self, action: TriageAction) -> Tuple[bool, str]:
        if action.action_type.value in self.current_task.config.forbidden_actions:
            return True, "forbidden_action_for_task"

        if action.action_type != ActionType.SUBMIT_TRIAGE and not action.email_id:
            return True, "email_id_required"

        if action.email_id and action.email_id not in self.inbox_ids:
            return True, "unknown_email_id"

        if action.action_type == ActionType.CLASSIFY and action.category is None:
            return True, "category_required"
        if action.action_type == ActionType.SET_PRIORITY and action.priority is None:
            return True, "priority_required"
        if action.action_type == ActionType.ROUTE and action.team is None:
            return True, "team_required"
        if action.action_type == ActionType.DRAFT_RESPONSE and not action.response_text:
            return True, "response_text_required"
        if action.action_type == ActionType.MARK_DUPLICATE and not action.original_email_id:
            return True, "original_email_id_required"
        if action.action_type == ActionType.ESCALATE and not action.reason:
            return True, "reason_required"
        if action.action_type == ActionType.REQUEST_INFO and not action.question:
            return True, "question_required"

        return False, ""

    def _apply_action(self, record: TriageRecord, action: TriageAction) -> None:
        record.actions_taken.append(action.action_type.value)

        if action.action_type == ActionType.CLASSIFY:
            record.category = action.category
            record.status = EmailStatus.CLASSIFIED
        elif action.action_type == ActionType.SET_PRIORITY:
            record.priority = action.priority
            record.status = EmailStatus.PRIORITIZED
        elif action.action_type == ActionType.ROUTE:
            record.team = action.team
            if action.team:
                self.team_queue_depths[action.team.value] = self.team_queue_depths.get(action.team.value, 0) + 1
            record.status = EmailStatus.ROUTED
        elif action.action_type == ActionType.DRAFT_RESPONSE:
            record.response_text = action.response_text
            record.status = EmailStatus.RESPONDED
        elif action.action_type == ActionType.MARK_DUPLICATE:
            record.marked_duplicate = True
            record.duplicate_of = action.original_email_id
            record.status = EmailStatus.DUPLICATE
        elif action.action_type == ActionType.ESCALATE:
            record.escalated = True
            record.escalation_reason = action.reason
            record.status = EmailStatus.ROUTED
        elif action.action_type == ActionType.REQUEST_INFO:
            record.requested_info = True

        if record.category and record.priority and record.team:
            record.status = EmailStatus.COMPLETE

    def _handle_submit(self) -> Tuple[float, Dict[str, Any]]:
        processed = self._count_processed()
        total = len(self.inbox_ids)
        coverage_ratio = processed / total if total else 0.0
        unprocessed = total - processed

        final_score = self.current_task.grade(self.records, self._task_truth(), self.inbox_ids, self.sender_tiers)
        completion = self.reward_calculator.completion_adjustment(
            fully_processed=(unprocessed == 0),
            unprocessed_count=unprocessed,
            coverage_ratio=coverage_ratio,
        )
        reward = final_score + completion.value

        return reward, {
            "final_score": final_score,
            "completion_adjustment": completion.components,
            "coverage_ratio": coverage_ratio,
            "processed": processed,
            "unprocessed": unprocessed,
        }

    def _task_truth(self) -> Dict[str, GroundTruthRecord]:
        return {email_id: self.ground_truth[email_id] for email_id in self.inbox_ids}

    def _count_processed(self) -> int:
        return sum(1 for record in self.records.values() if record.status == EmailStatus.COMPLETE)

    def _next_current_email(self) -> EmailObject | None:
        for email_id in self.inbox_ids:
            record = self.records[email_id]
            if record.status != EmailStatus.COMPLETE:
                return self.email_by_id[email_id]
        return None

    def _build_observation(self, last_result: str, action_valid: bool) -> TriageObservation:
        processed = self._count_processed()
        current_email = self._next_current_email() if not self.done else None
        snapshot = []
        for email_id in self.inbox_ids:
            email = self.email_by_id[email_id]
            record = self.records[email_id]
            snapshot.append(
                EmailSummary(
                    email_id=email.email_id,
                    subject=email.subject,
                    sender_tier=email.sender_tier,
                    status=record.status,
                    actions_taken=record.actions_taken,
                )
            )

        available = [a.value for a in ActionType]
        for forbidden in self.current_task.config.forbidden_actions:
            if forbidden in available:
                available.remove(forbidden)

        hint = self.current_task.get_hint() if self.current_task.config.hint_visible else None

        return TriageObservation(
            current_email=current_email,
            inbox_snapshot=snapshot,
            processed_count=processed,
            pending_count=max(0, len(self.inbox_ids) - processed),
            team_queue_depths=self.team_queue_depths,
            last_action_result=last_result,
            last_action_valid=action_valid,
            step_number=self.state_obj.step_count,
            episode_id=self.state_obj.episode_id,
            available_actions=available,
            task_id=self.current_task_id,
            hint=hint,
        )
