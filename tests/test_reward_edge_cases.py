from support_triage_env.models import (
    ActionType,
    Category,
    GroundTruthRecord,
    Priority,
    SenderTier,
    Team,
    TriageAction,
)
from support_triage_env.rewards.reward_calculator import RewardCalculator


def _truth(difficulty: int = 2) -> GroundTruthRecord:
    return GroundTruthRecord(
        email_id="email_001",
        category=Category.BILLING,
        priority=Priority.P2,
        team=Team.BILLING,
        difficulty_level=difficulty,
        requires_escalation=False,
    )


def test_request_info_helpful_then_overuse_penalty() -> None:
    rc = RewardCalculator()
    action = TriageAction(
        action_type=ActionType.REQUEST_INFO,
        email_id="email_001",
        question="Can you share invoice ID?",
    )

    helpful = rc.compute_step_reward(3, action, None, _truth(3), SenderTier.REGULAR, False, False, True, False)
    overuse = rc.compute_step_reward(3, action, None, _truth(3), SenderTier.REGULAR, False, False, True, False)

    assert helpful.value == 0.05
    assert overuse.value == -0.10


def test_request_info_unnecessary_penalty() -> None:
    rc = RewardCalculator()
    action = TriageAction(
        action_type=ActionType.REQUEST_INFO,
        email_id="email_009",
        question="Need more details?",
    )
    res = rc.compute_step_reward(2, action, None, _truth(1), SenderTier.REGULAR, False, False, False, True)
    assert res.value == -0.05


def test_early_submit_penalty_with_zero_processed_and_cap() -> None:
    rc = RewardCalculator()
    comp = rc.completion_adjustment(fully_processed=False, unprocessed_count=12, coverage_ratio=0.0)
    assert comp.value == -0.5
    assert comp.components["early_submit_penalty"] == -0.5


def test_completion_bonus_full_processed() -> None:
    rc = RewardCalculator()
    comp = rc.completion_adjustment(fully_processed=True, unprocessed_count=0, coverage_ratio=1.0)
    assert comp.value == 0.25
    assert comp.components["completion_bonus"] == 0.25


def test_vip_misprioritization_penalty_on_task3() -> None:
    rc = RewardCalculator()
    action = TriageAction(action_type=ActionType.SET_PRIORITY, email_id="email_001", priority=Priority.P4)
    res = rc.compute_step_reward(3, action, None, _truth(), SenderTier.VIP, False, False, False, False)
    assert res.value == -0.20
