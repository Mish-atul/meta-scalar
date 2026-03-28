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


def _truth() -> GroundTruthRecord:
    return GroundTruthRecord(
        email_id="email_001",
        category=Category.BILLING,
        priority=Priority.P2,
        team=Team.BILLING,
        difficulty_level=2,
    )


def test_classification_reward_positive() -> None:
    rc = RewardCalculator()
    action = TriageAction(action_type=ActionType.CLASSIFY, email_id="email_001", category=Category.BILLING)
    res = rc.compute_step_reward(1, action, None, _truth(), SenderTier.REGULAR, False, False, False, False)
    assert res.value > 0


def test_invalid_action_penalty() -> None:
    rc = RewardCalculator()
    action = TriageAction(action_type=ActionType.CLASSIFY, email_id="email_001", category=Category.BILLING)
    res = rc.compute_step_reward(1, action, None, _truth(), SenderTier.REGULAR, False, True, False, False)
    assert res.value == -0.05


def test_vip_penalty_task2_and_3_only() -> None:
    rc = RewardCalculator()
    action = TriageAction(action_type=ActionType.SET_PRIORITY, email_id="email_001", priority=Priority.P4)
    t2 = rc.compute_step_reward(2, action, None, _truth(), SenderTier.VIP, False, False, False, False)
    t1 = rc.compute_step_reward(1, action, None, _truth(), SenderTier.VIP, False, False, False, False)
    assert t2.value < 0
    assert t1.value >= 0
