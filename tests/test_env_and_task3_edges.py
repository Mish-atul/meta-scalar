from support_triage_env.graders.task3_grader import Task3Grader
from support_triage_env.models import (
    ActionType,
    Category,
    GroundTruthRecord,
    Priority,
    SenderTier,
    Team,
    TriageAction,
    TriageRecord,
)
from support_triage_env.server.environment import SupportTriageEnvironment


def test_task3_partial_vs_full_completion_score() -> None:
    grader = Task3Grader()
    truth = {
        "a": GroundTruthRecord(email_id="a", category=Category.BILLING, priority=Priority.P2, team=Team.BILLING),
        "b": GroundTruthRecord(email_id="b", category=Category.TECHNICAL, priority=Priority.P1, team=Team.TECH, requires_escalation=True),
    }
    sender_tiers = {"a": SenderTier.REGULAR, "b": SenderTier.VIP}

    full = {
        "a": TriageRecord(
            email_id="a",
            category=Category.BILLING,
            priority=Priority.P2,
            team=Team.BILLING,
            response_text="Hello, we are reviewing your billing issue and will update within 1 hour. Regards",
        ),
        "b": TriageRecord(
            email_id="b",
            category=Category.TECHNICAL,
            priority=Priority.P1,
            team=Team.TECH,
            response_text="Hello, this is urgent and we are escalating immediately. Regards",
            escalated=True,
        ),
    }

    partial = {
        "a": TriageRecord(email_id="a", category=Category.BILLING),
        "b": TriageRecord(email_id="b", category=Category.GENERAL, priority=Priority.P4),
    }

    score_full = grader.score(full, truth, ["a", "b"], sender_tiers)
    score_partial = grader.score(partial, truth, ["a", "b"], sender_tiers)
    assert score_full > score_partial


def test_environment_max_step_auto_termination_has_final_score() -> None:
    env = SupportTriageEnvironment()
    obs = env.reset(task_id=1, seed=42)
    email_id = obs.current_email.email_id

    info = {}
    done = False
    for _ in range(3):
        obs, _, done, info = env.step(
            TriageAction(action_type=ActionType.REQUEST_INFO, email_id=email_id, question="Need details")
        )

    assert done is True
    assert info.get("auto_terminated") is True
    assert "final_score" in info


def test_environment_early_submit_with_zero_processed_penalty() -> None:
    env = SupportTriageEnvironment()
    env.reset(task_id=3, seed=42)
    _, reward, done, info = env.step(TriageAction(action_type=ActionType.SUBMIT_TRIAGE))

    assert done is True
    assert info["processed"] == 0
    assert info["unprocessed"] > 0
    assert reward < 0
