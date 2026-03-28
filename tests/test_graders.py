from support_triage_env.graders.task1_grader import Task1Grader
from support_triage_env.models import Category, GroundTruthRecord, Priority, Team


def test_task1_grader_exact_match() -> None:
    grader = Task1Grader()
    gt = GroundTruthRecord(
        email_id="email_001",
        category=Category.BILLING,
        priority=Priority.P2,
        team=Team.BILLING,
    )
    score = grader.score(Category.BILLING, Priority.P2, gt)
    assert score == 1.0


def test_task1_grader_deterministic() -> None:
    grader = Task1Grader()
    gt = GroundTruthRecord(
        email_id="email_001",
        category=Category.TECHNICAL,
        priority=Priority.P1,
        team=Team.TECH,
    )
    first = grader.score(Category.GENERAL, Priority.P2, gt)
    second = grader.score(Category.GENERAL, Priority.P2, gt)
    assert first == second
