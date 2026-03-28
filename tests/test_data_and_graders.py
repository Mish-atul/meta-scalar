from support_triage_env.data.generate_dataset import generate
from support_triage_env.graders.common import (
    ClassificationMetrics,
    DuplicateGrader,
    EscalationGrader,
    PriorityMetrics,
    ResponseGrader,
    RoutingMetrics,
)
from support_triage_env.models import (
    Category,
    GroundTruthRecord,
    Priority,
    SenderTier,
    Team,
    TriageRecord,
)


def test_dataset_generation_integrity() -> None:
    emails, truth = generate()
    assert len(emails) >= 200
    assert len(truth) == len(emails)

    categories = {item["category"] for item in truth.values()}
    assert categories == {"billing", "technical", "account", "refund", "abuse", "general"}

    vip_ids = [email["email_id"] for email in emails if email["sender_tier"] == "vip"]
    for email_id in vip_ids:
        assert truth[email_id]["priority"] in {"P1", "P2"}


def test_common_graders_and_metrics_paths() -> None:
    truth = {
        "a": GroundTruthRecord(email_id="a", category=Category.BILLING, priority=Priority.P2, team=Team.BILLING),
        "b": GroundTruthRecord(email_id="b", category=Category.TECHNICAL, priority=Priority.P1, team=Team.TECH, requires_escalation=True),
        "c": GroundTruthRecord(email_id="c", category=Category.BILLING, priority=Priority.P2, team=Team.BILLING, is_duplicate=True, original_email_id="a"),
    }
    records = {
        "a": TriageRecord(email_id="a", category=Category.BILLING, priority=Priority.P2, team=Team.BILLING),
        "b": TriageRecord(email_id="b", category=Category.GENERAL, priority=Priority.P2, team=Team.TECH, escalated=True),
        "c": TriageRecord(email_id="c", category=Category.BILLING, priority=Priority.P2, team=Team.BILLING, marked_duplicate=True, duplicate_of="a"),
    }
    ids = ["a", "b", "c"]

    assert 0.0 <= ClassificationMetrics.f1_score(records, truth, ids) <= 1.0
    assert 0.0 <= PriorityMetrics.weighted_score(records, truth, ids) <= 1.0
    assert 0.0 <= RoutingMetrics.accuracy(records, truth, ids) <= 1.0
    assert 0.0 <= EscalationGrader.recall(records, truth, ids) <= 1.0
    assert 0.0 <= DuplicateGrader.precision(records, truth, ids) <= 1.0

    response_grader = ResponseGrader()
    response = "Hello, thank you for contacting support. We will investigate this billing charge and update within 1 hour. Regards"
    score = response_grader.quality_score(response, truth["a"])
    assert 0.0 <= score <= 1.0


def test_escalation_and_duplicate_empty_predictions() -> None:
    truth = {
        "x": GroundTruthRecord(
            email_id="x",
            category=Category.ABUSE,
            priority=Priority.P1,
            team=Team.ESCALATIONS,
            requires_escalation=True,
            is_duplicate=True,
            original_email_id="root",
        )
    }
    records = {}
    assert EscalationGrader.recall(records, truth, ["x"]) == 0.0
    assert DuplicateGrader.precision(records, truth, ["x"]) == 1.0
