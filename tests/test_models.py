from support_triage_env.models import ActionType, Category, Priority, TriageAction


def test_action_serialization_round_trip() -> None:
    action = TriageAction(
        action_type=ActionType.CLASSIFY,
        email_id="email_001",
        category=Category.BILLING,
    )
    payload = action.model_dump(mode="json")
    restored = TriageAction.model_validate(payload)
    assert restored.action_type == ActionType.CLASSIFY
    assert restored.category == Category.BILLING


def test_priority_enum_values() -> None:
    assert Priority.P1.value == "P1"
    assert Priority.P4.value == "P4"
