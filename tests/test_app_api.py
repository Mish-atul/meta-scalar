from fastapi.testclient import TestClient

from support_triage_env.server.app import app


client = TestClient(app)


def test_health_endpoints() -> None:
    root = client.get("/")
    health = client.get("/health")
    assert root.status_code == 200
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_reset_empty_body() -> None:
    """OpenEnv platform sends POST /reset with no body; must return 200."""
    reset = client.post("/reset")
    assert reset.status_code == 200
    payload = reset.json()
    assert payload["task_id"] == 1
    assert "current_email" in payload


def test_reset_step_state_contract() -> None:
    reset = client.post("/reset", json={"task_id": 1, "seed": 42})
    assert reset.status_code == 200
    payload = reset.json()
    assert payload["task_id"] == 1
    email_id = payload["current_email"]["email_id"]

    step = client.post(
        "/step",
        json={
            "action_type": "request_info",
            "email_id": email_id,
            "question": "Please share invoice details",
        },
    )
    assert step.status_code == 200
    step_payload = step.json()
    assert "observation" in step_payload
    assert "reward" in step_payload
    assert "done" in step_payload

    state = client.get("/state")
    assert state.status_code == 200
    state_payload = state.json()
    assert state_payload["task_id"] == 1
    assert state_payload["step_count"] >= 1
