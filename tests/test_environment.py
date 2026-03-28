from support_triage_env.models import ActionType, TriageAction
from support_triage_env.server.environment import SupportTriageEnvironment


def test_reset_returns_observation() -> None:
    env = SupportTriageEnvironment()
    obs = env.reset(task_id=1, seed=42)
    assert obs.current_email is not None
    assert obs.task_id == 1


def test_step_and_state_progress() -> None:
    env = SupportTriageEnvironment()
    obs = env.reset(task_id=1, seed=42)
    email_id = obs.current_email.email_id
    action = TriageAction(action_type=ActionType.REQUEST_INFO, email_id=email_id, question="Can you share invoice ID?")
    _, _, done, _ = env.step(action)
    state = env.state()
    assert state.step_count == 1
    assert done is False
