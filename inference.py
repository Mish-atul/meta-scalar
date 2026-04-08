from __future__ import annotations

import json
import os
import time
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from dotenv import load_dotenv

from support_triage_env.client import TriageEnv
from support_triage_env.models import ActionType, Category, Priority, Team, TriageAction

load_dotenv()


@dataclass
class RunResult:
    task_id: int
    score: float
    steps: int
    completed: bool


def _safe_json_action(text: str) -> Dict[str, Any] | None:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _heuristic_action(obs: Dict[str, Any]) -> TriageAction:
    current = obs.get("current_email")
    if current is None:
        return TriageAction(action_type=ActionType.SUBMIT_TRIAGE)

    email_id = current["email_id"]
    task_id = int(obs.get("task_id", 1))
    text = f"{current['subject']} {current['body']}".lower()

    category, team, priority = _predict_fields(text, current.get("sender_tier", "regular"))
    stage = _infer_stage(obs, email_id, task_id, priority)

    if stage == "classify":
        return TriageAction(action_type=ActionType.CLASSIFY, email_id=email_id, category=category)
    if stage == "priority":
        return TriageAction(action_type=ActionType.SET_PRIORITY, email_id=email_id, priority=priority)
    if stage == "respond":
        return TriageAction(
            action_type=ActionType.DRAFT_RESPONSE,
            email_id=email_id,
            response_text=(
                "Hello, thank you for contacting support. "
                "We understand this issue and will investigate immediately. "
                "Our team will provide an update within 1 hour with next steps. "
                "Regards, Support Team"
            ),
        )
    if stage == "route":
        return TriageAction(action_type=ActionType.ROUTE, email_id=email_id, team=team)
    if _all_emails_touched(obs):
        return TriageAction(action_type=ActionType.SUBMIT_TRIAGE)
    return TriageAction(action_type=ActionType.CLASSIFY, email_id=email_id, category=category)


def _predict_fields(text: str, sender_tier: str) -> Tuple[Category, Team, Priority]:
    if any(k in text for k in ["abuse", "harass", "profan", "threat"]):
        category = Category.ABUSE
        team = Team.ESCALATIONS
    elif any(k in text for k in ["refund", "return", "cancel", "prorated"]):
        category = Category.REFUND
        team = Team.BILLING
    elif any(k in text for k in ["charge", "invoice", "payment", "refund"]):
        category = Category.BILLING
        team = Team.BILLING
    elif any(k in text for k in ["login", "password", "account", "access"]):
        category = Category.ACCOUNT
        team = Team.ACCOUNT
    elif any(k in text for k in ["error", "crash", "bug", "not working", "outage", "2fa"]):
        category = Category.TECHNICAL
        team = Team.TECH
    elif any(k in text for k in ["feedback", "roadmap", "question"]):
        category = Category.GENERAL
        team = Team.GENERAL
    else:
        category = Category.REFUND
        team = Team.BILLING

    if category == Category.GENERAL and any(k in text for k in ["unacceptable", "profanity"]):
        team = Team.ESCALATIONS
    if "invoice" in text and category == Category.TECHNICAL:
        team = Team.BILLING
    if category == Category.BILLING and sender_tier == "vip" and "cancel" in text:
        team = Team.ACCOUNT

    if any(k in text for k in ["urgent", "outage", "legal", "critical", "production down", "unauthorized"]):
        priority = Priority.P1
    elif sender_tier == "vip":
        priority = Priority.P2
    elif any(k in text for k in ["frustrated", "angry", "cannot", "locked"]):
        priority = Priority.P2
    else:
        priority = Priority.P3

    return category, team, priority


def _infer_stage(obs: Dict[str, Any], email_id: str, task_id: int, predicted_priority: Priority) -> str:
    snapshot = obs.get("inbox_snapshot", [])
    row = next((item for item in snapshot if item.get("email_id") == email_id), None)
    actions = set(row.get("actions_taken", []) if row else [])

    if "classify" not in actions:
        return "classify"
    if "set_priority" not in actions:
        return "priority"
    if task_id == 1:
        return "done"
    if task_id == 3 and predicted_priority in {Priority.P1, Priority.P2} and "draft_response" not in actions:
        return "respond"
    if "route" not in actions:
        return "route"
    return "done"


def _all_emails_touched(obs: Dict[str, Any]) -> bool:
    snapshot = obs.get("inbox_snapshot", [])
    if not snapshot:
        return True
    for row in snapshot:
        actions = set(row.get("actions_taken", []))
        if not {"classify", "set_priority", "route"}.issubset(actions):
            return False
    return True


def wait_for_server(base_url: str, max_wait: int = 120) -> bool:
    """Poll the server health endpoint until it responds or timeout."""
    import httpx

    deadline = time.time() + max_wait
    delay = 1.0
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{base_url.rstrip('/')}/health")
                if resp.status_code == 200:
                    print(f"[INFO] Server is ready at {base_url}", flush=True)
                    return True
        except Exception:
            pass
        print(f"[INFO] Waiting for server at {base_url} (retry in {delay:.0f}s)...", flush=True)
        time.sleep(delay)
        delay = min(delay * 1.5, 10.0)
    return False


def run_task(task_id: int, seed: int = 42) -> RunResult:
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    env = TriageEnv(base_url=base_url)

    # Retry reset with backoff in case of transient connection errors
    obs = None
    last_err = None
    for attempt in range(5):
        try:
            obs = env.reset(task_id=task_id, seed=seed)
            break
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            print(f"[WARN] env.reset(task_id={task_id}) attempt {attempt+1} failed: {exc}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)

    if obs is None:
        print(f"[ERROR] env.reset(task_id={task_id}) failed after retries: {last_err}", flush=True)
        env.close()
        return RunResult(task_id=task_id, score=0.0, steps=0, completed=False)

    done = False
    steps = 0
    final_score = 0.0

    while not done and steps < 80:
        try:
            obs_dict = obs.model_dump(mode="json")
        except Exception:
            obs_dict = obs if isinstance(obs, dict) else {}

        action = _heuristic_action(obs_dict)

        if _all_emails_touched(obs_dict):
            action = TriageAction(action_type=ActionType.SUBMIT_TRIAGE)

        try:
            obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                final_score = float(info.get("final_score", reward))
        except Exception as exc:
            print(f"[WARN] env.step() failed at step {steps}: {exc}", flush=True)
            steps += 1
            break

    env.close()
    return RunResult(task_id=task_id, score=final_score, steps=steps, completed=done)


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    # Wait for the environment server container to become ready
    if not wait_for_server(base_url, max_wait=120):
        print(f"[ERROR] Server at {base_url} did not become ready in time.", flush=True)
        sys.exit(1)

    results = []
    for tid in [1, 2, 3]:
        try:
            result = run_task(tid, seed=42)
            results.append(result)
        except Exception as exc:
            print(f"[ERROR] Task {tid} raised unhandled exception: {exc}", flush=True)
            traceback.print_exc()
            results.append(RunResult(task_id=tid, score=0.0, steps=0, completed=False))

    for result in results:
        print(f"Task {result.task_id} score: {result.score:.3f} | steps={result.steps} | completed={result.completed}")


if __name__ == "__main__":
    main()
