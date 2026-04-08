from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from support_triage_env.models import ActionType, Category, Priority, Team, TriageAction
from support_triage_env.server.environment import SupportTriageEnvironment

load_dotenv()

BENCHMARK = "support-triage-env"
MAX_STEPS = 80
SUCCESS_THRESHOLD = {1: 0.50, 2: 0.30, 3: 0.15}


@dataclass
class RunResult:
    task_id: int
    score: float
    steps: int
    completed: bool


def log_start(task: int, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: TriageAction, reward: float, done: bool, error: str | None) -> None:
    action_payload = json.dumps(action.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=True)
    done_value = "true" if done else "false"
    error_value = "null" if error is None else json.dumps(error, ensure_ascii=True)
    print(
        f"[STEP] step={step} action={action_payload} reward={reward:.6f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_value = "true" if success else "false"
    rewards_payload = json.dumps([round(v, 6) for v in rewards], separators=(",", ":"), ensure_ascii=True)
    print(
        f"[END] success={success_value} steps={steps} score={score:.6f} rewards={rewards_payload}",
        flush=True,
    )


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


def _llm_action(client: OpenAI, model: str, obs: Dict[str, Any]) -> TriageAction | None:
    prompt = (
        "You are a support triage agent. Return exactly one JSON object with fields for a single action. "
        "Allowed action_type values: classify,set_priority,route,draft_response,mark_duplicate,escalate,request_info,submit_triage. "
        "Observation: " + json.dumps(obs)
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        payload = _safe_json_action(completion.choices[0].message.content or "")
        if not payload:
            return None
        return TriageAction.model_validate(payload)
    except Exception:
        return None


def run_task(task_id: int, seed: int = 42) -> RunResult:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN")
    api_base = os.getenv("API_BASE_URL", "")

    # Use the environment directly in-process — no HTTP server needed.
    env = SupportTriageEnvironment()
    llm_client = OpenAI(base_url=api_base, api_key=hf_token) if (hf_token and api_base) else None

    done = False
    steps = 0
    final_score = 0.0
    success = False
    rewards: list[float] = []

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    try:
        obs = env.reset(task_id=task_id, seed=seed)

        while not done and steps < MAX_STEPS:
            obs_dict = obs.model_dump(mode="json")
            action = _llm_action(llm_client, model_name, obs_dict) if llm_client else None
            if action is None:
                action = _heuristic_action(obs_dict)

            if _all_emails_touched(obs_dict):
                action = TriageAction(action_type=ActionType.SUBMIT_TRIAGE)

            step_error: str | None = None
            try:
                obs, reward, done, info = env.step(action)
            except Exception as exc:
                reward = 0.0
                done = True
                info = {}
                step_error = str(exc)

            steps += 1
            rewards.append(float(reward))
            log_step(step=steps, action=action, reward=float(reward), done=done, error=step_error)

            if done:
                final_score = float(info.get("final_score", reward))

        threshold = SUCCESS_THRESHOLD.get(task_id, 0.50)
        success = final_score >= threshold
    except Exception as exc:
        print(f"[ERROR] task={task_id} exception={exc}", flush=True)
    finally:
        log_end(success=success, steps=steps, score=final_score, rewards=rewards)

    return RunResult(task_id=task_id, score=final_score, steps=steps, completed=done)


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    results = [run_task(1, seed=42), run_task(2, seed=42), run_task(3, seed=42)]
    for result in results:
        print(
            f"Task {result.task_id} score: {result.score:.3f} | "
            f"steps={result.steps} | completed={result.completed}",
            flush=True,
        )


def _all_emails_touched(obs: Dict[str, Any]) -> bool:
    snapshot = obs.get("inbox_snapshot", [])
    if not snapshot:
        return True
    for row in snapshot:
        actions = set(row.get("actions_taken", []))
        if not {"classify", "set_priority", "route"}.issubset(actions):
            return False
    return True


if __name__ == "__main__":
    main()
