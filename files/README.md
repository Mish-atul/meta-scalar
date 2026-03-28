---
title: Support Triage Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
  - real-world
---

# SupportTriageEnv 📧

> A production-grade OpenEnv environment for training and evaluating AI agents on **real-world customer support email triage**.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.x-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![Validate Pipeline](../actions/workflows/validate-pipeline.yml/badge.svg)](../actions/workflows/validate-pipeline.yml)

---

## What Is This?

SupportTriageEnv simulates the **customer support triage workflow** that every customer-facing company
performs manually every day:

1. **Classify** incoming support emails by issue type
2. **Prioritize** tickets based on urgency, SLA, and customer tier
3. **Route** tickets to the correct support team
4. **Draft** initial acknowledgment responses for high-priority tickets
5. **Detect** duplicates and escalate critical issues

This is a genuine real-world task. A well-trained agent could dramatically reduce first-response
times and improve customer satisfaction scores (CSAT) at scale.

---

## Environment Description

**Domain:** Customer Support Operations  
**Task Type:** Sequential Decision Making over a structured inbox  
**Episode:** One triage session = one inbox of N emails to process

### Action Space

| Action Type       | Parameters                       | Description                          |
|-------------------|----------------------------------|--------------------------------------|
| `classify`        | `email_id`, `category`           | Assign issue category to email       |
| `set_priority`    | `email_id`, `priority`           | Assign P1–P4 urgency level           |
| `route`           | `email_id`, `team`               | Route to support team                |
| `draft_response`  | `email_id`, `response_text`      | Write initial customer response      |
| `mark_duplicate`  | `email_id`, `original_email_id`  | Flag as duplicate of existing ticket |
| `escalate`        | `email_id`, `reason`             | Escalate to senior team              |
| `request_info`    | `email_id`, `question`           | Ask customer for more details        |
| `submit_triage`   | —                                | Complete triage session (ends ep)    |

**Categories:** billing, technical, account, refund, abuse, general  
**Priorities:** P1 (≤1hr SLA), P2 (≤4hr), P3 (≤24hr), P4 (≤72hr)  
**Teams:** billing-team, tech-support, account-management, escalations, general-support

### Observation Space

Each observation includes:
- `current_email` — full email content (subject, body, sender, tier, thread history)
- `inbox_snapshot` — all emails with current processing status
- `team_queue_depths` — current load per support team
- `last_action_result` — feedback from the last action
- `available_actions` — valid actions for this step

---

## Tasks

### Task 1: Single-Email Classification (Easy)
**Objective:** Correctly classify and prioritize one unambiguous email  
**Max steps:** 3  
**Baseline score (GPT-4o):** ~0.78

### Task 2: Batch Priority + Routing (Medium)
**Objective:** Classify, prioritize, and route 5 emails including edge cases  
**Max steps:** 15  
**Baseline score (GPT-4o):** ~0.52

### Task 3: Full Pipeline with SLA + Drafts (Hard)
**Objective:** Full triage of 8–10 emails with VIPs, duplicates, escalations, and draft responses  
**Max steps:** 40  
**Baseline score (GPT-4o):** ~0.31

---

## Reward Function

The reward is **non-sparse** — the agent receives signal at every step:

| Action                     | Reward           |
|----------------------------|------------------|
| Correct classification     | +0.25 to +0.35   |
| Correct priority           | +0.20 to +0.25   |
| Correct routing            | +0.20            |
| Quality draft response     | +0.10 to +0.20   |
| Correct escalation         | +0.15            |
| Duplicate detection        | +0.20            |
| VIP assigned P3/P4         | **-0.30**        |
| False escalation           | -0.10            |
| Invalid/repeat action      | -0.05            |
| Clean completion bonus     | +0.25            |

---

## Setup and Usage

### Local Development (Exact Commands)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m support_triage_env.data.generate_dataset
```

Start API locally:

```bash
python -m uvicorn support_triage_env.server.app:app --host 127.0.0.1 --port 8000
```

Run baseline inference locally:

```bash
set API_BASE_URL=http://127.0.0.1:8000
python inference.py
```

### Docker Build and Runtime Validation

Build image:

```bash
docker build -t support-triage-env -f support_triage_env/server/Dockerfile .
```

Run container:

```bash
docker run --name support-triage-env-test -p 8010:8000 support-triage-env
```

Validate endpoints:

```bash
curl http://127.0.0.1:8010/health
curl -X POST http://127.0.0.1:8010/reset -H "Content-Type: application/json" -d "{\"task_id\":1,\"seed\":42}"
curl -X POST http://127.0.0.1:8010/step -H "Content-Type: application/json" -d "{\"action_type\":\"request_info\",\"email_id\":\"email_001\",\"question\":\"Please share invoice details\"}"
```

Stop and remove container:

```bash
docker rm -f support-triage-env-test
```

### Install and Connect (Hosted Space)

```bash
pip install openenv-core git+https://huggingface.co/spaces/YOUR_USERNAME/support-triage-env
```

```python
from support_triage_env import TriageEnv, TriageAction
from support_triage_env.models import ActionType, Category, Priority, Team

# Connect to the HF Space
with TriageEnv(base_url="https://YOUR_USERNAME-support-triage-env.hf.space").sync() as env:
    obs = env.reset()
    print(f"Inbox: {obs.pending_count} emails to triage")

    # Classify the first email
    result = env.step(TriageAction(
        action_type=ActionType.CLASSIFY,
        email_id=obs.current_email.email_id,
        category=Category.BILLING,
    ))
    print(f"Result: {result.observation.last_action_result}")
    print(f"Reward: {result.reward}")
```

### Run Locally with Docker

```bash
docker run -p 8000:8000 registry.hf.space/YOUR_USERNAME-support-triage-env:latest
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_xxxx"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export ENV_BASE_URL="https://YOUR_USERNAME-support-triage-env.hf.space"

python inference.py
```

## One-Command Validation Pipeline

Run the full judge-style validation flow with a single command:

```bash
python scripts/validate_pipeline.py
```

Windows PowerShell shortcut:

```powershell
.\validate.ps1
```

This runs, in order:

- test suite + coverage report
- Docker build
- container smoke tests (`/health`, `/reset`, `/step`)
- `openenv validate`

The pipeline is fail-fast and exits non-zero on any failed gate.

## Scoring and Evaluation Logic

Task-level graders are deterministic and return values in [0, 1].

- Task 1: category + priority correctness with partial credit for adjacent mistakes
- Task 2: per-email score (classification, priority, routing) multiplied by completion ratio
- Task 3: weighted composite:
  - Classification F1: 20%
  - Priority accuracy: 20%
  - Routing accuracy: 20%
  - Response quality (deterministic rubric): 20%
  - Escalation recall: 10%
  - Duplicate precision: 10%

Task 3 penalties are applied after component scoring and then clamped to [0, 1]:

- P1 without draft response: -0.15 each
- VIP downgraded to P3/P4: -0.20 each
- Incomplete triage records (missing classify/priority/route): -0.10 each
- Missed required escalations: -0.05 each
- Missed required duplicates: -0.05 each

Step rewards are non-sparse and include custom handling for request_info:

- +0.05 when ambiguity is real
- 0 when optional
- -0.05 when unnecessary
- -0.10 when repeated on same email

---

## Baseline Scores

Current deterministic baseline run in this repository (local API, fixed seed):

| Task   | Name                     | Score |
|--------|--------------------------|-------|
| Task 1 | Single Classification    | 0.75  |
| Task 2 | Batch Triage             | 0.80  |
| Task 3 | Full Pipeline            | 0.33  |

Hackathon minimum success gates are still satisfied:

- Task 1 >= 0.50
- Task 2 >= 0.30
- Task 3 >= 0.15

---

## Why This Environment Matters

Real customer support teams at companies like Stripe, Shopify, and Zendesk route
**tens of thousands of tickets daily**. Even a 10% improvement in triage accuracy
translates to:
- Faster resolution times → higher CSAT scores
- Fewer misdirected tickets → lower support costs
- Better SLA compliance → stronger enterprise contracts

SupportTriageEnv provides a realistic, safe sandbox to train and evaluate agents
for this critical business function.

---

## OpenEnv Spec Compliance

```bash
openenv validate  # ✅ Passes
```

- ✅ Typed Pydantic models (TriageAction, TriageObservation, TriageState)
- ✅ `reset()`, `step()`, `state()` implemented
- ✅ `openenv.yaml` manifest present
- ✅ 3 tasks with programmatic graders (scores 0.0–1.0)
- ✅ Non-sparse reward function
- ✅ Baseline inference script (`inference.py`)
- ✅ Working Dockerfile

---

## Authors

Built for the Meta AI × Scaler OpenEnv Hackathon 2026.

Team: Atul Kumar Mishra, Sarthak Lakhotia, Tanishk Bhanage
