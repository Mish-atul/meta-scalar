# requirements.md — SupportTriageEnv

## 1. Project Overview

**Environment Name:** SupportTriageEnv  
**Domain:** Customer Support Email Triage  
**Hackathon:** Meta AI × Scaler OpenEnv Hackathon, Round 1  
**Deadline:** 8 April 2026, 11:59 PM  
**Team:** Atul Kumar Mishra, Sarthak Lakhotia (lead), Tanishk Bhanage  

### What Problem Does This Solve?

Every customer-facing company — from SaaS startups to enterprise firms — receives hundreds to thousands
of support emails per day. A human support manager must:

1. **Classify** each email by issue type (billing, technical, account, refund, general, abuse, etc.)
2. **Prioritize** each ticket (P1 Critical → P4 Low) based on urgency, customer tier, SLA
3. **Route** each ticket to the correct team (billing-team, tech-support, account-management, escalations)
4. **Draft** an initial acknowledgment response
5. **Flag** duplicates, escalations, and VIP customers

This is currently done manually or with brittle rule-based systems. A well-trained RL agent could
dramatically reduce response times and improve customer satisfaction scores (CSAT).

SupportTriageEnv provides a sandboxed, reproducible OpenEnv environment where agents can be trained
and evaluated on this exact task.

---

## 2. Functional Requirements

### FR-01: OpenEnv Spec Compliance

| ID      | Requirement                                                                                          | Priority |
|---------|------------------------------------------------------------------------------------------------------|----------|
| FR-01-1 | Implement `reset()` → returns `TriageObservation` representing a fresh inbox                        | MUST     |
| FR-01-2 | Implement `step(action: TriageAction)` → returns `(TriageObservation, float, bool, dict)`           | MUST     |
| FR-01-3 | Implement `state()` → returns `State(episode_id, step_count, ...)`                                  | MUST     |
| FR-01-4 | All models (`TriageAction`, `TriageObservation`, `TriageState`, `TriageReward`) are Pydantic models  | MUST     |
| FR-01-5 | `openenv.yaml` manifest exists at env root with required fields                                      | MUST     |
| FR-01-6 | `openenv validate` command passes without errors                                                     | MUST     |

### FR-02: Email Dataset

| ID      | Requirement                                                                                          | Priority |
|---------|------------------------------------------------------------------------------------------------------|----------|
| FR-02-1 | Synthetic email dataset of 200+ unique emails across all categories                                  | MUST     |
| FR-02-2 | Each email has: id, subject, body, sender_email, sender_tier, timestamp, thread_history (optional)  | MUST     |
| FR-02-3 | Ground truth labels stored separately in `data/ground_truth.json`                                   | MUST     |
| FR-02-4 | Dataset covers all 6 categories: billing, technical, account, refund, abuse, general                | MUST     |
| FR-02-5 | Dataset includes VIP, regular, and first-time customer tiers                                        | MUST     |
| FR-02-6 | Dataset includes edge cases: ambiguous emails, multi-issue emails, duplicate tickets, angry tone     | SHOULD   |
| FR-02-7 | Dataset includes SLA breach risk scenarios (old tickets, explicitly urgent language)                 | SHOULD   |

### FR-03: Action Space

The agent may take exactly ONE of the following actions per step:

| Action Type         | Parameters Required                              | Description                                      |
|---------------------|--------------------------------------------------|--------------------------------------------------|
| `classify`          | `email_id`, `category`                           | Assign an issue category to the email            |
| `set_priority`      | `email_id`, `priority` (P1/P2/P3/P4)            | Assign urgency level                             |
| `route`             | `email_id`, `team`                               | Assign to a support team                         |
| `draft_response`    | `email_id`, `response_text`                      | Write an initial response to the customer        |
| `mark_duplicate`    | `email_id`, `original_email_id`                  | Flag as duplicate of an existing ticket          |
| `escalate`          | `email_id`, `reason`                             | Escalate to senior team / management             |
| `request_info`      | `email_id`, `question`                           | Ask customer for more info before processing     |
| `submit_triage`     | None                                             | Signal that triage session is complete (ends ep) |

### FR-04: Observation Space

Each observation returned by `reset()` or `step()` MUST include:

| Field                  | Type              | Description                                             |
|------------------------|-------------------|---------------------------------------------------------|
| `current_email`        | `EmailObject`     | The email the agent should act on next                  |
| `inbox_snapshot`       | `List[EmailSummary]` | Subject + ID + status for all emails in the inbox    |
| `processed_count`      | `int`             | Number of emails fully triaged so far                   |
| `pending_count`        | `int`             | Number of emails not yet fully triaged                  |
| `team_queue_depths`    | `Dict[str, int]`  | Current load per team (e.g., `{"tech-support": 12}`)   |
| `last_action_result`   | `str`             | Human-readable result of the last action taken          |
| `step_number`          | `int`             | Current step within episode                             |
| `episode_id`           | `str`             | UUID of current episode                                 |
| `available_actions`    | `List[str]`       | List of valid action types for the current step         |

### FR-05: Tasks

Three tasks MUST be implemented, each with a deterministic programmatic grader:

#### Task 1 — Single-Email Classification (Easy)
- **Objective:** Correctly classify ONE email into the right category
- **Episode length:** Max 3 steps (classify + optionally set_priority + submit)
- **Grader:** Returns 0.0–1.0 based on classification accuracy
- **Expected baseline score:** 0.70–0.85 for GPT-4o level model
- **Full spec:** See `tasks_and_graders.md`

#### Task 2 — Batch Triage: Priority + Routing (Medium)
- **Objective:** For a batch of 5 emails, assign correct priority AND route to correct team
- **Episode length:** Max 15 steps
- **Grader:** Returns 0.0–1.0 based on F1 score across both priority and routing
- **Expected baseline score:** 0.45–0.65 for GPT-4o level model
- **Full spec:** See `tasks_and_graders.md`

#### Task 3 — Full Pipeline with SLA + Draft Response (Hard)
- **Objective:** Full triage of 8–10 emails including VIPs, duplicates, SLA breaches, draft responses
- **Episode length:** Max 40 steps
- **Grader:** Composite score across all sub-tasks
- **Expected baseline score:** 0.25–0.45 for GPT-4o level model
- **Full spec:** See `tasks_and_graders.md`

### FR-06: Reward Function

| ID      | Requirement                                                                                               | Priority |
|---------|-----------------------------------------------------------------------------------------------------------|----------|
| FR-06-1 | Reward is computed at EVERY step, not only at episode end                                                 | MUST     |
| FR-06-2 | Correct `classify` action yields positive reward (+0.20 to +0.35 depending on difficulty)                | MUST     |
| FR-06-3 | Correct `set_priority` action yields positive reward (+0.15 to +0.25)                                    | MUST     |
| FR-06-4 | Correct `route` action yields positive reward (+0.15 to +0.20)                                           | MUST     |
| FR-06-5 | Quality `draft_response` yields positive reward (+0.10 to +0.20)                                         | MUST     |
| FR-06-6 | Misclassifying a VIP customer as low priority yields -0.30 penalty                                       | MUST     |
| FR-06-7 | Submitting triage with unmissed/unprocessed SLA-critical emails yields -0.20 penalty each                | MUST     |
| FR-06-8 | Repeated action on already-processed email yields -0.05 penalty                                          | MUST     |
| FR-06-9 | `submit_triage` with all emails processed yields completion bonus of +0.25                               | SHOULD   |
| FR-06-10| Full spec in `reward_design.md`                                                                           | —        |

### FR-07: Baseline Inference Script

| ID      | Requirement                                                                                          | Priority |
|---------|------------------------------------------------------------------------------------------------------|----------|
| FR-07-1 | Script named `inference.py` in project root                                                          | MUST     |
| FR-07-2 | Uses `openai.OpenAI` client with `base_url=API_BASE_URL`, `api_key=HF_TOKEN`                         | MUST     |
| FR-07-3 | Reads `API_BASE_URL`, `HF_TOKEN`, `MODEL_NAME` from environment variables                            | MUST     |
| FR-07-4 | Runs each of the 3 tasks and prints a score for each                                                 | MUST     |
| FR-07-5 | Total runtime under 20 minutes on 2 vCPU / 8 GB RAM machine                                         | MUST     |
| FR-07-6 | Produces deterministic, reproducible scores (fixed random seed, fixed test episodes)                 | MUST     |
| FR-07-7 | Script handles LLM errors gracefully (timeouts, bad JSON) and continues                              | SHOULD   |

---

## 3. Non-Functional Requirements

### NFR-01: Deployment

| ID       | Requirement                                                                                          |
|----------|------------------------------------------------------------------------------------------------------|
| NFR-01-1 | Deploys to a Hugging Face Space tagged with `openenv`                                                |
| NFR-01-2 | Space URL responds with HTTP 200 to a `GET /` ping                                                   |
| NFR-01-3 | `POST /reset` returns a valid `TriageObservation` JSON                                               |
| NFR-01-4 | `POST /step` accepts a `TriageAction` JSON and returns step result                                   |
| NFR-01-5 | `GET /state` returns current `State` JSON                                                            |

### NFR-02: Containerization

| ID       | Requirement                                                                            |
|----------|----------------------------------------------------------------------------------------|
| NFR-02-1 | `Dockerfile` at `server/Dockerfile`                                                    |
| NFR-02-2 | `docker build -t support-triage-env ./server` succeeds                                |
| NFR-02-3 | `docker run -p 8000:8000 support-triage-env` starts cleanly                           |
| NFR-02-4 | Container runs on 2 vCPU, 8 GB RAM without OOM                                        |
| NFR-02-5 | Container starts in under 30 seconds                                                   |

### NFR-03: Performance

| ID       | Requirement                                                                            |
|----------|----------------------------------------------------------------------------------------|
| NFR-03-1 | `reset()` completes in under 200ms                                                     |
| NFR-03-2 | `step()` (excluding LLM call) completes in under 100ms                                 |
| NFR-03-3 | Graders are deterministic and complete in under 50ms                                   |
| NFR-03-4 | Environment supports at least 1 concurrent session (more is a bonus)                  |

### NFR-04: Code Quality

| ID       | Requirement                                                                            |
|----------|----------------------------------------------------------------------------------------|
| NFR-04-1 | All Python code passes `ruff` or `flake8` linting                                     |
| NFR-04-2 | All Pydantic models have docstrings and field descriptions                             |
| NFR-04-3 | All public functions have type annotations                                             |
| NFR-04-4 | Test coverage for graders and reward function ≥ 80%                                   |
| NFR-04-5 | `pytest tests/` passes with zero failures                                              |

---

## 4. Out of Scope

- **No real email APIs** — all data is synthetic; no Gmail/Outlook integration in the env itself
- **No persistent storage** — environment is stateless across episodes (state lives in RAM)
- **No user authentication** — the HF Space is public
- **No multi-agent scenarios** — one agent per episode
- **No GUI frontend** — optional Gradio web interface is a nice-to-have but not required for judging

---

## 5. Success Criteria

The submission is successful if:

1. **Automated gate passes**: HF Space deploys, openenv validate passes, Dockerfile builds, inference.py runs
2. **Baseline scores are non-trivial**: Task 1 ≥ 0.50, Task 2 ≥ 0.30, Task 3 ≥ 0.15
3. **Task 3 challenges frontier models**: A GPT-4o level model scores < 0.50 on Task 3
4. **Human reviewers find it useful**: "Would I use this to train/evaluate a support triage agent?" = YES

---

## 6. Dependencies

### Runtime (server)
```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
openenv-core>=0.2.1
python-dotenv>=1.0.0
```

### Inference script (client-side)
```
openai>=1.0.0
openenv-core>=0.2.1
python-dotenv>=1.0.0
```

### Development only
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
ruff>=0.1.0
httpx>=0.25.0   # for testing FastAPI
```
