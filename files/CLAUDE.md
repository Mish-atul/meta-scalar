# CLAUDE.md — Project Context for Claude Code

## What This Project Is

This is a submission for the **Meta AI × Scaler OpenEnv Hackathon (Round 1, March–April 2026)**.

We are building **SupportTriageEnv** — a production-grade OpenEnv environment that trains and evaluates
AI agents on real-world **customer support email triage**: classification, prioritization, team routing,
and response drafting.

This is not a toy or game. It models a task that every customer-facing company does manually every day.
The environment is built to the OpenEnv 0.2.x spec (meta-pytorch/OpenEnv), deployed on Hugging Face Spaces,
and scored by an automated judge.

---

## Repository Layout (Final Target Structure)

```
support_triage_env/
├── CLAUDE.md                    ← You are here
├── README.md                    ← HF Space README (judged!)
├── openenv.yaml                 ← Environment manifest (judged!)
├── inference.py                 ← Baseline inference script (MUST be in root, MUST be named this)
├── pyproject.toml               ← Package deps
├── .env.example                 ← Sample env vars
├── .dockerignore
├── __init__.py                  ← Exports TriageAction, TriageObservation, TriageEnv
├── models.py                    ← Pydantic models: Action, Observation, State, Reward
├── client.py                    ← TriageEnv(EnvClient) — client-side class
│
├── data/
│   ├── emails.json              ← Synthetic email dataset (200+ emails, all tasks)
│   └── ground_truth.json        ← Ground truth labels per email_id
│
├── tasks/
│   ├── __init__.py
│   ├── base_task.py             ← BaseTask abstract class
│   ├── task1_classify.py        ← Task 1: Single-email classification (Easy)
│   ├── task2_batch_triage.py    ← Task 2: Batch priority + routing (Medium)
│   └── task3_full_pipeline.py   ← Task 3: Full pipeline with SLA + draft (Hard)
│
├── graders/
│   ├── __init__.py
│   ├── classification_grader.py ← Exact + partial match scoring
│   ├── priority_grader.py       ← Priority accuracy with severity weighting
│   ├── routing_grader.py        ← Team routing accuracy
│   └── response_grader.py       ← Programmatic response quality check
│
├── rewards/
│   ├── __init__.py
│   └── reward_calculator.py     ← Composite reward function (non-sparse!)
│
├── server/
│   ├── app.py                   ← FastAPI app entry point
│   ├── environment.py           ← SupportTriageEnvironment(Environment)
│   ├── requirements.txt         ← Server-side Python deps
│   └── Dockerfile               ← Container definition
│
├── tests/
│   ├── test_environment.py      ← reset/step/state smoke tests
│   ├── test_graders.py          ← Grader unit tests
│   └── test_reward.py           ← Reward function unit tests
│
└── outputs/                     ← Gitignored runtime output
    ├── logs/
    └── evals/
```

---

## Critical Rules — Read Before Writing Any Code

### 1. OpenEnv Spec Compliance (15% of score — but DISQUALIFYING if missed)
- All models MUST be Pydantic `BaseModel` subclasses
- `reset()` → returns `TriageObservation`
- `step(action: TriageAction)` → returns `(TriageObservation, float, bool, dict)`
- `state()` → returns current `State` object with `episode_id` and `step_count`
- `openenv.yaml` must exist at the root of the environment directory
- Validate with `openenv validate` before submitting

### 2. The Inference Script
- **MUST be named `inference.py`**
- **MUST be in the root directory of the project**
- MUST use `OpenAI` client (not Anthropic, not raw requests)
- MUST read credentials from env vars: `API_BASE_URL`, `HF_TOKEN`, `MODEL_NAME`
- MUST complete in under 20 minutes total runtime
- MUST produce scores for all 3 tasks

### 3. Reward Function — NOT Sparse
- The reward function MUST give signal at EVERY step, not just terminal
- Use `reward_calculator.py` which computes sub-rewards and sums them
- Never return a constant reward; always vary based on action quality

### 4. Graders — DETERMINISTIC
- Graders MUST produce the same score for the same input, always
- No randomness, no LLM calls inside graders
- All scores in range [0.0, 1.0]
- Use programmatic rules, not AI judgment

### 5. Docker
- The Dockerfile must work with `docker build . && docker run -p 8000:8000 <image>`
- Target machine: 2 vCPU, 8 GB RAM — keep it lean
- No GPU requirements

### 6. Data
- All email data is SYNTHETIC — do not use real customer data
- Generate 200+ emails spanning all categories and difficulty levels
- Ground truth labels must be stored in `data/ground_truth.json`

---

## Scoring Criteria Weights (from the judges)

| Criterion               | Weight | Our Strategy                                      |
|-------------------------|--------|---------------------------------------------------|
| Real-world utility      | 30%    | Support triage is a universally needed task       |
| Task & grader quality   | 25%    | 3 clear tasks, F1/accuracy graders, hard Task 3   |
| Environment design      | 20%    | Clean state, non-sparse reward, sensible episodes |
| Code quality + spec     | 15%    | Typed models, documented, openenv validate passes |
| Creativity & novelty    | 10%    | Support triage not yet in OpenEnv catalog         |

**Priority order when making trade-offs:**
1. Get OpenEnv spec 100% correct (disqualifying gate)
2. Make Task 3 genuinely hard (judges run a frontier model against it)
3. Make reward non-sparse and meaningful
4. Polish README and documentation

---

## Environment Variables

```bash
API_BASE_URL=https://router.huggingface.co/v1   # LLM router endpoint
HF_TOKEN=hf_xxxx                                 # Hugging Face token
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct    # Model for inference
OPENAI_API_KEY=${HF_TOKEN}                       # Alias — OpenAI client uses this
```

---

## Key Design Decisions (Do Not Change Without Reading design.md)

1. **Episode = one triage session** with N emails in the inbox
2. **One action per step** — the agent acts on one email at a time
3. **Episode ends** when agent calls `submit_triage()` action OR max_steps reached
4. **State tracks** which emails have been processed, current inbox snapshot
5. **Observation includes** the full email plus inbox metadata (queue depth, team loads)
6. **Ground truth** is loaded at reset() and never shown to the agent

---

## Files to Write First (Recommended Order)

1. `models.py` — all Pydantic types (nothing else works without this)
2. `data/emails.json` + `data/ground_truth.json` — test data
3. `graders/*.py` — deterministic scoring logic
4. `rewards/reward_calculator.py` — composite reward
5. `tasks/*.py` — task wrappers using graders
6. `server/environment.py` — main environment logic
7. `server/app.py` — FastAPI server
8. `client.py` — client-side wrapper
9. `server/Dockerfile` — containerization
10. `inference.py` — baseline script
11. `openenv.yaml` — manifest
12. `README.md` — documentation

---

## Testing Checklist Before Submission

- [ ] `python -c "from support_triage_env import TriageEnv, TriageAction"` — imports work
- [ ] `openenv validate` — passes
- [ ] `docker build -t support-triage-env ./server` — builds clean
- [ ] `docker run -p 8000:8000 support-triage-env` — starts, responds to `/reset`
- [ ] `python inference.py` — completes, prints scores for all 3 tasks, no errors
- [ ] All 3 task graders return scores in [0.0, 1.0] — verified by test suite
- [ ] `pytest tests/` — all green
- [ ] HF Space deploys and returns 200 on `/reset` ping
