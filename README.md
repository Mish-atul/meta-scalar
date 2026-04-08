---
title: Meta
emoji: 🏃
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
license: mit
---

# SupportTriageEnv

OpenEnv-compatible evaluation environment for training and benchmarking AI agents on customer support email triage.

## What It Does

The environment simulates real operational support flow:

1. **Classify** incoming emails into categories (billing, technical, account, refund, abuse, general)
2. **Set priority** levels (P1–P4) based on urgency and customer tier
3. **Route** to the correct team (billing, tech-support, account-management, escalations, general)
4. **Draft responses** for important tickets
5. **Handle** escalations, duplicates, and VIP customers
6. **Submit** episode and receive deterministic score

## Architecture

```
Agent (inference.py)
    └── SupportTriageEnvironment (in-process)
            ├── Task Layer (Task 1 / 2 / 3)
            ├── Reward Calculator (non-sparse step rewards)
            ├── Deterministic Graders
            └── Dataset (emails.json + ground_truth.json)
```

## API Endpoints

| Method | Path      | Description                        |
|--------|-----------|------------------------------------|
| GET    | /         | Health check                       |
| GET    | /health   | Health check                       |
| POST   | /reset    | Reset environment (body optional)  |
| POST   | /step     | Take an action                     |
| GET    | /state    | Get current episode state          |

**Hosted:** https://mishatul-meta.hf.space

## Tasks

| Task | Description                    | Max Steps | Baseline Threshold |
|------|--------------------------------|-----------|--------------------|
| 1    | Single email classification    | 3         | ≥ 0.50             |
| 2    | Batch priority + routing       | 15        | ≥ 0.30             |
| 3    | Full pipeline (8–10 emails)    | 40        | ≥ 0.15             |

## Quick Start

### Run the API Server

```bash
pip install -r requirements.txt
uvicorn support_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

### Run Inference (in-process, no server needed)

```bash
python inference.py
```

### Docker

```bash
docker build -t support-triage-env .
docker run --rm -p 7860:7860 support-triage-env
```

## Project Structure

```
├── inference.py                          # Baseline agent (runs in-process)
├── Dockerfile                            # Docker config (HF Spaces compatible)
├── openenv.yaml                          # OpenEnv manifest
├── requirements.txt                      # Python dependencies
├── support_triage_env/
│   ├── models.py                         # Pydantic models (actions, observations, state)
│   ├── client.py                         # HTTP client wrapper
│   ├── data/
│   │   ├── emails.json                   # 240 synthetic support emails
│   │   ├── ground_truth.json             # Ground truth labels
│   │   └── generate_dataset.py           # Dataset generator
│   ├── server/
│   │   ├── app.py                        # FastAPI application
│   │   └── environment.py                # Core environment engine
│   ├── tasks/
│   │   ├── task1_classify.py             # Single email classification
│   │   ├── task2_batch_triage.py         # Batch priority + routing
│   │   └── task3_full_pipeline.py        # Full pipeline
│   ├── graders/                          # Deterministic scoring
│   └── rewards/
│       └── reward_calculator.py          # Step-level reward logic
└── tests/                                # Test suite (23 tests, 86% coverage)
```

## Scoring

- Step-level rewards are **non-sparse** (reward at every step)
- Correct actions earn incremental reward
- Invalid/repeated actions incur penalties
- VIP mis-priority and incomplete triage are penalized
- Final episode score is deterministic and bounded

## Team

Atul Kumar Mishra, Sarthak Lakhotia, Tanishk Bhanage
