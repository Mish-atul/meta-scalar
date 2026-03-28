# Pre-Upload Requirement Checklist (From files/requirements.md)

Date: 2026-03-29
Validation source: local one-command pipeline (`python validate_pipeline.py`) + requirement-to-implementation mapping.

## Functional Requirements

- [x] FR-01 OpenEnv Spec Compliance
  - [x] `reset()` implemented
  - [x] `step(action)` implemented
  - [x] `state()` implemented
  - [x] Pydantic models present
  - [x] `openenv.yaml` present
  - [x] `openenv validate` passes

- [x] FR-02 Email Dataset
  - [x] 200+ synthetic emails (240 generated)
  - [x] Required fields present
  - [x] Ground truth in `support_triage_env/data/ground_truth.json`
  - [x] All categories covered
  - [x] VIP/regular/first_time included
  - [x] Edge cases included (ambiguous, duplicate, escalation, urgency)

- [x] FR-03 Action Space
  - [x] All required actions implemented: classify, set_priority, route, draft_response, mark_duplicate, escalate, request_info, submit_triage

- [x] FR-04 Observation Space
  - [x] current_email
  - [x] inbox_snapshot
  - [x] processed_count
  - [x] pending_count
  - [x] team_queue_depths
  - [x] last_action_result
  - [x] step_number
  - [x] episode_id
  - [x] available_actions

- [x] FR-05 Tasks
  - [x] Task 1 implemented with deterministic grader
  - [x] Task 2 implemented with deterministic grader
  - [x] Task 3 implemented with deterministic grader

- [x] FR-06 Reward Function
  - [x] Non-sparse step reward
  - [x] Action-specific rewards implemented
  - [x] VIP mis-priority penalty logic implemented
  - [x] Repeat/invalid penalties implemented
  - [x] Completion bonus/penalty logic implemented

- [x] FR-07 Baseline Inference Script
  - [x] Root `inference.py` exists
  - [x] Uses `openai.OpenAI`
  - [x] Reads `API_BASE_URL`, `HF_TOKEN`, `MODEL_NAME`
  - [x] Runs all 3 tasks and prints score
  - [x] Deterministic path implemented (fixed seeds)
  - [x] Handles LLM failures with fallback heuristics

## Non-Functional Requirements

- [x] NFR-01 Deployment API contract
  - [x] `GET /` and `GET /health` return 200
  - [x] `POST /reset` returns valid observation JSON
  - [x] `POST /step` returns valid step payload
  - [x] `GET /state` returns valid state JSON
  - [ ] HF Space deploy URL verification pending (requires push/deploy credentials)

- [x] NFR-02 Containerization
  - [x] Docker build passes
  - [x] Docker run passes
  - [x] Container smoke tests pass (`/health`, `/reset`, `/step`)
  - [ ] Strict path variant `docker build -t support-triage-env ./server` not used in current pipeline; active build command is `docker build -t support-triage-env -f support_triage_env/server/Dockerfile .`

- [x] NFR-03 Performance and determinism
  - [x] Deterministic graders
  - [x] Deterministic validation pipeline passing repeatedly
  - [ ] Explicit benchmark report for exact ms limits not added as artifact yet

- [x] NFR-04 Code Quality
  - [x] `pytest` passes
  - [x] Coverage >= 80% (current 86%)
  - [x] Type annotations and model field docs included
  - [ ] Lint gate (`ruff`/`flake8`) not yet wired in pipeline

## Success Criteria

- [x] openenv validate passes
- [x] Docker gate passes
- [x] Inference runs end-to-end
- [x] Baseline thresholds met (Task1 >= 0.50, Task2 >= 0.30, Task3 >= 0.15)
- [ ] First GitHub CI run green (pending push)
- [ ] Branch protection enforced (pending GitHub admin action/API)
- [ ] HF Space updated and reachable (pending deploy credentials)

## Final Tally

- Fully met now: core implementation, local validation, Docker, OpenEnv, coverage, CI workflow config.
- Pending external operations: GitHub push + remote CI confirmation, branch protection config, Hugging Face deployment verification.
