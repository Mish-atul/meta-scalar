# design.md — SupportTriageEnv Architecture & System Design

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    Inference Script / RL Trainer                   │
│                         (inference.py)                             │
│                                                                    │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  TriageEnv (EnvClient)  ←  client.py                     │    │
│   │                                                           │    │
│   │  env.reset()   →  TriageObservation                       │    │
│   │  env.step(a)   →  (obs, reward, done, info)               │    │
│   │  env.state()   →  TriageState                             │    │
│   └────────────────────────┬─────────────────────────────────┘    │
└────────────────────────────┼───────────────────────────────────────┘
                             │  WebSocket (OpenEnv protocol)
                             │
┌────────────────────────────▼───────────────────────────────────────┐
│             Docker Container / HF Space                            │
│                                                                    │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  FastAPI Server (server/app.py)                           │    │
│   │                                                           │    │
│   │  POST /reset    →  calls environment.reset()              │    │
│   │  POST /step     →  calls environment.step(action)         │    │
│   │  GET  /state    →  calls environment.state()              │    │
│   └────────────────────────┬─────────────────────────────────┘    │
│                            │                                       │
│   ┌────────────────────────▼─────────────────────────────────┐    │
│   │  SupportTriageEnvironment (server/environment.py)         │    │
│   │                                                           │    │
│   │  - Manages episode state (inbox, processed set, step ct)  │    │
│   │  - Validates actions against current state                │    │
│   │  - Calls RewardCalculator on each step                    │    │
│   │  - Delegates scoring to Task graders at episode end       │    │
│   └───────────────┬──────────────────┬────────────────────────┘    │
│                   │                  │                              │
│   ┌───────────────▼──────┐  ┌────────▼──────────────────────────┐ │
│   │  RewardCalculator    │  │  Task Graders                     │ │
│   │  (rewards/)          │  │  (graders/)                       │ │
│   │                      │  │                                   │ │
│   │  - Per-step reward   │  │  - ClassificationGrader           │ │
│   │  - Penalty logic     │  │  - PriorityGrader                 │ │
│   │  - Episode bonus     │  │  - RoutingGrader                  │ │
│   └──────────────────────┘  │  - ResponseGrader                 │ │
│                              └───────────────────────────────────┘ │
│                                                                    │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  Data Layer (data/)                                       │    │
│   │                                                           │    │
│   │  emails.json         — 200+ synthetic emails              │    │
│   │  ground_truth.json   — labels per email_id                │    │
│   └──────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Model Design

### 2.1 EmailObject (Input to Agent)

```python
class EmailObject(BaseModel):
    """A single email in the inbox."""
    email_id: str                          # Unique ID: "email_001"
    subject: str                           # Email subject line
    body: str                              # Full email body text
    sender_email: str                      # sender@example.com
    sender_tier: Literal["vip", "regular", "first_time"]
    timestamp: datetime                    # When email was received
    thread_history: List[str] = []         # Previous messages in thread
    attachments_mentioned: bool = False    # Mentions an attachment?
    language: str = "en"                   # For future multilingual support
```

### 2.2 TriageAction (Agent Output → Environment Input)

```python
class ActionType(str, Enum):
    CLASSIFY       = "classify"
    SET_PRIORITY   = "set_priority"
    ROUTE          = "route"
    DRAFT_RESPONSE = "draft_response"
    MARK_DUPLICATE = "mark_duplicate"
    ESCALATE       = "escalate"
    REQUEST_INFO   = "request_info"
    SUBMIT_TRIAGE  = "submit_triage"

class TriageAction(BaseModel):
    """One action taken by the agent."""
    action_type: ActionType

    # Used by: classify
    category: Optional[Literal[
        "billing", "technical", "account",
        "refund", "abuse", "general"
    ]] = None

    # Used by: classify, set_priority, route, draft_response,
    #          mark_duplicate, escalate, request_info
    email_id: Optional[str] = None

    # Used by: set_priority
    priority: Optional[Literal["P1", "P2", "P3", "P4"]] = None

    # Priority meanings:
    # P1 = Critical (SLA ≤ 1hr, VIP, data loss, outage)
    # P2 = High     (SLA ≤ 4hr, billing issue, feature broken)
    # P3 = Medium   (SLA ≤ 24hr, general question, minor issue)
    # P4 = Low      (SLA ≤ 72hr, feedback, enhancement request)

    # Used by: route
    team: Optional[Literal[
        "billing-team", "tech-support",
        "account-management", "escalations", "general-support"
    ]] = None

    # Used by: draft_response
    response_text: Optional[str] = None

    # Used by: mark_duplicate
    original_email_id: Optional[str] = None

    # Used by: escalate, request_info
    reason: Optional[str] = None
    question: Optional[str] = None
```

### 2.3 TriageObservation (Environment Output → Agent Input)

```python
class EmailSummary(BaseModel):
    """Lightweight email summary for inbox overview."""
    email_id: str
    subject: str
    sender_tier: str
    status: Literal["pending", "classified", "prioritized",
                    "routed", "responded", "complete", "duplicate"]
    actions_taken: List[str] = []

class TriageObservation(BaseModel):
    """What the agent sees after each step."""
    current_email: Optional[EmailObject]    # Email to focus on now
    inbox_snapshot: List[EmailSummary]      # All emails + status
    processed_count: int
    pending_count: int
    team_queue_depths: Dict[str, int]       # {"tech-support": 14, ...}
    last_action_result: str                 # "Successfully classified as billing"
    last_action_valid: bool                 # Was last action valid?
    step_number: int
    episode_id: str
    available_actions: List[str]            # What actions are legal now
    task_id: int                            # 1, 2, or 3
    hint: Optional[str] = None             # For Task 1 only (easy)
```

### 2.4 TriageState (Episode Metadata)

```python
class TriageState(BaseModel):
    """Internal episode state (returned by state() call)."""
    episode_id: str
    step_count: int
    task_id: int
    max_steps: int
    emails_in_inbox: int
    emails_processed: int
    cumulative_reward: float
    started_at: datetime
    is_done: bool
    # Private: ground truth is NOT exposed here
```

### 2.5 Ground Truth Format

```json
{
  "email_001": {
    "category": "billing",
    "priority": "P2",
    "team": "billing-team",
    "is_duplicate": false,
    "original_email_id": null,
    "requires_escalation": false,
    "sla_breach_risk": false,
    "difficulty_level": 1
  },
  "email_042": {
    "category": "technical",
    "priority": "P1",
    "team": "tech-support",
    "is_duplicate": false,
    "original_email_id": null,
    "requires_escalation": true,
    "sla_breach_risk": true,
    "difficulty_level": 3
  }
}
```

---

## 3. Episode Lifecycle

```
reset() called
    │
    ▼
Load task config (task_id determines inbox size + complexity)
    │
    ▼
Sample emails from dataset for this episode
    │
    ▼
Set episode_id = uuid4(), step_count = 0
    │
    ▼
Return TriageObservation (first email = highest urgency by timestamp)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Agent Action Loop                             │
│                                                                  │
│  Agent reads observation                                         │
│      │                                                           │
│      ▼                                                           │
│  Agent calls step(TriageAction)                                  │
│      │                                                           │
│      ▼                                                           │
│  Environment validates action                                    │
│      │                                                           │
│      ├─── INVALID: return obs with error message, reward=-0.05  │
│      │                                                           │
│      └─── VALID: execute action, update state                   │
│               │                                                  │
│               ▼                                                  │
│          RewardCalculator.compute(action, state, ground_truth)   │
│               │                                                  │
│               ▼                                                  │
│          step_count += 1                                         │
│               │                                                  │
│               ▼                                                  │
│          Check done conditions:                                  │
│            - action_type == SUBMIT_TRIAGE                        │
│            - step_count >= max_steps                             │
│               │                                                  │
│               ├─── NOT DONE: return next obs                     │
│               │                                                  │
│               └─── DONE: compute final episode score            │
│                         call task grader                         │
│                         return final obs, done=True              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Task Design Details

### Task 1: Single-Email Classification (Easy)

**Setup:**
- Inbox contains exactly 1 email
- Email is unambiguous (single clear category)
- No VIP tiers, no duplicates, no SLA risk
- `max_steps = 3`

**Required actions for full score:**
1. `classify(email_id, category)` — correct category
2. `set_priority(email_id, priority)` — correct priority
3. `submit_triage()` — finalize

**Grader logic:**
```
score = 0.0
if category_correct:
    score += 0.60
elif category_parent_correct:   # e.g., "technical" vs "account" but both non-billing
    score += 0.20
if priority_correct:
    score += 0.40
elif abs(priority_rank_diff) == 1:   # adjacent priority
    score += 0.15
return min(score, 1.0)
```

### Task 2: Batch Priority + Routing (Medium)

**Setup:**
- Inbox contains exactly 5 emails
- Mix of categories, priorities, customer tiers
- No duplicates, no draft responses required
- `max_steps = 15`

**Required actions for full score per email:**
1. `classify(email_id, category)`
2. `set_priority(email_id, priority)`
3. `route(email_id, team)`

**Grader logic:**
```
per_email_scores = []
for email_id in inbox_emails:
    s = 0.0
    if classification_correct[email_id]:
        s += 0.33
    if priority_correct[email_id]:
        s += 0.33
    elif adjacent_priority[email_id]:
        s += 0.10
    if routing_correct[email_id]:
        s += 0.34
    per_email_scores.append(s)

# Penalize missing emails (agent submitted without acting on some)
completion_ratio = len(acted_on) / len(inbox_emails)
raw_score = mean(per_email_scores)
return raw_score * completion_ratio
```

**What makes this medium difficulty:**
- Some emails are edge cases (e.g., billing issue that requires tech-support routing)
- VIP customer emails must be P2 or higher
- One email is intentionally ambiguous between two categories

### Task 3: Full Pipeline with SLA + Draft Response (Hard)

**Setup:**
- Inbox contains 8–10 emails
- Includes: 1 duplicate, 1 VIP P1, 1 SLA-breach-risk, 2 edge cases
- Draft responses required for all P1 and P2 emails
- `max_steps = 40`

**Required actions for full score per email:**
1. `classify`
2. `set_priority`
3. `route`
4. `draft_response` (for P1 and P2 emails only)
5. `mark_duplicate` (for the duplicate email)
6. `escalate` (for emails requiring escalation)

**Grader logic:**
```
components = {
    "classification_f1":   weighted 0.20,
    "priority_accuracy":   weighted 0.20,
    "routing_accuracy":    weighted 0.20,
    "response_quality":    weighted 0.20,
    "escalation_recall":   weighted 0.10,
    "duplicate_detection": weighted 0.10,
}

# SLA penalty: -0.15 per P1 email where response was NOT drafted
sla_penalty = count(p1_emails_without_response) * 0.15

# VIP penalty: -0.20 if VIP email assigned P3 or P4
vip_penalty = count(vip_as_low_priority) * 0.20

final_score = sum(component_scores) - sla_penalty - vip_penalty
return max(0.0, min(1.0, final_score))
```

**Response Quality Grader (programmatic, no LLM):**
```python
def score_response(response_text: str, email_context: EmailObject) -> float:
    score = 0.0
    checks = {
        "has_greeting":          bool(re.search(r"(dear|hello|hi|greetings)", text_lower)),
        "acknowledges_issue":    any(keyword in text_lower for keyword in issue_keywords),
        "has_apology_if_needed": (not needs_apology or apology_detected),
        "professional_tone":     not any(w in text_lower for w in unprofessional_words),
        "appropriate_length":    50 <= word_count <= 300,
        "no_placeholders":       "[" not in response_text and "TODO" not in response_text,
        "references_email":      any(word in text_lower for word in contact_references),
    }
    return sum(checks.values()) / len(checks)
```

---

## 5. Reward Function Design

Full details in `reward_design.md`. Summary:

```
step_reward = (
    classification_reward(action, ground_truth)   # 0.0 to +0.35
  + priority_reward(action, ground_truth)          # 0.0 to +0.25
  + routing_reward(action, ground_truth)           # 0.0 to +0.20
  + response_reward(action, ground_truth)          # 0.0 to +0.20
  - repeat_action_penalty                          # -0.05 per repeat
  - invalid_action_penalty                         # -0.05 per invalid
  - vip_downgrade_penalty                         # -0.30 if VIP → P3/P4
  - sla_breach_penalty                            # -0.20 per P1 ignored
)
```

---

## 6. File Implementation Guide

### models.py

```python
from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

# Define all models here: ActionType, TriageAction, EmailObject,
# EmailSummary, TriageObservation, TriageState
```

### server/environment.py

```python
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import TriageAction, TriageObservation

class SupportTriageEnvironment(Environment):
    """
    Main environment class. Manages episode state, validates actions,
    dispatches to RewardCalculator and Graders.
    """

    def __init__(self):
        self._state: Optional[TriageState] = None
        self._inbox: List[EmailObject] = []
        self._ground_truth: Dict[str, GroundTruth] = {}
        self._processed: Dict[str, TriageRecord] = {}
        self._reward_calc = RewardCalculator()
        self._data_loader = DataLoader()

    def reset(self) -> TriageObservation:
        ...

    def step(self, action: TriageAction) -> TriageObservation:
        ...

    @property
    def state(self) -> TriageState:
        ...
```

### server/app.py

```python
from openenv.core.env_server import HTTPEnvServer, create_app
from server.environment import SupportTriageEnvironment
from models import TriageAction, TriageObservation

server = HTTPEnvServer(
    env=SupportTriageEnvironment,
    action_cls=TriageAction,
    observation_cls=TriageObservation,
    max_concurrent_envs=4,
)

app = create_app(server)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### client.py

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import TriageAction, TriageObservation, TriageState

class TriageEnv(EnvClient[TriageAction, TriageObservation, TriageState]):
    """Client-side wrapper for SupportTriageEnv."""

    def _step_payload(self, action: TriageAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[TriageObservation]:
        ...

    def _parse_state(self, payload: dict) -> TriageState:
        ...
```

### openenv.yaml

```yaml
name: support-triage-env
version: "0.1.0"
description: >
  A real-world customer support email triage environment.
  Agents classify, prioritize, route, and draft responses for support tickets.
author: "Atul Kumar Mishra, Sarthak Lakhotia, Tanishk Bhanage"
tags:
  - openenv
  - customer-support
  - nlp
  - triage
  - real-world
tasks:
  - id: task1_classify
    name: "Single-Email Classification"
    difficulty: easy
    max_steps: 3
  - id: task2_batch_triage
    name: "Batch Priority and Routing"
    difficulty: medium
    max_steps: 15
  - id: task3_full_pipeline
    name: "Full Triage Pipeline with SLA and Drafts"
    difficulty: hard
    max_steps: 40
action_space:
  type: discrete_with_params
  actions:
    - classify
    - set_priority
    - route
    - draft_response
    - mark_duplicate
    - escalate
    - request_info
    - submit_triage
observation_space:
  type: structured_dict
  fields:
    - current_email
    - inbox_snapshot
    - team_queue_depths
    - last_action_result
    - available_actions
reward:
  type: shaped
  range: [-1.0, 1.0]
  sparse: false
```

---

## 7. Dockerfile Design

```dockerfile
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first (cache layer)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ ./server/
COPY models.py .
COPY data/ ./data/
COPY tasks/ ./tasks/
COPY graders/ ./graders/
COPY rewards/ ./rewards/

# Copy README for web interface
COPY README.md .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server.app:app",
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

---

## 8. Dataset Generation Strategy

The `data/emails.json` file must be generated as synthetic data. Distribution:

| Category    | Count | Notes                                         |
|-------------|-------|-----------------------------------------------|
| billing     | 35    | Charge disputes, invoice questions, upgrades  |
| technical   | 50    | Bugs, login failures, integrations, outages   |
| account     | 30    | Password resets, settings, team management    |
| refund      | 25    | Refund requests, cancellations                |
| abuse       | 15    | Spam reports, ToS violations, harassment      |
| general     | 20    | Feedback, questions, praise                   |
| duplicates  | 10    | Same issue as another email in dataset        |
| ambiguous   | 15    | Could plausibly be 2 categories               |
| **Total**   | 200   |                                               |

**Email structure template:**
```json
{
  "email_id": "email_001",
  "subject": "Charge on my account I didn't authorize",
  "body": "Hi support team,\n\nI just noticed a charge of $49.99 on my credit card...",
  "sender_email": "john.smith@gmail.com",
  "sender_tier": "regular",
  "timestamp": "2024-01-15T09:32:00Z",
  "thread_history": [],
  "attachments_mentioned": false,
  "language": "en"
}
```

---

## 9. Key Design Decisions & Rationale

### Decision 1: One action per step (not "full triage in one action")
**Rationale:** Makes the reward non-sparse. Agent gets feedback after every classify/route/etc.
action, not just at episode end. This is what the judges specifically look for (20% of score).

### Decision 2: State tracks what's been done per email
**Rationale:** Enables detecting repeat actions (penalty) and tracking completion.
The `processed` dict maps `email_id → TriageRecord` showing what actions have been taken.

### Decision 3: Ground truth NOT exposed in observation
**Rationale:** Prevents trivial cheating. Agent must read the email and reason.

### Decision 4: 200+ emails but episodes sample a small subset
**Rationale:** Prevents overfitting. Each `reset()` call samples a fresh set from the pool.
For Tasks 1/2/3, the pool is curated to match the task's difficulty requirements.

### Decision 5: ResponseGrader uses regex rules, not an LLM
**Rationale:** Deterministic, reproducible, fast. The judges require graders to be deterministic.
An LLM-based grader would be non-deterministic and slow.

### Decision 6: Separate task files in `tasks/` folder
**Rationale:** Makes it trivial to add Task 4 or 5 later, keeps environment.py clean,
and matches what judges expect when they read the codebase.
