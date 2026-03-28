# implementation_guide.md — Step-by-Step Build Plan

## Phase 0: Setup (Day 1 Morning)

```bash
# Install OpenEnv CLI
pip install openenv-core

# Scaffold the project using the CLI
openenv init support_triage_env
cd support_triage_env

# Initialize git
git init
git add .
git commit -m "Initial OpenEnv scaffold"
```

Then REPLACE the scaffolded files with our actual code below. Do NOT use the auto-generated
placeholder code — it won't match our models.

---

## Phase 1: Models (Day 1 — ~2 hours)

**File:** `models.py`  
**Goal:** Define ALL Pydantic types. Every other file depends on this.

```python
# models.py
from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Literal, Set
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


# ─── Enums ────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    CLASSIFY       = "classify"
    SET_PRIORITY   = "set_priority"
    ROUTE          = "route"
    DRAFT_RESPONSE = "draft_response"
    MARK_DUPLICATE = "mark_duplicate"
    ESCALATE       = "escalate"
    REQUEST_INFO   = "request_info"
    SUBMIT_TRIAGE  = "submit_triage"

class Category(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    ACCOUNT   = "account"
    REFUND    = "refund"
    ABUSE     = "abuse"
    GENERAL   = "general"

class Priority(str, Enum):
    P1 = "P1"  # Critical (≤1hr SLA)
    P2 = "P2"  # High     (≤4hr SLA)
    P3 = "P3"  # Medium   (≤24hr SLA)
    P4 = "P4"  # Low      (≤72hr SLA)

class Team(str, Enum):
    BILLING     = "billing-team"
    TECH        = "tech-support"
    ACCOUNT     = "account-management"
    ESCALATIONS = "escalations"
    GENERAL     = "general-support"

class EmailStatus(str, Enum):
    PENDING     = "pending"
    CLASSIFIED  = "classified"
    PRIORITIZED = "prioritized"
    ROUTED      = "routed"
    RESPONDED   = "responded"
    COMPLETE    = "complete"
    DUPLICATE   = "duplicate"

class SenderTier(str, Enum):
    VIP        = "vip"
    REGULAR    = "regular"
    FIRST_TIME = "first_time"


# ─── Email Data ────────────────────────────────────────────────────────────────

class EmailObject(BaseModel):
    """A single email visible to the agent in its full form."""
    email_id: str = Field(..., description="Unique email ID e.g. 'email_001'")
    subject: str
    body: str
    sender_email: str
    sender_tier: SenderTier
    timestamp: datetime
    thread_history: List[str] = Field(default_factory=list)
    attachments_mentioned: bool = False
    language: str = "en"

class EmailSummary(BaseModel):
    """Lightweight email summary for inbox overview panel."""
    email_id: str
    subject: str
    sender_tier: SenderTier
    status: EmailStatus
    actions_taken: List[str] = Field(default_factory=list)


# ─── Action ────────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    One action taken by the agent per step.

    Only fill in fields relevant to the action_type.
    Unused fields should be None (they will be ignored).
    """
    action_type: ActionType = Field(..., description="Type of action to perform")

    # For: classify
    category: Optional[Category] = Field(None, description="Issue category")

    # For: classify, set_priority, route, draft_response,
    #      mark_duplicate, escalate, request_info
    email_id: Optional[str] = Field(None, description="Target email ID")

    # For: set_priority
    priority: Optional[Priority] = Field(None, description="Priority level P1–P4")

    # For: route
    team: Optional[Team] = Field(None, description="Destination team")

    # For: draft_response
    response_text: Optional[str] = Field(None, description="Draft email text (50–400 words)")

    # For: mark_duplicate
    original_email_id: Optional[str] = Field(None, description="ID of the original ticket")

    # For: escalate, request_info
    reason: Optional[str] = Field(None, description="Reason for escalation")
    question: Optional[str] = Field(None, description="Question to ask the customer")


# ─── Observation ───────────────────────────────────────────────────────────────

class TriageObservation(BaseModel):
    """What the agent sees after reset() or step()."""
    current_email: Optional[EmailObject] = Field(
        None, description="Email to focus on. None if inbox is empty or episode done."
    )
    inbox_snapshot: List[EmailSummary] = Field(
        default_factory=list,
        description="All emails in inbox with current status"
    )
    processed_count: int = 0
    pending_count: int = 0
    team_queue_depths: Dict[str, int] = Field(
        default_factory=dict,
        description="Current tickets per team (context for routing decisions)"
    )
    last_action_result: str = ""
    last_action_valid: bool = True
    step_number: int = 0
    episode_id: str = ""
    available_actions: List[str] = Field(default_factory=list)
    task_id: int = 1
    hint: Optional[str] = None


# ─── State ─────────────────────────────────────────────────────────────────────

class TriageState(BaseModel):
    """Episode metadata returned by state()."""
    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    step_count: int = 0
    task_id: int = 1
    max_steps: int = 3
    emails_in_inbox: int = 0
    emails_processed: int = 0
    cumulative_reward: float = 0.0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    is_done: bool = False


# ─── Ground Truth (internal, never sent to agent) ─────────────────────────────

class GroundTruthRecord(BaseModel):
    """Ground truth labels for one email. NEVER exposed in observations."""
    email_id: str
    category: Category
    priority: Priority
    team: Team
    is_duplicate: bool = False
    original_email_id: Optional[str] = None
    requires_escalation: bool = False
    sla_breach_risk: bool = False
    difficulty_level: Literal[1, 2, 3] = 1


# ─── Triage Record (internal tracking) ────────────────────────────────────────

class TriageRecord(BaseModel):
    """Tracks what actions the agent took on a specific email."""
    email_id: str
    category: Optional[Category] = None
    priority: Optional[Priority] = None
    team: Optional[Team] = None
    response_text: Optional[str] = None
    marked_duplicate: bool = False
    duplicate_of: Optional[str] = None
    escalated: bool = False
    escalation_reason: Optional[str] = None
    actions_taken: List[str] = Field(default_factory=list)
    status: EmailStatus = EmailStatus.PENDING
```

---

## Phase 2: Synthetic Data Generation (Day 1 Afternoon — ~3 hours)

**File:** `data/generate_dataset.py`  
**Goal:** Create 200+ synthetic emails + ground truth labels.

Run this script ONCE to generate `data/emails.json` and `data/ground_truth.json`.

```python
# data/generate_dataset.py
"""
Run this script to generate the synthetic email dataset.
Usage: python data/generate_dataset.py
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Email templates organized by category and difficulty
TEMPLATES = {
    "billing": {
        "easy": [
            {
                "subject": "Unexpected charge on my account",
                "body": """Hi there,

I noticed a charge of $49.99 on my credit card statement dated {date}
that I don't recognize. My account email is {email}. I haven't upgraded
my subscription and I'd like to know what this charge is for.

Please help me understand this charge.

Best,
{name}""",
                "priority": "P2",
                "team": "billing-team",
                "difficulty_level": 1,
            },
            # ... add 10+ billing templates
        ],
        "medium": [
            # Templates where billing + something else is present
        ],
        "hard": [
            # VIP customer angry billing + threat to leave
        ],
    },
    # ... similar structure for technical, account, refund, abuse, general
}

def generate_email(template, email_id, sender_tier):
    """Fill in a template with synthetic data."""
    name = random.choice(["Alice Johnson", "Bob Smith", "Carlos Garcia", ...])
    email = f"{name.lower().replace(' ', '.')}@{random.choice(['gmail.com', 'outlook.com'])}"
    date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")

    return {
        "email_id": email_id,
        "subject": template["subject"],
        "body": template["body"].format(name=name, email=email, date=date),
        "sender_email": email,
        "sender_tier": sender_tier,
        "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
        "thread_history": [],
        "attachments_mentioned": random.random() < 0.1,
        "language": "en",
    }

def build_ground_truth(email_id, template, sender_tier):
    """Create ground truth record."""
    priority = template["priority"]
    # VIP customer override: never below P2
    if sender_tier == "vip" and priority in ("P3", "P4"):
        priority = "P2"

    return {
        "email_id": email_id,
        "category": template["category"],
        "priority": priority,
        "team": template["team"],
        "is_duplicate": False,
        "original_email_id": None,
        "requires_escalation": template.get("requires_escalation", False),
        "sla_breach_risk": priority == "P1",
        "difficulty_level": template["difficulty_level"],
    }

if __name__ == "__main__":
    emails = []
    ground_truth = {}
    # ... generation logic
    Path("data/emails.json").write_text(json.dumps(emails, indent=2))
    Path("data/ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
    print(f"Generated {len(emails)} emails")
```

**IMPORTANT:** You need to actually write the 200+ templates. Do this manually or use
an LLM (offline, before building the env) to generate varied, realistic email content.
Commit the generated JSON files to the repo.

---

## Phase 3: Graders (Day 2 — ~3 hours)

**Goal:** Build all 4 grader classes with unit tests.

See `tasks_and_graders.md` for full grader logic. Each grader file:
- `graders/classification_grader.py`
- `graders/priority_grader.py`
- `graders/routing_grader.py`
- `graders/response_grader.py`

After writing each grader, immediately write its tests in `tests/test_graders.py`.

**DO NOT PROCEED to Phase 4 until `pytest tests/test_graders.py` passes 100%.**

---

## Phase 4: Reward Calculator (Day 2 — ~2 hours)

**File:** `rewards/reward_calculator.py`  
**Goal:** Implement the composite reward function.

See `reward_design.md` for the complete specification.

After writing, test with `pytest tests/test_reward.py`.

---

## Phase 5: Task Wrappers (Day 2 — ~2 hours)

**Files:** `tasks/task1_classify.py`, `tasks/task2_batch_triage.py`, `tasks/task3_full_pipeline.py`

Each task wrapper:
1. Defines the config (inbox_size, max_steps, email_pool filter)
2. Implements `setup_episode(data_loader)` → returns `(inbox_emails, ground_truth)`
3. Implements `score_episode(triage_records, ground_truth)` → returns final score

```python
# tasks/base_task.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from models import EmailObject, GroundTruthRecord, TriageRecord

class BaseTask(ABC):
    task_id: int
    max_steps: int
    inbox_size: int

    @abstractmethod
    def setup_episode(self, data_loader) -> Tuple[List[EmailObject], Dict[str, GroundTruthRecord]]:
        """Sample emails for this episode from the data pool."""
        pass

    @abstractmethod
    def score_episode(
        self,
        triage_records: Dict[str, TriageRecord],
        ground_truth: Dict[str, GroundTruthRecord],
    ) -> float:
        """Compute final [0.0, 1.0] score for the completed episode."""
        pass

    def get_required_actions(self, email_id: str, ground_truth: Dict) -> List[str]:
        """Return list of required action types for this email in this task."""
        raise NotImplementedError
```

---

## Phase 6: Core Environment (Day 3 — ~4 hours)

**File:** `server/environment.py`  
**Goal:** The main environment class. This is the heart of the project.

```python
# server/environment.py
import random
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from openenv.core.env_server.interfaces import Environment

from models import (
    TriageAction, TriageObservation, TriageState, TriageRecord,
    EmailObject, EmailSummary, GroundTruthRecord,
    ActionType, EmailStatus, Priority, SenderTier,
)
from rewards.reward_calculator import RewardCalculator
from tasks.task1_classify import Task1
from tasks.task2_batch_triage import Task2
from tasks.task3_full_pipeline import Task3


class SupportTriageEnvironment(Environment):
    """
    Customer Support Email Triage environment.

    An AI agent acts as a support team manager, processing an inbox
    of customer emails. The agent must classify, prioritize, route,
    and draft responses for each email.
    """

    # Data loaded once at class level, shared across episodes
    _emails_db: Dict[str, EmailObject] = {}
    _ground_truth_db: Dict[str, GroundTruthRecord] = {}
    _loaded: bool = False

    def __init__(self, task_id: int = 1):
        self.task_id        = task_id
        self._task          = self._create_task(task_id)
        self._reward_calc   = RewardCalculator()

        # Episode state — reset per episode
        self._state:          Optional[TriageState]              = None
        self._inbox:          List[EmailObject]                  = []
        self._ground_truth:   Dict[str, GroundTruthRecord]       = {}
        self._triage_records: Dict[str, TriageRecord]            = {}
        self._done_actions:   Set[Tuple[str, str]]               = set()
        self._cumulative_reward: float                           = 0.0

        # Load dataset if not already loaded
        if not SupportTriageEnvironment._loaded:
            self._load_data()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def reset(self) -> TriageObservation:
        """Initialize a new episode."""
        # Sample fresh episode data from the task's pool
        inbox_emails, ground_truth = self._task.setup_episode(
            SupportTriageEnvironment._emails_db,
            SupportTriageEnvironment._ground_truth_db,
        )

        self._inbox          = inbox_emails
        self._ground_truth   = ground_truth
        self._triage_records = {e.email_id: TriageRecord(email_id=e.email_id)
                                 for e in inbox_emails}
        self._done_actions   = set()
        self._cumulative_reward = 0.0

        self._state = TriageState(
            episode_id      = str(uuid4()),
            step_count      = 0,
            task_id         = self.task_id,
            max_steps       = self._task.max_steps,
            emails_in_inbox = len(inbox_emails),
            emails_processed= 0,
            started_at      = datetime.utcnow(),
            is_done         = False,
        )

        return self._build_observation(
            last_result="Inbox loaded. You have {} emails to triage.".format(len(inbox_emails)),
            last_valid=True,
        )

    def step(self, action: TriageAction) -> TriageObservation:
        """Execute one agent action."""
        assert self._state is not None, "Call reset() before step()"

        # Validate action
        is_valid, error_msg = self._validate_action(action)
        if not is_valid:
            reward = -0.05
            self._cumulative_reward += reward
            self._state.step_count += 1
            return self._build_observation(
                last_result=f"Invalid action: {error_msg}",
                last_valid=False,
                reward=reward,
            )

        # Execute action
        result_msg = self._execute_action(action)

        # Compute reward
        reward = self._reward_calc.compute(
            action          = action,
            state           = self._state,
            email_objects   = {e.email_id: e for e in self._inbox},
            ground_truth    = self._ground_truth,
            processed_actions = self._done_actions,
        )

        # Track action
        self._done_actions.add((action.email_id or "", action.action_type.value))
        self._cumulative_reward += reward
        self._state.step_count  += 1
        self._state.cumulative_reward = self._cumulative_reward
        self._state.emails_processed = sum(
            1 for r in self._triage_records.values()
            if r.status != EmailStatus.PENDING
        )

        # Check done
        done = (
            action.action_type == ActionType.SUBMIT_TRIAGE
            or self._state.step_count >= self._state.max_steps
        )

        if done:
            self._state.is_done = True
            final_score = self._task.score_episode(
                self._triage_records,
                self._ground_truth,
            )
            result_msg += f" | Episode complete. Final score: {final_score:.3f}"

        return self._build_observation(
            last_result=result_msg,
            last_valid=True,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> TriageState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _validate_action(self, action: TriageAction) -> Tuple[bool, str]:
        """Returns (is_valid, error_message)."""
        if action.action_type == ActionType.SUBMIT_TRIAGE:
            return True, ""

        if action.email_id is None:
            return False, "email_id is required for this action type"

        email_ids = {e.email_id for e in self._inbox}
        if action.email_id not in email_ids:
            return False, f"email_id '{action.email_id}' not found in inbox"

        if action.action_type == ActionType.CLASSIFY and action.category is None:
            return False, "category is required for classify action"

        if action.action_type == ActionType.SET_PRIORITY and action.priority is None:
            return False, "priority is required for set_priority action"

        if action.action_type == ActionType.ROUTE and action.team is None:
            return False, "team is required for route action"

        if action.action_type == ActionType.DRAFT_RESPONSE and action.response_text is None:
            return False, "response_text is required for draft_response action"

        return True, ""

    def _execute_action(self, action: TriageAction) -> str:
        """Mutate triage record based on action. Returns result message."""
        record = self._triage_records[action.email_id]
        record.actions_taken.append(action.action_type.value)

        if action.action_type == ActionType.CLASSIFY:
            record.category = action.category
            record.status   = EmailStatus.CLASSIFIED
            return f"Classified email {action.email_id} as '{action.category.value}'"

        if action.action_type == ActionType.SET_PRIORITY:
            record.priority = action.priority
            if record.status == EmailStatus.CLASSIFIED:
                record.status = EmailStatus.PRIORITIZED
            return f"Set priority of {action.email_id} to {action.priority.value}"

        if action.action_type == ActionType.ROUTE:
            record.team   = action.team
            record.status = EmailStatus.ROUTED
            return f"Routed {action.email_id} to '{action.team.value}'"

        if action.action_type == ActionType.DRAFT_RESPONSE:
            record.response_text = action.response_text
            record.status = EmailStatus.RESPONDED
            return f"Draft response submitted for {action.email_id}"

        if action.action_type == ActionType.MARK_DUPLICATE:
            record.marked_duplicate = True
            record.duplicate_of     = action.original_email_id
            record.status           = EmailStatus.DUPLICATE
            return f"Marked {action.email_id} as duplicate of {action.original_email_id}"

        if action.action_type == ActionType.ESCALATE:
            record.escalated          = True
            record.escalation_reason  = action.reason
            return f"Escalated {action.email_id}: {action.reason}"

        if action.action_type == ActionType.REQUEST_INFO:
            return f"Information request sent for {action.email_id}: {action.question}"

        return "Action executed"

    def _build_observation(
        self,
        last_result: str,
        last_valid: bool,
        reward: float = 0.0,
        done: bool = False,
    ) -> TriageObservation:
        """Build the observation returned to the agent."""
        # Find the next email to focus on (oldest unprocessed)
        pending = [
            e for e in self._inbox
            if self._triage_records[e.email_id].status == EmailStatus.PENDING
        ]
        current = pending[0] if pending else (self._inbox[0] if self._inbox else None)

        inbox_snapshot = [
            EmailSummary(
                email_id     = e.email_id,
                subject      = e.subject,
                sender_tier  = e.sender_tier,
                status       = self._triage_records[e.email_id].status,
                actions_taken= self._triage_records[e.email_id].actions_taken,
            )
            for e in self._inbox
        ]

        available = self._get_available_actions(current)

        return TriageObservation(
            current_email      = current,
            inbox_snapshot     = inbox_snapshot,
            processed_count    = self._state.emails_processed if self._state else 0,
            pending_count      = len(pending),
            team_queue_depths  = {"tech-support": 12, "billing-team": 8,
                                   "account-management": 5, "escalations": 3,
                                   "general-support": 15},
            last_action_result = last_result,
            last_action_valid  = last_valid,
            step_number        = self._state.step_count if self._state else 0,
            episode_id         = self._state.episode_id if self._state else "",
            available_actions  = available,
            task_id            = self.task_id,
        )

    def _get_available_actions(self, current_email: Optional[EmailObject]) -> List[str]:
        """Return list of currently valid action type strings."""
        actions = ["submit_triage"]
        if current_email:
            actions += ["classify", "set_priority", "route",
                        "draft_response", "mark_duplicate",
                        "escalate", "request_info"]
        return actions

    def _create_task(self, task_id: int):
        return {1: Task1, 2: Task2, 3: Task3}[task_id]()

    @classmethod
    def _load_data(cls):
        """Load email dataset from disk. Called once."""
        data_dir   = Path(__file__).parent.parent / "data"
        emails_raw = json.loads((data_dir / "emails.json").read_text())
        gt_raw     = json.loads((data_dir / "ground_truth.json").read_text())

        for e in emails_raw:
            obj = EmailObject(**e)
            cls._emails_db[obj.email_id] = obj

        for email_id, gt in gt_raw.items():
            cls._ground_truth_db[email_id] = GroundTruthRecord(**gt)

        cls._loaded = True
```

---

## Phase 7: FastAPI Server (Day 3 — ~1 hour)

**File:** `server/app.py`

```python
# server/app.py
import os
from openenv.core.env_server import HTTPEnvServer, create_app
from server.environment import SupportTriageEnvironment
from models import TriageAction, TriageObservation

# Create a separate server instance per task
# The active task is set via TASK_ID env var (default: 1)
task_id = int(os.getenv("TASK_ID", "1"))

server = HTTPEnvServer(
    env=lambda: SupportTriageEnvironment(task_id=task_id),
    action_cls=TriageAction,
    observation_cls=TriageObservation,
    max_concurrent_envs=4,
)

app = create_app(server)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
```

---

## Phase 8: Client (Day 3 — ~1 hour)

**File:** `client.py`

```python
# client.py
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import TriageAction, TriageObservation, TriageState

class TriageEnv(EnvClient[TriageAction, TriageObservation, TriageState]):
    """
    Client-side wrapper for SupportTriageEnv.
    Use this to connect to a running server instance.
    """

    def _step_payload(self, action: TriageAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[TriageObservation]:
        obs = TriageObservation(**payload.get("observation", {}))
        return StepResult(
            observation = obs,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
            info        = payload.get("info", {}),
        )

    def _parse_state(self, payload: dict) -> TriageState:
        return TriageState(**payload)
```

---

## Phase 9: Dockerfile (Day 3 — ~1 hour)

**File:** `server/Dockerfile`

```dockerfile
FROM python:3.11-slim

# Build args
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TASK_ID=1

WORKDIR /app

# Layer 1: system deps (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Python deps (changes when requirements.txt changes)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3: Application code
COPY models.py .
COPY data/ ./data/
COPY tasks/ ./tasks/
COPY graders/ ./graders/
COPY rewards/ ./rewards/
COPY server/ ./server/
COPY README.md .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/').raise_for_status()"

CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**File:** `server/requirements.txt`
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
openenv-core>=0.2.1
python-dotenv>=1.0.0
httpx>=0.25.0
```

---

## Phase 10: Baseline Inference Script (Day 4 — ~3 hours)

**File:** `inference.py` (ROOT of project — NOT in server/)

This is the most scrutinized file. The judges run it directly.

```python
# inference.py
"""
Baseline inference script for SupportTriageEnv.

Runs a GPT-style model against all 3 tasks and prints reproducible scores.

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export HF_TOKEN="hf_xxxx"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    python inference.py

Expected output:
    Task 1 (Easy)   - Score: 0.78 (avg over 5 episodes)
    Task 2 (Medium) - Score: 0.52 (avg over 5 episodes)
    Task 3 (Hard)   - Score: 0.31 (avg over 5 episodes)
"""

import os
import json
import random
import textwrap
from typing import Optional

from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

SEED            = 42
EPISODES_PER_TASK = 5
MAX_STEPS         = 40

# ── Client setup ───────────────────────────────────────────────────────────────
llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support triage manager.
You will receive an inbox of customer emails and must triage them one by one.

For each email, you should:
1. Classify it into one of: billing, technical, account, refund, abuse, general
2. Set priority: P1 (critical, ≤1hr), P2 (high, ≤4hr), P3 (medium, ≤24hr), P4 (low, ≤72hr)
3. Route to a team: billing-team, tech-support, account-management, escalations, general-support
4. Draft a response for P1 and P2 emails
5. Mark duplicates if you see the same issue from same sender
6. Escalate abuse or critical technical issues

Priority guidelines:
- VIP customers (sender_tier=vip) should NEVER be P3 or P4
- Angry tone + billing issue = P2 minimum
- Account security/data loss = P1
- Abuse/harassment = always escalate

Respond ONLY with a JSON object for your next action:
{
  "action_type": "classify",
  "email_id": "email_001",
  "category": "billing"
}

Valid action_types: classify, set_priority, route, draft_response,
                   mark_duplicate, escalate, request_info, submit_triage
""").strip()


def observation_to_prompt(obs_dict: dict) -> str:
    """Convert observation JSON to a clear prompt for the LLM."""
    lines = []

    current = obs_dict.get("current_email")
    if current:
        lines.append("=== CURRENT EMAIL TO PROCESS ===")
        lines.append(f"ID:      {current['email_id']}")
        lines.append(f"From:    {current['sender_email']} ({current['sender_tier']})")
        lines.append(f"Subject: {current['subject']}")
        lines.append(f"Body:\n{current['body']}\n")

    inbox = obs_dict.get("inbox_snapshot", [])
    if inbox:
        lines.append("=== INBOX OVERVIEW ===")
        for e in inbox:
            lines.append(f"  [{e['status']}] {e['email_id']}: {e['subject']} ({e['sender_tier']})")

    lines.append(f"\nStep: {obs_dict.get('step_number')}")
    lines.append(f"Pending: {obs_dict.get('pending_count')} | Processed: {obs_dict.get('processed_count')}")
    lines.append(f"Last result: {obs_dict.get('last_action_result', '')}")
    lines.append("\nWhat is your next action? Respond with JSON only.")

    return "\n".join(lines)


def parse_llm_action(text: str) -> Optional[dict]:
    """Parse LLM output into an action dict. Returns None on failure."""
    text = text.strip()
    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        import re
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def run_episode(env_client, task_id: int) -> float:
    """Run one full episode and return the final score."""
    import httpx

    # Reset
    resp = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    obs  = resp.json()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_score = 0.0

    for step in range(MAX_STEPS):
        # Build prompt
        user_msg = observation_to_prompt(obs)
        messages.append({"role": "user", "content": user_msg})

        # LLM call
        try:
            completion = llm.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                max_tokens  = 300,
                temperature = 0.1,
            )
            assistant_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  LLM error at step {step}: {e}")
            # Fallback: submit_triage
            assistant_text = '{"action_type": "submit_triage"}'

        messages.append({"role": "assistant", "content": assistant_text})

        # Parse action
        action_dict = parse_llm_action(assistant_text)
        if action_dict is None:
            action_dict = {"action_type": "submit_triage"}

        # Step in environment
        try:
            step_resp = httpx.post(f"{ENV_BASE_URL}/step", json=action_dict, timeout=30)
            result    = step_resp.json()
            obs       = result.get("observation", obs)
            done      = result.get("done", False)
            reward    = result.get("reward", 0.0)
        except Exception as e:
            print(f"  Env error at step {step}: {e}")
            break

        if done:
            # Extract final score from info or last result
            info = result.get("info", {})
            last_score = info.get("final_score", 0.0)
            break

    return last_score


def main():
    random.seed(SEED)
    print(f"SupportTriageEnv Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Episodes per task: {EPISODES_PER_TASK}")
    print("=" * 50)

    results = {}

    for task_id in [1, 2, 3]:
        task_names = {1: "Easy — Single Classification",
                      2: "Medium — Batch Triage",
                      3: "Hard — Full Pipeline"}
        print(f"\nTask {task_id}: {task_names[task_id]}")

        scores = []
        for ep in range(EPISODES_PER_TASK):
            score = run_episode(None, task_id)
            scores.append(score)
            print(f"  Episode {ep+1}: {score:.3f}")

        avg = sum(scores) / len(scores)
        results[task_id] = avg
        print(f"  AVERAGE: {avg:.3f}")

    print("\n" + "=" * 50)
    print("FINAL SCORES:")
    print(f"  Task 1 (Easy):   {results[1]:.3f}")
    print(f"  Task 2 (Medium): {results[2]:.3f}")
    print(f"  Task 3 (Hard):   {results[3]:.3f}")
    print(f"  Overall Mean:    {sum(results.values())/3:.3f}")


if __name__ == "__main__":
    main()
```

---

## Phase 11: Tests (Day 4 — ~2 hours)

Minimum test files needed for validation:

### `tests/test_environment.py`
```python
def test_reset_returns_observation():
    env = SupportTriageEnvironment(task_id=1)
    obs = env.reset()
    assert obs.current_email is not None
    assert obs.episode_id != ""
    assert obs.step_number == 0
    assert len(obs.inbox_snapshot) > 0

def test_step_with_valid_classify_action():
    env = SupportTriageEnvironment(task_id=1)
    obs = env.reset()
    email_id = obs.current_email.email_id
    result_obs = env.step(TriageAction(
        action_type=ActionType.CLASSIFY,
        email_id=email_id,
        category=Category.BILLING
    ))
    assert result_obs.last_action_valid == True
    assert result_obs.step_number == 1

def test_state_returns_episode_metadata():
    env = SupportTriageEnvironment(task_id=1)
    env.reset()
    state = env.state
    assert state.step_count == 0
    assert state.task_id == 1
    assert state.is_done == False
```

---

## Phase 12: HF Space Deployment (Day 5)

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create the Space
huggingface-cli repo create support-triage-env --type space --space_sdk docker

# Push using OpenEnv CLI
openenv push --repo-id YOUR_USERNAME/support-triage-env

# Or manually with git
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/support-triage-env
git push space main
```

The HF Space `README.md` (with YAML front matter) must include:
```yaml
---
title: Support Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---
```

---

## Phase 13: openenv.yaml (Day 5)

Create this file in the project root based on the template in `design.md`.

Then validate:
```bash
openenv validate
```

Fix any errors before submitting.

---

## Final Checklist

```bash
# 1. Validate spec
openenv validate

# 2. Build container
docker build -t support-triage-env ./server

# 3. Start container
docker run -p 8000:8000 support-triage-env &
sleep 5

# 4. Test reset
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'

# 5. Test step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "email_id": "email_001", "category": "billing"}'

# 6. Run tests
pytest tests/ -v

# 7. Run inference (set env vars first)
python inference.py

# 8. Check runtime < 20 minutes
time python inference.py
```
