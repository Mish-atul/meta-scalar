# tasks_and_graders.md — Task Specifications & Grader Logic

## Overview

SupportTriageEnv has 3 tasks with increasing difficulty. Each task:
- Defines a concrete objective the agent must accomplish
- Defines which emails appear in the episode inbox
- Has a deterministic programmatic grader producing scores in [0.0, 1.0]
- Has clear success criteria and difficulty levels

All graders are DETERMINISTIC: same input always produces same output. No LLM calls.

---

## Task 1: Single-Email Classification (Easy)

### Objective
Given a single unambiguous email, the agent must correctly identify its category and priority.

### Episode Configuration
```python
TASK_1_CONFIG = {
    "task_id": 1,
    "inbox_size": 1,
    "max_steps": 3,
    "email_pool": "easy_unambiguous",  # filtered from full dataset
    "required_actions": ["classify", "set_priority"],
    "optional_actions": ["request_info"],
    "forbidden_actions": ["draft_response", "escalate"],  # too advanced
    "hint_visible": True,  # show difficulty hint in observation
}
```

### What "Fully Solved" Looks Like
```
Step 1: classify(email_id="email_001", category="billing")     → reward ~0.35
Step 2: set_priority(email_id="email_001", priority="P2")      → reward ~0.25
Step 3: submit_triage()                                        → completion bonus ~0.15
─────────────────────────────────────────────────────────────────
Total episode reward: ~0.75   |   Grader score: 1.0
```

### Grader: ClassificationGrader (Task 1)

```python
class Task1Grader:
    """
    Scores a Task 1 episode.

    Score breakdown:
      - Correct category:          0.60
      - Adjacent category:         0.20  (e.g. got "account" for "billing")
      - Correct priority:          0.40
      - Adjacent priority (±1):    0.15
      - Exact match both:          1.00
    """

    CATEGORY_ADJACENCY = {
        # categories that are "close enough" for partial credit
        ("billing", "account"):    True,
        ("technical", "general"):  True,
        ("refund", "billing"):     True,
        ("abuse", "general"):      False,  # these are NOT adjacent
    }

    PRIORITY_RANKS = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

    def score(
        self,
        submitted_category: Optional[str],
        submitted_priority: Optional[str],
        ground_truth: GroundTruthRecord,
    ) -> float:
        score = 0.0

        # --- Category scoring ---
        if submitted_category is None:
            pass  # 0 points for not classifying
        elif submitted_category == ground_truth.category:
            score += 0.60
        elif self._are_adjacent(submitted_category, ground_truth.category):
            score += 0.20

        # --- Priority scoring ---
        if submitted_priority is None:
            pass  # 0 points for not prioritizing
        elif submitted_priority == ground_truth.priority:
            score += 0.40
        else:
            submitted_rank = self.PRIORITY_RANKS.get(submitted_priority, 0)
            truth_rank     = self.PRIORITY_RANKS.get(ground_truth.priority, 0)
            if abs(submitted_rank - truth_rank) == 1:
                score += 0.15
            # More than 1 off: 0 additional points

        return min(score, 1.0)

    def _are_adjacent(self, cat_a: str, cat_b: str) -> bool:
        pair = tuple(sorted([cat_a, cat_b]))
        return self.CATEGORY_ADJACENCY.get(pair, False)
```

### Difficulty Calibration
- **GPT-4o baseline expected score:** 0.75–0.90
- **Random agent expected score:** ~0.10
- Emails in this pool have clear, unambiguous language

---

## Task 2: Batch Priority + Routing (Medium)

### Objective
Given a batch of 5 emails, correctly classify, prioritize, and route ALL of them
to the appropriate support team.

### Episode Configuration
```python
TASK_2_CONFIG = {
    "task_id": 2,
    "inbox_size": 5,
    "max_steps": 15,
    "email_pool": "medium_mixed",
    "required_actions": ["classify", "set_priority", "route"],
    "optional_actions": ["request_info", "mark_duplicate"],
    "forbidden_actions": ["draft_response"],
    "hint_visible": False,
    "guaranteed_composition": {
        "vip_emails": 1,       # exactly 1 VIP customer
        "edge_case_emails": 1, # 1 email ambiguous between 2 categories
        "standard_emails": 3,  # 3 clear-cut emails
    }
}
```

### What "Fully Solved" Looks Like
```
For each of 5 emails, agent must do:
  classify   → +reward
  set_priority → +reward
  route       → +reward

Step 1:  classify(email_001, "billing")          → +0.22
Step 2:  set_priority(email_001, "P2")           → +0.16
Step 3:  route(email_001, "billing-team")        → +0.16
Step 4:  classify(email_002, "technical")        → +0.22
Step 5:  set_priority(email_002, "P1")           → +0.16
Step 6:  route(email_002, "tech-support")        → +0.16
... (repeat for emails 003, 004, 005)
Step 15: submit_triage()                          → +completion bonus
─────────────────────────────────────────────────────────────────
Total grader score: 1.0
```

### Team Routing Logic (for agent to learn)

This is the routing matrix the ground truth is built on:

| Category    | Default Team          | If P1 or VIP            |
|-------------|-----------------------|-------------------------|
| billing     | billing-team          | billing-team            |
| technical   | tech-support          | tech-support + escalate |
| account     | account-management    | account-management      |
| refund      | billing-team          | billing-team            |
| abuse       | escalations           | escalations             |
| general     | general-support       | general-support         |

**Edge case rules (traps for the agent):**
- A billing issue from a VIP customer with "I'm canceling" in the body → route to `account-management` (retention)
- A technical issue that also involves unauthorized charges → route to `billing-team` (billing takes precedence)
- A "general feedback" that has profanity → route to `escalations` (detected by content keywords)

### Grader: Task2Grader

```python
class Task2Grader:
    """
    Scores a Task 2 episode using macro-averaged per-email scores.

    Per-email score:
      - Classification correct:   0.33
      - Priority correct:         0.33 (adjacent: 0.10)
      - Routing correct:          0.34

    Final score:
      = mean(per_email_scores) × completion_ratio

    completion_ratio = emails_acted_on / total_emails_in_inbox
    (Penalizes agent for ignoring emails entirely)
    """

    def score(
        self,
        triage_records: Dict[str, TriageRecord],  # email_id → what agent did
        ground_truth: Dict[str, GroundTruthRecord],
        inbox_emails: List[str],
    ) -> float:
        per_email = []

        for email_id in inbox_emails:
            record = triage_records.get(email_id)
            truth  = ground_truth[email_id]
            s = 0.0

            if record is None:
                # Agent never acted on this email
                per_email.append(0.0)
                continue

            # Classification
            if record.category == truth.category:
                s += 0.33
            elif self._adjacent_category(record.category, truth.category):
                s += 0.10

            # Priority
            if record.priority == truth.priority:
                s += 0.33
            elif record.priority and truth.priority:
                diff = abs(self.RANKS[record.priority] - self.RANKS[truth.priority])
                if diff == 1:
                    s += 0.10

            # Routing
            if record.team == truth.team:
                s += 0.34
            # No partial credit for routing (team assignment is explicit)

            per_email.append(s)

        raw_mean          = sum(per_email) / len(inbox_emails) if inbox_emails else 0.0
        completion_ratio  = len([e for e in inbox_emails if e in triage_records]) / len(inbox_emails)
        return min(raw_mean * completion_ratio, 1.0)
```

### Difficulty Calibration
- **GPT-4o baseline expected score:** 0.45–0.65
- **Key challenge:** The edge cases (VIP routing override, billing+technical conflict)
- **Trap:** Agent that just memorizes category→team mapping will fail edge cases

---

## Task 3: Full Pipeline with SLA + Draft Response (Hard)

### Objective
Complete triage of 8–10 emails including: classification, prioritization, routing,
draft responses (for P1/P2), duplicate detection, and escalations.

### Episode Configuration
```python
TASK_3_CONFIG = {
    "task_id": 3,
    "inbox_size_range": (8, 10),   # Varies per episode
    "max_steps": 40,
    "email_pool": "hard_full_pipeline",
    "required_actions": [
        "classify", "set_priority", "route",
        "draft_response",  # required for P1 and P2 emails
        "mark_duplicate",  # required for duplicate pairs
        "escalate",        # required for abuse + critical technical
    ],
    "optional_actions": ["request_info"],
    "hint_visible": False,
    "guaranteed_composition": {
        "p1_emails": 1,           # 1 critical email requiring immediate draft
        "vip_emails": 1,          # 1 VIP customer
        "duplicate_pair": 1,      # 1 pair of duplicates (2 emails)
        "sla_breach_risk": 1,     # 1 email with explicit urgency language
        "abuse_email": 1,         # 1 requiring escalation
        "ambiguous_emails": 2,    # 2 edge cases
        "standard_emails": 2,     # 2 routine emails
    }
}
```

### Score Components (Composite)

| Component               | Weight | Grader                          |
|-------------------------|--------|---------------------------------|
| Classification F1       | 20%    | ClassificationGrader.f1_score() |
| Priority Accuracy       | 20%    | PriorityGrader.weighted_score() |
| Routing Accuracy        | 20%    | RoutingGrader.accuracy()        |
| Response Quality        | 20%    | ResponseGrader.quality_score()  |
| Escalation Recall       | 10%    | EscalationGrader.recall()       |
| Duplicate Detection     | 10%    | DuplicateGrader.precision()     |

**Penalty deductions (applied AFTER component scoring):**
- `-0.15` per P1 email where no `draft_response` was submitted
- `-0.20` per VIP email assigned P3 or P4
- `-0.10` per email where `submit_triage` was called without classify+priority+route

```python
final_score = (
    0.20 * classification_f1
  + 0.20 * priority_weighted
  + 0.20 * routing_accuracy
  + 0.20 * response_quality_mean
  + 0.10 * escalation_recall
  + 0.10 * duplicate_precision
  - (sla_penalty * count_p1_without_response)
  - (vip_penalty * count_vip_downgraded)
  - (incomplete_penalty * count_incomplete_triaged)
)
return max(0.0, min(1.0, final_score))
```

### Grader: ResponseGrader (Programmatic, No LLM)

```python
class ResponseGrader:
    """
    Scores draft responses written by the agent.
    Uses deterministic text pattern checks only.
    """

    GREETING_PATTERNS = [
        r"dear\s+\w+",
        r"hello\s+\w+",
        r"hi\s+\w+",
        r"greetings",
        r"thank you for (contacting|reaching out|writing)",
    ]

    ACKNOWLEDGMENT_KEYWORDS_BY_CATEGORY = {
        "billing":   ["charge", "payment", "invoice", "account", "billing"],
        "technical": ["issue", "problem", "error", "bug", "not working"],
        "account":   ["account", "login", "access", "password", "settings"],
        "refund":    ["refund", "cancel", "money", "return"],
        "abuse":     ["report", "concern", "behavior", "policy"],
        "general":   ["feedback", "question", "information"],
    }

    UNPROFESSIONAL_WORDS = [
        "wtf", "dumb", "stupid", "idiot", "shut up",
        "not my problem", "deal with it", "whatever",
    ]

    CLOSING_PATTERNS = [
        r"best regards", r"sincerely", r"thank you",
        r"warm regards", r"kind regards", r"regards",
    ]

    def quality_score(self, response_text: str, email: EmailObject) -> float:
        """Returns score in [0.0, 1.0] based on 7 programmatic checks."""
        text_lower = response_text.lower()
        words      = response_text.split()

        checks = {}

        # 1. Has a greeting
        checks["greeting"] = any(
            re.search(p, text_lower) for p in self.GREETING_PATTERNS
        )

        # 2. Acknowledges the specific issue type
        issue_keywords = self.ACKNOWLEDGMENT_KEYWORDS_BY_CATEGORY.get(
            email.expected_category, []
        )
        checks["acknowledges_issue"] = any(kw in text_lower for kw in issue_keywords)

        # 3. Not too short (under 30 words = not helpful)
        checks["sufficient_length"] = len(words) >= 30

        # 4. Not too long (over 400 words = overwhelming)
        checks["not_too_long"] = len(words) <= 400

        # 5. Professional tone (no unprofessional language)
        checks["professional"] = not any(w in text_lower for w in self.UNPROFESSIONAL_WORDS)

        # 6. No placeholder text still in the response
        checks["no_placeholders"] = not bool(
            re.search(r"\[.*?\]|TODO|PLACEHOLDER|INSERT", response_text)
        )

        # 7. Has a closing
        checks["has_closing"] = any(
            re.search(p, text_lower) for p in self.CLOSING_PATTERNS
        )

        return sum(checks.values()) / len(checks)

    def score_all_responses(
        self,
        responses: Dict[str, str],          # email_id → response_text
        ground_truth: Dict[str, GroundTruthRecord],
        p1_p2_emails: List[str],
    ) -> float:
        """
        Average quality score across all P1/P2 emails.
        Emails without a response submitted score 0.0.
        """
        if not p1_p2_emails:
            return 1.0  # No responses required = perfect

        scores = []
        for email_id in p1_p2_emails:
            if email_id in responses:
                # Find the EmailObject from inbox (we need category for scoring)
                email_obj = self._get_email(email_id)
                scores.append(self.quality_score(responses[email_id], email_obj))
            else:
                scores.append(0.0)  # Missing response

        return sum(scores) / len(scores)
```

### Difficulty Calibration
- **GPT-4o baseline expected score:** 0.25–0.45
- **Key challenges:**
  - Must remember which emails are P1 to draft responses
  - Must detect duplicate without being explicitly told
  - Must escalate abuse emails which look like regular complaints
  - VIP penalty is severe and easy to trigger if not checking sender_tier
- **Frontier model score:** Estimated 0.45–0.60 for Claude 3.5 Sonnet

---

## Grader Unit Tests (Required)

All graders must pass these test cases:

### ClassificationGrader tests
```python
def test_exact_match():
    grader = ClassificationGrader()
    assert grader.score("billing", "billing") == 1.0

def test_wrong_category():
    assert grader.score("technical", "billing") == 0.0

def test_adjacent_category():
    score = grader.score("account", "billing")
    assert 0.0 < score < 0.5  # Partial credit

def test_none_submitted():
    assert grader.score(None, "billing") == 0.0
```

### PriorityGrader tests
```python
def test_exact_priority():
    assert grader.score("P2", "P2") == 1.0

def test_adjacent_priority():
    score = grader.score("P1", "P2")
    assert 0.0 < score < 1.0

def test_vip_downgrade_detected():
    # VIP downgrade to P3/P4 should be detectable
    result = grader.check_vip_violation(
        email={"sender_tier": "vip"},
        assigned_priority="P3"
    )
    assert result == True  # Violation detected
```

### ResponseGrader tests
```python
def test_good_response():
    response = """Dear John,

    Thank you for reaching out to us. We understand you have a billing concern
    and we take these matters seriously. Our billing team will review your account
    and get back to you within 4 hours.

    Best regards,
    Support Team"""
    score = grader.quality_score(response, email_with_billing_issue)
    assert score >= 0.80

def test_placeholder_response():
    response = "Dear [Customer Name], regarding your [ISSUE TYPE]..."
    score = grader.quality_score(response, mock_email)
    assert score < 0.60  # Penalized for placeholders

def test_unprofessional_response():
    response = "Hi, this is not my problem, deal with it."
    score = grader.quality_score(response, mock_email)
    assert score < 0.30
```

---

## Baseline Score Summary

These are the reproducible baseline scores inference.py must produce:

| Task   | Expected Score Range | What Baseline Agent Does                                    |
|--------|----------------------|-------------------------------------------------------------|
| Task 1 | 0.70 – 0.85          | GPT-4o reads email, classifies correctly most of the time   |
| Task 2 | 0.40 – 0.60          | Handles standard emails well, fails on edge cases           |
| Task 3 | 0.20 – 0.40          | Often misses duplicate detection, skips some draft responses|

These scores show the environment is neither trivial (would be boring for RL) nor
impossibly hard (agent would get no learning signal).
