# reward_design.md — Reward Function Design

## Core Philosophy

The judges explicitly look for "non-sparse reward" (20% of environment design score).
This means the agent MUST get useful signal at EVERY step, not just at the end.

Our reward function has 3 layers:
1. **Step-level reward** — immediate feedback per action
2. **Trajectory-level penalties** — discourages bad patterns over the episode
3. **Completion bonus** — reward for finishing cleanly

---

## Reward Components

### Layer 1: Step-Level Rewards

These are computed inside `RewardCalculator.compute_step_reward(action, state, ground_truth)`:

#### 1.1 Classification Reward
```python
def _classification_reward(action, ground_truth) -> float:
    """Reward for a classify action."""
    if action.action_type != ActionType.CLASSIFY:
        return 0.0
    if action.email_id is None or action.category is None:
        return -0.05  # Malformed action

    truth = ground_truth[action.email_id]

    if action.category == truth.category:
        # Exact match: full reward, scaled by email difficulty
        base = 0.25
        difficulty_bonus = {1: 0.0, 2: 0.05, 3: 0.10}
        return base + difficulty_bonus.get(truth.difficulty_level, 0.0)

    if _are_adjacent_categories(action.category, truth.category):
        return 0.08  # Partial credit: got in the right neighborhood

    return 0.0  # Wrong category: no reward (but no penalty either)
```

**Range:** 0.0 to +0.35

#### 1.2 Priority Reward
```python
def _priority_reward(action, ground_truth, email_metadata) -> float:
    """Reward for a set_priority action."""
    if action.action_type != ActionType.SET_PRIORITY:
        return 0.0

    truth = ground_truth[action.email_id]
    priority_ranks = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

    submitted_rank = priority_ranks.get(action.priority, 0)
    truth_rank     = priority_ranks.get(truth.priority, 0)
    diff           = abs(submitted_rank - truth_rank)

    if diff == 0:
        reward = 0.20
        # Bonus for getting P1 correct (hardest to detect, most impactful)
        if truth.priority == "P1":
            reward += 0.05
        return reward

    if diff == 1:
        return 0.07  # Close but not exact

    if diff == 2:
        return 0.02  # Far off

    # diff >= 3: completely wrong
    return 0.0
```

**Range:** 0.0 to +0.25

#### 1.3 Routing Reward
```python
def _routing_reward(action, ground_truth) -> float:
    """Reward for a route action."""
    if action.action_type != ActionType.ROUTE:
        return 0.0

    truth = ground_truth[action.email_id]

    if action.team == truth.team:
        return 0.20  # Full reward: correct team

    # Check if routed to a "reasonable" team (not completely wrong)
    reasonable_alternatives = REASONABLE_ROUTING_ALTERNATIVES.get(truth.team, set())
    if action.team in reasonable_alternatives:
        return 0.05  # Minor partial credit

    return 0.0
```

**REASONABLE_ROUTING_ALTERNATIVES mapping:**
```python
REASONABLE_ROUTING_ALTERNATIVES = {
    "billing-team":        {"account-management"},  # close, but wrong
    "tech-support":        {"general-support"},      # could happen
    "account-management":  {"billing-team"},          # close
    "escalations":         {"tech-support"},          # not great but understandable
    "general-support":     {"account-management"},
}
```

**Range:** 0.0 to +0.20

#### 1.4 Response Quality Reward
```python
def _response_reward(action, ground_truth, email_objects) -> float:
    """Reward for a draft_response action."""
    if action.action_type != ActionType.DRAFT_RESPONSE:
        return 0.0

    truth = ground_truth[action.email_id]

    # Drafting a response for P3/P4 email = minor negative (not needed)
    if truth.priority in ("P3", "P4"):
        return -0.03  # Slight penalty for unnecessary response

    if action.response_text is None or len(action.response_text.strip()) < 20:
        return -0.05  # Empty/trivial response

    email_obj      = email_objects[action.email_id]
    quality_score  = ResponseGrader().quality_score(action.response_text, email_obj)

    # Scale quality score to reward range [0.0, +0.20]
    return quality_score * 0.20
```

**Range:** -0.05 to +0.20

#### 1.5 Escalation Reward
```python
def _escalation_reward(action, ground_truth) -> float:
    """Reward for an escalate action."""
    if action.action_type != ActionType.ESCALATE:
        return 0.0

    truth = ground_truth[action.email_id]

    if truth.requires_escalation:
        return 0.15  # Correctly identified an escalation case
    else:
        return -0.10  # False escalation: wastes escalations team's time
```

**Range:** -0.10 to +0.15

#### 1.6 Duplicate Detection Reward
```python
def _duplicate_reward(action, ground_truth) -> float:
    """Reward for mark_duplicate action."""
    if action.action_type != ActionType.MARK_DUPLICATE:
        return 0.0

    truth = ground_truth[action.email_id]

    if not truth.is_duplicate:
        return -0.15  # False positive: incorrectly marked as duplicate

    if action.original_email_id == truth.original_email_id:
        return 0.20   # Correctly identified AND linked to right original

    if truth.is_duplicate and action.original_email_id != truth.original_email_id:
        return 0.05   # Detected it's a duplicate but linked to wrong original

    return 0.0
```

**Range:** -0.15 to +0.20

---

### Layer 2: Trajectory-Level Penalties

These are checked at every step and applied immediately:

#### 2.1 Repeat Action Penalty
```python
def _repeat_action_penalty(action, processed_actions) -> float:
    """Penalize agent for repeating the same action type on same email."""
    key = (action.email_id, action.action_type)
    if key in processed_actions:
        return -0.05  # Already did this action on this email
    return 0.0
```

#### 2.2 Invalid Action Penalty
```python
def _invalid_action_penalty(action, current_state) -> float:
    """Penalize agent for taking an action that's not currently valid."""
    # e.g., routing before classifying, or acting on an email not in inbox
    if not _is_action_valid(action, current_state):
        return -0.05
    return 0.0
```

#### 2.3 VIP Downgrade Penalty
```python
def _vip_penalty(action, email_metadata, ground_truth) -> float:
    """Heavy penalty for assigning low priority to VIP customer."""
    if action.action_type != ActionType.SET_PRIORITY:
        return 0.0

    email = email_metadata.get(action.email_id)
    if email and email.sender_tier == "vip":
        if action.priority in ("P3", "P4"):
            return -0.30  # Severe: VIP customer gets ignored = CSAT disaster

    return 0.0
```

---

### Layer 3: Completion Bonus

Applied when `submit_triage()` is called OR max_steps reached:

```python
def _completion_bonus(processed_set, inbox_emails, ground_truth) -> float:
    """Bonus for a clean, complete triage session."""
    if len(inbox_emails) == 0:
        return 0.0

    # What fraction of emails got all required actions?
    complete_count = 0
    for email_id in inbox_emails:
        record = processed_set.get(email_id, {})
        required = _get_required_actions(email_id, ground_truth)
        if all(action_type in record.actions_taken for action_type in required):
            complete_count += 1

    completion_ratio = complete_count / len(inbox_emails)

    # Bonus scales with completion: 0.25 max if 100% complete
    return 0.25 * completion_ratio
```

---

## Full Reward Calculation

```python
class RewardCalculator:
    """
    Computes per-step reward for any action in any task.
    """

    def compute(
        self,
        action: TriageAction,
        state: TriageState,
        email_objects: Dict[str, EmailObject],
        ground_truth: Dict[str, GroundTruthRecord],
        processed_actions: Set[Tuple[str, str]],
    ) -> float:

        if action.action_type == ActionType.SUBMIT_TRIAGE:
            return self._completion_bonus(
                state.processed_set,
                state.inbox_emails,
                ground_truth,
            )

        reward = 0.0

        # Step-level rewards
        reward += self._classification_reward(action, ground_truth)
        reward += self._priority_reward(action, ground_truth, email_objects)
        reward += self._routing_reward(action, ground_truth)
        reward += self._response_reward(action, ground_truth, email_objects)
        reward += self._escalation_reward(action, ground_truth)
        reward += self._duplicate_reward(action, ground_truth)

        # Trajectory penalties
        reward += self._repeat_action_penalty(action, processed_actions)
        reward += self._invalid_action_penalty(action, state)
        reward += self._vip_penalty(action, email_objects, ground_truth)

        # Clamp to reasonable range (prevent extreme values)
        return max(-1.0, min(1.0, reward))
```

---

## Reward Ranges Per Task

### Task 1 (Max possible per episode)
```
classify correct (difficulty 1):    +0.25
set_priority correct:               +0.20
submit_triage (100% complete):      +0.25
────────────────────────────────────────
Maximum possible reward:             0.70
Minimum (all wrong):                -0.15
Random agent expected:              ~0.05
GPT-4o expected:                    ~0.55
```

### Task 2 (Max possible per episode, 5 emails)
```
Per email: classify+priority+route all correct = 0.25+0.20+0.20 = 0.65
× 5 emails = 3.25
Completion bonus (100%):            +0.25
────────────────────────────────────────
Maximum (summed over episode):       3.50
Average per step:                   ~0.23 (for 15 steps)
```

### Task 3 (Max possible per episode, 10 emails)
```
Per standard email:                  0.65
P1 email with draft response:        0.65 + 0.20 (draft) = 0.85
Escalation (abuse email):            0.65 + 0.15 (escalate) = 0.80
Duplicate detection:                 0.65 + 0.20 (duplicate) = 0.85
Completion bonus:                   +0.25
────────────────────────────────────────
Maximum (summed over episode):       ~8.0 (over 40 steps)
```

---

## Why This Reward is "Meaningful"

Per the judging criteria ("Reward function provides useful varying signal — not just sparse"):

1. **Every classify/route/priority action gets immediate feedback.** The agent knows within 1 step whether its decision was right.

2. **Partial credit for near-misses.** Adjacent categories and adjacent priorities get partial reward rather than 0. This creates a smooth gradient for learning.

3. **Penalties are informative.** The VIP penalty (-0.30) is large enough to teach the agent to check `sender_tier` before assigning priority. The false escalation penalty (-0.10) teaches restraint.

4. **The completion bonus incentivizes coverage.** Agent can't just do perfect triage on 2 emails and ignore the rest; it needs to process all emails to maximize the bonus.

5. **No cliff effects.** Reward never jumps from 0 to 1 in one step. Every intermediate action contributes.
