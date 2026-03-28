from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Action types accepted by the environment."""

    CLASSIFY = "classify"
    SET_PRIORITY = "set_priority"
    ROUTE = "route"
    DRAFT_RESPONSE = "draft_response"
    MARK_DUPLICATE = "mark_duplicate"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"
    SUBMIT_TRIAGE = "submit_triage"


class Category(str, Enum):
    """Support issue categories."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    REFUND = "refund"
    ABUSE = "abuse"
    GENERAL = "general"


class Priority(str, Enum):
    """Ticket urgency levels."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Team(str, Enum):
    """Support team destinations."""

    BILLING = "billing-team"
    TECH = "tech-support"
    ACCOUNT = "account-management"
    ESCALATIONS = "escalations"
    GENERAL = "general-support"


class EmailStatus(str, Enum):
    """Lifecycle status of an email in an episode."""

    PENDING = "pending"
    CLASSIFIED = "classified"
    PRIORITIZED = "prioritized"
    ROUTED = "routed"
    RESPONDED = "responded"
    COMPLETE = "complete"
    DUPLICATE = "duplicate"


class SenderTier(str, Enum):
    """Customer tier."""

    VIP = "vip"
    REGULAR = "regular"
    FIRST_TIME = "first_time"


class EmailObject(BaseModel):
    """Full email object visible to agent for current step."""

    email_id: str = Field(..., description="Unique email ID")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email message body")
    sender_email: str = Field(..., description="Sender email address")
    sender_tier: SenderTier = Field(..., description="Customer tier")
    timestamp: datetime = Field(..., description="Original email timestamp")
    thread_history: List[str] = Field(default_factory=list, description="Previous thread messages")
    attachments_mentioned: bool = Field(default=False, description="Whether attachment is mentioned")
    language: str = Field(default="en", description="Language code")


class EmailSummary(BaseModel):
    """Compact inbox view row."""

    email_id: str = Field(..., description="Email ID")
    subject: str = Field(..., description="Email subject")
    sender_tier: SenderTier = Field(..., description="Customer tier")
    status: EmailStatus = Field(..., description="Current triage status")
    actions_taken: List[str] = Field(default_factory=list, description="Action history")


class TriageAction(BaseModel):
    """Single agent action for a step."""

    action_type: ActionType = Field(..., description="Action type")
    category: Optional[Category] = Field(default=None, description="Category for classify")
    email_id: Optional[str] = Field(default=None, description="Target email ID")
    priority: Optional[Priority] = Field(default=None, description="Priority for set_priority")
    team: Optional[Team] = Field(default=None, description="Team for route")
    response_text: Optional[str] = Field(default=None, description="Draft response text")
    original_email_id: Optional[str] = Field(default=None, description="Original email for duplicate link")
    reason: Optional[str] = Field(default=None, description="Escalation reason")
    question: Optional[str] = Field(default=None, description="Clarification question")


class TriageObservation(BaseModel):
    """Observation returned by reset and step."""

    current_email: Optional[EmailObject] = Field(
        default=None,
        description="Current email the agent should act on",
    )
    inbox_snapshot: List[EmailSummary] = Field(
        default_factory=list,
        description="Snapshot of all emails and status",
    )
    processed_count: int = Field(default=0, description="Count of fully triaged emails")
    pending_count: int = Field(default=0, description="Count of remaining emails")
    team_queue_depths: Dict[str, int] = Field(
        default_factory=dict,
        description="Current queue depth for each team",
    )
    last_action_result: str = Field(default="", description="Action result summary")
    last_action_valid: bool = Field(default=True, description="Whether the previous action was valid")
    step_number: int = Field(default=0, description="Step index in episode")
    episode_id: str = Field(default="", description="Current episode ID")
    available_actions: List[str] = Field(default_factory=list, description="Valid actions for current context")
    task_id: int = Field(default=1, description="Task identifier")
    hint: Optional[str] = Field(default=None, description="Optional instructional hint")


class TriageState(BaseModel):
    """Episode metadata returned by state."""

    episode_id: str = Field(default_factory=lambda: str(uuid4()), description="Episode UUID")
    step_count: int = Field(default=0, description="Number of steps taken")
    task_id: int = Field(default=1, description="Task ID")
    max_steps: int = Field(default=3, description="Maximum steps in episode")
    emails_in_inbox: int = Field(default=0, description="Emails sampled for episode")
    emails_processed: int = Field(default=0, description="Count fully processed")
    cumulative_reward: float = Field(default=0.0, description="Sum of rewards in episode")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Episode start timestamp")
    is_done: bool = Field(default=False, description="Episode completion flag")


class GroundTruthRecord(BaseModel):
    """Ground truth labels for a single email."""

    email_id: str = Field(..., description="Email ID")
    category: Category = Field(..., description="True category")
    priority: Priority = Field(..., description="True priority")
    team: Team = Field(..., description="True routing team")
    is_duplicate: bool = Field(default=False, description="Whether email is duplicate")
    original_email_id: Optional[str] = Field(default=None, description="Primary email if duplicate")
    requires_escalation: bool = Field(default=False, description="Whether escalation is required")
    sla_breach_risk: bool = Field(default=False, description="Whether issue has SLA breach risk")
    difficulty_level: Literal[1, 2, 3] = Field(default=1, description="Difficulty stratum")


class TriageRecord(BaseModel):
    """Agent action trace for one email during an episode."""

    email_id: str = Field(..., description="Email ID")
    category: Optional[Category] = Field(default=None, description="Submitted category")
    priority: Optional[Priority] = Field(default=None, description="Submitted priority")
    team: Optional[Team] = Field(default=None, description="Submitted team")
    response_text: Optional[str] = Field(default=None, description="Submitted response draft")
    requested_info: bool = Field(default=False, description="Whether clarification was requested")
    marked_duplicate: bool = Field(default=False, description="Whether duplicate was marked")
    duplicate_of: Optional[str] = Field(default=None, description="Duplicate mapping")
    escalated: bool = Field(default=False, description="Whether escalated")
    escalation_reason: Optional[str] = Field(default=None, description="Escalation reason")
    actions_taken: List[str] = Field(default_factory=list, description="Action history")
    status: EmailStatus = Field(default=EmailStatus.PENDING, description="Current status")


class TriageReward(BaseModel):
    """Structured reward debug object used in info payloads."""

    value: float = Field(..., description="Reward value for step")
    reason: str = Field(..., description="Human-readable explanation")
    components: Dict[str, float] = Field(default_factory=dict, description="Component breakdown")
