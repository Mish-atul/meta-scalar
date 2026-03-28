"""Generate synthetic email dataset and ground truth labels.

Usage:
    python -m support_triage_env.data.generate_dataset
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from support_triage_env.models import Category, Priority, SenderTier, Team

SEED = 42
TARGET_SIZE = 240


@dataclass(frozen=True)
class Template:
    category: Category
    subject: str
    body: str
    priority: Priority
    team: Team
    difficulty: int
    requires_escalation: bool = False
    is_ambiguous: bool = False
    duplicate_group: str | None = None


NAMES = [
    "Alice Johnson",
    "Bob Smith",
    "Carlos Garcia",
    "Divya Patel",
    "Elena Rossi",
    "Farhan Khan",
    "Grace Lee",
    "Hiro Tanaka",
    "Ivy Chen",
    "Jonas Meyer",
]

DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "proton.me"]


def build_templates() -> List[Template]:
    return [
        Template(Category.BILLING, "Unexpected invoice charge", "I was charged twice this month. Please investigate.", Priority.P2, Team.BILLING, 1),
        Template(Category.BILLING, "Refund still not processed", "My refund was promised 7 days ago and I have no update.", Priority.P2, Team.BILLING, 1),
        Template(Category.TECHNICAL, "App crashes on launch", "The app crashes immediately after login with error code 5002.", Priority.P1, Team.TECH, 1, requires_escalation=True),
        Template(Category.TECHNICAL, "2FA loop issue", "I cannot complete 2FA and I am locked in a verification loop.", Priority.P2, Team.TECH, 2),
        Template(Category.ACCOUNT, "Cannot update profile", "I get a permissions error when changing account settings.", Priority.P3, Team.ACCOUNT, 1),
        Template(Category.ACCOUNT, "Unauthorized login alert", "I received an alert for unknown sign-in from another country.", Priority.P1, Team.ACCOUNT, 2, requires_escalation=True),
        Template(Category.REFUND, "Cancel and refund request", "Please cancel my annual plan and issue a prorated refund.", Priority.P2, Team.BILLING, 1),
        Template(Category.REFUND, "Duplicate order refund", "I accidentally purchased twice, please refund one order.", Priority.P3, Team.BILLING, 2),
        Template(Category.ABUSE, "Harassment report", "Another user sent abusive language in chat. Please review urgently.", Priority.P1, Team.ESCALATIONS, 2, requires_escalation=True),
        Template(Category.ABUSE, "Profanity in forum thread", "A public post contains targeted abuse and slurs.", Priority.P2, Team.ESCALATIONS, 2, requires_escalation=True),
        Template(Category.GENERAL, "Feature feedback", "I love the product but would like better export options.", Priority.P4, Team.GENERAL, 1),
        Template(Category.GENERAL, "Question about roadmap", "Can you share whether SSO is on the upcoming roadmap?", Priority.P3, Team.GENERAL, 1),
        Template(Category.BILLING, "Billing and login blocked", "My card was billed and now I also cannot log in. Need urgent help.", Priority.P1, Team.BILLING, 3, is_ambiguous=True),
        Template(Category.TECHNICAL, "Data loss after update", "After update, some records disappeared. Also invoice seems wrong.", Priority.P1, Team.TECH, 3, is_ambiguous=True),
        Template(Category.ACCOUNT, "VIP account cancellation threat", "I am a premium enterprise customer and will cancel if this is not fixed today.", Priority.P1, Team.ACCOUNT, 3, requires_escalation=True),
        Template(Category.GENERAL, "General feedback with profanity", "The release is terrible and this is unacceptable. Fix this now.", Priority.P2, Team.ESCALATIONS, 3, requires_escalation=True),
        Template(Category.BILLING, "Duplicate invoice email #A", "You billed me for the same invoice twice. This seems duplicate.", Priority.P2, Team.BILLING, 2, duplicate_group="dup-a"),
        Template(Category.BILLING, "Duplicate invoice email #B", "Following up: same duplicate billing issue as my earlier email.", Priority.P2, Team.BILLING, 2, duplicate_group="dup-a"),
        Template(Category.TECHNICAL, "SLA risk: production outage", "Our production service is down for 2 hours, users cannot log in.", Priority.P1, Team.TECH, 3, requires_escalation=True),
        Template(Category.REFUND, "SLA risk: legal notice", "If no refund by end of day, we will file a legal complaint.", Priority.P1, Team.BILLING, 3, requires_escalation=True),
    ]


def random_sender(rng: random.Random) -> Tuple[str, SenderTier]:
    name = rng.choice(NAMES)
    handle = name.lower().replace(" ", ".")
    domain = rng.choice(DOMAINS)
    tier_roll = rng.random()
    if tier_roll < 0.12:
        tier = SenderTier.VIP
    elif tier_roll < 0.60:
        tier = SenderTier.REGULAR
    else:
        tier = SenderTier.FIRST_TIME
    return f"{handle}@{domain}", tier


def compose_body(template: Template, sender: str, ts: datetime, rng: random.Random) -> str:
    urgency = ""
    if template.priority == Priority.P1 or rng.random() < 0.2:
        urgency = "\nThis is urgent and impacts our SLA if not resolved immediately."
    anger = ""
    if rng.random() < 0.15:
        anger = "\nI am very frustrated with this experience."
    return (
        f"Hello support,\n\n{template.body}\n\nAccount: {sender}\n"
        f"Reported at: {ts.isoformat()}"
        f"{urgency}{anger}\n\nThanks"
    )


def generate() -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rng = random.Random(SEED)
    templates = build_templates()
    emails: List[Dict[str, object]] = []
    ground_truth: Dict[str, Dict[str, object]] = {}
    duplicate_anchor: Dict[str, str] = {}

    for idx in range(TARGET_SIZE):
        template = templates[idx % len(templates)]
        email_id = f"email_{idx + 1:03d}"
        sender_email, sender_tier = random_sender(rng)
        timestamp = datetime.now(timezone.utc) - timedelta(hours=rng.randint(1, 96), minutes=rng.randint(0, 59))

        priority = template.priority
        team = template.team

        # VIP priority override.
        if sender_tier == SenderTier.VIP and priority in {Priority.P3, Priority.P4}:
            priority = Priority.P2

        # Content-based routing edge cases.
        body = compose_body(template, sender_email, timestamp, rng)
        if sender_tier == SenderTier.VIP and "cancel" in template.body.lower() and template.category == Category.BILLING:
            team = Team.ACCOUNT
        if template.category == Category.TECHNICAL and "invoice" in template.body.lower():
            team = Team.BILLING
        if template.category == Category.GENERAL and "unacceptable" in template.body.lower():
            team = Team.ESCALATIONS

        email = {
            "email_id": email_id,
            "subject": template.subject,
            "body": body,
            "sender_email": sender_email,
            "sender_tier": sender_tier.value,
            "timestamp": timestamp.isoformat(),
            "thread_history": [],
            "attachments_mentioned": rng.random() < 0.1,
            "language": "en",
        }
        emails.append(email)

        is_duplicate = template.duplicate_group is not None
        original_email_id = None
        if template.duplicate_group:
            if template.duplicate_group in duplicate_anchor:
                original_email_id = duplicate_anchor[template.duplicate_group]
            else:
                duplicate_anchor[template.duplicate_group] = email_id
                is_duplicate = False

        ground_truth[email_id] = {
            "email_id": email_id,
            "category": template.category.value,
            "priority": priority.value,
            "team": team.value,
            "is_duplicate": is_duplicate,
            "original_email_id": original_email_id,
            "requires_escalation": template.requires_escalation,
            "sla_breach_risk": priority == Priority.P1,
            "difficulty_level": template.difficulty,
        }

    return emails, ground_truth


def main() -> None:
    emails, truth = generate()
    root = Path(__file__).resolve().parent
    (root / "emails.json").write_text(json.dumps(emails, indent=2), encoding="utf-8")
    (root / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Generated {len(emails)} emails and {len(truth)} ground truth records")


if __name__ == "__main__":
    main()
