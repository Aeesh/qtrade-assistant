from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Trigger types
# ---------------------------------------------------------------------------

class EscalationTrigger(Enum):
    SAFETY_HAZARD        = auto()   # physical danger keyword like "burning", "smoke", "explosion"
    EXPLICIT_HUMAN_REQ   = auto()   # "I want a manager", "speak to a person"
    REPEAT_FRUSTRATION   = auto()   # "third time", "again and again", etc.
    NO_GROUNDED_ANSWER   = auto()   # set by assistant when retrieval fails


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

_SAFETY_PATTERNS = re.compile(
    r"\b("
    # direct fire/heat words
    r"fire|flame(s)?|burn(ing|t|s)?|hot|heat(ing)?"
    r"|smok(e|ing)|spark(s|ing)?"
    # electrical
    r"|electric(al)?.?shock|short.?circuit"
    # physical damage indicators
    r"|melt(ing|ed)?|explod(e|ing|ed)?|explosion(s)?"
    r"|crack(ed|ing)?|shatter(ed|ing)?"
    # smell as a hazard signal — any unusual smell from a device is a hazard
    r"|smell(s|ing)?|odou?r|fume(s)?"
    # temperature language
    r"|overheating|overheat(s|ed)?|too (hot|warm)|really (hot|warm)|very (hot|warm)|gets? (hot|warm)"
    r"|warm(ing)?|scorch(ed|ing)?"
    r")\b",
    re.IGNORECASE,
)

_HUMAN_REQUEST_PATTERNS = re.compile(
    r"\b("
    r"(speak|talk|connect|transfer).{0,10}(to|with).{0,10}(a |an )?(human|person|agent|representative|manager|supervisor)"
    r"|i want a (human|person|agent|manager|supervisor)"
    r"|get me a (human|person|agent|manager|supervisor)"
    r"|real person"
    r"| is this a (human|person|agent|bot|manager|supervisor)\?"
    r")\b",
    re.IGNORECASE,
)

_REPEAT_FRUSTRATION_PATTERNS = re.compile(
    r"\b("
    r"(second|third|fourth|\d+(st|nd|rd|th)) (time|call|contact|attempt)"
    r"|called (again|back|multiple times)"
    r"|still (not|haven.?t|waiting)"
    r"|(very |extremely |so )?(frustrated|fed up|angry|furious|annoyed|tired|disappointed|upset|mad)"
    r"|this is (ridiculous|unacceptable)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Escalation Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    triggers: tuple[EscalationTrigger, ...]
    reason_summary: str          # used in handoff summary


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_escalation(
    message: str,
    *,
    retrieval_succeeded: bool = True,
) -> EscalationDecision:
    """
        This function evaluates message against all escalation rules.

        @param message              : raw customer message
        @param retrieval_succeeded  : False when the vector store found no grounded answer above the similarity floor

        @returns EscalationDecision : all triggers that matched, and a summary of the reasons for escalation
    """
    triggers: list[EscalationTrigger] = []
    reasons: list[str] = []

    # --- Hard trigger: safety hazard ----------------------------------------
    if _SAFETY_PATTERNS.search(message):
        triggers.append(EscalationTrigger.SAFETY_HAZARD)
        reasons.append("potential physical safety hazard detected in message")

    # --- Soft trigger: explicit human request --------------------------------
    if _HUMAN_REQUEST_PATTERNS.search(message):
        triggers.append(EscalationTrigger.EXPLICIT_HUMAN_REQ)
        reasons.append("customer explicitly requested a human agent")

    # --- Soft trigger: repeat frustration ------------------------------------
    if _REPEAT_FRUSTRATION_PATTERNS.search(message):
        triggers.append(EscalationTrigger.REPEAT_FRUSTRATION)
        reasons.append("customer indicated repeated contact or high frustration")

    # --- No grounded answer (set by caller) ----------------------------------
    if not retrieval_succeeded:
        triggers.append(EscalationTrigger.NO_GROUNDED_ANSWER)
        reasons.append("no answer grounded in QTrade documentation was found")

    should_escalate = bool(triggers)
    reason_summary = "; ".join(reasons) if reasons else "no escalation triggers matched"

    return EscalationDecision(
        should_escalate=should_escalate,
        triggers=tuple(triggers),
        reason_summary=reason_summary,
    )