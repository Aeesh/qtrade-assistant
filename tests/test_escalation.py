import pytest
from src.escalation.rules import (
    EscalationTrigger,
    evaluate_escalation,
)


# ---------------------------------------------------------------------------
# Safety hazard — hard trigger
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "My device is burning and there is smoke coming out",
    "I can see sparks from the back of the hub",
    "The unit is overheating badly",
    "There was an explosion sound from the device",
    "It smells like burning plastic near the charger",
    "I got an electric shock when I touched it",
])
def test_safety_hazard_triggers_escalation(message):
    decision = evaluate_escalation(message)
    assert decision.should_escalate
    assert EscalationTrigger.SAFETY_HAZARD in decision.triggers


# ---------------------------------------------------------------------------
# Explicit human request
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "I want to speak to a manager",
    "Connect me to a real person please",
    "Transfer me to a human agent",
    "I want to talk to a supervisor",
    "Get me a human NOW",
])
def test_explicit_human_request_triggers_escalation(message):
    decision = evaluate_escalation(message)
    assert decision.should_escalate
    assert EscalationTrigger.EXPLICIT_HUMAN_REQ in decision.triggers


# ---------------------------------------------------------------------------
# Repeat frustration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "This is the third time I've called about this",
    "I am very frustrated, this is my second attempt",
    "I'm extremely frustrated, still not resolved",
    "This is ridiculous, I called multiple times",
    "I am fed up, still not working",
])
def test_repeat_frustration_triggers_escalation(message):
    decision = evaluate_escalation(message)
    assert decision.should_escalate
    assert EscalationTrigger.REPEAT_FRUSTRATION in decision.triggers


# ---------------------------------------------------------------------------
# No grounded answer
# ---------------------------------------------------------------------------

def test_no_grounded_answer_triggers_escalation():
    decision = evaluate_escalation("Do you have bulk discount pricing?", retrieval_succeeded=False)
    assert decision.should_escalate
    assert EscalationTrigger.NO_GROUNDED_ANSWER in decision.triggers


# ---------------------------------------------------------------------------
# Normal queries — must NOT escalate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "How do I return an opened item?",
    "What is the refund policy?",
    "How do I reset my SmartHub?",
    "When will my order ship?",
    "Does the warranty cover defects?",
])
def test_normal_queries_do_not_escalate(message):
    decision = evaluate_escalation(message, retrieval_succeeded=True)
    assert not decision.should_escalate
    assert decision.triggers == ()


# ---------------------------------------------------------------------------
# Multiple triggers can fire together
# ---------------------------------------------------------------------------

def test_multiple_triggers_fire_simultaneously():
    message = (
        "This is the third time I've called and I want a manager — "
        "there is smoke coming from the device."
    )
    decision = evaluate_escalation(message)
    assert decision.should_escalate
    assert EscalationTrigger.SAFETY_HAZARD in decision.triggers
    assert EscalationTrigger.EXPLICIT_HUMAN_REQ in decision.triggers
    assert EscalationTrigger.REPEAT_FRUSTRATION in decision.triggers