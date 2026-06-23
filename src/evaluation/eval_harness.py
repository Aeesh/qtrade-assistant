"""
Lightweight evaluation harness for the QTrade support assistant.

Scores each Appendix B query on three dimensions:
  - escalated_correctly  : does the escalation decision match the expected label?
  - grounded             : does the answer cite a real doc (non-escalated only)?
  - no_hallucination     : simple keyword check — answer must not claim facts
                           the docs don't contain (heuristic; not LLM-as-judge)

Run from project root:
  python -m src.evaluation.eval_harness
  python -m src.evaluation.eval_harness --provider ollama
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import json
from dataclasses import asdict
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.assistant import QTradeAssistant, AssistantResponse

OUTPUT_DIR = "eval_results"


# ---------------------------------------------------------------------------
# Eval dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalCase:
    query: str
    should_escalate: bool
    expected_keyword_in_answer: str | None   # None for escalation cases
    label: str


EVAL_DATASET: list[EvalCase] = [
    EvalCase(
        query="I opened the box, can I still return it, and is there a fee?",
        should_escalate=False,
        expected_keyword_in_answer="15%",
        label="opened_return_fee",
    ),
    EvalCase(
        query="How do I reset my SmartHub?",
        should_escalate=False,
        expected_keyword_in_answer="amber",
        label="smarthub_reset",
    ),
    EvalCase(
        query="My order hasn't shipped in 4 days, where is it?",
        should_escalate=False,  # partially answerable, a grounded partial answer is expected
        expected_keyword_in_answer="support",
        label="shipping_delay",
    ),
    EvalCase(
        query="My SmartHub is getting very hot and smells like burning.",
        should_escalate=True,
        expected_keyword_in_answer=None,
        label="safety_hazard",
    ),
    EvalCase(
        query="This is the third time I've called, I want a refund and a manager NOW.",
        should_escalate=True,
        expected_keyword_in_answer=None,
        label="escalation_explicit_repeat",
    ),
    EvalCase(
        query="Do you offer bulk discounts for commercial installs?",
        should_escalate=True,   # not in docs → escalate
        expected_keyword_in_answer=None,
        label="out_of_scope",
    ),
    EvalCase(
        query="How long does a refund take?",
        should_escalate=False,
        expected_keyword_in_answer="5",
        label="refund_timeline",
    ),
    EvalCase(
        query="Does the warranty cover water damage?",
        should_escalate=False,
        expected_keyword_in_answer="not cover",
        label="warranty_water",
    ),
]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    case: EvalCase
    response: AssistantResponse
    escalation_correct: bool
    grounding_pass: bool      # True if answer cites a doc OR case expects escalation
    keyword_pass: bool        # True if expected keyword found (or None expected)

    @property
    def overall_pass(self) -> bool:
        return self.escalation_correct and self.grounding_pass and self.keyword_pass


def score(case: EvalCase, response: AssistantResponse) -> EvalResult:
    escalation_correct = response.is_escalated == case.should_escalate
    grounding_pass = (
        case.should_escalate
        or bool(response.cited_docs)
    )
    keyword_pass = (
        case.expected_keyword_in_answer is None
        or case.expected_keyword_in_answer.lower() in response.answer.lower()
    )
    return EvalResult(
        case=case,
        response=response,
        escalation_correct=escalation_correct,
        grounding_pass=grounding_pass,
        keyword_pass=keyword_pass,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(assistant: QTradeAssistant, provider_name: str) -> None:
    results: list[EvalResult] = []

    for case in EVAL_DATASET:
        response = assistant.handle(case.query)
        result = score(case, response)
        results.append(result)

        status = "✓" if result.overall_pass else "✗"
        print(
            f"{status} [{case.label}]\n"
            f"   Escalation: {'correct' if result.escalation_correct else 'WRONG'} "
            f"(expected={case.should_escalate}, got={response.is_escalated})\n"
            f"   Grounding : {'pass' if result.grounding_pass else 'FAIL'}\n"
            f"   Keyword   : {'pass' if result.keyword_pass else 'FAIL'} "
            f"(looking for {case.expected_keyword_in_answer!r})\n"
        )

    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    esc_acc = sum(1 for r in results if r.escalation_correct) / total * 100
    ground_acc = sum(1 for r in results if r.grounding_pass) / total * 100
    kw_acc = sum(1 for r in results if r.keyword_pass) / total * 100

    print("=" * 50)
    print(f"Overall pass    : {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Escalation acc  : {esc_acc:.0f}%")
    print(f"Grounding acc   : {ground_acc:.0f}%")
    print(f"Keyword acc     : {kw_acc:.0f}%")
    print("=" * 50)

    report = {
        "provider": provider_name,
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total * 100,
            "escalation_accuracy": esc_acc,
            "grounding_accuracy": ground_acc,
            "keyword_accuracy": kw_acc,
        },
        "results": [
            {
                "label": r.case.label,
                "query": r.case.query,
                "expected_escalation": r.case.should_escalate,
                "actual_escalation": r.response.is_escalated,
                "answer": r.response.answer,
                "cited_docs": r.response.cited_docs,
                "escalation_correct": r.escalation_correct,
                "grounding_pass": r.grounding_pass,
                "keyword_pass": r.keyword_pass,
                "overall_pass": r.overall_pass,
            }
            for r in results
        ],
    }

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / f"eval_results_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    import os
    import argparse
    from src.llm import GeminiProvider, OllamaProvider
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file if present

    parser = argparse.ArgumentParser(description="Run QTradeAssistant evaluation harness")

    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )

    args = parser.parse_args()

    provider_name = args.provider
    if provider_name == "ollama":
        provider = OllamaProvider()
    else:
        provider = GeminiProvider()

    assistant = QTradeAssistant(docs_dir="data/help-docs", provider=provider)
    run_eval(assistant, provider_name=provider_name)