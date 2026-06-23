"""
Command-line interface for the QTrade support assistant.

Usage
-----
  # Interactive mode (default)
  python cli.py

  # Single query mode
  python cli.py --query "How do I reset my SmartHub?"

  # Use Ollama instead of Gemini
  python cli.py --provider ollama --model llama3.2

  # Run the Appendix B sanity-check queries
  python cli.py --run-samples
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.assistant import QTradeAssistant
from src.llm import OllamaProvider, GeminiProvider

load_dotenv()


logging.basicConfig(
    level=logging.WARNING,  # flip to INFO to see retrieval debug output
    format="%(levelname)s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Appendix B sample queries from the challenge doc spec.
# ---------------------------------------------------------------------------

APPENDIX_B_QUERIES = [
    "I opened the box, can I still return it, and is there a fee?",
    "How do I reset my SmartHub?",
    "My order hasn't shipped in 4 days, where is it?",
    "My SmartHub is getting very hot and smells like burning.",
    "This is the third time I've called, I want a refund and a manager NOW.",
    "Do you offer bulk discounts for commercial installs?",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BANNER = """\
╔══════════════════════════════════════╗
║   QTrade AI Support Assistant        ║
║   Type 'exit' or Ctrl-C to quit.     ║
╚══════════════════════════════════════╝"""


def _build_provider(args: argparse.Namespace):
    """
      Build an LLMProvider instance based on CLI args.
    """
    if args.provider == "ollama":
        return OllamaProvider(
            model=args.model or "llama3.2",
            base_url=args.ollama_url,
        )
    # default: gemini
    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(
            "\n[ERROR] No GEMINI_API_KEY found.\n"
            "  Set it:  export GEMINI_API_KEY=your_key\n"
            "  Or use Ollama:  python cli.py --provider ollama\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return GeminiProvider(model=args.model or "gemini-2.0-flash", api_key=api_key)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QTrade customer support assistant (RAG + escalation)"
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (e.g. llama3.2, gemini-2.0-flash)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key (overrides GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--docs-dir",
        default="data/help-docs",
        help="Path to help doc .txt files (default: data/help-docs)",
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--run-samples",
        action="store_true",
        help="Run all Appendix B sample queries and exit",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    args = parser.parse_args()

    print("Initialising QTrade assistant …", flush=True)
    provider = _build_provider(args)
    assistant = QTradeAssistant(
        docs_dir=args.docs_dir,
        provider=provider,
        top_k=args.top_k,
        persist_dir=args.persist_dir,
    )
    print("Ready.\n")

    if args.run_samples:
        print("Running Appendix B sample queries …")
        for query in APPENDIX_B_QUERIES:
            response = assistant.handle(query)
            print(response)
        return

    if args.query:
        response = assistant.handle(args.query)
        print(response)
        return

    # Interactive REPL
    print(BANNER)
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Goodbye!")
                break
            response = assistant.handle(user_input)
            print(response)
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()