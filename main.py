"""
EIOS - Step 5: Terminal Engine

The main interactive entry point. Type a question, get a real,
evidence-backed answer, see exactly which tools were used and how long
each step took.

Each question is handled independently, no shared conversation memory
between questions in the same session. This is deliberate: an audit
question should be its own complete, self-contained investigation, and
it keeps token usage predictable on the free tier rather than growing
with every question asked.

Run with:  python main.py
"""

import asyncio
import json
import os
import time
from datetime import datetime

from agent import connect_tools, ask

LOG_DIR = "logs"

EXAMPLE_QUESTIONS = [
    "Is applicant 100002 a high risk applicant, and does approving their loan violate policy?",
    "Are there any unusually large loan amounts in AMT_CREDIT?",
    "Segment applicants by income, credit amount, and annuity into 4 groups.",
    "What does our credit risk policy say about the maximum loan to income ratio?",
    "What is the forecasted loan application volume for the next 3 months?",
]


def print_banner():
    print("\n" + "=" * 70)
    print("  EIOS — Enterprise Intelligence Operating System")
    print("=" * 70)
    print("Type a question, or one of these commands:")
    print("  help    - show example questions")
    print("  stats   - show session performance summary")
    print("  exit    - quit and show final summary")
    print("=" * 70)


def print_help():
    print("\nExample questions you can ask:")
    for q in EXAMPLE_QUESTIONS:
        print(f"  - {q}")
    print()


def print_stats(session_log):
    if not session_log:
        print("\nNo questions asked yet this session.\n")
        return

    durations = [entry["duration_seconds"] for entry in session_log]
    print("\n--- Session performance summary ---")
    print(f"Questions asked: {len(session_log)}")
    print(f"Total time: {sum(durations):.1f}s")
    print(f"Average time per question: {sum(durations) / len(durations):.1f}s")
    print(f"Fastest: {min(durations):.1f}s   Slowest: {max(durations):.1f}s")
    print()


def start_session_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    path = os.path.join(LOG_DIR, filename)
    print(f"Session log: {path}")
    return path


def log_interaction(log_path, question, answer, duration):
    # written immediately after each question, not buffered until exit,
    # so a crash or Ctrl+C mid-session doesn't lose earlier answers
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "duration_seconds": round(duration, 2),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


async def main():
    print("Connecting to EIOS backend...")
    raw_tools = await connect_tools()
    print_banner()

    log_path = start_session_log()
    session_log = []

    while True:
        try:
            question = input("\nAsk EIOS > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "help":
            print_help()
            continue
        if question.lower() == "stats":
            print_stats(session_log)
            continue

        start = time.time()
        try:
            answer = await ask(raw_tools, question)
        except KeyboardInterrupt:
            print("\n[Cancelled] That question was interrupted. Session continues.")
            continue
        except Exception as e:
            print(f"\n[ERROR] Something went wrong answering that question: {e}")
            print("Try rephrasing, or ask something else.")
            continue
        duration = time.time() - start

        entry = log_interaction(log_path, question, answer, duration)
        session_log.append(entry)

    print("\nShutting down.")
    print_stats(session_log)
    if session_log:
        print(f"Full session saved to: {log_path}")
    print("Thanks for using EIOS.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye.")
