"""
EIOS - Step 4: The Agent

This is the reasoning brain. It connects to mcp_server.py (all six tools),
uses Llama 3.3 70B (via Groq's free API) to decide which tools to call
and in what order, and follows a strict Standard Operating Procedure so
it doesn't skip steps or answer from guesswork.

WHY GROQ AND NOT GEMINI: Gemini's free tier for gemini-3.6-flash allows
only 20 requests per day, a single well-behaved question can use 8-10 of
those, leaving almost no room for the iterative debugging this project
needs. Groq's free tier allows 14,400 requests per day on Llama 3.3 70B,
a genuinely workable limit for daily development, at no cost and no
credit card, with strong tool-calling support.

Before running this, get a free API key (no credit card required):

    1. Go to https://console.groq.com/keys
    2. Sign in and create a key
    3. Set it as an environment variable:

    Windows (permanent, all future sessions):
        setx GROQ_API_KEY "your-key-here"
        (then CLOSE and REOPEN your terminal, setx does not apply to
        the terminal window you ran it in)

    Verify it worked in the NEW terminal window with:
        echo %GROQ_API_KEY%

Run with:  python agent.py

NOTE: To switch providers later (e.g. for a live demo, as insurance
against free-tier rate limits), change only the MODEL line below, e.g.
"anthropic:claude-haiku-4-5-20251001" or "google_genai:gemini-3.6-flash".
Nothing else in this file needs to change.
"""

import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# swap this one line to change LLM provider
MODEL = "groq:openai/gpt-oss-120b"

# the SOP: this is what stops the agent from wandering, skipping steps,
# or answering from guesswork instead of actually using its tools
SYSTEM_PROMPT = """You are an autonomous enterprise risk and compliance
analyst for EIOS Financial Services. You answer questions about loan
applications by combining real database queries, real statistical
analysis, and real policy documents. You never guess or invent numbers,
you always use your tools to get real answers.

Follow this Standard Operating Procedure for every question:

1. INVESTIGATE: ALWAYS call query_database first, for every question,
   with no exceptions. This step is mandatory even if you believe another
   tool (like assess_default_risk) might return overlapping information.
   Do not skip it based on your own judgment that it seems redundant, an
   auditable system must follow the same first step every time, so that
   its behavior is predictable and explainable, not just correct.
   Write narrow, specific queries, select only the columns relevant to
   the question, avoid SELECT * when joining tables, since one applicant
   can have many related records in bureau_credit_history or
   previous_applications, and a wide join can return an oversized,
   automatically truncated result. If you are not certain a column name
   exists in a table, call get_schema first to check the real column
   names, rather than guessing, a wrong guess causes a query error and
   wastes a step.

2. ANALYZE: Once you have raw data, apply the relevant analysis tool:
   - find_anomalies to detect statistical outliers
   - segment_applicants to group applicants by behavior
   - forecast_trend to project future volume
   - assess_default_risk to score a specific applicant's risk
   Only use the tools relevant to the question asked, not all of them
   every time.

3. CONTEXTUALIZE: If the question involves policy compliance, violations,
   or "should this have been approved" type questions, use search_policies
   ONCE with a single well-formed query to retrieve the relevant policy
   clause. Use the best-matching passages you get back, even if they are
   not a perfect exact match, real policy documents rarely phrase things
   exactly like a question does. Do not call search_policies more than
   twice total for a single question. Cite whichever specific section
   number and document you actually retrieved.

4. SYNTHESIZE: Combine what you found, the real numbers and the real
   policy text, into a clear, direct answer. State specific numbers and
   reference the specific policy section by name. If nothing relevant
   was found, say so plainly, don't fabricate a plausible-sounding answer.

SCOPE: You can only answer questions about loan applications, the three
database tables, and the policy/report documents you have access to. If
a question is outside this scope, for example general knowledge
questions, questions about other companies or datasets, or anything your
tools genuinely cannot address, say so clearly and directly: state that
this is outside what EIOS has data for, and do not attempt to answer
from general knowledge. Do not guess or produce a plausible-sounding
answer for something you have no real data on.

Never skip step 1. Never state a number you did not get from a tool.
Never state a policy rule you did not retrieve from search_policies.
"""

MCP_CONFIG = {
    "eios": {
        "command": "python",
        "args": ["mcp_server.py"],
        "transport": "stdio",
    }
}


async def build_agent():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. See the instructions at the top "
            "of this file for how to get a free key and set it."
        )

    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()
    print(f"Connected to MCP server, loaded {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}")

    agent = create_agent(
        model=MODEL,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


async def ask(agent, question: str):
    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 70}")

    import time
    all_messages = []
    step_num = 0
    last_step_time = time.time()

    print("\n(waiting for first response, this can take 10-60+ seconds "
          "on the free tier, especially after a retry, this is normal)")

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
        config={"recursion_limit": 14},  # hard safety net: caps total steps
        # so a misbehaving loop (like repeatedly re-querying search_policies)
        # can never run away and exhaust the day's quota again, regardless
        # of what causes it. If this limit is ever hit, that's a signal the
        # SOP needs further tightening, not something to raise blindly.
    ):
        step_num += 1
        elapsed = time.time() - last_step_time
        last_step_time = time.time()
        for node_name, node_output in chunk.items():
            print(f"\n[Step {step_num}] Node: {node_name}  (+{elapsed:.1f}s)")
            messages = node_output.get("messages", [])
            for msg in messages:
                all_messages.append(msg)
                msg_type = type(msg).__name__
                if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
                    for call in msg.tool_calls:
                        print(f"  [TOOL CALL] {call['name']}({call['args']})")
                elif msg_type == "ToolMessage":
                    content_preview = str(msg.content)[:150]
                    print(f"  [TOOL RESULT] {content_preview}...")
                elif msg_type == "AIMessage" and msg.content:
                    print(f"  [MODEL RESPONSE] {str(msg.content)[:150]}...")

    final_answer = all_messages[-1].content if all_messages else "(no response)"
    print(f"\n--- Final answer ---\n{final_answer}\n")
    return final_answer


async def main():
    agent = await build_agent()

    # a single test question for now, to confirm the agent works end to
    # end before we build the terminal loop (main.py) around it
    await ask(
        agent,
        "Is applicant 100002 a high risk applicant, and if so, does "
        "approving their loan violate our credit risk policy?",
    )


if __name__ == "__main__":
    asyncio.run(main())
