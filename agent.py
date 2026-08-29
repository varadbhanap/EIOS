"""
EIOS - Step 4: The Agent

This is the reasoning brain. It connects to mcp_server.py (all 7 tools),
uses an LLM to decide which tools to call and in what order, and follows
a strict Standard Operating Procedure so it doesn't skip steps or answer
from guesswork.

Before running this, get a free API key (no credit card required):

    1. Go to https://console.groq.com/keys
    2. Sign in and create a key
    3. Set it as an environment variable:

    Windows (permanent, all future sessions):
        setx GROQ_API_KEY "your-key-here"
        (then CLOSE and REOPEN your terminal)

Run with:  python main.py  (this file is imported, not run directly)

NOTE: To switch providers later, change only the MODEL line below, e.g.
"anthropic:claude-haiku-4-5-20251001" or "google_genai:gemini-3.6-flash".
"""

import asyncio
import copy
import os
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# swap this one line to change LLM provider
MODEL = "groq:openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Structural tool-call limits
#
# WHY THIS EXISTS: relying on the system prompt alone to say "call
# search_policies at most twice" is not reliable, on harder or compound
# questions, the model has called it 3+ times anyway, eating the entire
# recursion-limit budget before ever reaching a final answer. This
# enforces the limit in code instead of just asking nicely: once a tool
# hits its cap for a given question, further calls to it return a short
# message telling the agent to work with what it already has, rather
# than actually re-running the tool. This is the same "defense in depth,
# not just a prompt tweak" pattern used earlier for the oversized-query
# and schema-hallucination fixes.
# ---------------------------------------------------------------------------
TOOL_CALL_LIMITS = {
    "search_policies": 2,
    "get_schema": 3,          # up to 3 real tables may legitimately need checking
    "query_database": 6,
    "find_anomalies": 3,
    "segment_applicants": 3,
    "forecast_trend": 3,
    "assess_default_risk": 3,
}


def wrap_tools_with_limits(raw_tools):
    """
    Returns a fresh, independently-limited copy of the tool list. Called
    once per question (not once per session), so every question gets its
    own clean call budget rather than limits accumulating across an
    entire terminal session.
    """
    wrapped = []
    for tool in raw_tools:
        limit = TOOL_CALL_LIMITS.get(tool.name)
        if limit is None:
            wrapped.append(tool)
            continue

        tool_copy = copy.copy(tool)
        original_coroutine = tool_copy.coroutine
        # some MCP tools are configured with response_format
        # "content_and_artifact", meaning callers expect a (content,
        # artifact) tuple back, not a bare string. The "limit exceeded"
        # message below must match that shape for those tools, or the
        # framework rejects it with a type error.
        expects_tuple = getattr(tool_copy, "response_format", None) == "content_and_artifact"
        call_count = {"n": 0}

        # IMPORTANT: call_count must be captured as a default argument,
        # exactly like _orig/_name/_limit below. Python for-loops don't
        # create a new scope per iteration, so without this, every
        # wrapped tool's closure would share one single counter (the
        # last tool processed in this loop) instead of each tool having
        # its own independent count, this was a real bug caught during
        # testing: a compound question's 4th tool call overall tripped
        # search_policies's limit on its very first actual call, because
        # every prior tool call in the question had been silently
        # incrementing the same shared counter.
        async def limited_coroutine(*args, _orig=original_coroutine, _name=tool.name,
                                     _limit=limit, _count=call_count,
                                     _tuple=expects_tuple, **kwargs):
            _count["n"] += 1
            if _count["n"] > _limit:
                message = (
                    f"You have already called {_name} {_limit} time(s) for this "
                    f"question, that is the limit. Do not call it again, use the "
                    f"results you already have to answer the question now."
                )
                return (message, None) if _tuple else message
            return await _orig(*args, **kwargs)

        tool_copy.coroutine = limited_coroutine
        wrapped.append(tool_copy)
    return wrapped


# the SOP: this is what stops the agent from wandering, skipping steps,
# or answering from guesswork instead of actually using its tools
SYSTEM_PROMPT = """You are an autonomous enterprise risk and compliance
analyst for EIOS Financial Services. You answer questions about loan
applications by combining real database queries, real statistical
analysis, and real policy documents. You never guess or invent numbers,
you always use your tools to get real answers.

TOOLS: You have exactly these tools available: get_schema, query_database,
find_anomalies, segment_applicants, forecast_trend, assess_default_risk,
search_policies. You do NOT have a code execution tool, a Python
interpreter, or any tool named run_code or similar. Never attempt to call
a tool that is not in this list. If you need to do arithmetic (like
averaging a few numbers), do it directly yourself in your own reasoning
and final answer text, do not call a tool for it.

TOOL BUDGET: Each tool has a strict limit on how many times it can be
called for a single question (typically 2-6 depending on the tool). Use
each tool efficiently, don't call search_policies multiple times with
minor wording variations of the same query, one well-formed query is
usually enough. If a tool tells you that you've reached its limit, stop
calling it and answer using what you already have.

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
   - segment_applicants to group applicants by behavior. This tool
     already returns a real, correctly computed default rate for each
     individual segment, included in the segment_profiles output. If
     asked to reason about risk per segment, use that value directly,
     do not run a separate query trying to approximate a default rate,
     a separate aggregate query will not match the real per-segment
     figure and will produce a misleading answer.
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
   Do any necessary arithmetic yourself, directly in this step.

SCOPE: You can only answer questions about loan applications, the three
database tables, and the policy/report documents you have access to. If
a question is outside this scope, for example general knowledge
questions, questions about other companies or datasets, or anything your
tools genuinely cannot address, say so clearly and directly: state that
this is outside what EIOS has data for, and do not attempt to answer
from general knowledge. Do not guess or produce a plausible-sounding
answer for something you have no real data on.

Never skip step 1. Never state a number you did not get from a tool or
calculate yourself from tool results. Never state a policy rule you did
not retrieve from search_policies. Never call a tool that isn't in your
tools list.
"""

# a short, emphatic reminder appended only on a retry, after the model
# has already attempted to call a nonexistent tool once
RETRY_REMINDER = (
    "\n\nIMPORTANT REMINDER (this is a retry): on the previous attempt, you "
    "tried to call a tool that does not exist (such as run_code or a Python "
    "interpreter). You do not have any code execution tool. For this "
    "attempt, do all arithmetic yourself directly in your reasoning and "
    "final answer text. Only call tools from your actual tools list: "
    "get_schema, query_database, find_anomalies, segment_applicants, "
    "forecast_trend, assess_default_risk, search_policies."
)

MCP_CONFIG = {
    "eios": {
        "command": "python",
        "args": ["mcp_server.py"],
        "transport": "stdio",
    }
}


async def connect_tools():
    """
    Connects to the MCP server once and returns the raw tool list. This
    is done once per session (called from main.py at startup), not once
    per question, since launching the MCP subprocess has real overhead.
    The actual agent, with freshly-limited tool copies, is built fresh
    inside ask() for every question instead.
    """
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
    return tools


def _is_nonexistent_tool_error(exc) -> bool:
    msg = str(exc)
    return "which was not in request.tools" in msg or "tool_use_failed" in msg


async def ask(raw_tools, question: str):
    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 70}")

    print("\n(waiting for first response, this can take 10-60+ seconds "
          "on the free tier, especially after a retry, this is normal)")

    for attempt in (1, 2):
        system_prompt = SYSTEM_PROMPT if attempt == 1 else SYSTEM_PROMPT + RETRY_REMINDER

        # fresh, independently-limited tool copies every question, and
        # every attempt, so a retry gets a clean call budget too
        agent = create_agent(
            model=MODEL,
            tools=wrap_tools_with_limits(raw_tools),
            system_prompt=system_prompt,
        )

        all_messages = []
        step_num = 0
        last_step_time = time.time()

        try:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="updates",
                config={"recursion_limit": 26},
                # raised from 14: the per-tool call limits above now
                # structurally prevent runaway loops, so a lower ceiling
                # is no longer needed to protect against them, and a
                # higher one gives genuinely complex, multi-part
                # questions enough room to actually finish.
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

        except Exception as e:
            if _is_nonexistent_tool_error(e) and attempt == 1:
                print(f"\n[RETRY] The model tried to call a tool that doesn't exist "
                      f"({e}). Retrying once with a corrective reminder...")
                continue
            raise

    raise RuntimeError("Failed after retry.")
