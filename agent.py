"""
Lesson 01 — Your first agent (bare metal, zero framework)
=========================================================
A minimal agent that uses a local LLM (LM Studio or Ollama) + 2 tools.

What you learn here:
  1. The "agent loop" (the heart of EVERY agent, including framework-based ones)
  2. How to define tools the LLM can call
  3. How to pass tool results back to the LLM and continue reasoning
  4. When the agent decides it is done (stop condition)

Run:
  python3 agent.py

Prerequisite:
  An active OpenAI-compatible server.
    LM Studio → port 1234 with a model loaded
    Ollama    → port 11434 with `ollama serve` in the background
  Recommended models (support tool calling):
    - qwen2.5-coder:7b  (best for tool use at 7B)
    - llama3.1:8b
"""

# lezione-01-primo-agent/agent.py
from __future__ import annotations

import json
import os
from datetime import datetime

import requests

# ============================================================
# CONFIG — endpoint and model
# ============================================================
# Default: LM Studio on :1234. For Ollama export:
#   export LLM_URL="http://localhost:11434/v1/chat/completions"
#   export LLM_MODEL="qwen2.5-coder:7b"
LLM_URL = os.getenv("LLM_URL", "http://localhost:1234/v1/chat/completions")
MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder-7b-instruct")
MAX_ITERATIONS = 5  # safety net against infinite loops


# ============================================================
# STEP 1 — DEFINE THE FUNCTIONS THE AGENT CAN USE ("tools")
# ============================================================
# These are the REAL Python functions that run when the LLM
# decides to call them. Ordinary code.

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def get_current_time() -> str:
    """Return the current date and time in ISO-8601 format."""
    return datetime.now().isoformat(timespec="seconds")


def multiply(a, b) -> float:
    return a * b


# Map "tool name" → Python function.
# The agent loop uses this dict to know WHICH function to run
# when the LLM says "call tool X".
TOOL_REGISTRY: dict[str, callable] = {
    "add": add,
    "get_current_time": get_current_time,
    "multiply": multiply
}


# ============================================================
# STEP 2 — DESCRIBE THE TOOLS TO THE LLM (JSON Schema)
# ============================================================
# The LLM does not see the Python code. We pass a SCHEMA that describes:
#   - the tool name
#   - what it does (description → the LLM uses this to decide WHETHER to call it)
#   - which parameters it accepts, their types, which are required
# The format is the standard OpenAI/JSON Schema, accepted by LM Studio,
# Ollama, Anthropic (with small differences), and practically every other LLM.

TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First summand"},
                    "b": {"type": "number", "description": "Second summand"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First factor"},
                    "b": {"type": "number", "description": "Second factor"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current date and time. Takes no parameters.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ============================================================
# STEP 3 — FUNCTION THAT TALKS TO THE LLM
# ============================================================
def call_llm(messages: list[dict]) -> dict:
    """Send the full conversation to the LLM and return its raw response."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "tool_choice": "auto",  # the LLM chooses whether/which tool to use
        "temperature": 0.1,     # low → more deterministic answers (useful for debugging)
    }
    try:
        resp = requests.post(LLM_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Error calling the LLM at {LLM_URL}. "
            f"Is the server running? Details: {exc}"
        ) from exc
    return resp.json()


# ============================================================
# STEP 4 — THE AGENT LOOP (the heart of everything)
# ============================================================
def run_agent(user_input: str) -> str:
    """
    The agent loop. Structure:

        ┌─────────────────────────────────────────────────┐
        │ 1. Send the ENTIRE conversation to the LLM    │
        │ 2. The LLM replies with ONE of two things:      │
        │      (a) final text       → STOP                │
        │      (b) list of tool_calls → we execute them   │
        │ 3. Append tool results to the history           │
        │ 4. Goto 1                                       │
        └─────────────────────────────────────────────────┘

    Stops when the LLM no longer requests tools, or when we hit
    MAX_ITERATIONS (safety net against infinite loops).
    """
    # The agent's "working memory": the message list.
    # It grows each iteration. The LLM always sees the full history.
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are an assistant that uses tools to answer. "
                "When you need to compute or know the time, call the appropriate tool. "
                "After you receive the results, reply to the user in English."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    for step in range(1, MAX_ITERATIONS + 1):
        print(f"\n─── Step {step} ──────────────────────────────")

        response = call_llm(messages)
        assistant_msg = response["choices"][0]["message"]

        # Always append the LLM reply to the history
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls")

        # --- CASE A: no tool requested → the LLM has a final answer ---
        if not tool_calls:
            final = (assistant_msg.get("content") or "").strip()
            print("LLM → final answer ready")
            return final

        # --- CASE B: the LLM wants to call one or more tools ---
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", "")

            # Note: arguments arrive as a JSON STRING, not a dict.
            # Always parse. If the LLM returns bad JSON, we handle it.
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
                print(f"  ⚠️  Malformed JSON from LLM: {raw_args!r}")

            print(f"LLM → calling {name}({args})")

            # Run the actual tool (with error handling)
            fn = TOOL_REGISTRY.get(name)
            if fn is None:
                result = f"Error: tool '{name}' does not exist"
            else:
                try:
                    result = fn(**args)
                except Exception as exc:  # noqa: BLE001 — we want to catch everything
                    result = f"Error running {name}: {exc}"

            print(f"Tool → {result}")

            # Send the result back to the LLM as a role="tool" message.
            # tool_call_id links result ↔ request: essential when the LLM
            # asked for MULTIPLE tools in parallel.
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": json.dumps(result, default=str),
            })

    return "⚠️ Iteration limit reached without a final answer."


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    question = "What time is it now? And what is 137 + 456?"
    print(f"👤 User: {question}")

    answer = run_agent(question)
    print(f"\n🤖 Agent: {answer}")
