"""
Direct LLM Tool Calling Demonstration — NO FRAMEWORKS.

Demonstrates:
1. How to declare tool schemas (Anthropic & OpenAI raw JSON schemas).
2. How to dispatch requests with tool definitions via raw HTTP.
3. How to intercept model tool calls, execute the local function safely.
4. How to return the tool execution result back into the message history.
5. How the model formulates its final answer incorporating the tool result.

This fulfills HANDOFF.md §9 Step 4:
"Read docs.claude.com -> Tool use, then build a calculator tool and get the model to call it."
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


# ── 1. Local Tool Implementation ───────────────────────────────────────────────

def execute_calculator(operation: str, a: float, b: float) -> dict[str, Any]:
    """Execute mathematical operations deterministically."""
    op = operation.lower().strip()
    if op in ("add", "+"):
        result = a + b
    elif op in ("subtract", "-"):
        result = a - b
    elif op in ("multiply", "*"):
        result = a * b
    elif op in ("divide", "/"):
        if b == 0:
            return {"error": "Division by zero is undefined"}
        result = a / b
    elif op in ("power", "^", "**"):
        result = math.pow(a, b)
    else:
        return {"error": f"Unknown operation: {operation}"}

    return {"operation": op, "a": a, "b": b, "result": result}


# ── 2. Tool Schemas ────────────────────────────────────────────────────────────

TOOL_DESCRIPTION = "Performs mathematical operations (add, subtract, multiply, divide, power) on two numbers."

OPENAI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power"],
                    "description": "The math operation to perform.",
                },
                "a": {"type": "number", "description": "The first operand."},
                "b": {"type": "number", "description": "The second operand."},
            },
            "required": ["operation", "a", "b"],
        },
    },
}

ANTHROPIC_TOOL_SCHEMA = {
    "name": "calculate",
    "description": TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide", "power"],
                "description": "The math operation to perform.",
            },
            "a": {"type": "number", "description": "The first operand."},
            "b": {"type": "number", "description": "The second operand."},
        },
        "required": ["operation", "a", "b"],
    },
}


# ── 3. Tool Calling with OpenAI API ────────────────────────────────────────────

def run_openai_tool_call(query: str, api_key: str | None = None) -> None:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        print("  [OpenAI] Skipped: OPENAI_API_KEY not configured.")
        return

    print(f"\n[OpenAI Wire Call] Prompt: '{query}'")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the calculator tool for math questions."},
        {"role": "user", "content": query},
    ]

    # Step 1: Send query with available tools
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "tools": [OPENAI_TOOL_SCHEMA],
        "tool_choice": "auto",
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        print(f"  Model answered directly without tools: {message.get('content')}")
        return

    print(f"  ✓ Model requested {len(tool_calls)} tool call(s):")
    messages.append(message)

    # Step 2: Execute requested tools locally
    for tc in tool_calls:
        call_id = tc["id"]
        fn_name = tc["function"]["name"]
        fn_args = json.loads(tc["function"]["arguments"])
        print(f"    -> Invoking: {fn_name}({fn_args}) [Call ID: {call_id}]")

        if fn_name == "calculate":
            result = execute_calculator(
                operation=fn_args["operation"],
                a=float(fn_args["a"]),
                b=float(fn_args["b"]),
            )
        else:
            result = {"error": f"Tool {fn_name} not found"}

        print(f"    <- Local execution returned: {result}")

        # Step 3: Append tool response back to message history
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result),
        })

    # Step 4: Call model again with tool output to get final answer
    follow_up_payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
    }

    with httpx.Client(timeout=30.0) as client:
        final_resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=follow_up_payload)
        final_resp.raise_for_status()
        final_data = final_resp.json()

    final_content = final_data["choices"][0]["message"]["content"]
    print(f"  ✓ Final Assistant Answer:\n    {final_content.strip()}")


# ── 4. Tool Calling with Anthropic API ─────────────────────────────────────────

def run_anthropic_tool_call(query: str, api_key: str | None = None) -> None:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("sk-ant-..."):
        print("  [Anthropic] Skipped: ANTHROPIC_API_KEY not configured.")
        return

    print(f"\n[Anthropic Wire Call] Prompt: '{query}'")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": query},
    ]

    # Step 1: Send query with tools
    payload = {
        "model": "claude-haiku-3-5-20241022",
        "max_tokens": 1024,
        "system": "You are a helpful assistant. Use the calculator tool for math operations.",
        "messages": messages,
        "tools": [ANTHROPIC_TOOL_SCHEMA],
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content_blocks = data.get("content", [])
    tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

    if not tool_uses:
        direct_text = "".join(b.get("text", "") for b in content_blocks)
        print(f"  Model answered directly: {direct_text}")
        return

    print(f"  ✓ Anthropic requested {len(tool_uses)} tool_use block(s):")
    messages.append({"role": "assistant", "content": content_blocks})

    tool_results = []
    for tu in tool_uses:
        tool_id = tu["id"]
        tool_name = tu["name"]
        tool_input = tu["input"]
        print(f"    -> Invoking: {tool_name}({tool_input}) [ID: {tool_id}]")

        if tool_name == "calculate":
            result = execute_calculator(
                operation=tool_input["operation"],
                a=float(tool_input["a"]),
                b=float(tool_input["b"]),
            )
        else:
            result = {"error": f"Tool {tool_name} not found"}

        print(f"    <- Local execution returned: {result}")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": json.dumps(result),
        })

    # Step 3: Append tool results as a user turn
    messages.append({
        "role": "user",
        "content": tool_results,
    })

    # Step 4: Call model for final text response
    follow_up_payload = {
        "model": "claude-haiku-3-5-20241022",
        "max_tokens": 1024,
        "messages": messages,
        "tools": [ANTHROPIC_TOOL_SCHEMA],
    }

    with httpx.Client(timeout=30.0) as client:
        final_resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=follow_up_payload)
        final_resp.raise_for_status()
        final_data = final_resp.json()

    final_text = "".join(b.get("text", "") for b in final_data.get("content", []) if b.get("type") == "text")
    print(f"  ✓ Final Anthropic Answer:\n    {final_text.strip()}")


# ── 5. Offline Simulation Mode ─────────────────────────────────────────────────

def run_simulation() -> None:
    """Simulate tool calling locally without external API keys."""
    print("\n--- Running Offline Simulation of Tool Dispatch Loop ---")
    query = "What is 234 multiplied by 876?"
    print(f"User Prompt: {query}")

    # Simulated model decision
    print("1. Router/LLM parses query -> chooses tool 'calculate'")
    args = {"operation": "multiply", "a": 234.0, "b": 876.0}
    print(f"2. LLM emits tool call: calculate({args})")

    # Local execution
    result = execute_calculator(**args)
    print(f"3. Python runtime calculates: {result['result']}")

    # Formulate answer
    final_output = f"234 multiplied by 876 equals {int(result['result']):,}."
    print(f"4. LLM receives tool_result -> responds: '{final_output}'")
    print("--- Simulation Complete ---\n")


def main() -> None:
    print("=" * 65)
    print("  RAW LLM TOOL CALLING DEMO — Safe Calculator")
    print("=" * 65)

    has_openai = bool(os.environ.get("OPENAI_API_KEY") and not os.environ["OPENAI_API_KEY"].startswith("sk-..."))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY") and not os.environ["ANTHROPIC_API_KEY"].startswith("sk-ant-..."))

    if not has_openai and not has_anthropic:
        print("\nNote: No live API keys detected in .env. Running offline verification simulation.")
        run_simulation()
        print("To run against live APIs, add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env")
        return

    sample_query = "What is 347 multiplied by 892?"
    if has_openai:
        run_openai_tool_call(sample_query)
    if has_anthropic:
        run_anthropic_tool_call(sample_query)


if __name__ == "__main__":
    main()
