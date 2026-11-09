import json
import time
import uuid
from pathlib import Path

import mock_llm
from agent_tools import TOOL_MAP

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"

MAX_ITERATIONS = 6
DEFAULT_TOKEN_LIMIT = 6000
DEFAULT_TIME_LIMIT_SECONDS = 10

SYSTEM_PROMPT = """AGENT
You are an Instamart customer support agent.
Use the available tools to verify orders, search policies, calculate
refunds, issue eligible refunds, and escalate cases when required.

Rules:
- Never invent order information.
- Never claim a refund was issued unless the tool confirms it.
- Refunds above INR 1000 require human approval.
- Cancelled orders are not eligible for refunds.
- If an order cannot be found, return NOT_FOUND.
- Use tool observations to decide the next action.
- For [LOOP] messages, continue requesting tools until the iteration limit.
- Finish with a status word followed by a colon:
  REFUND:, DENIED:, ESCALATE:, NOT_FOUND:, or FALLBACK:.
"""


def _json_safe(value):
    return json.loads(json.dumps(value, default=str))


def _append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def _save_state(run_id, state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{run_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(_json_safe(state), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temporary.replace(path)


def _load_state(run_id):
    path = STATE_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved state for run {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_message(response):
    return response.choices[0].message.content or ""


def _extract_tokens(response, messages):
    usage = getattr(response, "usage", None)
    total = getattr(usage, "total_tokens", None) if usage else None
    if total is not None:
        return int(total)
    return sum(
        mock_llm.est_tokens(str(message.get("content", "")))
        for message in messages
    )


def _parse_action(text):
    """Parse a JSON action from the model's response."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        payload = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    tool_name = payload.get("tool")
    arguments = payload.get("arguments", {})
    if tool_name not in TOOL_MAP or not isinstance(arguments, dict):
        return None
    return {"tool": tool_name, "arguments": arguments}


def _tool_observation(tool_name, arguments):
    try:
        return _json_safe(TOOL_MAP[tool_name](**arguments))
    except Exception as exc:
        return {
            "status": "TOOL_ERROR",
            "tool": tool_name,
            "message": str(exc)
        }


def _final_status(answer):
    if ":" in answer:
        return answer.split(":", 1)[0].strip().upper()
    return "FALLBACK"


def _finish(state, status, answer):
    state["status"] = status
    state["final_answer"] = answer
    state["finished"] = True
    _save_state(state["run_id"], state)
    _append_jsonl(RUNS_PATH, {
        "scenario_id": state.get("scenario_id"),
        "run_id": state["run_id"],
        "status": status,
        "final_answer": answer,
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "steps": state["steps"]
    })
    return {
        "run_id": state["run_id"],
        "status": status,
        "final_answer": answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"]
    }


def run_agent(
    user_message,
    scenario_id=None,
    max_iterations=MAX_ITERATIONS,
    token_limit=DEFAULT_TOKEN_LIMIT,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    run_id=None
):
    run_id = run_id or str(uuid.uuid4())
    started = time.monotonic()
    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "steps": [],
        "llm_calls": 0,
        "tokens_used": 0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
        "started_elapsed": 0,
        "finished": False,
        "status": None,
        "final_answer": None
    }
    _save_state(run_id, state)
    return _run_loop(state, started)


def _run_loop(state, started):
    while state["iteration"] < state["max_iterations"]:
        if time.monotonic() - started >= state["time_limit_seconds"]:
            return _finish(
                state, "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: Time limit reached. "
                "Please hand off to a human support agent."
            )

        try:
            response = mock_llm.chat(state["messages"])
        except mock_llm.TransientLLMError:
            state["steps"].append({
                "type": "llm_error",
                "message": "Transient mock LLM error."
            })
            _save_state(state["run_id"], state)
            continue

        call_tokens = _extract_tokens(response, state["messages"])
        if state["tokens_used"] + call_tokens > state["token_limit"]:
            return _finish(
                state, "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: Token limit reached. "
                "Please hand off to a human support agent."
            )

        state["llm_calls"] += 1
        state["tokens_used"] += call_tokens
        state["iteration"] += 1

        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            status = _final_status(content)
            if not content or status == "FALLBACK":
                return _finish(
                    state, "FALLBACK",
                    "FALLBACK: Unable to complete the request safely. "
                    "Please hand off to a human support agent."
                )

            # Do not accept a refund claim unless the tool confirms it.
            if status == "REFUND":
                refund_steps = [
                    step for step in state["steps"]
                    if step.get("tool") == "issue_refund"
                ]
                confirmed = any(
                    step.get("observation", {}).get("status") == "REFUND"
                    and float(
                        step.get("observation", {}).get("refund_inr", 0)
                    ) > 0
                    for step in refund_steps
                )
                if not confirmed:
                    return _finish(
                        state, "FALLBACK",
                        "FALLBACK: The refund could not be confirmed. "
                        "Please hand off to a human support agent."
                    )
            return _finish(state, status, content)

        # Save the assistant's tool-call message in the conversation.
        assistant_tool_calls = []
        for call in tool_calls:
            assistant_tool_calls.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, default=str)
                    if isinstance(call.arguments, dict)
                    else call.arguments
                }
            })

        state["messages"].append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": assistant_tool_calls
        })

        # Execute each requested tool and save its observation.
        for call in tool_calls:
            tool_name = call.name
            arguments = call.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            if tool_name not in TOOL_MAP or not isinstance(arguments, dict):
                observation = {
                    "status": "TOOL_ERROR",
                    "tool": tool_name,
                    "message": "Invalid tool name or arguments."
                }
            else:
                observation = _tool_observation(tool_name, arguments)

            state["steps"].append({
                "iteration": state["iteration"],
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })
            state["messages"].append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(observation, default=str)
            })
            _save_state(state["run_id"], state)

            if time.monotonic() - started >= state["time_limit_seconds"]:
                return _finish(
                    state, "FALLBACK_TIMEOUT",
                    "FALLBACK_TIMEOUT: Time limit reached. "
                    "Please hand off to a human support agent."
                )

    return _finish(
        state, "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: Maximum iterations reached. "
        "Please hand off to a human support agent."
    )


def resume_agent(run_id):
    state = _load_state(run_id)
    if state.get("finished"):
        return {
            "run_id": run_id,
            "status": state["status"],
            "final_answer": state["final_answer"],
            "steps": state["steps"],
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]
        }
    return _run_loop(state, time.monotonic())


if __name__ == "__main__":
    print("ReAct agent module loaded.")
    print("Available tools:", ", ".join(TOOL_MAP.keys()))
