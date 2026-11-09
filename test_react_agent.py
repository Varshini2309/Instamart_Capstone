
"""
Part D: ReAct agent tests.

Uses explicit fixture dates and an isolated refund ledger.
Does not modify mock_llm.py or the production refund ledger.
"""

import json
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import agent_tools
import react_agent
from react_agent import run_agent, resume_agent


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TEST_LEDGER = OUTPUT_DIR / "test_refund_ledger.jsonl"
TEST_RESULTS = OUTPUT_DIR / "react_agent_test_results.json"

# Explicit test fixture dates; these are not dates read from the orders database.
DELIVERY_DATE = date(2025, 9, 16)
REQUEST_DATE = DELIVERY_DATE + timedelta(days=3)


def fixture_issue_refund(*args, **kwargs):
    """Supply explicit fixture dates only when testing the agent."""
    kwargs.setdefault("delivery_date", DELIVERY_DATE.isoformat())
    kwargs.setdefault("request_date", REQUEST_DATE.isoformat())
    return original_issue_refund(*args, **kwargs)


def print_result(label, result):
    print(
        f"{label}: status={result['status']}, "
        f"llm_calls={result['llm_calls']}, "
        f"tokens_used={result['tokens_used']}, "
        f"steps={len(result['steps'])}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the original tool and ledger configuration.
    global original_issue_refund
    original_issue_refund = agent_tools.issue_refund
    original_ledger_path = agent_tools.LEDGER_PATH
    original_tool_map_refund = react_agent.TOOL_MAP["issue_refund"]

    # This harness writes only to a separate test ledger.
    agent_tools.LEDGER_PATH = TEST_LEDGER
    react_agent.TOOL_MAP["issue_refund"] = fixture_issue_refund

    results = []

    try:
        # Remove only this harness's old artifacts.
        if TEST_LEDGER.exists():
            TEST_LEDGER.unlink()
        if TEST_RESULTS.exists():
            TEST_RESULTS.unlink()

        scenarios = [
            (
                "S1",
                "My order ORD-100048 reached me very late. I would like a refund.",
                "REFUND",
            ),
            (
                "S2",
                "My order ORD-100067 reached me very late. I would like a refund.",
                "REFUND",
            ),
            (
                "S3",
                "Order ORD-100145 arrived far too late, please refund me.",
                "ESCALATE",
            ),
            (
                "S4",
                "Please refund order ORD-100001, I did not like the delivery.",
                "DENIED",
            ),
            (
                "S5",
                "I want my money back for order ORD-100025.",
                "DENIED",
            ),
            (
                "S6",
                "Refund for order ORD-999999 please.",
                "NOT_FOUND",
            ),
            (
                "S7",
                "[LOOP] Refund order ORD-100048.",
                "FALLBACK_MAX_ITERATIONS",
            ),
        ]

        for scenario_id, message, expected_status in scenarios:
            result = run_agent(
                message,
                scenario_id=scenario_id,
                run_id=f"test-{scenario_id}-{uuid.uuid4().hex[:8]}",
            )

            record = {
                "test": scenario_id,
                "expected": expected_status,
                "actual": result["status"],
                "passed": result["status"] == expected_status,
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }
            results.append(record)
            print_result(scenario_id, result)
            print(
                f"  Expected: {expected_status} | "
                f"{'PASS' if record['passed'] else 'CHECK'}"
            )

        # Verify token-limit handling.
        token_result = run_agent(
            "My order ORD-100048 reached me very late. I would like a refund.",
            scenario_id="TOKEN_LIMIT",
            token_limit=200,
            run_id=f"test-token-{uuid.uuid4().hex[:8]}",
        )
        token_pass = token_result["status"] == "FALLBACK_TOKEN_LIMIT"
        results.append({
            "test": "TOKEN_LIMIT",
            "expected": "FALLBACK_TOKEN_LIMIT",
            "actual": token_result["status"],
            "passed": token_pass,
            "llm_calls": token_result["llm_calls"],
            "tokens_used": token_result["tokens_used"],
            "steps": token_result["steps"],
        })
        print_result("TOKEN_LIMIT", token_result)
        print(f"  {'PASS' if token_pass else 'CHECK'}")

        # Verify timeout before the first LLM call/tool.
        timeout_result = run_agent(
            "My order ORD-100048 reached me very late. I would like a refund.",
            scenario_id="TIMEOUT",
            time_limit_seconds=0,
            run_id=f"test-timeout-{uuid.uuid4().hex[:8]}",
        )
        timeout_pass = (
            timeout_result["status"] == "FALLBACK_TIMEOUT"
            and timeout_result["llm_calls"] == 0
            and len(timeout_result["steps"]) == 0
        )
        results.append({
            "test": "TIMEOUT",
            "expected": "FALLBACK_TIMEOUT with no tool execution",
            "actual": timeout_result["status"],
            "passed": timeout_pass,
            "llm_calls": timeout_result["llm_calls"],
            "tokens_used": timeout_result["tokens_used"],
            "steps": timeout_result["steps"],
        })
        print_result("TIMEOUT", timeout_result)
        print(f"  {'PASS' if timeout_pass else 'CHECK'}")

       
        # Crash-and-resume test: simulate a crash after the second
        # tool observation has been saved to the checkpoint.
        checkpoint_id = f"test-resume-{uuid.uuid4().hex[:8]}"
        crash_triggered = {"value": False}
        original_save_state = react_agent._save_state

        class SimulatedCrash(Exception):
            pass

        def crash_after_second_checkpoint(run_id, state):
            original_save_state(run_id, state)

            if (
                run_id == checkpoint_id
                and len(state.get("steps", [])) == 2
                and not crash_triggered["value"]
            ):
                crash_triggered["value"] = True
                raise SimulatedCrash(
                    "Simulated crash after the second saved tool observation."
                )

        react_agent._save_state = crash_after_second_checkpoint

        try:
            try:
                run_agent(
                    "My order ORD-100048 reached me very late. "
                    "I would like a refund.",
                    scenario_id="RESUME",
                    run_id=checkpoint_id,
                )
            except SimulatedCrash as exc:
                print(f"\nCRASH SIMULATED: {exc}")
        finally:
            react_agent._save_state = original_save_state

        # Verify that the checkpoint contains exactly two tool steps.
        saved_state = react_agent._load_state(checkpoint_id)
        checkpoint_steps = saved_state.get("steps", [])
        checkpoint_tool_names = [
            step["tool"]
            for step in checkpoint_steps
            if "tool" in step
        ]

        print("Checkpoint tool steps:", checkpoint_tool_names)

        # Resume the unfinished run from its saved checkpoint.
        resumed_result = resume_agent(checkpoint_id)

        # Collect tool names from the complete saved execution history.
        all_steps = resumed_result.get("steps", [])
        all_tool_names = [
            step["tool"]
            for step in all_steps
            if "tool" in step
        ]

        expected_tool_names = [
            "get_order",
            "search_policy",
            "calculate_refund",
            "issue_refund",
        ]

        resume_pass = (
            crash_triggered["value"]
            and len(checkpoint_steps) == 2
            and resumed_result["run_id"] == checkpoint_id
            and resumed_result["status"] == "REFUND"
            and "128.25" in resumed_result["final_answer"]
            and all_tool_names == expected_tool_names
        )

        results.append({
            "test": "CRASH_AND_RESUME",
            "expected": (
                "Recover after two saved tool steps without repeating tools; "
                "finish with REFUND INR 128.25"
            ),
            "actual": resumed_result["status"],
            "passed": resume_pass,
            "llm_calls": resumed_result["llm_calls"],
            "tokens_used": resumed_result["tokens_used"],
            "checkpoint_tool_names": checkpoint_tool_names,
            "all_tool_names": all_tool_names,
            "steps": resumed_result["steps"],
        })

        print_result("CRASH_AND_RESUME", resumed_result)
        print("  Complete tool sequence:", all_tool_names)
        print(f"  {'PASS' if resume_pass else 'CHECK'}")


        TEST_RESULTS.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        passed = sum(item["passed"] for item in results)
        print(f"\nTest summary: {passed}/{len(results)} checks passed.")
        print(f"Test results saved to: {TEST_RESULTS}")
        print(f"Isolated refund ledger: {TEST_LEDGER}")

    finally:
        # Restore the normal tool mapping and production ledger path.
        react_agent.TOOL_MAP["issue_refund"] = original_tool_map_refund
        agent_tools.LEDGER_PATH = original_ledger_path


if __name__ == "__main__":
    main()
