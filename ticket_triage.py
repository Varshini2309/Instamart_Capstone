from sklearn.metrics import accuracy_score, confusion_matrix
import re
import pandas as pd
import mock_llm
import json
import random
import time
from pathlib import Path

# Configuration
MAX_INPUT_TOKENS = 400

RAW_TICKETS = "data/raw/tickets.csv"

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_ID_PATTERN = r"ORD-\d{6}"


ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

ZERO_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category.
Allowed categories: billing, delivery, technical, fraud.
Return a JSON-only reply in this form:
{"category": "...", "reason": "..."}
Do not include markdown or any text outside the JSON object.
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category.
Allowed categories: billing, delivery, technical, fraud.
Return a JSON-only reply in this form:
{"category": "...", "reason": "..."}
Do not include markdown or any text outside the JSON object.

EXAMPLES:
Ticket: I was charged twice for the same purchase.
Reply: {"category": "billing", "reason": "Duplicate charge"}

Ticket: My account was hacked and I see unauthorized purchases.
Reply: {"category": "fraud", "reason": "Unauthorized account activity"}

Ticket: The delivery is late and the package has not arrived.
Reply: {"category": "delivery", "reason": "Late delivery"}
"""


def redact_text(text):
    """Redact email, phone number, and order ID in the required order."""
    text = str(text or "")

    text, email_count = re.subn(
        EMAIL_PATTERN, "[EMAIL]", text
    )
    text, phone_count = re.subn(
        PHONE_PATTERN, "[PHONE]", text
    )
    text, order_count = re.subn(
        ORDER_ID_PATTERN, "[ORDER_ID]", text
    )

    counts = {
        "emails": email_count,
        "phones": phone_count,
        "order_ids": order_count,
    }
    return text, counts


def remaining_sensitive_data(text):
    """Count any sensitive patterns that remain after redaction."""
    return {
        "emails": len(re.findall(EMAIL_PATTERN, text)),
        "phones": len(re.findall(PHONE_PATTERN, text)),
        "order_ids": len(re.findall(ORDER_ID_PATTERN, text)),
    }


BASE_DELAY = 0.05
MAX_RETRIES = 3


def call_with_retry(messages, ticket_id, tools=None):
    """Retry transient model failures up to three times."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages, tools=tools)

        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={attempt} wait_seconds={wait:.4f}"
            )
            time.sleep(wait)


def validate_response(reply):
    """Validate JSON output or recover one unambiguous category keyword."""

    # First, try to parse the response as JSON.
    try:
        data = json.loads(reply)

        if (
            isinstance(data, dict)
            and data.get("category") in ALLOWED_CATEGORIES
        ):
            return data["category"], "classified_json"

        # Valid JSON with an invalid category must not be guessed.
        return None, "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        pass

    # For non-JSON replies, extract lowercase words.
    words = re.findall(r"[a-z]+", str(reply).lower())
    found_categories = ALLOWED_CATEGORIES.intersection(words)

    if len(found_categories) == 1:
        return next(iter(found_categories)), "classified_keyword"

    return None, "flagged_invalid_output"



def run_pipeline(prompt, token_budget=None):
    """Run the ticket pipeline with input, privacy and budget guardrails."""
    mock_llm.reset()

    tickets = prepare_tickets()
    running_tokens = 0
    results = []

    for ticket in tickets:
        ticket_id = ticket["ticket_id"]
        status = ticket["status"]
        category = None
        tokens_used = 0

        if status in {"skipped_empty", "flagged_long"}:
            results.append({
                **ticket,
                "category": category,
                "status": status,
                "tokens_used": tokens_used,
            })
            continue

        # Budget is checked before every model call.
        if (
            token_budget is not None
            and running_tokens >= token_budget
        ):
            results.append({
                **ticket,
                "category": category,
                "status": "budget_exhausted",
                "tokens_used": 0,
            })
            continue

        # Only the redacted ticket text is sent as the user message.
        user_message = ticket["redacted_text"]

        assert sum(
            remaining_sensitive_data(user_message).values()
        ) == 0, f"Sensitive data would reach model for {ticket_id}"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = call_with_retry(messages, ticket_id)
        except mock_llm.TransientLLMError:
            status = "flagged_invalid_output"
        else:
            tokens_used = response.usage.total_tokens
            running_tokens += tokens_used

            reply = response.choices[0].message.content
            category, status = validate_response(reply)

        results.append({
            **ticket,
            "category": category,
            "status": status,
            "tokens_used": tokens_used,
        })

    print("Running token total:", running_tokens)
    return results

def evaluate_run(results, run_name):
    """Report accuracy and token usage for classified tickets."""
    classified = [
        item for item in results
        if item["status"] in {
            "classified_json",
            "classified_keyword",
        }
    ]

    y_true = [item["true_category"] for item in classified]
    y_pred = [item["category"] for item in classified]

    total_tokens = sum(item["tokens_used"] for item in results)
    count = len(classified)
    accuracy = accuracy_score(y_true, y_pred) if count else 0.0

    print(f"\n--- {run_name} ---")
    print("Classified tickets:", count)
    print(f"Accuracy: {accuracy:.1%}")
    print("Total tokens:", total_tokens)
    print(
        "Tokens per classified ticket:",
        round(total_tokens / count, 2) if count else 0,
    )

    return classified


def compare_prompts():
    """Run each prompt independently and compare performance."""
    zero_results = run_pipeline(ZERO_SHOT_PROMPT)
    zero_classified = evaluate_run(zero_results, "Zero-shot")

    few_results = run_pipeline(FEW_SHOT_PROMPT)
    few_classified = evaluate_run(few_results, "Few-shot")

    labels = ["billing", "delivery", "technical", "fraud"]

    matrix = confusion_matrix(
        [item["true_category"] for item in zero_classified],
        [item["category"] for item in zero_classified],
        labels=labels,
    )

    print("\nZero-shot confusion matrix")
    print("Rows = actual categories; columns = predicted categories")
    print(pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    ))

    return zero_results, few_results


OUTPUT_DIR = Path("outputs")
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"


def save_budget_demo():
    """Run zero-shot classification with the required token budget."""
    budget = 6000
    results = run_pipeline(
        ZERO_SHOT_PROMPT,
        token_budget=budget,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv",
        index=False,
    )

    exhausted = sum(
        item["status"] == "budget_exhausted"
        for item in results
    )

    print(f"\nBudget demo limit: {budget}")
    print("Tickets stopped by budget:", exhausted)
    print("Saved outputs/triage_budget_demo.csv")

    return results


def apply_routing(results):
    """Route tickets using output status and Part B customer segments."""
    segments = pd.read_csv(
        SEGMENTS_FILE,
        dtype={"customer_id": str},
    )

    if not {"customer_id", "segment_name", "is_premium"}.issubset(
        segments.columns
    ):
        raise ValueError(
            "customer_segments.csv must contain customer_id, "
            "segment_name and is_premium"
        )

    # Build a customer lookup; missing customers are treated as non-premium.
    segment_lookup = {}
    for _, row in segments.iterrows():
        segment_lookup[row["customer_id"]] = {
            "segment_name": row["segment_name"],
            "is_premium": str(row["is_premium"]).strip().lower()
            in {"true", "1", "yes"},
        }

    output_rows = []

    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    for item in results:
        customer_id = item["customer_id"]
        segment = segment_lookup.get(customer_id, {})
        segment_name = segment.get("segment_name", "")
        is_premium = segment.get("is_premium", False)

        if item["status"] in human_statuses:
            route = "human_review"
            priority = "NORMAL"
        elif item["category"] == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_premium:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        output_rows.append({
            "ticket_id": item["ticket_id"],
            "customer_id": customer_id,
            "status": item["status"],
            "category": item["category"] or "",
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": item["tokens_used"],
        })

    return pd.DataFrame(
        output_rows,
        columns=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ],
    )


def save_final_results():
    """Run few-shot classification with the final token budget."""
    results = run_pipeline(
        FEW_SHOT_PROMPT,
        token_budget=100000,
    )

    final_df = apply_routing(results)
    final_df.to_csv(
        OUTPUT_DIR / "triage_results.csv",
        index=False,
    )

    print("\nFinal ticket count:", len(final_df))
    print("\nFinal status counts:")
    print(final_df["status"].value_counts())
    print("\nFinal routing counts:")
    print(final_df["route"].value_counts())
    print("Saved outputs/triage_results.csv")

    return final_df


def prepare_tickets():
    tickets = pd.read_csv(
        RAW_TICKETS,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }
    missing_columns = required_columns - set(tickets.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    results = []
    totals = {"emails": 0, "phones": 0, "order_ids": 0}

    # Preserve the original ticket order.
    for _, ticket in tickets.iterrows():
        original_text = ticket["text"]

        # Empty or whitespace-only tickets must never reach the model.
        if not original_text.strip():
            status = "skipped_empty"
            redacted_text = ""
            counts = {"emails": 0, "phones": 0, "order_ids": 0}

        # Long tickets must never reach the model.
        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
            redacted_text, counts = redact_text(original_text)

        else:
            status = "ready"
            redacted_text, counts = redact_text(original_text)

        for key, count in counts.items():
            totals[key] += count

        # Verify that all three sensitive patterns are absent
        # from every redacted ticket.
        remaining = remaining_sensitive_data(redacted_text)
        assert sum(remaining.values()) == 0, (
            f"Sensitive data remains in ticket {ticket['ticket_id']}: "
            f"{remaining}"
        )

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "order_id": ticket["order_id"],
            "true_category": ticket["true_category"],
            "original_text": original_text,
            "redacted_text": redacted_text,
            "status": status,
            "tokens_estimated": (
                mock_llm.est_tokens(redacted_text)
                if status == "ready"
                else 0
            ),
            "emails_redacted": counts["emails"],
            "phones_redacted": counts["phones"],
            "order_ids_redacted": counts["order_ids"],
        })

    print("Tickets loaded:", len(results))
    print("Input guardrail counts:")
    print(pd.Series([r["status"] for r in results]).value_counts())

    print("\nRedaction totals:")
    for key, value in totals.items():
        print(f"{key}: {value}")

    print("Total substitutions:", sum(totals.values()))
    print("Remaining sensitive patterns: 0")

    return results



if __name__ == "__main__":
    save_budget_demo()
    save_final_results()