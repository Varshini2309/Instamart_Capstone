
from pathlib import Path
import sqlite3
import json
from datetime import datetime, timezone

import pandas as pd
import mock_llm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DB_PATH = DATA_DIR / "instamart.db"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"



def get_order(order_id):
    """Look up an order in the SQLite database."""
    if not DB_PATH.exists():
        return {
        "status": "NOT_FOUND",
        "error": True,
        "message": "Order database was not found."
    }

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
            columns = [row[1] for row in cursor.fetchall()]

            if "order_id" not in columns:
                continue

            cursor.execute(
                f'SELECT * FROM "{table}" WHERE order_id = ? LIMIT 1',
                (order_id,)
            )
            row = cursor.fetchone()

            if row:
                order_data = dict(zip(columns, row))
                return {
                    "status": "FOUND",
                    **order_data
                }

    return {
        "status": "NOT_FOUND",
        "error": True,
        "message": f"No order found for {order_id}."
    }



def search_policy(query):
    """Retrieve relevant policy chunks using policy_rag."""
    from policy_rag import load_policy_documents, chunk_documents, build_retriever

    documents = load_policy_documents()
    chunks = chunk_documents(
        documents,
        chunk_size=40,
        overlap=10
    )
    retrieve = build_retriever(chunks)
    results = retrieve(query, k=3)

    return {
        "status": "OK",
        "results": [
            {
                "source": item["source"],
                "text": item["text"],
                "score": round(float(item["score"]), 4)
            }
            for item in results
        ]
    }

def calculate_refund(
    order_id=None,
    request_date=None,
    delivery_date=None,
    order_amount=None,
    delivery_minutes=None
):
    """Calculate refund eligibility using order data and refund policy."""
    from datetime import date

    # Support the mock LLM interface, which supplies order amount
    # and delivery duration instead of an order ID.
    if order_id is None:
        if order_amount is None or delivery_minutes is None:
            return {
                "status": "ERROR",
                "refund_inr": 0,
                "message": "Order ID or complete order details are required."
            }

        try:
            amount = round(float(order_amount), 2)
            minutes = float(delivery_minutes)
        except (TypeError, ValueError):
            return {
                "status": "ERROR",
                "refund_inr": 0,
                "message": "Invalid order amount or delivery duration."
            }

        if minutes <= 45:
            return {
                "status": "DENIED",
                "refund_inr": 0,
                "message": "Delivery took 45 minutes or less."
            }

        return {
            "status": (
                "NEEDS_HUMAN_APPROVAL"
                if amount > 1000
                else "ELIGIBLE"
            ),
            "refund_inr": amount,
            "order_amount": amount,
            "message": (
                "Human approval required."
                if amount > 1000
                else "Late-delivery refund calculated."
            )
        }

    order = get_order(order_id)

    if order.get("status") == "NOT_FOUND":
        return {
            "status": "NOT_FOUND",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Order was not found."
        }

    if str(order.get("status", "")).lower() != "completed":
        return {
            "status": "DENIED",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Only completed orders are eligible for a refund."
        }

    delivery_minutes = order.get("delivery_minutes")

    if delivery_minutes is None:
        return {
            "status": "ERROR",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Delivery time is unavailable."
        }

    if float(delivery_minutes) <= 45:
        return {
            "status": "DENIED",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Delivery took 45 minutes or less."
        }

    try:
        amount = round(float(order["order_amount"]), 2)
    except (KeyError, TypeError, ValueError):
        return {
            "status": "ERROR",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Could not identify the order amount."
        }

    # The database does not contain a delivery date, so the
    # seven-day rule cannot be verified without explicit dates.
    if request_date is None or delivery_date is None:
        return {
            "status": "DATE_REQUIRED",
            "order_id": order_id,
            "refund_inr": 0,
            "order_amount": amount,
            "message": (
                "Provide both delivery_date and request_date "
                "to validate the seven-day refund window."
            )
        }

    try:
        if isinstance(request_date, str):
            request_date = date.fromisoformat(request_date)
        if isinstance(delivery_date, str):
            delivery_date = date.fromisoformat(delivery_date)
    except (ValueError, TypeError):
        return {
            "status": "ERROR",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Dates must use YYYY-MM-DD format."
        }

    days_since_delivery = (request_date - delivery_date).days

    if days_since_delivery < 0 or days_since_delivery > 7:
        return {
            "status": "DENIED",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Refund requests must be raised within seven days."
        }

    return {
        "status": (
            "NEEDS_HUMAN_APPROVAL"
            if amount > 1000
            else "ELIGIBLE"
        ),
        "order_id": order_id,
        "refund_inr": amount,
        "order_amount": amount,
        "message": (
            "Refund above INR 1000 requires human approval."
            if amount > 1000
            else "Order qualifies for a late-delivery refund."
        )
    }

def _append_ledger(tool_name, arguments, observation):
    """Append a JSON record to the refund ledger."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "arguments": arguments,
        "observation": observation
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def issue_refund(
    order_id,
    amount=None,
    reason="Customer refund request",
    request_date=None,
    delivery_date=None,
    refund_inr=None
):
    """Issue an eligible refund or route it for human approval."""

    # Accept both the normal function argument and the mock LLM argument.
    if amount is None:
        amount = refund_inr

    try:
        requested_amount = round(float(amount), 2)
    except (TypeError, ValueError):
        requested_amount = 0.0

    calculation = calculate_refund(
        order_id,
        request_date=request_date,
        delivery_date=delivery_date
    )

    if calculation.get("status") == "NOT_FOUND":
        observation = {
            "status": "NOT_FOUND",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Order was not found."
        }

    elif calculation.get("status") == "NEEDS_HUMAN_APPROVAL":
        observation = {
            "status": "ESCALATE",
            "order_id": order_id,
            "refund_inr": calculation.get("refund_inr", 0),
            "currency": "INR",
            "reason": "Human approval is required for refunds above INR 1000."
        }

    elif calculation.get("status") != "ELIGIBLE":
        observation = {
            "status": calculation.get("status", "ERROR"),
            "order_id": order_id,
            "refund_inr": 0,
            "message": calculation.get(
                "message",
                "The refund request is not eligible."
            )
        }

    elif requested_amount <= 0:
        observation = {
            "status": "DENIED",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Refund amount must be greater than zero."
        }

    elif requested_amount != round(
        float(calculation.get("refund_inr", 0)), 2
    ):
        observation = {
            "status": "DENIED",
            "order_id": order_id,
            "refund_inr": 0,
            "message": "Requested amount does not match the eligible refund."
        }

    else:
        observation = {
            "status": "REFUND",
            "order_id": order_id,
            "refund_inr": requested_amount,
            "currency": "INR",
            "reason": reason,
            "message": "Refund approved under the automatic refund policy."
        }

    _append_ledger(
        "issue_refund",
        {
            "order_id": order_id,
            "amount": requested_amount,
            "reason": reason,
            "request_date": request_date,
            "delivery_date": delivery_date
        },
        observation
    )

    return observation




def escalate_to_human(order_id, reason):
    """Record a human-review escalation."""
    observation = {
        "status": "ESCALATE",
        "order_id": order_id,
        "reason": reason,
        "message": "Human support review is required."
    }

    _append_ledger(
        "escalate_to_human",
        {"order_id": order_id, "reason": reason},
        observation
    )
    return observation


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve an order's details and status using its order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for information relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },

    {
        "name": "calculate_refund",
        "description": "Calculate refund eligibility and the permitted refund amount.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "request_date": {"type": "string"},
                "delivery_date": {"type": "string"},
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": []
        }
    },
    {
        "name": "issue_refund",
        "description": "Validate and record an eligible refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
                "reason": {"type": "string"},
                "request_date": {"type": "string"},
                "delivery_date": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id"]
        }
    },

    {
        "name": "escalate_to_human",
        "description": "Record a refund or customer issue for human support review.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["order_id", "reason"]
        }
    }
]
