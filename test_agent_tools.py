
from datetime import date, timedelta

from agent_tools import issue_refund

delivery_date = date(2025, 9, 16)
request_date = delivery_date + timedelta(days=3)

tests = [
    ("S1", "ORD-100048", 128.25, "REFUND"),
    ("S2", "ORD-100067", 135.00, "REFUND"),
    ("S3", "ORD-100145", 1999.00, "ESCALATE"),
    ("S4", "ORD-100001", 109.65, "DENIED"),
    ("S5", "ORD-100025", 45.00, "DENIED"),
]

for scenario, order_id, amount, expected in tests:
    result = issue_refund(
        order_id,
        amount,
        reason=f"Test scenario {scenario}",
        request_date=request_date,
        delivery_date=delivery_date
    )

    print(
        f"{scenario}: expected={expected}, "
        f"actual={result['status']}, result={result}"
    )
