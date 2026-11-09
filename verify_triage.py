
import pandas as pd

results = pd.read_csv(
    "outputs/triage_results.csv",
    dtype={"customer_id": str},
    keep_default_na=False,
)

segments = pd.read_csv(
    "outputs/customer_segments.csv",
    dtype={"customer_id": str},
    keep_default_na=False,
)

assert len(results) == 164, "Expected 164 result rows"

required_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]
assert list(results.columns) == required_columns

segment_lookup = {}
for _, row in segments.iterrows():
    segment_lookup[row["customer_id"]] = {
        "segment_name": row["segment_name"],
        "is_premium": str(row["is_premium"]).strip().lower()
        in {"true", "1", "yes"},
    }

human_statuses = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted",
}

errors = []

for _, row in results.iterrows():
    customer_id = row["customer_id"]
    status = row["status"]
    category = row["category"]
    segment = segment_lookup.get(customer_id, {})
    premium = segment.get("is_premium", False)

    if status in human_statuses:
        expected_route, expected_priority = "human_review", "NORMAL"
    elif category == "fraud":
        expected_route, expected_priority = "security_team", "HIGH"
    elif premium:
        expected_route, expected_priority = "premium_desk", "HIGH"
    else:
        expected_route, expected_priority = "support_queue", "NORMAL"

    if row["route"] != expected_route:
        errors.append(
            f"{row['ticket_id']}: route {row['route']}, "
            f"expected {expected_route}"
        )

    if row["priority"] != expected_priority:
        errors.append(
            f"{row['ticket_id']}: priority {row['priority']}, "
            f"expected {expected_priority}"
        )

    expected_segment = segment.get("segment_name", "")
    if row["segment_name"] != expected_segment:
        errors.append(
            f"{row['ticket_id']}: incorrect segment name"
        )

if errors:
    print("ROUTING CHECK FAILED")
    for error in errors:
        print(error)
    raise SystemExit(1)

print("ROUTING CHECK PASSED")
print("Rows verified:", len(results))
print("\nVerified route counts:")
print(results["route"].value_counts())
print("\nVerified status counts:")
print(results["status"].value_counts())

