
from pathlib import Path
import re
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parent
SQL_FILE = ROOT / "sql" / "queries.sql"
DB_FILE = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sql_text = SQL_FILE.read_text(encoding="utf-8")

    # Split the file into the seven named queries.
    sections = re.split(r"(?m)^-- Q([1-7]):[^\n]*\n", sql_text)
    queries = {
        int(sections[i]): sections[i + 1].strip()
        for i in range(1, len(sections), 2)
    }

    assert set(queries) == set(range(1, 8)), (
        "Expected exactly seven queries, Q1 through Q7."
    )

    results = {}

    with sqlite3.connect(DB_FILE) as connection:
        for number in range(1, 8):
            df = pd.read_sql_query(queries[number], connection)
            results[number] = df

            print(f"\n{'=' * 60}")
            print(f"Q{number}: {len(df)} rows")
            print(df.head(10).to_string(index=False))

            df.to_csv(
                OUTPUT_DIR / f"q{number}_results.csv",
                index=False,
            )

    # ---------------------------------------------------------
    # Reference checks from the capstone brief
    # ---------------------------------------------------------
    q1 = results[1]
    q2 = results[2]
    q3 = results[3]
    q4 = results[4]
    q5 = results[5]
    q6 = results[6]
    q7 = results[7]

    print(f"\n{'=' * 60}")
    print("REFERENCE CHECKS")

    checks = {
        "Q1 customers": (len(q1), 30),
        "Q2 city/category rows": (len(q2), 30),
        "Q3 products": (len(q3), 11),
        "Q4 customers without completed orders": (len(q4), 25),
        "Q5 above-average spend customers": (len(q5), 550),
        "Q7 daily revenue rows": (len(q7), 183),
    }

    for label, (actual, expected) in checks.items():
        print(
            f"{label}: actual={actual}, expected={expected}, "
            f"{'PASS' if actual == expected else 'CHECK'}"
        )

    q2_total = round(q2["revenue_inr"].sum(), 2)
    print(f"Q2 total revenue: INR {q2_total:,.2f}")

    if not q2.empty:
        largest = q2.loc[q2["revenue_inr"].idxmax()]
        print(
            "Q2 largest cell: "
            f"{largest['city']} / {largest['category']} — "
            f"INR {largest['revenue_inr']:,.2f}"
        )

    if not q5.empty:
        average_spend = q5["total_spend"].sum()
        # Calculate the average across all customers with completed spend.
        with sqlite3.connect(DB_FILE) as connection:
            avg_df = pd.read_sql_query(
                """
                SELECT AVG(total_spend) AS average_spend
                FROM (
                    SELECT customer_id, SUM(order_amount) AS total_spend
                    FROM orders
                    WHERE status = 'completed'
                    GROUP BY customer_id
                )
                """,
                connection,
            )
        print(
            "Q5 average completed spend: "
            f"INR {avg_df.iloc[0]['average_spend']:,.2f}"
        )

    if not q6.empty:
        tie_count = int(
            (q6["spend_rank"] != q6["spend_row_number"]).sum()
        )
        print(f"Q6 rows where RANK differs from ROW_NUMBER: {tie_count}")

        city_tops = (
            q6.sort_values(["city", "spend_rank", "customer_id"])
            .groupby("city", as_index=False)
            .first()
        )
        print("\nQ6 top customer per city:")
        print(
            city_tops[
                ["city", "customer_id", "total_spend"]
            ].to_string(index=False)
        )

    if not q7.empty:
        last = q7.sort_values("day").iloc[-1]
        print(f"Q7 final cumulative revenue: INR {last['cumulative_revenue_inr']:,.2f}")

    print(f"\nIndividual query results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
