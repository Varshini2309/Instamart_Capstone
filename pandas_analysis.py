
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"


def main():
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    completed = orders.loc[orders["status"] == "completed"].copy()

    # ---------------------------------------------------------
    # A. Rebuild Q2 using pandas pivot_table with margins
    # ---------------------------------------------------------
    revenue_data = (
        completed
        .merge(customers[["customer_id", "city"]], on="customer_id")
        .merge(products[["product_id", "category"]], on="product_id")
    )

    pivot = pd.pivot_table(
        revenue_data,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All",
    ).round(2)

    print("\nPANDAS PIVOT TABLE — REVENUE BY CITY AND CATEGORY")
    print(pivot.to_string())
    print(f"\nPivot grand total: INR {pivot.loc['All', 'All']:,.2f}")

    # ---------------------------------------------------------
    # B. Named aggregation: completed spend by city
    # ---------------------------------------------------------
    city_summary = (
        revenue_data.groupby("city")
        .agg(
            total_spend_inr=("order_amount", "sum"),
            mean_order_amount_inr=("order_amount", "mean"),
            completed_order_count=("order_amount", "count"),
        )
        .round({"total_spend_inr": 2, "mean_order_amount_inr": 2})
        .sort_index()
    )

    print("\nCOMPLETED ORDER SUMMARY BY CITY")
    print(city_summary.to_string())

    # ---------------------------------------------------------
    # C. Distribution and IQR outlier analysis
    # ---------------------------------------------------------
    amount = completed["order_amount"].dropna()

    q1 = amount.quantile(0.25)
    median = amount.median()
    q3 = amount.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    print("\nCOMPLETED ORDER AMOUNT DISTRIBUTION")
    print(f"Count: {amount.count()}")
    print(f"Mean: INR {amount.mean():.2f}")
    print(f"Median: INR {median:.2f}")
    print(f"Skewness: {amount.skew():.2f}")
    print(f"Minimum: INR {amount.min():.2f}")
    print(f"Q1: INR {q1:.2f}")
    print(f"Q3: INR {q3:.2f}")
    print(f"Maximum: INR {amount.max():.2f}")
    print(f"IQR: INR {iqr:.2f}")
    print(f"Lower fence: INR {lower_fence:.2f}")
    print(f"Upper fence: INR {upper_fence:.2f}")
    print(f"Outliers below lower fence: {(amount < lower_fence).sum()}")
    print(f"Outliers above upper fence: {(amount > upper_fence).sum()}")

    # ---------------------------------------------------------
    # D. Correlation matrix for completed orders
    # ---------------------------------------------------------
    numeric_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes",
    ]

    correlation = completed[numeric_columns].corr().round(4)

    print("\nCORRELATION MATRIX")
    print(correlation.to_string())

    # Identify the off-diagonal pair with the largest absolute correlation.
    pairs = []
    for i, col_a in enumerate(numeric_columns):
        for col_b in numeric_columns[i + 1:]:
            value = correlation.loc[col_a, col_b]
            pairs.append((col_a, col_b, value))

    strongest = max(pairs, key=lambda item: abs(item[2]))
    print(
        "\nLargest absolute correlation: "
        f"{strongest[0]} and {strongest[1]} = {strongest[2]:.4f}"
    )

    # Save analysis outputs for later reproducibility.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUTPUT_DIR / "pandas_revenue_pivot.csv")
    city_summary.to_csv(OUTPUT_DIR / "city_spend_summary.csv")
    correlation.to_csv(OUTPUT_DIR / "completed_order_correlations.csv")

    # Independent SQL total for cross-checking.
    with sqlite3.connect(DB_PATH) as connection:
        sql_total = pd.read_sql_query(
            """
            SELECT SUM(order_amount) AS revenue
            FROM orders
            WHERE status = 'completed'
            """,
            connection,
        ).iloc[0]["revenue"]

    print(f"\nSQL revenue total: INR {sql_total:,.2f}")
    print(f"Pandas revenue total: INR {pivot.loc['All', 'All']:,.2f}")
    print(
        "Grand totals match:",
        round(float(sql_total), 2) == round(float(pivot.loc["All", "All"]), 2),
    )


if __name__ == "__main__":
    main()
