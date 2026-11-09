
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CLEAN.mkdir(parents=True, exist_ok=True)


def main():
    # ---------------------------------------------------------
    # 1. Load the raw tables
    # ---------------------------------------------------------
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(
        RAW / "orders.csv",
        dtype={"order_amount": "string"},
    )

    changes = {}

    # ---------------------------------------------------------
    # 2. Clean customers
    # ---------------------------------------------------------
    customers["city"] = customers["city"].str.strip().str.title()

    gender_map = {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
    }

    original_gender = customers["gender"].copy()
    cleaned_gender = (
        customers["gender"]
        .fillna("")
        .str.strip()
        .str.lower()
        .map(gender_map)
        .fillna("unknown")
    )
    changes["Customer gender values standardized"] = int(
        (original_gender.fillna("unknown") != cleaned_gender).sum()
    )
    customers["gender"] = cleaned_gender

    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    )

    invalid_age = (
        customers["age"].notna()
        & ~customers["age"].between(10, 100)
    )
    changes["Customer ages outside 10-100 set missing"] = int(
        invalid_age.sum()
    )
    customers.loc[invalid_age, "age"] = pd.NA

    missing_age_before_fill = int(customers["age"].isna().sum())
    city_median_age = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_median_age)

    changes["Missing customer ages filled by city median"] = (
        missing_age_before_fill
    )

    customers["age"] = customers["age"].round().astype("int64")
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="raise"
    )

    # ---------------------------------------------------------
    # 3. Clean orders in the required order
    # ---------------------------------------------------------
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="raise"
    )

    exact_duplicates = int(orders.duplicated().sum())
    orders = orders.drop_duplicates().copy()
    changes["Exact duplicate order rows removed"] = exact_duplicates

    # Sort so the latest record for each order_id is retained.
    orders = orders.sort_values(
        ["order_id", "order_ts"], kind="mergesort"
    )
    duplicate_ids = int(orders["order_id"].duplicated().sum())
    orders = orders.drop_duplicates(
        subset=["order_id"], keep="last"
    ).copy()
    changes["Older duplicate order records removed"] = duplicate_ids

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    # Join product prices and categories for imputation.
    product_lookup = products[
        ["product_id", "price_inr", "category"]
    ]
    orders = orders.merge(
        product_lookup,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    missing_amount = orders["order_amount"].isna()
    changes["Missing or nonnumeric order amounts filled"] = int(
        missing_amount.sum()
    )
    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)
    orders.loc[missing_amount, "order_amount"] = calculated_amount[
        missing_amount
    ]

    negative_delivery = orders["delivery_minutes"] < 0
    changes["Negative delivery times set missing"] = int(
        negative_delivery.sum()
    )
    orders.loc[negative_delivery, "delivery_minutes"] = pd.NA

    completed = orders["status"].eq("completed")
    missing_completed_delivery = (
        completed & orders["delivery_minutes"].isna()
    )
    changes["Missing completed delivery times filled"] = int(
        missing_completed_delivery.sum()
    )

    # Median by product category, using completed orders only.
    completed_category_median = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )

    completed_rows = orders.loc[completed].copy()
    completed_rows["category_median"] = completed_category_median
    fill_delivery = completed_rows["delivery_minutes"].isna()
    completed_rows.loc[fill_delivery, "delivery_minutes"] = (
        completed_rows.loc[fill_delivery, "category_median"]
    )

    orders.loc[completed_rows.index, "delivery_minutes"] = (
        completed_rows["delivery_minutes"]
    )

    # Cancelled orders must never receive an invented delivery time.
    cancelled = orders["status"].eq("cancelled")
    orders.loc[cancelled, "delivery_minutes"] = pd.NA

    # Remove helper columns before saving the clean orders table.
    orders = orders.drop(columns=["price_inr", "category"])

    # Round monetary values to two decimal places.
    orders["order_amount"] = orders["order_amount"].round(2)

    # ---------------------------------------------------------
    # 4. Save clean CSV files
    # ---------------------------------------------------------
    customers.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)

    # Store timestamps in the required text format.
    orders["order_ts"] = orders["order_ts"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    customers["signup_date"] = customers["signup_date"].dt.strftime(
        "%Y-%m-%d"
    )
    orders.to_csv(CLEAN / "orders.csv", index=False)

    # ---------------------------------------------------------
    # 5. Store clean tables in SQLite
    # ---------------------------------------------------------
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as connection:
        customers.to_sql(
            "customers", connection, if_exists="replace", index=False
        )
        products.to_sql(
            "products", connection, if_exists="replace", index=False
        )
        orders.to_sql(
            "orders", connection, if_exists="replace", index=False
        )

    # ---------------------------------------------------------
    # 6. Report cleaning changes
    # ---------------------------------------------------------
    print("\nCLEANING CHANGE LOG")
    for label, count in changes.items():
        print(f"{label}: {count}")

    print("\nCLEAN TABLE SHAPES")
    print(f"customers: {customers.shape}")
    print(f"products: {products.shape}")
    print(f"orders: {orders.shape}")

    print(f"\nClean CSV files saved to: {CLEAN}")
    print(f"SQLite database saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
