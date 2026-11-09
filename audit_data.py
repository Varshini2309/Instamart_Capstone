
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"


def report_table(name, df):
    print(f"\n{'=' * 65}")
    print(f"TABLE: {name}")
    print(f"Shape: {df.shape}")

    print("\nData types:")
    print(df.dtypes.to_string())

    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percent,
    })

    print("\nMissing-value report:")
    print(missing_report.to_string())


def main():
    orders = pd.read_csv(RAW / "orders.csv")
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")

    # Preserve empty ticket fields as empty strings when loading tickets.
    tickets = pd.read_csv(
        RAW / "tickets.csv",
        keep_default_na=False,
    )

    for name, df in [
        ("orders.csv", orders),
        ("customers.csv", customers),
        ("products.csv", products),
        ("tickets.csv", tickets),
    ]:
        report_table(name, df)

    numeric_amount = pd.to_numeric(
        orders["order_amount"],
        errors="coerce",
    )

    numeric_age = pd.to_numeric(
        customers["age"],
        errors="coerce",
    )

    print(f"\n{'=' * 65}")
    print("REQUIRED AUDIT CHECKPOINTS")

    checks = {
        "Raw order rows": len(orders),
        "Exact duplicate order rows": int(orders.duplicated().sum()),
        "Pending order rows": int(
            orders["status"].eq("pending").sum()
        ),
        "Distinct order IDs": int(orders["order_id"].nunique()),
        "Nonnumeric order amounts": int(numeric_amount.isna().sum()),
        "Negative delivery times": int(
            (orders["delivery_minutes"] < 0).sum()
        ),
        "Blank delivery times": int(
            orders["delivery_minutes"].isna().sum()
        ),
        "Blank customer ages": int(customers["age"].isna().sum()),
        "Ages outside 10-100": int(
            ((numeric_age < 10) | (numeric_age > 100)).sum()
        ),
        "Distinct raw city spellings": int(
            customers["city"].nunique()
        ),
        "Distinct nonblank raw gender spellings": int(
            customers.loc[
                customers["gender"].notna()
                & customers["gender"].astype(str).str.strip().ne(""),
                "gender",
            ].nunique()
        ),
        "Blank customer genders": int(
            customers["gender"].isna().sum()
        ),
    }

    for label, value in checks.items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
