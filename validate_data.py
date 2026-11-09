
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"


def main():
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    completed = orders.loc[orders["status"] == "completed"]
    cancelled = orders.loc[orders["status"] == "cancelled"]

    # Customer checkpoints
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }

    assert len(customers) == 1200
    assert customers["city"].value_counts().to_dict() == expected_cities
    assert customers["gender"].value_counts().to_dict() == {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }
    assert customers["age"].notna().all()
    assert customers["age"].between(18, 60).all()

    # Order checkpoints
    assert len(orders) == 12309
    assert orders["order_id"].nunique() == 12309
    assert completed.shape[0] == 11110
    assert cancelled.shape[0] == 1199
    assert not orders["status"].eq("pending").any()
    assert orders["order_amount"].notna().all()
    assert completed["delivery_minutes"].notna().all()
    assert cancelled["delivery_minutes"].isna().all()
    assert round(completed["order_amount"].sum(), 2) == 2998668.80

    # Product checkpoint
    assert len(products) == 40

    # SQLite checkpoint
    with sqlite3.connect(DB_PATH) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"customers", "products", "orders"}.issubset(tables)

        for table in ("customers", "products", "orders"):
            database_count = connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            expected_count = {
                "customers": len(customers),
                "products": len(products),
                "orders": len(orders),
            }[table]

            assert database_count == expected_count

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
