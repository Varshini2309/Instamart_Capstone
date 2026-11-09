
import sqlite3
from pathlib import Path

db_path = Path("data/instamart.db")

with sqlite3.connect(db_path) as connection:
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()

    for (table_name,) in tables:
        print(f"\nTable: {table_name}")

        columns = connection.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()

        print("Columns:", [column[1] for column in columns])

        if "order" in table_name.lower():
            rows = connection.execute(
                f'SELECT * FROM "{table_name}" LIMIT 2'
            ).fetchall()
            print("Sample rows:", rows)
