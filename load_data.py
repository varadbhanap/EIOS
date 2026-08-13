"""
EIOS - Step 1: Load Home Credit Default Risk data into SQLite

Before running this:
1. Download application_train.csv, bureau.csv, previous_application.csv
   from https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Place them inside:  EIOS/data/raw/
3. Run this script from inside the EIOS/ folder:  python load_data.py
"""

import pandas as pd
import sqlite3
import os

RAW_DIR = "data/raw"
DB_PATH = "data/enterprise_data.db"

FILES = {
    "application_train.csv": "loan_applications",
    "bureau.csv": "bureau_credit_history",
    "previous_application.csv": "previous_applications",
}


def load_file(filename, table_name, conn):
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"[MISSING] {path} not found. Skipping {table_name}.")
        return

    print(f"Loading {filename} into table '{table_name}'...")
    # read in chunks since bureau.csv and previous_application.csv are large
    chunk_size = 100_000
    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunk.to_sql(
            table_name,
            conn,
            if_exists="replace" if first_chunk else "append",
            index=False,
        )
        first_chunk = False
        total_rows += len(chunk)

    print(f"  -> {total_rows:,} rows loaded into '{table_name}'")


def main():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    for filename, table_name in FILES.items():
        load_file(filename, table_name, conn)

    # quick sanity check: show table names and row counts
    print("\n--- Database summary ---")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"{table}: {count:,} rows")

    conn.close()
    print(f"\nDone. Database saved at {DB_PATH}")


if __name__ == "__main__":
    main()
