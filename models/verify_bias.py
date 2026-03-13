import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test_dataset.csv"))

sqlite_faster = (test_df["sqlite_time_sec"] < test_df["duckdb_time_sec"]).sum()
duckdb_faster = (test_df["duckdb_time_sec"] < test_df["sqlite_time_sec"]).sum()
total = len(test_df)

print(f"Total test queries: {total}")
print(f"SQLite actually faster: {sqlite_faster} ({sqlite_faster/total*100:.1f}%)")
print(f"DuckDB actually faster: {duckdb_faster} ({duckdb_faster/total*100:.1f}%)")