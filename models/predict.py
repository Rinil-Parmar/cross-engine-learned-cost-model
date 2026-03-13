# -*- coding: utf-8 -*-
"""
==========================================================
Cross-Engine Learned Cost Model — Query Router / Predictor
==========================================================

Given a SQL query, predicts the best engine (SQLite or DuckDB)
by estimating execution cost on each engine and selecting the minimum.

Can be run as:
    - Script:     python -m models.predict
    - Import:     from models.predict import predict_best_engine

Usage:
    python -m models.predict                        # Interactive mode
    python -m models.predict --query "SELECT ..."   # Single query
    python -m models.predict --file queries.sql     # Batch from file
"""

# ============================================================
# CELL 1: Imports & Setup
# ============================================================

import os
import sys
import json
import re
import argparse

import numpy as np
import joblib

# ============================================================
# CELL 2: Configuration & Paths
# ============================================================

# Adjust if running in Colab:
# PROJECT_ROOT = "/content/cross-engine-learned-cost-model"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SQLITE_MODEL_PATH = os.path.join(MODELS_DIR, "model_sqlite.joblib")
DUCKDB_MODEL_PATH = os.path.join(MODELS_DIR, "model_duckdb.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# ============================================================
# CELL 3: Feature Extraction (same 25 features as pipeline)
# ============================================================

FEATURE_COLS = [
    "num_joins",
    "has_subquery",
    "num_conditions",
    "has_groupby",
    "has_orderby",
    "has_having",
    "has_limit",
    "has_distinct",
    "has_like",
    "has_exists",
    "has_case",
    "num_aggregations",
    "num_tables",
    "query_length",
    "num_select_cols",
    "has_between",
    "has_in",
    "has_left_join",
    "join_complexity",
    "num_tokens",
    "num_string_literals",
    "num_numeric_literals",
    "nesting_depth",
    "has_string_func",
    "has_arithmetic",
]


def extract_features(sql: str) -> dict:
    """
    Extract 25 engine-agnostic features from a SQL query string.
    Must match the pipeline's feature extraction exactly.
    """
    sql_upper = sql.upper()
    sql_clean = " ".join(sql.split())

    # --- Joins ---
    num_joins = len(re.findall(r'\bJOIN\b', sql_upper))
    has_left_join = 1 if re.search(r'\bLEFT\s+(OUTER\s+)?JOIN\b', sql_upper) else 0

    # --- Subquery ---
    select_count = len(re.findall(r'\bSELECT\b', sql_upper))
    has_subquery = 1 if select_count > 1 else 0

    # --- WHERE conditions ---
    where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', sql_upper, re.DOTALL)
    num_conditions = 0
    if where_match:
        where_clause = where_match.group(1)
        num_conditions = 1 + len(re.findall(r'\bAND\b|\bOR\b', where_clause))

    # --- Clauses ---
    has_groupby = 1 if re.search(r'\bGROUP\s+BY\b', sql_upper) else 0
    has_orderby = 1 if re.search(r'\bORDER\s+BY\b', sql_upper) else 0
    has_having = 1 if re.search(r'\bHAVING\b', sql_upper) else 0
    has_limit = 1 if re.search(r'\bLIMIT\b', sql_upper) else 0
    has_distinct = 1 if re.search(r'\bDISTINCT\b', sql_upper) else 0

    # --- Predicates ---
    has_like = 1 if re.search(r'\bLIKE\b', sql_upper) else 0
    has_exists = 1 if re.search(r'\bEXISTS\b', sql_upper) else 0
    has_between = 1 if re.search(r'\bBETWEEN\b', sql_upper) else 0
    has_in = 1 if re.search(r'\bIN\s*\(', sql_upper) else 0
    has_case = 1 if re.search(r'\bCASE\b', sql_upper) else 0

    # --- Aggregations ---
    agg_funcs = ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX']
    num_aggregations = sum(len(re.findall(rf'\b{f}\s*\(', sql_upper)) for f in agg_funcs)

    # --- Tables ---
    from_tables = re.findall(r'\bFROM\s+(\w+)', sql_upper)
    join_tables = re.findall(r'\bJOIN\s+(\w+)', sql_upper)
    all_tables = set(from_tables + join_tables)
    # Remove SQL keywords that might be caught
    sql_keywords = {'WHERE', 'SELECT', 'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'UNION', 'EXCEPT', 'INTERSECT'}
    all_tables = all_tables - sql_keywords
    num_tables = max(len(all_tables), 1)

    # --- Query length ---
    query_length = len(sql_clean)

    # --- SELECT columns ---
    select_match = re.search(r'\bSELECT\b(.+?)\bFROM\b', sql_upper, re.DOTALL)
    num_select_cols = 1
    if select_match:
        select_clause = select_match.group(1).strip()
        if select_clause == '*':
            num_select_cols = 1
        else:
            # Count commas outside parentheses
            depth = 0
            commas = 0
            for ch in select_clause:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == ',' and depth == 0:
                    commas += 1
            num_select_cols = commas + 1

    # --- Join complexity ---
    join_complexity = num_joins * num_conditions

    # --- Tokens ---
    num_tokens = len(sql_clean.split())

    # --- Literals ---
    num_string_literals = len(re.findall(r"'[^']*'", sql))
    num_numeric_literals = len(re.findall(r'\b\d+\.?\d*\b', sql))

    # --- Nesting depth ---
    nesting_depth = 0
    current_depth = 0
    for ch in sql:
        if ch == '(':
            current_depth += 1
            nesting_depth = max(nesting_depth, current_depth)
        elif ch == ')':
            current_depth -= 1

    # --- String functions ---
    string_funcs = ['SUBSTR', 'SUBSTRING', 'TRIM', 'UPPER', 'LOWER', 'REPLACE', 'CONCAT']
    has_string_func = 1 if any(re.search(rf'\b{f}\b', sql_upper) for f in string_funcs) else 0

    # --- Arithmetic in SELECT ---
    has_arithmetic = 0
    if select_match:
        select_clause = select_match.group(1)
        if re.search(r'[+\-*/]', select_clause):
            has_arithmetic = 1

    return {
        "num_joins": num_joins,
        "has_subquery": has_subquery,
        "num_conditions": num_conditions,
        "has_groupby": has_groupby,
        "has_orderby": has_orderby,
        "has_having": has_having,
        "has_limit": has_limit,
        "has_distinct": has_distinct,
        "has_like": has_like,
        "has_exists": has_exists,
        "has_case": has_case,
        "num_aggregations": num_aggregations,
        "num_tables": num_tables,
        "query_length": query_length,
        "num_select_cols": num_select_cols,
        "has_between": has_between,
        "has_in": has_in,
        "has_left_join": has_left_join,
        "join_complexity": join_complexity,
        "num_tokens": num_tokens,
        "num_string_literals": num_string_literals,
        "num_numeric_literals": num_numeric_literals,
        "nesting_depth": nesting_depth,
        "has_string_func": has_string_func,
        "has_arithmetic": has_arithmetic,
    }


# ============================================================
# CELL 4: Load Models
# ============================================================


def load_models() -> tuple:
    """Load saved model pair and metadata."""

    if not os.path.exists(SQLITE_MODEL_PATH):
        print(f"❌ SQLite model not found: {SQLITE_MODEL_PATH}")
        print("   Run training first: python -m models.train")
        sys.exit(1)

    if not os.path.exists(DUCKDB_MODEL_PATH):
        print(f"❌ DuckDB model not found: {DUCKDB_MODEL_PATH}")
        print("   Run training first: python -m models.train")
        sys.exit(1)

    model_sqlite = joblib.load(SQLITE_MODEL_PATH)
    model_duckdb = joblib.load(DUCKDB_MODEL_PATH)

    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

    print(f"✅ Models loaded successfully.")
    if metadata:
        print(f"   Model type:    {metadata.get('best_model', 'unknown')}")
        print(f"   Train samples: {metadata.get('training_samples', 'unknown')}")
        print(f"   Test accuracy: {metadata.get('test_accuracy', 'unknown')}")

    return model_sqlite, model_duckdb, metadata


# ============================================================
# CELL 5: Prediction Engine
# ============================================================


def predict_best_engine(sql: str, model_sqlite, model_duckdb) -> dict:
    """
    Core prediction function.

    Input:  SQL query string
    Output: dict with predicted times, best engine, confidence
    """

    # Validate input
    sql = sql.strip()
    if not sql:
        return {"error": "Empty query provided."}

    if not re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', sql.upper()):
        return {"error": "Input does not appear to be a valid SQL query."}

    # Extract features
    try:
        features = extract_features(sql)
    except Exception as e:
        return {"error": f"Feature extraction failed: {str(e)}"}

    # Build feature vector in correct order
    feature_vector = np.array([[features[col] for col in FEATURE_COLS]])

    # Predict
    try:
        pred_sqlite = float(model_sqlite.predict(feature_vector)[0])
        pred_duckdb = float(model_duckdb.predict(feature_vector)[0])
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Ensure non-negative predictions
    pred_sqlite = max(pred_sqlite, 0.0)
    pred_duckdb = max(pred_duckdb, 0.0)

    # Select best engine
    if pred_sqlite <= pred_duckdb:
        best_engine = "sqlite"
        confidence = pred_duckdb / pred_sqlite if pred_sqlite > 0 else float("inf")
    else:
        best_engine = "duckdb"
        confidence = pred_sqlite / pred_duckdb if pred_duckdb > 0 else float("inf")

    # Determine margin
    diff = abs(pred_sqlite - pred_duckdb)
    if diff < 0.001:
        decision_strength = "negligible"
    elif diff < 0.01:
        decision_strength = "weak"
    elif diff < 0.05:
        decision_strength = "moderate"
    else:
        decision_strength = "strong"

    return {
        "query": sql[:100] + ("..." if len(sql) > 100 else ""),
        "predicted_sqlite_sec": round(pred_sqlite, 6),
        "predicted_duckdb_sec": round(pred_duckdb, 6),
        "best_engine": best_engine,
        "confidence_ratio": round(confidence, 2),
        "decision_strength": decision_strength,
        "features": features,
    }


# ============================================================
# CELL 6: Batch Prediction
# ============================================================


def predict_batch(queries: list, model_sqlite, model_duckdb) -> list:
    """Predict best engine for a list of SQL queries."""
    results = []
    for i, sql in enumerate(queries):
        result = predict_best_engine(sql, model_sqlite, model_duckdb)
        result["query_index"] = i
        results.append(result)
    return results


def predict_from_file(filepath: str, model_sqlite, model_duckdb) -> list:
    """Load queries from a .sql file (one per line or semicolon-separated)."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return []

    with open(filepath, "r") as f:
        content = f.read()

    # Split by semicolons or newlines
    if ";" in content:
        queries = [q.strip() for q in content.split(";") if q.strip()]
    else:
        queries = [q.strip() for q in content.splitlines() if q.strip() and not q.strip().startswith("--")]

    print(f"📄 Loaded {len(queries)} queries from {filepath}")
    return predict_batch(queries, model_sqlite, model_duckdb)


# ============================================================
# CELL 7: Display Functions
# ============================================================


def display_result(result: dict):
    """Pretty-print a single prediction result."""

    if "error" in result:
        print(f"  ❌ Error: {result['error']}")
        return

    best = result["best_engine"]
    emoji = "🟦" if best == "sqlite" else "🟩"

    print(f"\n  {'─' * 55}")
    print(f"  Query: {result['query']}")
    print(f"  {'─' * 55}")
    print(f"  Predicted SQLite time: {result['predicted_sqlite_sec']:.6f}s")
    print(f"  Predicted DuckDB time: {result['predicted_duckdb_sec']:.6f}s")
    print(f"  {emoji} Best Engine: {best.upper()}")
    print(f"  Confidence: {result['confidence_ratio']:.2f}x ({result['decision_strength']})")


def display_batch_summary(results: list):
    """Print summary of batch predictions."""

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid:
        print("  ❌ No valid predictions.")
        return

    sqlite_count = sum(1 for r in valid if r["best_engine"] == "sqlite")
    duckdb_count = sum(1 for r in valid if r["best_engine"] == "duckdb")
    total_sqlite = sum(r["predicted_sqlite_sec"] for r in valid)
    total_duckdb = sum(r["predicted_duckdb_sec"] for r in valid)
    total_model = sum(
        r["predicted_sqlite_sec"] if r["best_engine"] == "sqlite"
        else r["predicted_duckdb_sec"]
        for r in valid
    )

    print(f"\n{'=' * 55}")
    print(f"  BATCH PREDICTION SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Total queries:    {len(results)}")
    print(f"  Valid:            {len(valid)}")
    if errors:
        print(f"  Errors:           {len(errors)}")
    print(f"\n  Engine Routing:")
    print(f"    → SQLite: {sqlite_count} ({sqlite_count/len(valid)*100:.1f}%)")
    print(f"    → DuckDB: {duckdb_count} ({duckdb_count/len(valid)*100:.1f}%)")
    print(f"\n  Estimated Total Runtime:")
    print(f"    All on SQLite:  {total_sqlite:.4f}s")
    print(f"    All on DuckDB:  {total_duckdb:.4f}s")
    print(f"    Model routing:  {total_model:.4f}s")

    savings = ((min(total_sqlite, total_duckdb) - total_model) / min(total_sqlite, total_duckdb)) * 100
    print(f"    Savings:        {savings:.2f}% over best static")
    print(f"{'=' * 55}")


# ============================================================
# CELL 8: Interactive Mode
# ============================================================


def interactive_mode(model_sqlite, model_duckdb):
    """Interactive SQL query router — type queries, get engine recommendations."""

    print(f"\n{'=' * 55}")
    print("  🚀 Interactive Query Router")
    print(f"{'=' * 55}")
    print("  Type a SQL query to get the best engine.")
    print("  Commands:")
    print("    exit / quit   — stop")
    print("    example       — show example queries")
    print()

    examples = [
        "SELECT * FROM lineitem WHERE l_quantity > 30;",
        "SELECT l_returnflag, SUM(l_extendedprice) FROM lineitem GROUP BY l_returnflag;",
        "SELECT c.c_name, o.o_totalprice FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey WHERE o.o_totalprice > 1000;",
        "SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount)) FROM lineitem l JOIN supplier s ON l.l_suppkey = s.s_suppkey JOIN nation n ON s.s_nationkey = n.n_nationkey GROUP BY n.n_name ORDER BY 2 DESC;",
    ]

    while True:
        try:
            sql = input("\n  SQL> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not sql:
            continue

        if sql.lower() in ("exit", "quit", "q"):
            print("  Goodbye!")
            break

        if sql.lower() == "example":
            print("\n  Example queries:")
            for i, ex in enumerate(examples, 1):
                print(f"    {i}. {ex}")
            continue

        # Handle example number selection
        if sql.isdigit() and 1 <= int(sql) <= len(examples):
            sql = examples[int(sql) - 1]
            print(f"  Using: {sql}")

        result = predict_best_engine(sql, model_sqlite, model_duckdb)
        display_result(result)


# ============================================================
# CELL 9: Main Entry Point
# ============================================================


def main():
    """Main entry point with CLI argument support."""

    parser = argparse.ArgumentParser(
        description="Cross-Engine Query Router — Predict the best SQL engine."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single SQL query to predict.",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to .sql file with queries (one per line or semicolon-separated).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )

    args = parser.parse_args()

    # Load models
    model_sqlite, model_duckdb, metadata = load_models()

    # --- Mode 1: Single query ---
    if args.query:
        result = predict_best_engine(args.query, model_sqlite, model_duckdb)
        if args.json:
            # Remove features from JSON output for cleaner output
            result.pop("features", None)
            print(json.dumps(result, indent=2))
        else:
            display_result(result)
        return

    # --- Mode 2: Batch from file ---
    if args.file:
        results = predict_from_file(args.file, model_sqlite, model_duckdb)
        if args.json:
            for r in results:
                r.pop("features", None)
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                display_result(r)
            display_batch_summary(results)
        return

    # --- Mode 3: Interactive ---
    interactive_mode(model_sqlite, model_duckdb)


if __name__ == "__main__":
    main()