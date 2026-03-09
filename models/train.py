# Cell 1:  Imports
# Cell 2:  Config & paths
# Cell 3:  Load train/val/test (auto-splits master if needed)
# Cell 4:  Prepare X, Y matrices
# Cell 5:  Define 4 model types
# Cell 6:  Metric functions
# Cell 7:  Train on TRAIN, evaluate on VAL, pick best
# Cell 8:  Final evaluation on TEST (touched once)
# Cell 9:  Baseline comparisons on TEST
# Cell 10: Retrain best on TRAIN+VAL, save .joblib
# Cell 11: Generate 5 plots
# Cell 12: Save metrics JSON, print summary
"""
==========================================================
Cross-Engine Learned Cost Model — Model Training Pipeline
==========================================================

This script trains per-engine regression models to predict
SQL query execution time, then selects the best model type
based on engine selection accuracy.

Can be run as:
    - Python script:  python -m models.train
    - Colab notebook: copy each cell block

Input:
    data/train_dataset.csv
    data/val_dataset.csv
    data/test_dataset.csv

Output:
    models/model_sqlite.joblib
    models/model_duckdb.joblib
    models/model_metadata.json
    results/metrics/evaluation_results.json
    results/figures/*.png
"""

# ============================================================
# CELL 1: Imports & Setup
# ============================================================

import os
import sys
import json
import warnings
import time

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Remove this line if running in Colab
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.base import clone
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

print("✅ All libraries imported successfully.")

# ============================================================
# CELL 2: Configuration & Paths
# ============================================================

# --- Paths ---
# Adjust PROJECT_ROOT if running in Colab
# PROJECT_ROOT = "/content/cross-engine-learned-cost-model"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Create output directories
for d in [DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Dataset Paths ---
TRAIN_PATH = os.path.join(DATA_DIR, "train_dataset.csv")
VAL_PATH = os.path.join(DATA_DIR, "val_dataset.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_dataset.csv")
MASTER_PATH = os.path.join(DATA_DIR, "master_dataset.csv")

# --- 25 Feature Columns (must match pipeline output) ---
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

# --- Target Columns ---
SQLITE_TARGET = "sqlite_time_sec"
DUCKDB_TARGET = "duckdb_time_sec"

print(f"✅ Configuration ready.")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Features:     {len(FEATURE_COLS)}")
print(f"   Targets:      {SQLITE_TARGET}, {DUCKDB_TARGET}")

# ============================================================
# CELL 3: Load & Validate Datasets
# ============================================================


def validate_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate a dataset has required columns and clean data."""

    # Check feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"   ❌ {name}: Missing features: {missing}")
        sys.exit(1)

    # Check target columns
    for t in [SQLITE_TARGET, DUCKDB_TARGET]:
        if t not in df.columns:
            print(f"   ❌ {name}: Missing target: {t}")
            sys.exit(1)

    # Drop invalid rows (negative or zero runtimes)
    original = len(df)
    df = df[(df[SQLITE_TARGET] > 0) & (df[DUCKDB_TARGET] > 0)].copy()
    dropped = original - len(df)

    # Drop rows with NaN in features
    nan_before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [SQLITE_TARGET, DUCKDB_TARGET]).copy()
    nan_dropped = nan_before - len(df)

    # Drop infinite values
    inf_mask = np.isinf(df[FEATURE_COLS + [SQLITE_TARGET, DUCKDB_TARGET]]).any(axis=1)
    inf_dropped = inf_mask.sum()
    df = df[~inf_mask].copy()

    status = "✅" if len(df) > 0 else "❌ EMPTY"
    print(f"   {status} {name}: {len(df)} rows", end="")
    if dropped > 0:
        print(f" (dropped {dropped} invalid)", end="")
    if nan_dropped > 0:
        print(f" (dropped {nan_dropped} NaN)", end="")
    if inf_dropped > 0:
        print(f" (dropped {inf_dropped} Inf)", end="")
    print()

    return df.reset_index(drop=True)


def load_datasets() -> tuple:
    """
    Load train/val/test datasets.
    Falls back to master_dataset.csv with auto-split if splits don't exist.
    """
    print("\n📂 Loading datasets...")

    # Case 1: All three split files exist
    if all(os.path.exists(p) for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]):
        print("   Found train/val/test split files.")
        train_df = validate_dataframe(pd.read_csv(TRAIN_PATH), "Train")
        val_df = validate_dataframe(pd.read_csv(VAL_PATH), "Val")
        test_df = validate_dataframe(pd.read_csv(TEST_PATH), "Test")
        return train_df, val_df, test_df

    # Case 2: Only master_dataset.csv exists — auto-split
    if os.path.exists(MASTER_PATH):
        print("   ⚠️  Split files not found. Auto-splitting master_dataset.csv...")
        master_df = validate_dataframe(pd.read_csv(MASTER_PATH), "Master")

        from sklearn.model_selection import train_test_split

        # 70% train, 15% val, 15% test
        train_df, temp_df = train_test_split(
            master_df, test_size=0.30, random_state=42,
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Save for future use
        train_df.to_csv(TRAIN_PATH, index=False)
        val_df.to_csv(VAL_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)

        print(f"   ✅ Auto-split saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    # Case 3: Nothing found
    print("   ❌ No dataset files found in data/ directory.")
    print("      Run the pipeline first: python scripts/pipeline.py")
    sys.exit(1)


train_df, val_df, test_df = load_datasets()

print(f"\n📊 Dataset Summary:")
print(f"   Train: {len(train_df)} rows")
print(f"   Val:   {len(val_df)} rows")
print(f"   Test:  {len(test_df)} rows")
print(f"   Total: {len(train_df) + len(val_df) + len(test_df)} rows")

# ============================================================
# CELL 4: Prepare Feature Matrices
# ============================================================

X_train = train_df[FEATURE_COLS].values
X_val = val_df[FEATURE_COLS].values
X_test = test_df[FEATURE_COLS].values

y_train_sqlite = train_df[SQLITE_TARGET].values
y_train_duckdb = train_df[DUCKDB_TARGET].values

y_val_sqlite = val_df[SQLITE_TARGET].values
y_val_duckdb = val_df[DUCKDB_TARGET].values

y_test_sqlite = test_df[SQLITE_TARGET].values
y_test_duckdb = test_df[DUCKDB_TARGET].values

print(f"✅ Feature matrices prepared.")
print(f"   X_train: {X_train.shape}")
print(f"   X_val:   {X_val.shape}")
print(f"   X_test:  {X_test.shape}")

# Quick sanity check
print(f"\n📈 Runtime Statistics (Train):")
print(f"   SQLite  — mean: {y_train_sqlite.mean():.4f}s, "
      f"median: {np.median(y_train_sqlite):.4f}s, "
      f"max: {y_train_sqlite.max():.4f}s")
print(f"   DuckDB  — mean: {y_train_duckdb.mean():.4f}s, "
      f"median: {np.median(y_train_duckdb):.4f}s, "
      f"max: {y_train_duckdb.max():.4f}s")

faster_sqlite = (y_train_sqlite < y_train_duckdb).sum()
faster_duckdb = (y_train_duckdb < y_train_sqlite).sum()
print(f"\n   SQLite faster: {faster_sqlite} ({faster_sqlite/len(y_train_sqlite)*100:.1f}%)")
print(f"   DuckDB faster: {faster_duckdb} ({faster_duckdb/len(y_train_duckdb)*100:.1f}%)")

# ============================================================
# CELL 5: Define Models
# ============================================================


def get_models() -> dict:
    """Return dict of model_name → sklearn estimator."""
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        ),
    }


print(f"✅ 4 model types defined:")
for name in get_models():
    print(f"   • {name}")

# ============================================================
# CELL 6: Engine Selection Accuracy Calculator
# ============================================================


def compute_selection_metrics(
    y_true_sqlite: np.ndarray,
    y_true_duckdb: np.ndarray,
    y_pred_sqlite: np.ndarray,
    y_pred_duckdb: np.ndarray,
) -> dict:
    """
    Core metric: does the model pick the actually faster engine?

    Returns accuracy, correct count, overhead from wrong picks.
    """
    true_best = np.where(y_true_sqlite <= y_true_duckdb, "sqlite", "duckdb")
    pred_best = np.where(y_pred_sqlite <= y_pred_duckdb, "sqlite", "duckdb")

    correct = (true_best == pred_best).sum()
    total = len(true_best)
    accuracy = correct / total if total > 0 else 0.0

    # Overhead = extra time from wrong selections
    true_optimal = np.minimum(y_true_sqlite, y_true_duckdb)
    pred_selected = np.where(
        y_pred_sqlite <= y_pred_duckdb, y_true_sqlite, y_true_duckdb
    )
    total_overhead = float((pred_selected - true_optimal).sum())

    return {
        "selection_accuracy": round(accuracy, 4),
        "correct": int(correct),
        "total": int(total),
        "total_overhead_sec": round(total_overhead, 6),
        "avg_overhead_per_query_sec": round(total_overhead / total, 6) if total > 0 else 0,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics for one engine."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "median_ae": round(float(median_absolute_error(y_true, y_pred)), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(float(
            np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-10, None))) * 100
        ), 2),
    }


print("✅ Metric functions defined.")

# ============================================================
# CELL 7: Train All Models on Train Set, Evaluate on Val Set
# ============================================================

print("\n" + "=" * 65)
print("  MODEL TRAINING (Train Set) & EVALUATION (Val Set)")
print("=" * 65)

model_configs = get_models()
all_results = {}
trained_models = {}  # Store trained model pairs

best_val_accuracy = -1
best_model_name = None

for model_name, model_template in model_configs.items():
    print(f"\n{'─' * 55}")
    print(f"  📦 {model_name}")
    print(f"{'─' * 55}")

    start_time = time.perf_counter()

    # Clone fresh models
    model_s = clone(model_template)
    model_d = clone(model_template)

    # Train on train set
    model_s.fit(X_train, y_train_sqlite)
    model_d.fit(X_train, y_train_duckdb)

    train_time = time.perf_counter() - start_time

    # --- Evaluate on TRAIN set (to check for overfitting) ---
    train_pred_s = model_s.predict(X_train)
    train_pred_d = model_d.predict(X_train)

    train_selection = compute_selection_metrics(
        y_train_sqlite, y_train_duckdb, train_pred_s, train_pred_d
    )
    train_reg_s = compute_regression_metrics(y_train_sqlite, train_pred_s)
    train_reg_d = compute_regression_metrics(y_train_duckdb, train_pred_d)

    # --- Evaluate on VAL set (model selection) ---
    val_pred_s = model_s.predict(X_val)
    val_pred_d = model_d.predict(X_val)

    val_selection = compute_selection_metrics(
        y_val_sqlite, y_val_duckdb, val_pred_s, val_pred_d
    )
    val_reg_s = compute_regression_metrics(y_val_sqlite, val_pred_s)
    val_reg_d = compute_regression_metrics(y_val_duckdb, val_pred_d)

    # Print results
    print(f"  Training time: {train_time:.3f}s")
    print()
    print(f"  {'Metric':<30} {'Train':<12} {'Val':<12}")
    print(f"  {'─' * 54}")
    print(f"  {'Selection Accuracy':<30} {train_selection['selection_accuracy']:<12.4f} {val_selection['selection_accuracy']:<12.4f}")
    print(f"  {'MAE (SQLite)':<30} {train_reg_s['mae']:<12.6f} {val_reg_s['mae']:<12.6f}")
    print(f"  {'MAE (DuckDB)':<30} {train_reg_d['mae']:<12.6f} {val_reg_d['mae']:<12.6f}")
    print(f"  {'R² (SQLite)':<30} {train_reg_s['r2']:<12.4f} {val_reg_s['r2']:<12.4f}")
    print(f"  {'R² (DuckDB)':<30} {train_reg_d['r2']:<12.4f} {val_reg_d['r2']:<12.4f}")

    # Check for overfitting
    acc_gap = train_selection["selection_accuracy"] - val_selection["selection_accuracy"]
    if acc_gap > 0.15:
        print(f"  ⚠️  Possible overfitting! Train-Val accuracy gap: {acc_gap:.4f}")

    # Store results
    all_results[model_name] = {
        "training_time_sec": round(train_time, 3),
        "train": {
            "selection": train_selection,
            "regression_sqlite": train_reg_s,
            "regression_duckdb": train_reg_d,
        },
        "val": {
            "selection": val_selection,
            "regression_sqlite": val_reg_s,
            "regression_duckdb": val_reg_d,
        },
    }

    trained_models[model_name] = (model_s, model_d)

    # Track best
    if val_selection["selection_accuracy"] > best_val_accuracy:
        best_val_accuracy = val_selection["selection_accuracy"]
        best_model_name = model_name

print(f"\n{'=' * 65}")
print(f"  🏆 Best Model (by Val Accuracy): {best_model_name} ({best_val_accuracy:.4f})")
print(f"{'=' * 65}")

# ============================================================
# CELL 8: Final Evaluation on TEST Set (One Time Only)
# ============================================================

print(f"\n{'=' * 65}")
print(f"  FINAL EVALUATION — TEST SET (untouched until now)")
print(f"{'=' * 65}")

final_model_s, final_model_d = trained_models[best_model_name]

test_pred_s = final_model_s.predict(X_test)
test_pred_d = final_model_d.predict(X_test)

test_selection = compute_selection_metrics(
    y_test_sqlite, y_test_duckdb, test_pred_s, test_pred_d
)
test_reg_s = compute_regression_metrics(y_test_sqlite, test_pred_s)
test_reg_d = compute_regression_metrics(y_test_duckdb, test_pred_d)

print(f"\n  Model: {best_model_name}")
print(f"\n  🎯 Engine Selection Accuracy: {test_selection['selection_accuracy']:.4f}")
print(f"     Correct: {test_selection['correct']}/{test_selection['total']}")
print(f"     Total overhead: {test_selection['total_overhead_sec']:.6f}s")
print(f"     Avg overhead/query: {test_selection['avg_overhead_per_query_sec']:.6f}s")
print(f"\n  📊 Regression Metrics:")
print(f"     {'Metric':<15} {'SQLite':<12} {'DuckDB':<12}")
print(f"     {'─' * 39}")
print(f"     {'MAE':<15} {test_reg_s['mae']:<12.6f} {test_reg_d['mae']:<12.6f}")
print(f"     {'RMSE':<15} {test_reg_s['rmse']:<12.6f} {test_reg_d['rmse']:<12.6f}")
print(f"     {'R²':<15} {test_reg_s['r2']:<12.4f} {test_reg_d['r2']:<12.4f}")
print(f"     {'MAPE (%)':<15} {test_reg_s['mape']:<12.2f} {test_reg_d['mape']:<12.2f}")

all_results["test"] = {
    "best_model": best_model_name,
    "selection": test_selection,
    "regression_sqlite": test_reg_s,
    "regression_duckdb": test_reg_d,
}

# ============================================================
# CELL 9: Baseline Comparisons on TEST Set
# ============================================================

print(f"\n{'=' * 65}")
print("  BASELINE COMPARISONS (Test Set)")
print(f"{'=' * 65}")

# Total runtimes
total_sqlite = float(y_test_sqlite.sum())
total_duckdb = float(y_test_duckdb.sum())
total_oracle = float(np.minimum(y_test_sqlite, y_test_duckdb).sum())

# Model selection
model_selected = np.where(test_pred_s <= test_pred_d, y_test_sqlite, y_test_duckdb)
total_model = float(model_selected.sum())

# Heuristic: joins > 2 → DuckDB, else SQLite
test_joins = test_df["num_joins"].values
heuristic_selected = np.where(test_joins > 2, y_test_duckdb, y_test_sqlite)
total_heuristic = float(heuristic_selected.sum())

# Heuristic accuracy
h_pred_s = np.where(test_joins <= 2, 0.0, 1.0)
h_pred_d = np.where(test_joins > 2, 0.0, 1.0)
heuristic_metrics = compute_selection_metrics(
    y_test_sqlite, y_test_duckdb, h_pred_s, h_pred_d
)

# Always-X accuracies
always_sqlite_acc = float((y_test_sqlite <= y_test_duckdb).mean())
always_duckdb_acc = float((y_test_duckdb <= y_test_sqlite).mean())

print(f"\n  {'Strategy':<25} {'Total Time (s)':<18} {'Accuracy':<12}")
print(f"  {'─' * 55}")
print(f"  {'Always SQLite':<25} {total_sqlite:<18.4f} {always_sqlite_acc:<12.4f}")
print(f"  {'Always DuckDB':<25} {total_duckdb:<18.4f} {always_duckdb_acc:<12.4f}")
print(f"  {'Heuristic (joins>2)':<25} {total_heuristic:<18.4f} {heuristic_metrics['selection_accuracy']:<12.4f}")
print(f"  {'Model (learned)':<25} {total_model:<18.4f} {test_selection['selection_accuracy']:<12.4f}")
print(f"  {'Oracle (perfect)':<25} {total_oracle:<18.4f} {'1.0000':<12}")

best_static = min(total_sqlite, total_duckdb)
improvement = ((best_static - total_model) / best_static) * 100
print(f"\n  📈 Improvement over best static baseline: {improvement:.2f}%")
print(f"  📈 Improvement over heuristic: {((total_heuristic - total_model) / total_heuristic) * 100:.2f}%")

baselines = {
    "always_sqlite": {"total_time_sec": round(total_sqlite, 4), "accuracy": round(always_sqlite_acc, 4)},
    "always_duckdb": {"total_time_sec": round(total_duckdb, 4), "accuracy": round(always_duckdb_acc, 4)},
    "heuristic": {"total_time_sec": round(total_heuristic, 4), "accuracy": heuristic_metrics["selection_accuracy"]},
    "model": {"total_time_sec": round(total_model, 4), "accuracy": test_selection["selection_accuracy"]},
    "oracle": {"total_time_sec": round(total_oracle, 4), "accuracy": 1.0},
    "improvement_over_best_static_pct": round(improvement, 2),
}
all_results["baselines"] = baselines

# ============================================================
# CELL 10: Retrain Best Model on Train+Val, Save
# ============================================================

print(f"\n{'=' * 65}")
print(f"  RETRAINING BEST MODEL ON TRAIN + VAL")
print(f"{'=' * 65}")

# Combine train + val for final model
X_trainval = np.vstack([X_train, X_val])
y_trainval_sqlite = np.concatenate([y_train_sqlite, y_val_sqlite])
y_trainval_duckdb = np.concatenate([y_train_duckdb, y_val_duckdb])

print(f"  Combined train+val: {len(X_trainval)} samples")

final_model_s = clone(get_models()[best_model_name])
final_model_d = clone(get_models()[best_model_name])

final_model_s.fit(X_trainval, y_trainval_sqlite)
final_model_d.fit(X_trainval, y_trainval_duckdb)

# Save models
sqlite_path = os.path.join(MODELS_DIR, "model_sqlite.joblib")
duckdb_path = os.path.join(MODELS_DIR, "model_duckdb.joblib")
joblib.dump(final_model_s, sqlite_path)
joblib.dump(final_model_d, duckdb_path)
print(f"  ✅ Saved: {sqlite_path}")
print(f"  ✅ Saved: {duckdb_path}")

# Save metadata
metadata = {
    "best_model": best_model_name,
    "feature_columns": FEATURE_COLS,
    "engines": ["sqlite", "duckdb"],
    "training_samples": len(X_trainval),
    "val_accuracy": best_val_accuracy,
    "test_accuracy": test_selection["selection_accuracy"],
}
metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✅ Saved: {metadata_path}")

# ============================================================
# CELL 11: Generate Plots
# ============================================================

print(f"\n{'=' * 65}")
print("  GENERATING PLOTS")
print(f"{'=' * 65}")

# --- Plot 1: Runtime Scatter (SQLite vs DuckDB) ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test_sqlite, y_test_duckdb, alpha=0.5, s=25, c="#2196F3", edgecolors="none")
max_val = max(y_test_sqlite.max(), y_test_duckdb.max()) * 1.05
ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Equal time")
ax.fill_between([0, max_val], [0, 0], [0, max_val], alpha=0.05, color="green", label="DuckDB faster")
ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], alpha=0.05, color="orange", label="SQLite faster")
ax.set_xlabel("SQLite Time (sec)", fontsize=12)
ax.set_ylabel("DuckDB Time (sec)", fontsize=12)
ax.set_title("SQLite vs DuckDB Execution Time (Test Set)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "runtime_comparison.png"), dpi=150)
plt.close()
print("  ✅ runtime_comparison.png")

# --- Plot 2: Model Accuracy Comparison ---
model_names = [k for k in all_results if k not in ("baselines", "test")]
val_accs = [all_results[k]["val"]["selection"]["selection_accuracy"] for k in model_names]
train_accs = [all_results[k]["train"]["selection"]["selection_accuracy"] for k in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x_pos - width / 2, train_accs, width, label="Train", color="#90CAF9", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x_pos + width / 2, val_accs, width, label="Validation", color="#2196F3", edgecolor="black", linewidth=0.5)

ax.set_ylabel("Engine Selection Accuracy", fontsize=12)
ax.set_title("Engine Selection Accuracy: Train vs Validation", fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)

for bar, acc in zip(bars2, val_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{acc:.2%}", ha="center", fontsize=10, fontweight="bold")

ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_accuracy_comparison.png"), dpi=150)
plt.close()
print("  ✅ model_accuracy_comparison.png")

# --- Plot 3: Baseline Comparison ---
bl = all_results["baselines"]
labels = ["Always\nSQLite", "Always\nDuckDB", "Heuristic", "Model", "Oracle"]
values = [
    bl["always_sqlite"]["total_time_sec"],
    bl["always_duckdb"]["total_time_sec"],
    bl["heuristic"]["total_time_sec"],
    bl["model"]["total_time_sec"],
    bl["oracle"]["total_time_sec"],
]
accs = [
    bl["always_sqlite"]["accuracy"],
    bl["always_duckdb"]["accuracy"],
    bl["heuristic"]["accuracy"],
    bl["model"]["accuracy"],
    bl["oracle"]["accuracy"],
]
bar_colors = ["#EF5350", "#EF5350", "#FFA726", "#66BB6A", "#42A5F5"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Runtime bars
bars = ax1.bar(labels, values, color=bar_colors, edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Total Runtime (sec)", fontsize=12)
ax1.set_title("Total Workload Runtime", fontsize=14)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.2f}s", ha="center", fontsize=9, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# Accuracy bars
bars2 = ax2.bar(labels, accs, color=bar_colors, edgecolor="black", linewidth=0.5)
ax2.set_ylabel("Selection Accuracy", fontsize=12)
ax2.set_title("Engine Selection Accuracy", fontsize=14)
ax2.set_ylim(0, 1.15)
for bar, acc in zip(bars2, accs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{acc:.2%}", ha="center", fontsize=9, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "baseline_comparison.png"), dpi=150)
plt.close()
print("  ✅ baseline_comparison.png")

# --- Plot 4: Feature Importance ---
rf_for_importance = RandomForestRegressor(
    n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
)
y_diff = y_train_sqlite - y_train_duckdb
rf_for_importance.fit(X_train, y_diff)
importances = rf_for_importance.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = range(len(FEATURE_COLS))
ax.barh(y_pos, importances[indices], color="#2196F3", edgecolor="black", linewidth=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels([FEATURE_COLS[i] for i in indices], fontsize=10)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title("Feature Importance for Runtime Difference (SQLite − DuckDB)", fontsize=13)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("  ✅ feature_importance.png")

# --- Plot 5: Prediction Error Distribution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

error_s = y_test_sqlite - test_pred_s
error_d = y_test_duckdb - test_pred_d

ax1.hist(error_s, bins=30, color="#FF9800", edgecolor="black", alpha=0.8)
ax1.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
ax1.set_xlabel("Prediction Error (sec)", fontsize=11)
ax1.set_ylabel("Frequency", fontsize=11)
ax1.set_title("SQLite Prediction Error", fontsize=13)
ax1.grid(axis="y", alpha=0.3)

ax2.hist(error_d, bins=30, color="#4CAF50", edgecolor="black", alpha=0.8)
ax2.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Prediction Error (sec)", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.set_title("DuckDB Prediction Error", fontsize=13)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "prediction_error_distribution.png"), dpi=150)
plt.close()
print("  ✅ prediction_error_distribution.png")

# ============================================================
# CELL 12: Save All Metrics & Final Summary
# ============================================================

metrics_path = os.path.join(METRICS_DIR, "evaluation_results.json")
with open(metrics_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Metrics saved: {metrics_path}")

# --- Final Summary ---
print(f"\n{'=' * 65}")
print("  ✅ TRAINING PIPELINE COMPLETE")
print(f"{'=' * 65}")
print(f"  Best model:              {best_model_name}")
print(f"  Val selection accuracy:  {best_val_accuracy:.4f}")
print(f"  Test selection accuracy: {test_selection['selection_accuracy']:.4f}")
print(f"  Test correct picks:      {test_selection['correct']}/{test_selection['total']}")
print(f"")
print(f"  📁 Saved Files:")
print(f"     models/model_sqlite.joblib")
print(f"     models/model_duckdb.joblib")
print(f"     models/model_metadata.json")
print(f"     results/metrics/evaluation_results.json")
print(f"     results/figures/runtime_comparison.png")
print(f"     results/figures/model_accuracy_comparison.png")
print(f"     results/figures/baseline_comparison.png")
print(f"     results/figures/feature_importance.png")
print(f"     results/figures/prediction_error_distribution.png")
print(f"")
print(f"  Next step: python -m models.predict")
print(f"{'=' * 65}")