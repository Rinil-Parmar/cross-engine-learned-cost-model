# 🔄 Cross-Engine Learned Cost Model

A machine learning system that predicts the **fastest SQL engine** (SQLite or DuckDB) for any given query — without running it on both.

> **Input:** SQL query → **Output:** Best engine + predicted runtime

---

## 📌 What This Does

```
SQL Query → Extract 25 Features → Model predicts SQLite time
                                 → Model predicts DuckDB time
                                 → Pick the faster one ✅
```

- Generates **3,100+** query variants from all **22 TPC-H** benchmark queries
- Benchmarks each on **SQLite** and **DuckDB**
- Trains **per-engine regression models** (one for SQLite, one for DuckDB)
- Routes new queries to the predicted faster engine — **zero-shot, no execution needed**

---

## 🏗️ Project Structure

```
cross-engine-learned-cost-model/
├── data/
│   ├── train_dataset.csv             # Training split (70%)
│   ├── val_dataset.csv               # Validation split (15%)
│   └── test_dataset.csv              # Test split (15%)
├── docs/
│   └── Final_Proposal_Report_ADT.pdf
├── models/
│   ├── __init__.py
│   ├── train.py                      # Model training pipeline
│   ├── predict.py                    # Query router / predictor
│   ├── verify_bias.py                # Bias verification script
│   ├── model_sqlite.joblib           # Trained SQLite cost model
│   ├── model_duckdb.joblib           # Trained DuckDB cost model
│   └── model_metadata.json           # Model info & feature list
├── results/
│   ├── figures/
│   │   ├── runtime_comparison.png
│   │   ├── model_accuracy_comparison.png
│   │   ├── baseline_comparison.png
│   │   ├── feature_importance.png
│   │   └── prediction_error_distribution.png
│   └── metrics/
│       └── evaluation_results.json
├── scripts/
│   └── pipeline.py                   # Data generation & benchmarking
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/Rinil-Parmar/cross-engine-learned-cost-model.git
cd cross-engine-learned-cost-model
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
```

### Step 1: Generate Dataset

```bash
cd scripts
python pipeline.py
```

Generates TPC-H data (SF=0.5), benchmarks 3,100+ queries on both engines, extracts features, saves train/val/test splits.


### Step 2: Train Models

```bash
python -m models.train
```

Trains 4 model types (Linear, Ridge, Random Forest, Gradient Boosting) × 2 engines, picks the best, saves it.


### Step 3: Predict

```bash
# Interactive mode
python -m models.predict

# Single query
python -m models.predict --query "SELECT * FROM lineitem WHERE l_quantity > 30"

# Batch from file
python -m models.predict --file queries.sql

# JSON output
python -m models.predict --query "SELECT * FROM lineitem" --json
```

---

## 🧠 How It Works

### Training Flow

```
train_dataset.csv  → Train 4 model types (fit)
val_dataset.csv    → Evaluate all 4, pick best model
test_dataset.csv   → Final accuracy (touched ONCE)
train + val        → Retrain best model, save .joblib
```

### Prediction Flow

```
Input SQL → Extract 25 features
         → model_sqlite.joblib → predicted SQLite time
         → model_duckdb.joblib → predicted DuckDB time
         → Pick minimum → Best Engine
```

---

## 📊 25 Features Extracted

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `num_joins` | int | Number of JOIN operations |
| 2 | `has_subquery` | 0/1 | Contains subquery |
| 3 | `num_conditions` | int | WHERE conditions count |
| 4 | `has_groupby` | 0/1 | Contains GROUP BY |
| 5 | `has_orderby` | 0/1 | Contains ORDER BY |
| 6 | `has_having` | 0/1 | Contains HAVING |
| 7 | `has_limit` | 0/1 | Contains LIMIT |
| 8 | `has_distinct` | 0/1 | Contains DISTINCT |
| 9 | `has_like` | 0/1 | Contains LIKE |
| 10 | `has_exists` | 0/1 | Contains EXISTS |
| 11 | `has_case` | 0/1 | Contains CASE/WHEN |
| 12 | `num_aggregations` | int | SUM/AVG/COUNT/MIN/MAX count |
| 13 | `num_tables` | int | Tables referenced |
| 14 | `query_length` | int | Character count |
| 15 | `num_select_cols` | int | Columns in SELECT |
| 16 | `has_between` | 0/1 | Contains BETWEEN |
| 17 | `has_in` | 0/1 | Contains IN clause |
| 18 | `has_left_join` | 0/1 | Contains LEFT JOIN |
| 19 | `join_complexity` | int | joins × conditions |
| 20 | `num_tokens` | int | Token count |
| 21 | `num_string_literals` | int | String literal count |
| 22 | `num_numeric_literals` | int | Numeric literal count |
| 23 | `nesting_depth` | int | Max subquery depth |
| 24 | `has_string_func` | 0/1 | SUBSTR/TRIM/UPPER etc. |
| 25 | `has_arithmetic` | 0/1 | +−×÷ in SELECT |

---

## 📈 Results

### Engine Selection Accuracy

| Strategy | Accuracy |
|----------|----------|
| Always SQLite | 16% |
| Always DuckDB | 84% |
| Heuristic (joins > 2) | ~65% |
| **Model (Learned)** | **See evaluation_results.json** |
| Oracle (Perfect) | 100% |

### Key Findings

- DuckDB is faster on **86%** of TPC-H analytical queries
- SQLite wins on **simple single-table scans** with filters/LIMIT
- The learned model correctly identifies **when SQLite is the better choice**

---

## 📂 Dataset

Available on Hugging Face:
[Rinil-Parmar/tpch-query-routing-dataset](https://huggingface.co/datasets/Rinil-Parmar/tpch-query-routing-dataset)

| Split | Size | Purpose |
|-------|------|---------|
| Train | 70% | Model training |
| Validation | 15% | Model selection |
| Test | 15% | Final evaluation (one-time) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| DuckDB | Columnar engine + TPC-H data generator |
| SQLite | Row-based engine |
| scikit-learn | Model training & evaluation |
| pandas / numpy | Data processing |
| matplotlib / seaborn | Visualization |
| joblib | Model serialization |

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch (`git checkout -b dev-yourname`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push (`git push origin dev-yourname`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
