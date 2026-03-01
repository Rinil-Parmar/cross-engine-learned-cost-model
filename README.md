# 🔄 Cross-Engine Learned Cost Model

A cross-engine learned cost modeling system for **zero-shot SQL engine selection** using TPC-H workloads and lightweight machine learning models.

> **Goal:** Given a SQL query, automatically predict whether **SQLite** or **DuckDB** will execute it faster — without running it on both engines.

---

## 📌 Overview

This project builds a machine learning pipeline that:

1. **Generates** 1,100+ SQL query variants from all 22 TPC-H benchmark queries
2. **Benchmarks** each query on both SQLite and DuckDB engines
3. **Extracts** 25 structural features from each query
4. **Trains** a lightweight ML model to predict the faster engine
5. **Enables** zero-shot engine selection for unseen queries

---

## 🏗️ Project Structure

```
cross-engine-learned-cost-model/
├── data/
│   ├── test_dataset.csv              # Test split (15%)
│   ├── train_dataset.csv             # Training split (70%)
│   └── val_dataset.csv               # Validation split (15%)
├── docs/
│   └── Final_Proposal_Report_ADT.pdf # Project proposal report
├── models/                           # Trained ML models
├── notebooks/                        # Jupyter notebooks for exploration
├── results/
│   ├── figures/                      # Visualizations & plots
│   └── metrics/                      # Model evaluation metrics
├── scripts/
│   ├── pipeline.py                   # Data generation & benchmarking pipeline
│   └── upload_dataset_to_hf.py       # Upload dataset to Hugging Face
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Rinil-Parmar/cross-engine-learned-cost-model.git
cd cross-engine-learned-cost-model

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
cd scripts
python pipeline.py
```

This will:
- Generate TPC-H data (SF=0.5) using DuckDB
- Load data into both SQLite and DuckDB
- Generate 1,100 query variants with random parameters
- Benchmark each query on both engines
- Extract 25 features per query
- Save train/val/test splits to `data/`

---

## 📊 Features Extracted (25)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `num_joins` | Number of JOIN operations |
| 2 | `has_subquery` | Contains subquery (0/1) |
| 3 | `num_conditions` | Number of WHERE conditions |
| 4 | `has_groupby` | Contains GROUP BY (0/1) |
| 5 | `has_orderby` | Contains ORDER BY (0/1) |
| 6 | `has_having` | Contains HAVING (0/1) |
| 7 | `has_limit` | Contains LIMIT (0/1) |
| 8 | `has_distinct` | Contains DISTINCT (0/1) |
| 9 | `has_like` | Contains LIKE (0/1) |
| 10 | `has_exists` | Contains EXISTS (0/1) |
| 11 | `has_case` | Contains CASE/WHEN (0/1) |
| 12 | `num_aggregations` | Count of SUM/AVG/COUNT/MIN/MAX |
| 13 | `num_tables` | Number of tables referenced |
| 14 | `query_length` | Character count of query |
| 15 | `num_select_cols` | Columns in SELECT clause |
| 16 | `has_between` | Contains BETWEEN (0/1) |
| 17 | `has_in` | Contains IN clause (0/1) |
| 18 | `has_left_join` | Contains LEFT JOIN (0/1) |
| 19 | `join_complexity` | joins × conditions |
| 20 | `num_tokens` | Token count |
| 21 | `num_string_literals` | Count of string literals |
| 22 | `num_numeric_literals` | Count of numeric literals |
| 23 | `nesting_depth` | Subquery nesting depth |
| 24 | `has_string_func` | Contains string functions (0/1) |
| 25 | `has_arithmetic` | Arithmetic in SELECT (0/1) |

---

## 📂 Dataset

The generated dataset is available on Hugging Face:
[Rinil-Parmar/tpch-query-routing-dataset](https://huggingface.co/datasets/Rinil-Parmar/tpch-query-routing-dataset)

### Dataset Splits

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

---

## 🛠️ Tech Stack

- **Python** — Core language
- **DuckDB** — Columnar analytical database + TPC-H data generator
- **SQLite** — Row-based relational database
- **scikit-learn** — ML model training and evaluation
- **pandas** — Data manipulation
- **matplotlib** — Visualization

---

## 📈 Results

![Dataset Summary](https://github.com/Rinil-Parmar/cross-engine-learned-cost-model/blob/main/results/figures/dataset_summary.png)

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b dev-yourname`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin dev-yourname`)
5. Open a Pull Request to `develop`

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](https://github.com/Rinil-Parmar/cross-engine-learned-cost-model/blob/main/LICENSE) file for details.
