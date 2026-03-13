# -*- coding: utf-8 -*-
"""
Cross-Engine SQL Query Router — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.predict import load_models, predict_best_engine, FEATURE_COLS

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Cross-Engine SQL Query Router",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        padding: 10px 0 0 0;
    }
    .sub-header {
        text-align: center;
        opacity: 0.7;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 18px 24px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .sqlite-win {
        border: 2px solid #42A5F5;
        background: rgba(66,165,245,0.12);
    }
    .duckdb-win {
        border: 2px solid #66BB6A;
        background: rgba(102,187,106,0.12);
    }
    .live-box {
        border: 2px solid #FFA726;
        background: rgba(255,167,38,0.12);
        padding: 18px 24px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stat-box {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        margin-bottom: 8px;
    }
    .stat-label {
        font-size: 0.8rem;
        opacity: 0.6;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 1.3rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.markdown('<p class="main-header">🔄 Cross-Engine SQL Query Router</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict the fastest SQL engine (SQLite vs DuckDB) — without executing the query</p>', unsafe_allow_html=True)

# ============================================================
# Load Models (cached)
# ============================================================
@st.cache_resource
def get_models():
    return load_models()

try:
    model_s, model_d, metadata = get_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"❌ Failed to load models: {e}")
    st.info("Run `python -m models.train` first.")
    st.stop()

# ============================================================
# DB Path Resolution
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# SQLITE_CANDIDATES = [
#     os.path.join(PROJECT_ROOT, "data", "tpch.db"),
#     os.path.join(PROJECT_ROOT, "scripts", "tpch.sqlite"),
#     os.path.join(PROJECT_ROOT, "data", "tpch.sqlite"),
#     os.path.join(PROJECT_ROOT, "tpch.db"),
# ]
# DUCKDB_CANDIDATES = [
#     os.path.join(PROJECT_ROOT, "data", "tpch.duckdb"),
#     os.path.join(PROJECT_ROOT, "scripts", "tpch.duckdb"),
#     os.path.join(PROJECT_ROOT, "tpch.duckdb"),
# ]

# SQLITE_DB = next((p for p in SQLITE_CANDIDATES if os.path.exists(p)), None)
# DUCKDB_DB = next((p for p in DUCKDB_CANDIDATES if os.path.exists(p)), None)
SQLITE_DB = os.path.join(PROJECT_ROOT, "scripts", "tpch.sqlite")
DUCKDB_DB = os.path.join(PROJECT_ROOT, "scripts", "tpch.duckdb")

# ============================================================
# Live Test Functions
# ============================================================
def run_on_sqlite(sql: str) -> dict:
    if SQLITE_DB is None:
        return {"time_sec": None, "rows": None,
                "error": f"SQLite DB not found. Searched: {SQLITE_CANDIDATES}"}
    try:
        conn = sqlite3.connect(SQLITE_DB)
        start = time.perf_counter()
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.perf_counter() - start
        conn.close()
        return {"time_sec": round(elapsed, 6), "rows": len(rows), "error": None}
    except Exception as e:
        return {"time_sec": None, "rows": None, "error": str(e)}


def run_on_duckdb(sql: str) -> dict:
    if DUCKDB_DB is None:
        return {"time_sec": None, "rows": None,
                "error": f"DuckDB file not found. Searched: {DUCKDB_CANDIDATES}"}
    try:
        import duckdb
        conn = duckdb.connect(DUCKDB_DB, read_only=True)
        start = time.perf_counter()
        result = conn.execute(sql).fetchall()
        elapsed = time.perf_counter() - start
        conn.close()
        return {"time_sec": round(elapsed, 6), "rows": len(result), "error": None}
    except Exception as e:
        return {"time_sec": None, "rows": None, "error": str(e)}


def run_live_test(sql: str) -> dict:
    sq = run_on_sqlite(sql)
    dk = run_on_duckdb(sql)
    actual_best = None
    if sq["time_sec"] is not None and dk["time_sec"] is not None:
        actual_best = "sqlite" if sq["time_sec"] <= dk["time_sec"] else "duckdb"
    return {"sqlite": sq, "duckdb": dk, "actual_best": actual_best}

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        "Enter a SQL query → the model extracts **25 features** "
        "→ predicts runtime on SQLite & DuckDB → recommends the faster engine."
    )
    st.markdown("**No database connection needed.** Predictions come from query structure alone.")

    st.divider()

    st.markdown("### 📊 Model Info")
    model_name   = metadata.get("best_model", "N/A")
    train_samples = metadata.get("training_samples", "N/A")
    test_acc     = metadata.get("test_accuracy", None)

    st.markdown(f"- **Model:** {model_name}")
    st.markdown(f"- **Training Samples:** {train_samples}")
    if test_acc is not None:
        st.markdown(f"- **Test Accuracy:** {test_acc:.1%}")
    st.markdown(f"- **Features:** {len(FEATURE_COLS)}")

    st.divider()

    st.markdown("### 🗄️ Database Status")
    if SQLITE_DB:
        st.success(f"✅ SQLite: `{os.path.basename(SQLITE_DB)}`")
    else:
        st.error("❌ SQLite DB not found")
    if DUCKDB_DB:
        st.success(f"✅ DuckDB: `{os.path.basename(DUCKDB_DB)}`")
    else:
        st.error("❌ DuckDB file not found")

    st.divider()

    st.markdown("### 🧪 Examples")
    examples = {
        "(none)": "",
        "Simple scan → SQLite":     "SELECT o_orderkey, o_totalprice FROM orders WHERE o_totalprice > 5000 LIMIT 10",
        "Count filter → SQLite":    "SELECT COUNT(*) FROM customer WHERE c_mktsegment = 'BUILDING'",
        "Aggregation → DuckDB":     "SELECT l_returnflag, l_linestatus, SUM(l_quantity), AVG(l_extendedprice) FROM lineitem WHERE l_shipdate <= '1998-09-01' GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag",
        "Multi-join → DuckDB":      "SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount)) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey GROUP BY n.n_name ORDER BY 2 DESC",
        "Subquery + EXISTS → DuckDB":"SELECT s.s_name FROM supplier s JOIN nation n ON s.s_nationkey = n.n_nationkey WHERE n.n_name = 'CANADA' AND EXISTS (SELECT 1 FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey WHERE l.l_suppkey = s.s_suppkey)",
    }
    selected = st.selectbox("Load example:", list(examples.keys()), label_visibility="collapsed")

# ============================================================
# Query Input
# ============================================================
st.markdown("### ✍️ Enter SQL Query")

default_query = examples.get(selected, "")
sql_input = st.text_area(
    "query_input",
    value=default_query,
    height=140,
    placeholder="SELECT * FROM lineitem WHERE l_quantity > 30",
    label_visibility="collapsed",
)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    predict_clicked = st.button("🚀 Predict Best Engine", type="primary", use_container_width=True)
with btn_col2:
    live_clicked = st.button(
        "⚡ Live Test on Databases",
        type="secondary",
        use_container_width=True,
        disabled=(SQLITE_DB is None and DUCKDB_DB is None),
    )

# ============================================================
# Session State — store results independently
# ============================================================
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None
if "live_result" not in st.session_state:
    st.session_state.live_result = None
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""

# Handle predict button
if predict_clicked:
    if not sql_input.strip():
        st.warning("⚠️ Enter a SQL query first.")
    else:
        st.session_state.pred_result = predict_best_engine(sql_input.strip(), model_s, model_d)
        st.session_state.last_sql    = sql_input.strip()
        st.session_state.live_result = None   # clear old live results

# Handle live test button
if live_clicked:
    if not sql_input.strip():
        st.warning("⚠️ Enter a SQL query first.")
    else:
        # Always (re)run prediction so comparison table works
        st.session_state.pred_result = predict_best_engine(sql_input.strip(), model_s, model_d)
        st.session_state.last_sql    = sql_input.strip()
        with st.spinner("⚡ Executing query on SQLite and DuckDB — please wait..."):
            st.session_state.live_result = run_live_test(sql_input.strip())

# ============================================================
# SECTION 1 — ML Prediction Output
# ============================================================
if st.session_state.pred_result is not None:
    result = st.session_state.pred_result

    if "error" in result:
        st.error(f"❌ Prediction error: {result['error']}")
    else:
        best       = result["best_engine"]
        pred_s     = result["predicted_sqlite_sec"]
        pred_d     = result["predicted_duckdb_sec"]
        confidence = result["confidence_ratio"]
        strength   = result["decision_strength"]
        features   = result["features"]

        st.markdown("---")
        st.markdown("## 🤖 ML Prediction")

        if best == "sqlite":
            st.markdown(
                f'<div class="result-box sqlite-win">'
                f'🟦 Recommended: <strong>SQLite</strong> — '
                f'predicted {pred_s:.6f}s vs DuckDB {pred_d:.6f}s — {confidence:.2f}x faster'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-box duckdb-win">'
                f'🟩 Recommended: <strong>DuckDB</strong> — '
                f'predicted {pred_d:.6f}s vs SQLite {pred_s:.6f}s — {confidence:.2f}x faster'
                f'</div>',
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in [
            (c1, "Predicted SQLite", f"{pred_s:.6f}s"),
            (c2, "Predicted DuckDB", f"{pred_d:.6f}s"),
            (c3, "Confidence",       f"{confidence:.2f}x"),
            (c4, "Strength",         strength.capitalize()),
        ]:
            col.markdown(
                f'<div class="stat-box">'
                f'<div class="stat-label">{label}</div>'
                f'<div class="stat-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # --- Feature Analysis ---
        st.markdown("---")
        st.markdown("### 🔍 Feature Analysis")

        active_features   = [f for f in FEATURE_COLS if features[f] != 0]
        inactive_features = [f for f in FEATURE_COLS if features[f] == 0]
        st.markdown(f"**{len(active_features)}/{len(FEATURE_COLS)}** features active in this query.")

        descriptions = {
            "num_joins":            "Number of JOIN operations",
            "has_subquery":         "Contains nested SELECT",
            "num_conditions":       "WHERE conditions count",
            "has_groupby":          "Uses GROUP BY",
            "has_orderby":          "Uses ORDER BY",
            "has_having":           "Uses HAVING",
            "has_limit":            "Uses LIMIT",
            "has_distinct":         "Uses DISTINCT",
            "has_like":             "Uses LIKE pattern",
            "has_exists":           "Uses EXISTS",
            "has_case":             "Uses CASE/WHEN",
            "num_aggregations":     "SUM/AVG/COUNT/MIN/MAX count",
            "num_tables":           "Tables referenced",
            "query_length":         "Character count",
            "num_select_cols":      "SELECT column count",
            "has_between":          "Uses BETWEEN",
            "has_in":               "Uses IN (...)",
            "has_left_join":        "Uses LEFT JOIN",
            "join_complexity":      "Joins × conditions",
            "num_tokens":           "Word count",
            "num_string_literals":  "String literal count",
            "num_numeric_literals": "Numeric constant count",
            "nesting_depth":        "Max nesting depth",
            "has_string_func":      "Uses SUBSTR/TRIM/UPPER etc.",
            "has_arithmetic":       "Has +−×÷ in SELECT",
        }

        tab1, tab2, tab3 = st.tabs(["📋 Features", "📊 Chart", "🧠 How It Works"])

        with tab1:
            if active_features:
                st.markdown("**✅ Active Features**")
                st.dataframe(
                    pd.DataFrame([
                        {"Feature": f, "Value": features[f], "Description": descriptions.get(f, "")}
                        for f in active_features
                    ]),
                    use_container_width=True, hide_index=True,
                )
            if inactive_features:
                with st.expander(f"⬜ Inactive Features ({len(inactive_features)})"):
                    st.dataframe(
                        pd.DataFrame([
                            {"Feature": f, "Value": 0, "Description": descriptions.get(f, "")}
                            for f in inactive_features
                        ]),
                        use_container_width=True, hide_index=True,
                    )

        with tab2:
            if active_features:
                chart_df = pd.DataFrame({
                    "Feature": active_features,
                    "Value":   [features[f] for f in active_features],
                }).sort_values("Value", ascending=True)
                st.bar_chart(chart_df.set_index("Feature"), horizontal=True, use_container_width=True)
            else:
                st.info("No active features to chart.")

        with tab3:
            st.markdown("""
**Step 1 — Feature Extraction**
Your SQL query is parsed to extract 25 structural features: join count, subqueries, aggregations, nesting depth, etc.

**Step 2 — Dual Prediction**
Two ML models run on the same feature vector:
- `model_sqlite.joblib` → predicted SQLite time
- `model_duckdb.joblib` → predicted DuckDB time

**Step 3 — Engine Selection**
The engine with the lower predicted time wins. Confidence = ratio of the two times.
            """)
            st.divider()
            complexity = (
                features["num_joins"] * 3
                + features["has_subquery"] * 2
                + features["num_aggregations"] * 2
                + features["has_groupby"]
                + features["has_orderby"]
                + features["nesting_depth"]
            )
            reason = (
                "High complexity → DuckDB's vectorized engine wins" if complexity >= 5
                else "Moderate complexity → DuckDB likely has an edge" if complexity >= 2
                else "Low complexity → SQLite's lightweight engine wins"
            )
            st.info(f"Complexity score: **{complexity}/15** — {reason}")

# ============================================================
# SECTION 2 — Live Test Output (completely separate section)
# ============================================================
if st.session_state.live_result is not None:
    live        = st.session_state.live_result
    sq          = live["sqlite"]
    dk          = live["duckdb"]
    actual_best = live["actual_best"]

    st.markdown("---")
    st.markdown("## ⚡ Live Database Execution Results")

    # Winner banner
    if actual_best == "sqlite":
        st.markdown(
            f'<div class="result-box sqlite-win">'
            f'🟦 Actual Winner: <strong>SQLite</strong> — '
            f'{sq["time_sec"]:.6f}s vs DuckDB {dk["time_sec"]:.6f}s'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif actual_best == "duckdb":
        st.markdown(
            f'<div class="result-box duckdb-win">'
            f'🟩 Actual Winner: <strong>DuckDB</strong> — '
            f'{dk["time_sec"]:.6f}s vs SQLite {sq["time_sec"]:.6f}s'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="live-box">⚡ Live test ran — see individual results below</div>', unsafe_allow_html=True)

    # Individual engine result cards
    lc1, lc2 = st.columns(2)

    with lc1:
        st.markdown("#### 🔵 SQLite")
        if sq["error"]:
            st.error(f"**Error:** {sq['error']}")
        else:
            st.markdown(
                f'<div class="stat-box sqlite-win">'
                f'<div class="stat-label">Actual Execution Time</div>'
                f'<div class="stat-value">{sq["time_sec"]:.6f}s</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"✅ Query succeeded  |  Rows returned: **{sq['rows']}**")

    with lc2:
        st.markdown("#### 🟢 DuckDB")
        if dk["error"]:
            st.error(f"**Error:** {dk['error']}")
        else:
            st.markdown(
                f'<div class="stat-box duckdb-win">'
                f'<div class="stat-label">Actual Execution Time</div>'
                f'<div class="stat-value">{dk["time_sec"]:.6f}s</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"✅ Query succeeded  |  Rows returned: **{dk['rows']}**")

    # Prediction vs Reality comparison table
    if st.session_state.pred_result is not None and "error" not in st.session_state.pred_result:
        pred    = st.session_state.pred_result
        pred_s  = pred["predicted_sqlite_sec"]
        pred_d  = pred["predicted_duckdb_sec"]
        best_ml = pred["best_engine"]

        st.markdown("---")
        st.markdown("### 🎯 Prediction vs Reality")

        if actual_best:
            if actual_best == best_ml:
                st.success(f"✅ Prediction **CORRECT** — ML predicted and actual both agree: **{actual_best.upper()}** is faster")
            else:
                st.warning(
                    f"⚠️ Prediction **INCORRECT** — ML predicted **{best_ml.upper()}** "
                    f"but actual winner is **{actual_best.upper()}**"
                )

        rows = []
        if sq["time_sec"] is not None:
            rows.append({
                "Engine":         "🔵 SQLite",
                "ML Predicted (s)": f"{pred_s:.6f}",
                "Actual (s)":       f"{sq['time_sec']:.6f}",
                "Absolute Error (s)": f"{abs(pred_s - sq['time_sec']):.6f}",
                "Rows Returned":    sq["rows"],
            })
        if dk["time_sec"] is not None:
            rows.append({
                "Engine":         "🟢 DuckDB",
                "ML Predicted (s)": f"{pred_d:.6f}",
                "Actual (s)":       f"{dk['time_sec']:.6f}",
                "Absolute Error (s)": f"{abs(pred_d - dk['time_sec']):.6f}",
                "Rows Returned":    dk["rows"],
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if sq["time_sec"] and dk["time_sec"] and dk["time_sec"] > 0:
            ratio    = sq["time_sec"] / dk["time_sec"]
            faster   = "SQLite" if ratio < 1 else "DuckDB"
            faster_by = (1 / ratio) if ratio < 1 else ratio
            st.info(f"🏁 **{faster}** was **{faster_by:.2f}x faster** in actual execution")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Cross-Engine Learned Cost Model • TPC-H Benchmark • "
    "[GitHub](https://github.com/Rinil-Parmar/cross-engine-learned-cost-model)"
)