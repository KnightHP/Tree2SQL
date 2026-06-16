# Tree2SQL

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-SQL%20native%20ML-0f62fe.svg)](https://duckdb.org/)
[![License](https://img.shields.io/badge/License-Unlicensed-lightgrey.svg)](#license)

Tree2SQL converts trained scikit-learn decision trees into native SQL `CASE` expressions so predictions can run inside DuckDB without row-by-row Python execution.

Instead of relying on a Python UDF for every record, Tree2SQL compiles the tree logic into SQL that DuckDB can execute directly using its vectorized engine. The result is the same model behavior with dramatically less overhead and much faster inference for analytical workloads.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Why Tree2SQL](#why-tree2sql)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Main Components](#main-components)
- [Repository Structure](#repository-structure)
- [Supported Models](#supported-models)
- [Datasets and Demos](#datasets-and-demos)
- [Performance Highlights](#performance-highlights)
- [Safety and Correctness](#safety-and-correctness)
- [Testing](#testing)
- [Use Cases](#use-cases)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

Tree2SQL is a small but practical Python package for database-native machine learning inference. It focuses on one idea: decision trees can be translated exactly into SQL conditionals.

That makes it possible to:

- train a tree in scikit-learn
- convert it into SQL `CASE` logic
- run predictions inside DuckDB
- compare SQL inference to Python UDF inference
- benchmark performance and correctness

This is especially useful when you want fast batch scoring, explainable tree logic, and a workflow that stays inside SQL instead of switching back and forth between Python and the database.

## How It Works

A decision tree splits on feature thresholds until it reaches a leaf. Tree2SQL walks the tree recursively and converts each branch into nested SQL logic.

A simplified flow looks like this:

1. Fit a scikit-learn decision tree.
2. Read the tree structure and feature names.
3. Convert each root-to-leaf path into SQL `WHEN` conditions.
4. Emit a full `CASE WHEN ... THEN ... END` expression.
5. Use that SQL expression directly in DuckDB queries.

Because the model is translated rather than approximated, the generated SQL is intended to match the original tree’s predictions exactly.

## Why Tree2SQL

Traditional Python-based inference in DuckDB can be slow because each row may trigger Python execution, serialization overhead, and GIL contention. Tree2SQL avoids that by keeping the prediction logic in SQL.

### Key advantages

- **Faster inference** — avoids Python row-by-row execution
- **Native DuckDB execution** — uses the database engine directly
- **Exact logic preservation** — mirrors the trained tree structure
- **Readable output** — generated SQL can be inspected and audited
- **Easy integration** — works with SQL queries, DataFrames, and notebooks

## Features

- Convert `DecisionTreeClassifier` models to SQL
- Convert `DecisionTreeRegressor` models to SQL
- Support binary and multi-class classification
- Generate nested `CASE` expressions for any tree depth
- Embed predictions directly inside `SELECT` statements
- Rewrite `predict_*()` calls automatically
- Benchmark SQL inference against Python UDF execution
- Validate correctness between Python and SQL predictions
- Safely handle quoted column names and special characters

## Installation

### Basic install

```bash
pip install -e .
```

### Install with data utilities and development tools

```bash
pip install -e ".[data,dev]"
```

## Requirements

Tree2SQL is designed for Python 3.9+ and depends on:

- DuckDB
- scikit-learn
- pandas
- numpy
- matplotlib

Optional extras include:

- `ucimlrepo` for dataset loading
- `pytest` for testing
- `jupyter` and `ipykernel` for notebooks

## Quick Start

### 1. Train a decision tree

```python
from sklearn.tree import DecisionTreeClassifier
from tree2sql import TreeToSQL

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
```

### 2. Convert it to SQL

```python
converter = TreeToSQL(
    clf,
    feature_names=X_train.columns.tolist(),
    model_name="bank_clf"
)

sql_expr = converter.to_sql()
print(sql_expr)
```

### 3. Use it inside DuckDB

```python
from tree2sql import Database

db = Database("tree2sql.duckdb")
db.load_dataframe(df, "bank_data")
db.register_model("bank_clf", converter)

results = db.query_df("""
    SELECT
        age,
        duration,
        predict_bank_clf(age, balance, duration, ...) AS subscription_pred
    FROM bank_data
    WHERE age > 30
""")
```

### 4. Embed the SQL manually

```python
sql = f"""
SELECT
    customer_id,
    {converter.to_sql()} AS churn_risk
FROM customers
WHERE last_activity_days > 30
"""

db.execute(sql, rewrite=False)
```

## Usage Examples

### Generate a SQL expression from a tree

```python
from sklearn.tree import DecisionTreeClassifier
from tree2sql import TreeToSQL

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

converter = TreeToSQL(model, feature_names=X_train.columns.tolist(), model_name="demo_model")
print(converter.to_sql())
```

### Run predictions from SQL

```python
from tree2sql import Database

db = Database()
db.load_dataframe(df, "input_table")
db.register_model("demo_model", converter)

predictions = db.query_df("""
    SELECT
        *,
        predict_demo_model(feature_1, feature_2, feature_3) AS prediction
    FROM input_table
""")
```

### Use the expression in custom SQL

```python
sql = f"""
SELECT
    id,
    {converter.to_select_expr(alias='prediction')}
FROM input_table
"""
```

## Main Components

### `TreeToSQL`

Core converter that transforms a fitted decision tree into SQL.

Common methods include:

- `to_sql()`
- `to_select_expr()`
- `wrap_in_select()`
- `tree_stats()`

### `Database`

A lightweight DuckDB wrapper for loading data, registering models, and executing rewritten SQL.

### `Benchmark`

Utility for comparing SQL-native inference against Python UDF inference and for plotting speedups.

## Repository Structure

```text
tree2sql_project/
├── tree2sql/            # Core package code
├── notebooks/           # Demo and benchmark notebooks
├── data/                # Dataset loading helpers
├── tests/               # Automated tests
├── README.md            # Project documentation
├── requirements.txt    # Runtime/dev dependencies
└── setup.py             # Package configuration
```

## Supported Models

Tree2SQL currently supports:

- `DecisionTreeClassifier` for binary classification
- `DecisionTreeClassifier` for multi-class classification
- `DecisionTreeRegressor`

It is designed to work with trees of different depths, and it can safely handle feature names that contain spaces or special characters.

## Datasets and Demos

The repository includes notebooks and data helpers for exploring the project in more depth.

### Example demos

- **Demo notebook** — step-by-step walkthrough of the workflow
- **Benchmark notebook** — performance analysis and plots
- **Stroke classification demo** — classification example with benchmark results
- **California housing demo** — regression example with benchmark results

### Dataset support

The project includes a Bank Marketing dataset loader and uses `ucimlrepo` with a synthetic fallback if the dataset cannot be fetched.

## Performance Highlights

Tree2SQL was validated on both classification and regression workloads.

Examples from the included benchmark results:

- Stroke prediction classification: 100% agreement between SQL and Python predictions
- California housing regression: numerically identical predictions with negligible floating-point differences
- Speedups ranging from tens to hundreds of times faster depending on tree depth and workload

These results show that converting decision trees to SQL can preserve model fidelity while significantly improving execution speed.

## Safety and Correctness

Tree2SQL generates SQL carefully to preserve model behavior and reduce risk:

- feature names are validated and quoted
- threshold values are represented exactly
- generated SQL avoids user-controlled string interpolation
- SQL and Python outputs can be checked for parity with tests and benchmarks

## Testing

Run the test suite with:

```bash
pytest tests/ -v
```

You can also validate the project manually by running the notebooks and comparing SQL and Python outputs on the included datasets.

## Use Cases

- scoring predictions directly in SQL warehouses or embedded analytics engines
- replacing slow Python UDF-based inference paths
- explaining and auditing tree-based model logic
- running batch predictions during reporting or ETL jobs
- benchmarking native SQL inference vs Python execution

## Future Improvements

Possible future enhancements for this project could include:

- support for more tree-based estimators
- export helpers for notebooks and reports
- richer visualization of tree-to-SQL mappings
- additional database backends beyond DuckDB
- automatic packaging and release workflows

## License

No license file is currently included in the repository. If you plan to share or reuse this project publicly, add a license file so usage rights are clear.

## Contributing

Contributions are welcome. A good contribution workflow would include:

1. Fork the repository
2. Create a feature branch
3. Add or update tests
4. Run the test suite
5. Open a pull request with a clear description

## Acknowledgements

Tree2SQL is inspired by research on co-optimizing SQL and machine learning inference, including the CactusDB paper referenced in the project materials.
