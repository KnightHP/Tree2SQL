# Tree2SQL

Tree2SQL converts trained scikit-learn decision trees into native SQL `CASE` expressions so predictions can run inside DuckDB without calling Python row by row.

In practice, that means a model that would normally be evaluated through a Python UDF can instead be embedded directly into SQL and executed by DuckDB’s vectorized engine. The result is the same prediction logic, but with much faster inference and no Python overhead at query time.

## What this project does

Tree2SQL takes a fitted decision tree model and translates each root-to-leaf path into nested SQL `CASE WHEN ... THEN ... END` logic. The generated SQL can then be:

- printed and inspected as a standalone expression
- embedded directly into larger SQL queries
- rewritten automatically when using helper methods in the package
- benchmarked against Python UDF execution in DuckDB

Because decision trees are rule-based and branch cleanly on feature thresholds, they map naturally to SQL conditionals. That makes Tree2SQL especially useful for database-native machine learning inference, analytics pipelines, and performance-sensitive batch scoring.

## Why use Tree2SQL?

Traditional Python-based inference in DuckDB can be slow because each row may require a Python call, data serialization, and GIL contention. Tree2SQL avoids that by keeping inference entirely in SQL.

### Key benefits

- **Faster inference**: often dramatically faster than Python UDFs
- **Native DuckDB execution**: runs inside the SQL engine
- **Exact logic preservation**: generated SQL mirrors the trained tree structure
- **Easy integration**: works with normal SQL queries and DataFrames
- **Transparent output**: generated `CASE` expressions are readable and debuggable

## Features

- Convert `DecisionTreeClassifier` models to SQL
- Convert `DecisionTreeRegressor` models to SQL
- Support binary and multi-class classification
- Generate nested `CASE` expressions for any tree depth
- Embed predictions directly into `SELECT` statements
- Rewrite `predict_*()` calls into SQL automatically
- Benchmark SQL execution against Python UDF execution
- Validate correctness between Python and SQL predictions
- Handle quoted column names and special characters safely

## Project highlights

The repository includes validated demos showing that Tree2SQL can achieve large speedups while maintaining prediction fidelity.

Examples from the included benchmark results:

- Stroke prediction classification: 100% agreement between SQL and Python predictions
- California housing regression: numerically identical results with negligible floating-point differences
- Speedups ranging from tens to hundreds of times faster depending on model depth and workload

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

## Quick start

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

## Main components

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

## Repository structure

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

## Example use cases

- scoring predictions directly in SQL warehouses or embedded analytics engines
- replacing slow Python UDF-based inference paths
- explaining and auditing tree-based model logic
- running batch predictions during reporting or ETL jobs
- benchmarking native SQL inference vs Python execution

## Testing

Run the test suite with:

```bash
pytest tests/ -v
```

## Notebooks

The repository includes notebooks for interactive exploration and benchmarking, such as:

- a step-by-step demo notebook
- depth sweep benchmark analysis
- stroke classification demo
- California housing regression demo

## Supported models

Tree2SQL currently supports:

- `DecisionTreeClassifier` for binary classification
- `DecisionTreeClassifier` for multi-class classification
- `DecisionTreeRegressor`

## Safety and correctness

Tree2SQL generates SQL carefully to preserve model behavior and reduce risk:

- feature names are validated and quoted
- threshold values are represented exactly
- generated SQL avoids user-controlled string interpolation
- SQL and Python outputs can be checked for parity with tests and benchmarks

## Future improvements

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
