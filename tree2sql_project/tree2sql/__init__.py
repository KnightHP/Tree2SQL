"""
tree2sql - Convert scikit-learn decision trees to native SQL CASE statements.

Enables 50-200x faster inference directly in DuckDB by replacing Python UDF
calls with vectorized SQL execution.

Example usage:
    from tree2sql import TreeToSQL, Database
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    converter = TreeToSQL(clf, feature_names=X.columns.tolist(), model_name="bank_model")
    sql_expr = converter.to_sql()

    db = Database("tree2sql.duckdb")
    db.register_model("bank_model", converter)
    result = db.execute("SELECT predict_class(bank_model, *) FROM bank_data")
"""

from .converter import TreeToSQL
from .database import Database
from .parser import QueryRewriter
from .benchmark import Benchmark

__version__ = "1.0.0"
__all__ = ["TreeToSQL", "Database", "QueryRewriter", "Benchmark"]
