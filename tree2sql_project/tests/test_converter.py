"""
test_converter.py - Unit tests for Tree2SQL.

Run with:  pytest tests/test_converter.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tree2sql.converter import TreeToSQL
from tree2sql.parser import QueryRewriter
from tree2sql.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_clf():
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    feature_names = [f"feat_{i}" for i in range(5)]
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    return clf, feature_names, X


@pytest.fixture
def regression_model():
    X, y = make_regression(n_samples=500, n_features=4, random_state=42)
    feature_names = [f"col_{i}" for i in range(4)]
    reg = DecisionTreeRegressor(max_depth=4, random_state=42)
    reg.fit(X, y)
    return reg, feature_names, X


@pytest.fixture
def multiclass_clf():
    X, y = make_classification(
        n_samples=600, n_features=6, n_classes=3, n_informative=4, random_state=7
    )
    feature_names = [f"x{i}" for i in range(6)]
    clf = DecisionTreeClassifier(max_depth=5, random_state=7)
    clf.fit(X, y)
    return clf, feature_names, X


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------

class TestTreeToSQL:
    def test_basic_sql_generation(self, binary_clf):
        clf, features, _ = binary_clf
        converter = TreeToSQL(clf, features)
        sql = converter.to_sql()
        assert "CASE WHEN" in sql
        assert "THEN" in sql
        assert "ELSE" in sql
        assert "END" in sql

    def test_sql_matches_predict_classification(self, binary_clf):
        clf, features, X = binary_clf
        converter = TreeToSQL(clf, features, model_name="test_clf")

        db = Database(":memory:")
        df = pd.DataFrame(X, columns=features)
        db.load_dataframe(df, "test_table")
        db.register_model("test_clf", converter)

        case_expr = converter.to_sql()
        sql_preds = db.execute(
            f"SELECT {case_expr} AS pred FROM test_table", rewrite=False
        ).df()["pred"].tolist()

        py_preds = [str(p) for p in clf.predict(X)]
        sql_preds_str = [str(p) for p in sql_preds]

        assert sql_preds_str == py_preds, "SQL predictions must match model.predict()"
        db.close()

    def test_sql_matches_predict_regression(self, regression_model):
        reg, features, X = regression_model
        converter = TreeToSQL(reg, features, model_name="test_reg")

        db = Database(":memory:")
        df = pd.DataFrame(X, columns=features)
        db.load_dataframe(df, "reg_table")

        case_expr = converter.to_sql()
        sql_preds = db.execute(
            f"SELECT {case_expr} AS pred FROM reg_table", rewrite=False
        ).df()["pred"].tolist()

        py_preds = reg.predict(X).tolist()

        for sql_val, py_val in zip(sql_preds, py_preds):
            assert abs(float(sql_val) - float(py_val)) < 1e-6, (
                f"Regression mismatch: SQL={sql_val}, Python={py_val}"
            )
        db.close()

    def test_multiclass_classification(self, multiclass_clf):
        clf, features, X = multiclass_clf
        converter = TreeToSQL(clf, features, model_name="multiclass")

        db = Database(":memory:")
        df = pd.DataFrame(X, columns=features)
        db.load_dataframe(df, "mc_table")

        case_expr = converter.to_sql()
        sql_preds = db.execute(
            f"SELECT {case_expr} AS pred FROM mc_table", rewrite=False
        ).df()["pred"].tolist()

        py_preds = [str(p) for p in clf.predict(X)]
        assert [str(p) for p in sql_preds] == py_preds
        db.close()

    def test_single_node_tree(self):
        """A tree with max_depth=1 is a stump — should still produce a valid CASE expression."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        clf = DecisionTreeClassifier(max_depth=1, random_state=0)
        clf.fit(X, y)
        converter = TreeToSQL(clf, [f"f{i}" for i in range(4)])
        sql = converter.to_sql()
        # Must still be valid SQL with CASE or a leaf literal
        assert sql != ""

    def test_special_character_feature_names(self):
        """Column names with spaces must be double-quoted."""
        X, y = make_classification(
            n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0
        )
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        converter = TreeToSQL(clf, ["col a", "col-b"])
        sql = converter.to_sql()
        assert '"col a"' in sql
        assert '"col-b"' in sql

    def test_wrong_feature_count_raises(self):
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        with pytest.raises(ValueError, match="Expected 4 feature names"):
            TreeToSQL(clf, ["only_two", "features"])

    def test_unfitted_model_raises(self):
        clf = DecisionTreeClassifier()
        with pytest.raises(ValueError, match="fitted"):
            TreeToSQL(clf, ["f0", "f1"])

    def test_tree_stats(self, binary_clf):
        clf, features, _ = binary_clf
        converter = TreeToSQL(clf, features)
        stats = converter.tree_stats()
        assert stats["is_classifier"] is True
        assert stats["max_depth"] == clf.get_depth()
        assert stats["n_leaves"] == clf.get_n_leaves()


# ---------------------------------------------------------------------------
# Deep tree test
# ---------------------------------------------------------------------------

class TestDeepTree:
    def test_deep_tree_correctness(self):
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        features = [f"f{i}" for i in range(10)]
        clf = DecisionTreeClassifier(max_depth=15, random_state=42)
        clf.fit(X, y)
        converter = TreeToSQL(clf, features, model_name="deep")

        db = Database(":memory:")
        df = pd.DataFrame(X, columns=features)
        db.load_dataframe(df, "deep_table")

        case_expr = converter.to_sql()
        sql_preds = db.execute(
            f"SELECT {case_expr} AS pred FROM deep_table", rewrite=False
        ).df()["pred"].tolist()

        py_preds = [str(p) for p in clf.predict(X)]
        assert [str(p) for p in sql_preds] == py_preds
        db.close()


# ---------------------------------------------------------------------------
# Parser / rewriter tests
# ---------------------------------------------------------------------------

class TestQueryRewriter:
    def test_simple_rewrite(self):
        X, y = make_classification(
            n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=0
        )
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        converter = TreeToSQL(clf, ["col1", "col2"], model_name="mymodel")
        rewriter = QueryRewriter({"mymodel": converter})

        sql = "SELECT predict_mymodel(col1, col2) FROM t"
        rewritten = rewriter.rewrite(sql)
        assert "predict_mymodel" not in rewritten
        assert "CASE WHEN" in rewritten

    def test_unknown_model_raises(self):
        rewriter = QueryRewriter({})
        with pytest.raises(KeyError, match="ghost"):
            rewriter.rewrite("SELECT predict_ghost(x) FROM t")

    def test_has_predict_calls(self):
        rewriter = QueryRewriter({})
        assert rewriter.has_predict_calls("SELECT predict_foo(x) FROM t")
        assert not rewriter.has_predict_calls("SELECT x FROM t")

    def test_list_predict_calls(self):
        rewriter = QueryRewriter({})
        calls = rewriter.list_predict_calls("SELECT predict_model1(a,b), predict_model2(c) FROM t")
        names = [c[0] for c in calls]
        assert "model1" in names
        assert "model2" in names


# ---------------------------------------------------------------------------
# Database integration tests
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_load_and_query(self):
        db = Database(":memory:")
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        db.load_dataframe(df, "nums")
        result = db.query_df("SELECT count(*) AS n FROM nums", rewrite=False)
        assert result["n"].iloc[0] == 3
        db.close()

    def test_register_and_predict(self):
        X, y = make_classification(n_samples=200, n_features=3,
                                   n_informative=2, n_redundant=1, random_state=1)
        features = ["a", "b", "c"]
        clf = DecisionTreeClassifier(max_depth=3, random_state=1)
        clf.fit(X, y)

        db = Database(":memory:")
        df = pd.DataFrame(X, columns=features)
        db.load_dataframe(df, "data")

        converter = TreeToSQL(clf, features, model_name="m")
        db.register_model("m", converter)

        result = db.query_df("SELECT predict_m(a, b, c) AS p FROM data")
        assert len(result) == 200
        db.close()

    def test_table_exists(self):
        db = Database(":memory:")
        df = pd.DataFrame({"x": [1, 2]})
        db.load_dataframe(df, "mytable")
        assert db.table_exists("mytable")
        assert not db.table_exists("nonexistent")
        db.close()
