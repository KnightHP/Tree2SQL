"""
converter.py - Core Decision Tree → SQL CASE statement converter.

Recursively traverses scikit-learn tree internals and emits nested
CASE WHEN ... THEN ... ELSE ... END expressions that are semantically
identical to model.predict().
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class TreeToSQL:
    """
    Convert a trained scikit-learn DecisionTreeClassifier or
    DecisionTreeRegressor into a single SQL expression.

    Parameters
    ----------
    model : DecisionTreeClassifier | DecisionTreeRegressor
        A fitted decision tree.
    feature_names : list[str]
        Column names in the same order as the training features.
    model_name : str, optional
        Human-readable identifier used when registering with Database.
    """

    def __init__(
        self,
        model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
        feature_names: List[str],
        model_name: str = "model",
    ) -> None:
        if not hasattr(model, "tree_"):
            raise ValueError("Model must be a fitted scikit-learn decision tree.")
        if len(feature_names) != model.n_features_in_:
            raise ValueError(
                f"Expected {model.n_features_in_} feature names, "
                f"got {len(feature_names)}."
            )

        self.model = model
        self.feature_names = [self._safe_column_name(f) for f in feature_names]
        self.model_name = model_name
        self.is_classifier = isinstance(model, DecisionTreeClassifier)

        tree = model.tree_
        self._children_left = tree.children_left
        self._children_right = tree.children_right
        self._feature = tree.feature
        self._threshold = tree.threshold
        self._value = tree.value
        self._n_node_samples = tree.n_node_samples

        # TREE_UNDEFINED sentinel used by scikit-learn for leaf nodes
        self._LEAF = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_sql(self, indent: int = 0) -> str:
        """Return the full SQL CASE expression for the tree."""
        return self._node_to_sql(0, indent)

    def to_select_expr(self, alias: str = "prediction") -> str:
        """Return a SELECT-ready expression, e.g. for embedding in a query."""
        sql = self.to_sql(indent=4)
        return f"{sql} AS {alias}"

    def wrap_in_select(self, table_name: str, extra_cols: str = "*") -> str:
        """Return a complete SELECT statement that scores every row."""
        sql = self.to_sql(indent=8)
        return (
            f"SELECT\n"
            f"    {extra_cols},\n"
            f"    {sql} AS prediction\n"
            f"FROM {table_name}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_leaf(self, node_id: int) -> bool:
        return self._children_left[node_id] == self._LEAF

    def _leaf_value(self, node_id: int) -> str:
        """Return the SQL literal for a leaf node."""
        value = self._value[node_id]
        if self.is_classifier:
            class_idx = int(np.argmax(value[0]))
            label = self.model.classes_[class_idx]
            # Always return string labels as quoted SQL strings
            return f"'{str(label)}'"
        else:
            # Regression: return mean value at leaf (tree.value shape is (n_nodes, n_outputs, 1))
            return repr(float(value[0][0]))

    def _node_to_sql(self, node_id: int, indent: int) -> str:
        """Recursively build a nested CASE expression."""
        if self._is_leaf(node_id):
            return self._leaf_value(node_id)

        feature_col = self.feature_names[self._feature[node_id]]
        # Cast to plain Python float so repr() yields '0.5' not 'np.float64(0.5)'
        threshold = float(self._threshold[node_id])
        left_sql = self._node_to_sql(self._children_left[node_id], indent + 4)
        right_sql = self._node_to_sql(self._children_right[node_id], indent + 4)

        pad = " " * indent
        inner = " " * (indent + 4)
        return (
            f"CASE WHEN {feature_col} <= {threshold!r}\n"
            f"{inner}THEN {left_sql}\n"
            f"{inner}ELSE {right_sql}\n"
            f"{pad}END"
        )

    @staticmethod
    def _safe_column_name(name: str) -> str:
        """
        Wrap column names that contain spaces or special characters in
        double-quotes.  Pure alphanumeric + underscore names are left as-is.
        Double-quotes inside the name are escaped to prevent SQL injection.
        """
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            return name
        # Escape any embedded double-quotes
        safe = name.replace('"', '""')
        return f'"{safe}"'

    # ------------------------------------------------------------------
    # Diagnostics / introspection
    # ------------------------------------------------------------------

    def tree_stats(self) -> dict:
        """Return a summary of the tree structure."""
        tree = self.model.tree_
        return {
            "n_nodes": tree.node_count,
            "max_depth": self.model.get_depth(),
            "n_leaves": self.model.get_n_leaves(),
            "n_features": self.model.n_features_in_,
            "model_name": self.model_name,
            "is_classifier": self.is_classifier,
        }

    def __repr__(self) -> str:
        stats = self.tree_stats()
        return (
            f"TreeToSQL(model_name={self.model_name!r}, "
            f"depth={stats['max_depth']}, "
            f"leaves={stats['n_leaves']}, "
            f"classifier={self.is_classifier})"
        )
