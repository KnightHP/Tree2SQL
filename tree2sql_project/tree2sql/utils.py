"""
utils.py - Shared helper functions.
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace problematic characters in DataFrame column names.
    Returns a copy with cleaned names.
    """
    new_cols = {}
    for col in df.columns:
        cleaned = re.sub(r'[^A-Za-z0-9_]', '_', col)
        # Strip leading digits
        if cleaned and cleaned[0].isdigit():
            cleaned = "_" + cleaned
        new_cols[col] = cleaned
    return df.rename(columns=new_cols)


def print_tree_sql(
    model: DecisionTreeClassifier | DecisionTreeRegressor,
    feature_names: List[str],
    model_name: str = "model",
) -> str:
    """
    Convenience: print and return the SQL CASE expression for a tree.
    """
    from .converter import TreeToSQL
    converter = TreeToSQL(model, feature_names, model_name=model_name)
    sql = converter.to_sql()
    print(f"-- SQL for '{model_name}' (depth={model.get_depth()}, leaves={model.get_n_leaves()})")
    print(sql)
    return sql


def onehot_to_sql_features(df: pd.DataFrame, prefix_sep: str = "_") -> List[str]:
    """
    Return the list of column names after one-hot encoding, suitable for
    passing to TreeToSQL as feature_names.
    """
    return list(df.columns)


def format_speedup_message(speedup: float) -> str:
    """Return a human-readable speedup description."""
    if speedup >= 100:
        return f"🚀 {speedup:.0f}x faster (>100x tier)"
    elif speedup >= 10:
        return f"⚡ {speedup:.1f}x faster"
    else:
        return f"📈 {speedup:.1f}x faster"
