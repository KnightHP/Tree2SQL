"""
database.py - DuckDB connection wrapper and model registry.

Provides a thin, convenient interface around duckdb for:
  - Loading DataFrames as persistent tables
  - Registering Tree2SQL models
  - Executing rewritten SQL queries
  - Registering Python UDF fallbacks for benchmarking
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import duckdb
import numpy as np
import pandas as pd

from .converter import TreeToSQL
from .parser import QueryRewriter


class Database:
    """
    Wrapper around a DuckDB connection with Tree2SQL model awareness.

    Parameters
    ----------
    db_path : str
        Path to the DuckDB database file, or ":memory:" for in-process.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._models: Dict[str, TreeToSQL] = {}
        self._rewriter = QueryRewriter(self._models)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> None:
        """
        Persist a pandas DataFrame as a DuckDB table.

        Parameters
        ----------
        df : pd.DataFrame
        table_name : str
        if_exists : 'replace' | 'append' | 'fail'
        """
        if if_exists == "replace":
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        elif if_exists == "fail":
            existing = self.conn.execute(
                "SELECT count(*) FROM information_schema.tables "
                f"WHERE table_name = '{table_name}'"
            ).fetchone()[0]
            if existing:
                raise ValueError(f"Table '{table_name}' already exists.")

        # Register DataFrame as a view then materialise as a table
        self.conn.register("_tmp_df", df)
        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_df")
        self.conn.unregister("_tmp_df")

    def table_exists(self, table_name: str) -> bool:
        result = self.conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            f"WHERE table_name = '{table_name}'"
        ).fetchone()
        return result[0] > 0

    def row_count(self, table_name: str) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_model(self, model_name: str, converter: TreeToSQL) -> None:
        """Register a TreeToSQL converter under a named alias."""
        self._models[model_name] = converter
        self._rewriter = QueryRewriter(self._models)

    def list_models(self) -> list:
        return list(self._models.keys())

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, rewrite: bool = True) -> duckdb.DuckDBPyRelation:
        """
        Execute a SQL query.  If ``rewrite=True`` any predict_*() calls
        are first replaced with their inline CASE expressions.
        """
        if rewrite:
            sql = self._rewriter.rewrite(sql)
        return self.conn.execute(sql)

    def query_df(self, sql: str, rewrite: bool = True) -> pd.DataFrame:
        """Execute and return results as a DataFrame."""
        return self.execute(sql, rewrite=rewrite).df()

    # ------------------------------------------------------------------
    # UDF registration (for benchmarking the "slow" path)
    # ------------------------------------------------------------------

    def register_predict_udf(self, model_name: str) -> None:
        """
        Register a Python UDF ``predict_<model_name>(features...)`` that
        calls model.predict() row-by-row.  Used only in benchmarks to
        demonstrate the speedup.
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not registered.")

        converter = self._models[model_name]
        model = converter.model
        feature_names = converter.feature_names
        n_features = len(feature_names)

        # Build the UDF dynamically.  DuckDB 1.x requires the Python function
        # to have exactly n_features explicit parameters (not *args), so we
        # construct it at runtime.
        if converter.is_classifier:
            return_type = "VARCHAR"

            def _inner_predict(*args):
                arr = np.array(args, dtype=float).reshape(1, -1)
                return str(model.predict(arr)[0])
        else:
            return_type = "DOUBLE"

            def _inner_predict(*args):
                arr = np.array(args, dtype=float).reshape(1, -1)
                return float(model.predict(arr)[0])

        # Create a wrapper whose signature has exactly n_features arguments
        param_names = ", ".join(f"_x{i}" for i in range(n_features))
        call_args   = ", ".join(f"_x{i}" for i in range(n_features))
        globs = {"_inner_predict": _inner_predict}
        exec(
            f"def _udf({param_names}): return _inner_predict({call_args})",
            globs,
        )
        udf_func = globs["_udf"]

        udf_name = f"predict_{model_name}"
        arg_types = ["DOUBLE"] * n_features
        try:
            self.conn.remove_function(udf_name)
        except Exception:
            pass
        self.conn.create_function(udf_name, udf_func, arg_types, return_type)
