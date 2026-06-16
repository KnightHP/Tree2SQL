"""
benchmark.py - Performance benchmarking suite.

Compares two execution strategies:
  1. Naive UDF  - Python function called per-row via DuckDB UDF mechanism
  2. Native SQL - Inline CASE expression executed by DuckDB's vectorised engine

Reports execution time, speedup factor, and optionally generates charts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .converter import TreeToSQL
from .database import Database


@dataclass
class BenchmarkResult:
    model_name: str
    max_depth: int
    n_rows: int
    n_runs: int
    udf_times: List[float]
    sql_times: List[float]

    @property
    def udf_mean(self) -> float:
        return float(np.mean(self.udf_times))

    @property
    def sql_mean(self) -> float:
        return float(np.mean(self.sql_times))

    @property
    def speedup(self) -> float:
        if self.sql_mean == 0:
            return float("inf")
        return self.udf_mean / self.sql_mean

    def summary(self) -> str:
        return (
            f"[{self.model_name}] depth={self.max_depth}, rows={self.n_rows:,}\n"
            f"  UDF mean : {self.udf_mean:.3f}s\n"
            f"  SQL mean : {self.sql_mean:.3f}s\n"
            f"  Speedup  : {self.speedup:.1f}x"
        )


class Benchmark:
    """
    Run head-to-head benchmarks between naive Python UDF and native SQL
    CASE expression prediction in DuckDB.

    Parameters
    ----------
    db : Database
        Connected Database instance with a preloaded table.
    table_name : str
        Table to query during benchmarks.
    n_runs : int
        Number of timed repetitions (average is reported).
    """

    def __init__(
        self,
        db: Database,
        table_name: str = "bank_data",
        n_runs: int = 5,
    ) -> None:
        self.db = db
        self.table_name = table_name
        self.n_runs = n_runs
        self.results: List[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Core timing helpers
    # ------------------------------------------------------------------

    def _time_sql(self, sql: str) -> float:
        """Return wall-clock seconds for one execution of *sql*."""
        start = time.perf_counter()
        self.db.execute(sql, rewrite=False).fetchall()
        return time.perf_counter() - start

    # ------------------------------------------------------------------
    # Single-model benchmark
    # ------------------------------------------------------------------

    def run_single(
        self,
        model: DecisionTreeClassifier | DecisionTreeRegressor,
        feature_columns: List[str],
        model_name: str = "bench_model",
        max_depth: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Benchmark one model against the configured table.

        Returns a BenchmarkResult with per-run timings and speedup.
        """
        depth = max_depth or model.get_depth()

        converter = TreeToSQL(model, feature_columns, model_name=model_name)
        self.db.register_model(model_name, converter)
        self.db.register_predict_udf(model_name)

        # Build the UDF argument list (raw column names, no safe-quoting)
        udf_args = ", ".join(feature_columns)
        udf_sql = (
            f"SELECT predict_{model_name}({udf_args}) FROM {self.table_name}"
        )

        # Build the inline CASE expression SQL
        case_expr = converter.to_sql()
        case_sql = f"SELECT {case_expr} FROM {self.table_name}"

        # Warm up (not counted)
        self.db.execute(udf_sql, rewrite=False).fetchall()
        self.db.execute(case_sql, rewrite=False).fetchall()

        udf_times, sql_times = [], []
        for _ in range(self.n_runs):
            udf_times.append(self._time_sql(udf_sql))
            sql_times.append(self._time_sql(case_sql))

        n_rows = self.db.row_count(self.table_name)
        result = BenchmarkResult(
            model_name=model_name,
            max_depth=depth,
            n_rows=n_rows,
            n_runs=self.n_runs,
            udf_times=udf_times,
            sql_times=sql_times,
        )
        self.results.append(result)
        print(result.summary())
        return result

    # ------------------------------------------------------------------
    # Multi-depth benchmark
    # ------------------------------------------------------------------

    def run_depth_sweep(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: List[str],
        depths: List[int] = None,
        task: str = "classification",
    ) -> List[BenchmarkResult]:
        """
        Train trees at each depth in *depths* and benchmark each one.

        Parameters
        ----------
        depths : list[int]
            Tree max_depth values to sweep, default [3, 5, 7, 10].
        task : 'classification' | 'regression'
        """
        if depths is None:
            depths = [3, 5, 7, 10]

        sweep_results = []
        for depth in depths:
            name = f"model_d{depth}"
            if task == "classification":
                model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            result = self.run_single(model, feature_columns, model_name=name, max_depth=depth)
            sweep_results.append(result)

        return sweep_results

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------

    def verify_correctness(
        self,
        model: DecisionTreeClassifier | DecisionTreeRegressor,
        feature_columns: List[str],
        model_name: str = "verify_model",
        sample_n: int = 1000,
    ) -> float:
        """
        Return the fraction of rows where the SQL prediction matches
        model.predict().  Should be 1.0 (100% agreement).
        """
        converter = TreeToSQL(model, feature_columns, model_name=model_name)
        self.db.register_model(model_name, converter)

        case_expr = converter.to_sql()
        sql = f"SELECT {case_expr} AS sql_pred FROM {self.table_name} LIMIT {sample_n}"
        sql_preds = self.db.execute(sql, rewrite=False).df()["sql_pred"].tolist()

        df = self.db.query_df(
            f"SELECT {', '.join(feature_columns)} FROM {self.table_name} LIMIT {sample_n}",
            rewrite=False,
        )
        py_preds = [str(p) for p in model.predict(df.values)]
        sql_preds_str = [str(p) for p in sql_preds]

        matches = sum(a == b for a, b in zip(py_preds, sql_preds_str))
        accuracy = matches / len(py_preds)
        print(f"Correctness check: {matches}/{len(py_preds)} = {accuracy:.4%}")
        return accuracy

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_speedup(
        self,
        results: Optional[List[BenchmarkResult]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Bar chart comparing UDF vs native SQL execution time per tree depth.
        """
        if results is None:
            results = self.results
        if not results:
            raise ValueError("No benchmark results to plot.")

        depths = [r.max_depth for r in results]
        udf_means = [r.udf_mean for r in results]
        sql_means = [r.sql_mean for r in results]
        speedups = [r.speedup for r in results]

        x = np.arange(len(depths))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Tree2SQL: Native SQL vs Python UDF Inference Performance", fontsize=13)

        # Left: execution time comparison
        bars1 = ax1.bar(x - width / 2, udf_means, width, label="Python UDF", color="#e74c3c", alpha=0.85)
        bars2 = ax1.bar(x + width / 2, sql_means, width, label="Native SQL", color="#2ecc71", alpha=0.85)
        ax1.set_xlabel("Tree Max Depth")
        ax1.set_ylabel("Mean Execution Time (s)")
        ax1.set_title("Execution Time by Tree Depth")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(d) for d in depths])
        ax1.legend()
        ax1.set_yscale("log")

        # Annotate bars with values
        for bar in bars1:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{bar.get_height():.2f}s",
                ha="center", va="bottom", fontsize=8,
            )
        for bar in bars2:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{bar.get_height():.3f}s",
                ha="center", va="bottom", fontsize=8,
            )

        # Right: speedup factor
        bars3 = ax2.bar(x, speedups, color="#3498db", alpha=0.85)
        ax2.set_xlabel("Tree Max Depth")
        ax2.set_ylabel("Speedup Factor (×)")
        ax2.set_title("Speedup: Native SQL vs Python UDF")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(d) for d in depths])
        ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)

        for bar, sp in zip(bars3, speedups):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(speedups) * 0.01,
                f"{sp:.1f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        return fig

    def summary_dataframe(self, results: Optional[List[BenchmarkResult]] = None) -> pd.DataFrame:
        """Return benchmark results as a tidy DataFrame."""
        if results is None:
            results = self.results
        return pd.DataFrame([
            {
                "model_name": r.model_name,
                "max_depth": r.max_depth,
                "n_rows": r.n_rows,
                "udf_mean_s": round(r.udf_mean, 4),
                "sql_mean_s": round(r.sql_mean, 4),
                "speedup_x": round(r.speedup, 1),
            }
            for r in results
        ])
