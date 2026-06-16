"""
parser.py - SQL query parser and rewriter.

Finds ``predict_<model_name>(col1, col2, ...)`` pseudo-function calls
in user SQL and replaces them with the equivalent inline CASE expression
produced by TreeToSQL.

The rewriter is intentionally minimal: it handles the common pattern of
predict calls in SELECT lists and WHERE clauses without implementing a
full SQL parser.
"""

from __future__ import annotations

import re
from typing import Dict

from .converter import TreeToSQL


# Matches:  predict_<name>(<anything>)
# Group 1 = model name, Group 2 = argument list (unparsed)
_PREDICT_PATTERN = re.compile(
    r'\bpredict_([A-Za-z0-9_]+)\s*\(([^)]*)\)',
    re.IGNORECASE,
)


class QueryRewriter:
    """
    Rewrite predict_*() calls in SQL strings to inline CASE expressions.

    Parameters
    ----------
    models : dict[str, TreeToSQL]
        Model registry populated by Database.register_model().
    """

    def __init__(self, models: Dict[str, TreeToSQL]) -> None:
        self._models = models

    def rewrite(self, sql: str) -> str:
        """
        Replace every ``predict_<name>(...)`` occurrence in *sql* with
        the corresponding SQL CASE expression.

        Raises
        ------
        KeyError
            If a referenced model has not been registered.
        """
        def _replace(match: re.Match) -> str:
            model_name = match.group(1).lower()
            if model_name not in self._models:
                raise KeyError(
                    f"No model named '{model_name}' is registered. "
                    f"Available models: {list(self._models.keys())}"
                )
            converter = self._models[model_name]
            return f"({converter.to_sql()})"

        return _PREDICT_PATTERN.sub(_replace, sql)

    def has_predict_calls(self, sql: str) -> bool:
        """Return True if *sql* contains any predict_*() call."""
        return bool(_PREDICT_PATTERN.search(sql))

    def list_predict_calls(self, sql: str) -> list:
        """Return a list of (model_name, args_str) tuples found in *sql*."""
        return [
            (m.group(1).lower(), m.group(2).strip())
            for m in _PREDICT_PATTERN.finditer(sql)
        ]
