"""
load_dataset.py - Bank Marketing dataset loader.

Downloads Bank Marketing (UCI ID 222) via ucimlrepo, one-hot encodes
categorical columns, and returns clean train/test splits ready for DuckDB
and scikit-learn.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_bank_marketing(test_size: float = 0.2, random_state: int = 42):
    """
    Fetch Bank Marketing dataset (45,211 rows, 16 features) from UCI
    and return processed X, y plus train/test splits.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, full_df
    """
    try:
        from ucimlrepo import fetch_ucirepo
        print("Downloading Bank Marketing dataset from UCI repository...")
        bank_marketing = fetch_ucirepo(id=222)
        X_raw = bank_marketing.data.features
        y_raw = bank_marketing.data.targets.squeeze()
    except Exception as e:
        print(f"ucimlrepo fetch failed ({e}), generating synthetic fallback dataset.")
        X_raw, y_raw = _synthetic_bank_data()

    # One-hot encode categorical columns
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False)

    # Ensure all columns are numeric
    X_encoded = X_encoded.astype(float)

    # Sanitize column names (replace spaces / dots with underscores)
    X_encoded.columns = [
        c.replace(" ", "_").replace(".", "_").replace("-", "_")
        for c in X_encoded.columns
    ]

    feature_names = X_encoded.columns.tolist()

    # Binary target: 'yes' → 1, 'no' → 0 (keep as string for classification demo)
    y = y_raw.astype(str).str.strip().str.lower()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Full dataframe for loading into DuckDB
    full_df = X_encoded.copy()
    full_df["y"] = y.values

    print(
        f"Dataset loaded: {len(X_encoded):,} rows, "
        f"{len(feature_names)} features, "
        f"target distribution: {y.value_counts().to_dict()}"
    )
    return X_train, X_test, y_train, y_test, feature_names, full_df


def _synthetic_bank_data(n: int = 45211, random_state: int = 42):
    """
    Generate a synthetic stand-in for Bank Marketing when ucimlrepo is
    unavailable.  Column names mirror the real dataset.
    """
    import numpy as np
    rng = np.random.default_rng(random_state)

    n_samples = n
    X = pd.DataFrame({
        "age": rng.integers(18, 95, n_samples).astype(float),
        "balance": rng.normal(1422, 3009, n_samples),
        "day": rng.integers(1, 32, n_samples).astype(float),
        "duration": rng.integers(0, 3000, n_samples).astype(float),
        "campaign": rng.integers(1, 50, n_samples).astype(float),
        "pdays": rng.choice([-1] + list(range(1, 400)), n_samples).astype(float),
        "previous": rng.integers(0, 50, n_samples).astype(float),
        "job": rng.choice(
            ["admin.", "blue-collar", "entrepreneur", "housemaid",
             "management", "retired", "self-employed", "services",
             "student", "technician", "unemployed", "unknown"],
            n_samples,
        ),
        "marital": rng.choice(["divorced", "married", "single"], n_samples),
        "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n_samples),
        "default": rng.choice(["no", "yes"], n_samples, p=[0.98, 0.02]),
        "housing": rng.choice(["no", "yes"], n_samples),
        "loan": rng.choice(["no", "yes"], n_samples, p=[0.85, 0.15]),
        "contact": rng.choice(["cellular", "telephone", "unknown"], n_samples),
        "month": rng.choice(
            ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"],
            n_samples,
        ),
        "poutcome": rng.choice(["failure", "other", "success", "unknown"], n_samples),
    })

    # Synthetic target: roughly 11.7% positive rate like the real dataset
    probs = (
        0.05
        + 0.15 * (X["duration"] > 300).astype(float)
        + 0.10 * (X["age"] < 30).astype(float)
        + 0.05 * (X["previous"] > 0).astype(float)
    )
    probs = probs.clip(0, 1)
    y_arr = rng.random(n_samples) < probs
    y = pd.Series(["yes" if v else "no" for v in y_arr], name="y")
    return X, y


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, full_df = load_bank_marketing()
    print(f"Training set : {X_train.shape}")
    print(f"Test set     : {X_test.shape}")
    print(f"Features     : {feature_names[:5]} ... ({len(feature_names)} total)")
