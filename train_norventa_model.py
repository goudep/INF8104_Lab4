"""Reproducibly train the TEAM-BUILT NorVenta teaching baseline.

This is NOT an original NorVenta model. Do not run this for normal auditing:
use test_norventa_model.py instead. To intentionally replace the bundled model,
run: python train_norventa_model.py --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "norventa_credit.csv"
MODEL = ROOT / "norventa_logreg_model.joblib"
MANIFEST = ROOT / "model_manifest.json"
FEATURES = ["revenu", "score_credit"]
LABEL = "approbation"
GROUP = "groupe"
SEED = 42
TEST_FRACTION = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Retrain the team-built teaching baseline (requires explicit --force)")
    parser.add_argument("--force", action="store_true", help="Explicitly replace the model and manifest")
    args = parser.parse_args()
    if not args.force and (MODEL.exists() or MANIFEST.exists()):
        raise SystemExit("Model already exists. For auditing run: python test_norventa_model.py\n"
                         "Only to intentionally retrain: python train_norventa_model.py --force")
    data = pd.read_csv(SOURCE)
    if data.empty or not set([GROUP, *FEATURES, LABEL]).issubset(data.columns):
        raise ValueError("Missing expected columns or empty input CSV")
    if data[LABEL].isna().any() or not set(data[LABEL].unique()).issubset({0, 1}):
        raise ValueError("approbation labels must be 0 or 1")
    if data[FEATURES].isna().all().any():
        raise ValueError("At least one input feature is entirely missing")

    train_idx, test_idx = train_test_split(
        data.index.to_numpy(), test_size=TEST_FRACTION, random_state=SEED,
        stratify=data[LABEL],
    )
    train = data.iloc[train_idx]
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipeline.fit(train[FEATURES], train[LABEL])
    joblib.dump(pipeline, MODEL)
    metadata = {
        "project": "INF8104 Lab 4 NorVenta - teaching baseline only",
        "model_origin": "Team-built example, not an original NorVenta production model",
        "input_csv": SOURCE.name,
        "csv_sha256": sha256_file(SOURCE),
        "model_file": MODEL.name,
        "model_sha256": sha256_file(MODEL),
        "split_method": "train_test_split with stratify=approbation",
        "split_seed": SEED,
        "test_fraction": TEST_FRACTION,
        "source_row_index_definition": "Zero-based position in unchanged source CSV, excluding header",
        "features": FEATURES,
        "label": LABEL,
        "fairness_group_only": GROUP,
        "pipeline": ["SimpleImputer(median)", "StandardScaler", "LogisticRegression(max_iter=1000)"],
        "decision_threshold": 0.5,
        "train_indices": [int(v) for v in train_idx],
        "test_indices": [int(v) for v in test_idx],
        "training_environment": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
    }
    MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model + manifest. TRAIN={len(train_idx)}, TEST={len(test_idx)}. Run test_norventa_model.py")


if __name__ == "__main__":
    main()
