"""INF8104 Lab 4 — baseline model verification and prediction only.

Execute: python test_norventa_model.py [--synthetic-count 100] [--synthetic-seed 2026]

Does NOT retrain, re-split, alter the source CSV, or perform Phase 3/4 audits.
Outputs (under outputs/):
  test_result.csv             400 held-out rows with labels/predictions
  split_trace.csv             2000 source rows with TRAIN/TEST membership
  evaluation_summary.csv      overall held-out performance, including ROC-AUC
  synthetic_inputs.csv        new illustrative, unlabelled applicant records
  synthetic_predictions.csv   decisions/probabilities for the new records

Only load the supplied .joblib file from a trusted source. Synthetic data are not
independent real applicants and have no labels; no synthetic accuracy is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "norventa_credit.csv"
MODEL = ROOT / "norventa_logreg_model.joblib"
MANIFEST = ROOT / "model_manifest.json"
OUTPUT = ROOT / "outputs"
FEATURES = ["revenu", "score_credit"]
LABEL = "approbation"
GROUP = "groupe"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_synthetic(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample TRAIN examples, gently perturb features; NO approval labels.

    This retains some correlation between income and score. It does NOT
    represent independently collected new applicants or provide ground truth.
    """
    rng = np.random.default_rng(seed)
    bases = train.iloc[rng.integers(0, len(train), size=n)]
    income = bases["revenu"].to_numpy(dtype=float)
    score = bases["score_credit"].to_numpy(dtype=float)
    income_low, income_high = np.nanquantile(train["revenu"], [0.01, 0.99])
    score_low, score_high = np.nanquantile(train["score_credit"], [0.01, 0.99])
    income_new = np.clip(
        income * (1 + rng.normal(0, 0.05, size=n)), max(0, income_low), income_high
    )
    score_new = np.clip(score + rng.normal(0, 12, size=n), score_low, score_high)
    return pd.DataFrame({
        "simulation_id": [f"S{i:04d}" for i in range(1, n + 1)],
        GROUP: bases[GROUP].to_numpy(),
        "revenu": np.round(income_new, 6),
        "score_credit": np.round(score_new, 6),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen NorVenta baseline; export basic evaluation and synthetic predictions")
    parser.add_argument("--synthetic-count", type=int, default=100,
                        help="Number of simulated examples to predict, default 100")
    parser.add_argument("--synthetic-seed", type=int, default=2026,
                        help="Random seed for simulated examples, default 2026")
    args = parser.parse_args()
    require(1 <= args.synthetic_count <= 100000, "--synthetic-count must be 1..100000")
    for path in (CSV, MODEL, MANIFEST):
        require(path.is_file(), f"Required file not found: {path.name}")

    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(file_sha256(CSV) == metadata["csv_sha256"],
            "Original CSV has changed since training; restore it before testing")
    require(file_sha256(MODEL) == metadata["model_sha256"],
            "Frozen model file has changed; restore model and manifest together")
    require(metadata["features"] == FEATURES and metadata["label"] == LABEL,
            "Features or target label do not match model metadata")

    # Read source twice: typed values for calculations, exact original text for CSV exports.
    data = pd.read_csv(CSV)
    raw = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    require(len(data) == len(raw) and len(data) > 0, "Source CSV is invalid/empty")
    require(set([GROUP, *FEATURES, LABEL]).issubset(data.columns), "Required CSV columns missing")
    require(not data[LABEL].isna().any() and set(data[LABEL].unique()).issubset({0, 1}),
            "Approval labels must be 0 or 1")

    train_ids = np.asarray(metadata["train_indices"], dtype=int)
    test_ids = np.asarray(metadata["test_indices"], dtype=int)
    n = len(data)
    require(len(train_ids) + len(test_ids) == n, "Frozen split does not cover every row")
    require(len(set(train_ids)) == len(train_ids) and len(set(test_ids)) == len(test_ids),
            "Repeated row index in frozen split")
    require(set(train_ids).isdisjoint(set(test_ids)), "TRAIN/TEST overlap detected")
    require(set(train_ids) | set(test_ids) == set(range(n)),
            "Frozen split misses/introduces invalid row indices")

    # Important: verifying the recorded split is NOT making a new split.
    expected_train, expected_test = train_test_split(
        data.index.to_numpy(), test_size=metadata["test_fraction"],
        random_state=metadata["split_seed"], stratify=data[LABEL],
    )
    require(np.array_equal(train_ids, expected_train) and np.array_equal(test_ids, expected_test),
            "Frozen split does not match the documented original random split")

    # Only load a trusted team-generated model artifact.
    model = joblib.load(MODEL)
    require(np.array_equal(model.classes_, np.array([0, 1])),
            "Model classes must be [0, 1] for probability column 1")
    X_test = data.iloc[test_ids][FEATURES]
    y_test = data.iloc[test_ids][LABEL].to_numpy(dtype=int)
    predicted = model.predict(X_test).astype(int)
    probs = model.predict_proba(X_test)[:, 1]
    require(len(predicted) == len(y_test), "Prediction row count mismatch")

    OUTPUT.mkdir(exist_ok=True)
    # Trace CSV: annotate a COPY, not the original CSV. Saved row order is original order.
    membership = np.full(n, "TRAIN", dtype=object)
    membership[test_ids] = "TEST"
    trace = raw.copy()
    trace.insert(0, "source_row_index", np.arange(n))
    trace["dataset_split"] = membership
    trace.to_csv(OUTPUT / "split_trace.csv", index=False)

    # Held-out TEST records with original features and model predictions.
    result = raw.iloc[test_ids].copy()
    result.insert(0, "source_row_index", test_ids)
    result["approbation_predite"] = predicted
    result["decision_predite"] = np.where(predicted == 1, "Approved", "Not approved")
    result["prob_approbation_1"] = np.round(probs, 8)
    result["matches_recorded_label"] = predicted == y_test
    result.sort_values("source_row_index").to_csv(OUTPUT / "test_result.csv", index=False)

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_test, predicted, labels=[0, 1]).ravel())
    performance = {
        "holdout_samples": len(test_ids),
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    pd.DataFrame([performance]).to_csv(OUTPUT / "evaluation_summary.csv", index=False)

    # Separately: synthetic predictions illustrate model usage; no true labels.
    synthetic = make_synthetic(data.iloc[train_ids], args.synthetic_count, args.synthetic_seed)
    synthetic.to_csv(OUTPUT / "synthetic_inputs.csv", index=False)
    synth_pred = model.predict(synthetic[FEATURES]).astype(int)
    synth_prob = model.predict_proba(synthetic[FEATURES])[:, 1]
    synth_result = synthetic.copy()
    synth_result["approbation_predite"] = synth_pred
    synth_result["decision_predite"] = np.where(synth_pred == 1, "Approved", "Not approved")
    synth_result["prob_approbation_1"] = np.round(synth_prob, 8)
    synth_result.to_csv(OUTPUT / "synthetic_predictions.csv", index=False)

    print("[PASS] Original model/CSV hashes match training manifest.")
    print(f"[PASS] Frozen split verified: TRAIN={len(train_ids)} | TEST={len(test_ids)}; no overlap.")
    print(f"[PASS] TEST accuracy={performance['accuracy']:.4f}; ROC-AUC={performance['roc_auc']:.4f}.")
    print(f"[PASS] Synthetic examples predicted: {len(synthetic)} (no ground-truth labels).")
    print("Saved in outputs/:")
    for name in ("test_result.csv", "split_trace.csv", "evaluation_summary.csv",
                 "synthetic_inputs.csv", "synthetic_predictions.csv"):
        print(f" - {name}")
    print("Scope: baseline only. Phase 3 fairness and Phase 4 robustness/perturbations are NOT run.")


if __name__ == "__main__":
    main()
