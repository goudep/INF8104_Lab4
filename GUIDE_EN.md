# INF8104 · Lab 4 — NorVenta Baseline Model: Complete User Guide (English)

**Version:** Baseline V2 (basic model verification and prediction demonstration only)  
**Project:** NorVenta credit-approval classification example  
**Audience:** INF8104 Lab 4 team members (handoff across Phases 1–4)

> **Important:** This is a **teaching baseline model built by the team** using the instructor-provided `norventa_credit.csv`. It is **not** an original production model supplied or deployed by NorVenta. The `approbation` column contains given approval labels, not independently verified, objectively correct credit decisions. This package does **not** complete the fairness audit in Phase 3, robustness audit in Phase 4, or log analysis in Phase 5.

## 1. What is included and what remains to be done?

| Component | Status |
|---|---|
| Original 2,000 application records | Included; source CSV preserved |
| Random stratified 80% TRAIN / 20% TEST split | Completed and fixed at training time |
| Logistic regression model and preprocessing pipeline | Trained and saved |
| Holdout Accuracy, Precision, Recall, ROC-AUC and confusion matrix | Included as **baseline performance verification** |
| 100 simulated applications and their predictions | Included; no ground-truth labels |
| Source-row indices, TRAIN/TEST membership and SHA-256 integrity checks | Included |
| Fairness metrics and interpretation by group A/B | **Not completed: Phase 3** |
| Edge cases, missing values and small-perturbation robustness tests | **Not completed: Phase 4** |
| Anomaly detection on Lab 2 `auth.log` | **Not completed: Phase 5** |

`test_norventa_model.py` handles **basic performance checks, traceability verification and simulated prediction**, not the full audits assigned to later phases.

## 2. Folder structure and purpose of every file

```text
INF8104_Lab4/
├── norventa_credit.csv            # Original input data; do not modify
├── train_norventa_model.py        # Script used to train and save the model
├── norventa_logreg_model.joblib   # ★ Trained model (core model artifact)
├── model_manifest.json           # ★ Frozen split and integrity metadata
├── test_norventa_model.py         # ★ Main everyday verification script
├── predict_example.py            # Single-application prediction example
├── requirements_lab4.txt         # Python dependency versions
├── GUIDE_EN.md                   # This English guide
└── outputs/
    ├── test_result.csv           # Predictions on 400 held-out TEST rows
    ├── split_trace.csv           # TRAIN/TEST trace for all 2,000 rows
    ├── evaluation_summary.csv    # Overall holdout evaluation metrics
    ├── synthetic_inputs.csv      # 100 newly generated example inputs
    └── synthetic_predictions.csv # Predictions for simulated inputs
```

| File | Purpose and notes |
|---|---|
| `norventa_credit.csv` | Original instructor-provided dataset containing `groupe`, `revenu`, `score_credit`, `approbation`. **Do not edit it.** |
| **`norventa_logreg_model.joblib`** | **The actual trained model.** Stores a scikit-learn Pipeline with missing-value imputation, scaling and logistic regression. |
| `train_norventa_model.py` | Makes the original stratified random 80/20 split, trains the pipeline and saves the model plus manifest. **Not part of routine use.** |
| **`test_norventa_model.py`** | **Main script to run.** Verifies original file hashes and split indices, loads the model, computes basic evaluation metrics and writes five outputs. **Does not retrain.** |
| `model_manifest.json` | Training metadata: row indices for both splits, seed, features, label, dependency versions, and SHA-256 hashes. The test script verifies the original split rather than drawing a new one. |
| `predict_example.py` | Takes income and credit score from the command line and prints class 0/1, Approved/Not approved, and the model probability. |
| `requirements_lab4.txt` | Lists pinned versions of NumPy, pandas, SciPy, scikit-learn and joblib. |
| `.venv` / `.venv313` | Locally created Python environments; **not part of the model, and should not be uploaded or shared**. |

**Key distinction:** `train_norventa_model.py` **trains** → `norventa_logreg_model.joblib` **stores the model** → `test_norventa_model.py` **uses and checks the model**.

## 3. Complete setup: Windows / VS Code / PowerShell

**Recommended setup:** Windows PowerShell and 64-bit Python **3.13**. The earlier Python 3.15 environment could not install the pinned older pandas wheel. Python 3.13 is the safer environment for this provided model and dependency set.

### Step 1 — Extract the project and open a terminal

Open the extracted folder in VS Code and select **Terminal → New Terminal**. 

### Step 2 — Check Python 3.13

```powershell
py -3.13 --version
```

If it is not installed:

```powershell
py install 3.13
py -3.13 --version
```

### Step 3 — Create a project-specific virtual environment

```powershell
py -3.13 -m venv .venv
```

A `venv` isolates the project's Python dependencies and versions from other projects. It is **neither a virtual machine nor the AI model**.

### Step 4 — Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements_lab4.txt
```

The `--only-binary=:all:` option installs pre-built wheels only, preventing pip from attempting local C/C++ compilation of libraries such as pandas.

### Step 5 — Verify imports

```powershell
.\.venv\Scripts\python.exe -c "import pandas, sklearn, joblib; print('Environment OK')"
```

Expected output:

```text
Environment OK
```

### Step 6 — Run the main verification script

```powershell
.\.venv\Scripts\python.exe test_norventa_model.py
```

**Expected console output (minor differences may occur across versions):**

```text
[PASS] Original model/CSV hashes match training manifest.
[PASS] Frozen split verified: TRAIN=1600 | TEST=400; no overlap.
[PASS] TEST accuracy=0.8600; ROC-AUC=0.9400.
[PASS] Synthetic examples predicted: 100 (no ground-truth labels).
Saved in outputs/:
 - test_result.csv
 - split_trace.csv
 - evaluation_summary.csv
 - synthetic_inputs.csv
 - synthetic_predictions.csv
Scope: baseline only. Phase 3 fairness and Phase 4 robustness/perturbations are NOT run.
```

Each run regenerates the **five files** under `outputs/`. It does **not** modify the original source CSV, the saved `.joblib` model, or `model_manifest.json`.

### Step 7 — Optional: change the synthetic sample count

```powershell
.\.venv\Scripts\python.exe test_norventa_model.py --synthetic-count 200
```

This generates 200 simulated examples **without true labels**. It does not change the 400 held-out TEST records.

### Step 8 — Optional: predict a single application

```powershell
.\.venv\Scripts\python.exe predict_example.py --revenu 50000 --score-credit 700
```

The result includes class 0/1, `Approved`/`Not approved`, and the model's predicted probability of class 1. This baseline accepts only `revenu` and `score_credit` as predictive features.

## 4. How the model was created

1. **Load the data:** `norventa_credit.csv` contains 2,000 observations and four columns: `groupe`, `revenu`, `score_credit`, `approbation`.
2. **Define features:** `X = [revenu, score_credit]`. The `groupe` field is **not** a predictive feature in this baseline; it is kept for later group-level fairness analysis.
3. **Define the label:** `y = approbation`; 1 means the provided label says approved, and 0 means not approved.
4. **Random train/test split:** `train_test_split(test_size=0.20, stratify=y, random_state=42)` yields 1,600 TRAIN and 400 TEST rows. Stratification approximately preserves the label proportions, while the fixed seed makes the split reproducible.
5. **Fit on TRAIN only:** `SimpleImputer(strategy='median')` (missing-value imputation) → `StandardScaler()` (feature standardization) → `LogisticRegression(max_iter=1000, random_state=42)`. Preprocessing statistics are learned on TRAIN only to avoid leakage.
6. **Save artifacts:** The trained pipeline is serialized to `norventa_logreg_model.joblib`. Original split indices, SHA-256 hashes and settings are saved in `model_manifest.json`.
7. **Evaluate on unseen records:** Basic classification metrics are computed only on the 400 held-out TEST rows.
8. **Demonstrate on simulated cases:** The script samples base observations from TRAIN, applies small random changes to income and score (within reasonable training-data ranges), and predicts their labels. The resulting examples have **no ground-truth approval labels** and are not independently collected real applicants.

### Why is the data not split again during testing?

The saved model has already been trained on **specific** 1,600 observations. Resampling a new TEST set could accidentally include rows the model has seen, leading to data leakage and overly optimistic results. `model_manifest.json` stores the original row indices; the test script only verifies that frozen split. `split_trace.csv` is an **annotated copy**, while the source CSV stays unchanged.

### What does ROC-AUC mean?

- **ROC — Receiver Operating Characteristic:** a curve plotting the true-positive rate (TPR) against the false-positive rate (FPR) at different decision thresholds.
- **AUC — Area Under the Curve:** the area beneath that ROC curve.
- **ROC-AUC ≈ 0.9400:** on this holdout, a randomly chosen positive-labeled observation will receive a higher score than a randomly chosen negative-labeled observation about 94% of the time (standard AUC awards half-credit to ties).
- **Limitations:** It does not demonstrate fairness, probability calibration, robustness, or production readiness.

## 5. Explanation of all five generated output files

### A. `outputs/test_result.csv` — 400 labeled holdout rows and predictions

Compares the **provided label** with the model's prediction for each TEST observation. Use it to reproduce baseline metrics and as input for the Phase 3 fairness analysis.

| Column | Meaning |
|---|---|
| `source_row_index` | Zero-based position of the observation in the original source CSV (excluding its header) |
| `groupe` | Group A/B; the real-world meaning is not documented |
| `revenu` | Income |
| `score_credit` | Credit score |
| `approbation` | Provided approval label (0/1) |
| `approbation_predite` | Model prediction (0/1) |
| `decision_predite` | `Approved` or `Not approved` |
| `prob_approbation_1` | Model's predicted probability for class 1 |
| `matches_recorded_label` | Whether prediction equals the provided label (True/False) |

### B. `outputs/split_trace.csv` — traceability for all 2,000 original rows

Contains all original source columns plus `source_row_index` and `dataset_split` (`TRAIN` or `TEST`). It documents which records were used to train and which were held out, without altering the source CSV. **TRAIN/TEST is never supplied to the model as a feature.**

### C. `outputs/evaluation_summary.csv` — baseline performance on TEST only

| Field | Expected value | Meaning |
|---|---:|---|
| `holdout_samples` | 400 | Number of held-out observations |
| `accuracy` | 0.86 | 86% of predictions match provided labels |
| `precision` | ≈ 0.8377 | Among predicted approvals, share with an approval label |
| `recall` | 0.80625 | Among approval-labeled rows, share predicted approved |
| `roc_auc` | ≈ 0.9400 | Ranking/discrimination of positive and negative labels |
| `tn` | 215 | Labeled reject, predicted reject |
| `fp` | 25 | Labeled reject, predicted approve |
| `fn` | 31 | Labeled approve, predicted reject |
| `tp` | 129 | Labeled approve, predicted approve |

Here, “correct” means **agreement with `approbation` in the provided dataset**, not proof that any real credit decision was appropriate or fair.

### D. `outputs/synthetic_inputs.csv` — 100 illustrative new inputs

Columns: `simulation_id`, `groupe`, `revenu`, `score_credit`. By default, the script generates 100 examples by sampling TRAIN cases and introducing small random changes. There is **no true `approbation` value**. These records demonstrate new predictions and **do not substitute** for Phase 4's systematic robustness tests.

### E. `outputs/synthetic_predictions.csv` — predictions for simulated applications

Adds `approbation_predite`, `decision_predite`, and `prob_approbation_1` to the input fields. Two actual rows included in the supplied project package:

```csv
simulation_id,groupe,revenu,score_credit,approbation_predite,decision_predite,prob_approbation_1
S0001,B,77780.047594,719.923682,1,Approved,0.99723777
S0002,A,26091.659818,616.439273,0,Not approved,0.01306651
```

Because synthetic applications have **no known true labels**, they cannot be used to calculate Accuracy, true false-positive rate, or other metrics requiring ground truth.

## 6. Command reference

```powershell
# 1. Create Python 3.13 environment
py -3.13 -m venv .venv

# 2. Install pinned dependencies
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements_lab4.txt

# 3. Verify imports
.\.venv\Scripts\python.exe -c "import pandas, sklearn, joblib; print('Environment OK')"

# 4. Main verification: create the five outputs
.\.venv\Scripts\python.exe test_norventa_model.py

# 5. Optional: generate 200 synthetic applications
.\.venv\Scripts\python.exe test_norventa_model.py --synthetic-count 200

# 6. Optional: manual single-application prediction
.\.venv\Scripts\python.exe predict_example.py --revenu 50000 --score-credit 700
```

**The trained model is already supplied. Routine use does not require rerunning the training script.**

## 7. Team collaboration and version-control rules

1. Everyone must use the **same versions** of the `.joblib`, `model_manifest.json`, and unmodified `norventa_credit.csv`; do not edit any of them independently.
2. Normally run only `test_norventa_model.py`. If retraining is genuinely required, back up the entire directory, obtain team agreement, then run `train_norventa_model.py --force`. Recheck all outputs and integrity information afterward.
3. Do not add a TRAIN/TEST column to the original source CSV. Use `outputs/split_trace.csv` as the separate traceability copy.
4. Do not submit `.venv` or `.venv313`; each person creates their own local environment.
5. The Phase 3 owner must **independently calculate and interpret** group fairness metrics on TEST data. The Phase 4 owner must **independently design and document** edge-case, missing-value and perturbation tests.
6. Load `.joblib` files only from trusted sources. `joblib.load()` uses deserialization that can execute code.
7. Phase 5 also requires the Lab 2 `auth.log`; log-anomaly analysis is outside this baseline package.
8. Phase 1 (scoping) may use these files to define the audited teaching system, business context, preliminary risks and scope. The detailed Model Card belongs to Phase 2. Do not present the baseline as NorVenta's official production model.

## 8. Troubleshooting

| Issue | Suggested action |
|---|---|
| `py -3.13` is not found | Run `py install 3.13` and reopen the terminal after installation. |
| pip downloads `pandas*.tar.gz` and requests Visual Studio | Check that you are using Python 3.13 and `--only-binary=:all:`. |
| `Application Control policy has blocked this file` | A Windows security policy blocked a binary. Do not disable security controls blindly; investigate the policy or use a course-approved environment. |
| SHA-256 verification fails | The CSV or `.joblib` was changed. Restore matching files from the same project version rather than bypassing the check. |
| Output files are missing | Look in `outputs/` next to the Python script. |
| An old folder still includes `audit_edge_cases.csv`, etc. | Those are leftover outputs from an earlier package, not Baseline V2. Extract into a fresh directory rather than mixing versions. |

**Final reminder:** The 400 held-out TEST rows have provided labels and support baseline performance evaluation. The 100 synthetic cases have no true labels and are useful only for demonstrating predictions. Full audits for Phases 3, 4 and 5 are still pending.
