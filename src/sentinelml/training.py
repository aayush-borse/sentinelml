import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sentinelml.artifacts import promote_candidate
from sentinelml.config import ARTIFACTS_DIR, FEATURE_NAMES, MIN_ROC_AUC, MODEL_VERSION_PREFIX, RANDOM_STATE, TARGET_NAME
from sentinelml.data import generate_dataset
from sentinelml.validation import validate_frame

def _distribution_summary(frame: pd.DataFrame) -> dict:
    summary = {}
    for column in FEATURE_NAMES:
        values = frame[column].astype(float)
        summary[column] = {"quantiles": [float(x) for x in values.quantile(np.linspace(0, 1, 11)).tolist()],
                           "mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return summary

def train_model(frame: pd.DataFrame | None = None, output_dir: Path | None = None,
                minimum_roc_auc: float | None = None) -> dict:
    if frame is None:
        frame = generate_dataset()
    report = validate_frame(frame, require_target=True)
    if not report.valid:
        raise ValueError("Training data validation failed: " + "; ".join(report.errors))
    X, y = frame[FEATURE_NAMES].astype(float), frame[TARGET_NAME].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=180, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    version = f"{MODEL_VERSION_PREFIX}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    root = Path(output_dir) if output_dir else ARTIFACTS_DIR / "candidates" / version
    root.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, root / "model.joblib")
    metadata = {
        "model_version": version, "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier", "task": "synthetic_binary_classification",
        "features": FEATURE_NAMES, "training_rows": int(len(frame)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "metrics": metrics, "data_validation": report.to_dict(), "random_state": RANDOM_STATE,
        "dataset": {"type": "synthetic", "generator": "sklearn.datasets.make_classification"},
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    baseline = {"features": FEATURE_NAMES, "rows": int(len(X_train)),
        "distributions": _distribution_summary(X_train)}
    (root / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    minimum = MIN_ROC_AUC if minimum_roc_auc is None else minimum_roc_auc
    pointer = promote_candidate(root, metric_name="roc_auc", minimum=minimum)
    metadata["promotion"] = pointer
    return metadata
