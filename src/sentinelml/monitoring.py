import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sentinelml.config import FEATURE_NAMES
from sentinelml.validation import validate_frame

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    reference, current = reference[np.isfinite(reference)], current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        span = max(abs(float(reference[0])) * 1e-6, 1e-6)
        edges = np.array([float(reference.min()) - span, float(reference.max()) + span])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def monitor_frame(frame: pd.DataFrame, baseline_path: Path) -> dict[str, Any]:
    report = validate_frame(frame)
    if not report.valid:
        raise ValueError("Monitoring data validation failed: " + "; ".join(report.errors))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    results = {}
    for feature in FEATURE_NAMES:
        ref = np.asarray(baseline["distributions"][feature]["quantiles"], dtype=float)
        score = _psi(ref, frame[feature].astype(float).to_numpy())
        level = "significant" if score >= 0.20 else "moderate" if score >= 0.10 else "low"
        results[feature] = {"psi": round(score, 5), "level": level}
    significant = [k for k, v in results.items() if v["level"] == "significant"]
    moderate = [k for k, v in results.items() if v["level"] == "moderate"]
    status = "drift_detected" if significant else "review" if moderate else "no_material_drift_flagged"
    recommendation = (
        "Investigate significant feature drift; inspect the data pipeline and evaluate on newly labelled outcomes before retraining."
        if significant else
        "Review moderate shifts and monitor labelled performance; drift alone does not prove quality loss."
        if moderate else
        "No material drift flagged by this heuristic. Continue monitoring and verify quality when labels are available."
    )
    return {"rows": int(len(frame)), "features": results, "summary": {
        "significant_features": significant, "moderate_features": moderate,
        "status": status, "recommendation": recommendation},
        "method": "Population Stability Index (PSI), reference-quantile bins",
        "caveat": "Heuristic monitoring only; feature drift is not equivalent to model performance degradation."}
