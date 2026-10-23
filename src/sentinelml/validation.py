from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from sentinelml.config import FEATURE_NAMES, TARGET_NAME

@dataclass
class ValidationReport:
    valid: bool
    rows: int
    missing_values: int
    non_numeric_values: int
    duplicate_rows: int
    missing_columns: list[str]
    unexpected_columns: list[str]
    errors: list[str]
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def validate_frame(frame: pd.DataFrame, *, require_target: bool = False,
                   allow_extra_columns: bool = False) -> ValidationReport:
    errors = []
    expected = set(FEATURE_NAMES + ([TARGET_NAME] if require_target else []))
    actual = set(frame.columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected) if not allow_extra_columns else []
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    if unexpected:
        errors.append(f"Unexpected columns: {', '.join(unexpected)}")
    relevant = [c for c in FEATURE_NAMES + ([TARGET_NAME] if require_target else []) if c in frame]
    missing_values = int(frame[relevant].isna().sum().sum()) if relevant else 0
    non_numeric = 0
    for col in [c for c in FEATURE_NAMES if c in frame]:
        original_missing = int(frame[col].isna().sum())
        converted = pd.to_numeric(frame[col], errors="coerce")
        non_numeric += max(0, int(converted.isna().sum()) - original_missing)
        if not np.isfinite(converted.dropna().to_numpy(dtype=float)).all():
            errors.append(f"Column '{col}' contains infinite values.")
    if missing_values:
        errors.append(f"Found {missing_values} missing values.")
    if non_numeric:
        errors.append(f"Found {non_numeric} non-numeric feature values.")
    if frame.empty:
        errors.append("Dataset is empty.")
    return ValidationReport(not errors, int(len(frame)), missing_values, non_numeric,
        int(frame.duplicated().sum()) if len(frame) else 0, missing, unexpected, errors)
