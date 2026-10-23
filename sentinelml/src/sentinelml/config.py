import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_STORE = ARTIFACTS_DIR / "models"
ACTIVE_MODEL_POINTER = ARTIFACTS_DIR / "active_model.json"
FEATURE_NAMES = [f"feature_{i}" for i in range(10)]
TARGET_NAME = "target"
RANDOM_STATE = 42
MODEL_VERSION_PREFIX = "sentinelml"
MIN_ROC_AUC = float(os.getenv("SENTINELML_MIN_ROC_AUC", "0.60"))
