import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from sentinelml.config import ACTIVE_MODEL_POINTER, MODEL_STORE

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def promote_candidate(candidate_dir: Path, *, metric_name: str, minimum: float) -> dict:
    '''Promote a candidate only when its recorded metric passes policy.'''
    metadata_path = candidate_dir / "metadata.json"
    model_path = candidate_dir / "model.joblib"
    baseline_path = candidate_dir / "baseline.json"
    if not all(path.exists() for path in (metadata_path, model_path, baseline_path)):
        raise ValueError("Candidate package is incomplete; promotion denied.")
    metadata = _read_json(metadata_path)
    metric = metadata.get("metrics", {}).get(metric_name)
    if not isinstance(metric, (int, float)) or not (metric >= minimum):
        raise ValueError(f"Promotion denied: {metric_name}={metric!r} is below required minimum {minimum}.")
    MODEL_STORE.mkdir(parents=True, exist_ok=True)
    final_dir = MODEL_STORE / metadata["model_version"]
    if final_dir.exists():
        raise FileExistsError(f"Model version already exists: {final_dir.name}")
    shutil.copytree(candidate_dir, final_dir)
    pointer = {
        "model_version": metadata["model_version"],
        "model_path": str((final_dir / "model.joblib").relative_to(ACTIVE_MODEL_POINTER.parent)),
        "metadata_path": str((final_dir / "metadata.json").relative_to(ACTIVE_MODEL_POINTER.parent)),
        "baseline_path": str((final_dir / "baseline.json").relative_to(ACTIVE_MODEL_POINTER.parent)),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promotion_policy": {"metric": metric_name, "minimum": minimum, "observed": metric},
    }
    ACTIVE_MODEL_POINTER.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_MODEL_POINTER.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    return pointer

def active_paths() -> dict:
    if not ACTIVE_MODEL_POINTER.exists():
        raise FileNotFoundError("No active model. Run `python -m sentinelml.cli train` first.")
    pointer = _read_json(ACTIVE_MODEL_POINTER)
    base = ACTIVE_MODEL_POINTER.parent
    return {
        "version": pointer["model_version"],
        "model": base / pointer["model_path"],
        "metadata": base / pointer["metadata_path"],
        "baseline": base / pointer["baseline_path"],
        "pointer": pointer,
    }
