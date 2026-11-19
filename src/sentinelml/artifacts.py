import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from sentinelml.config import ACTIVE_MODEL_POINTER, ARTIFACTS_DIR, MODEL_STORE

try:
    import fcntl
except ImportError:  # Windows fallback: fcntl is POSIX-only.
    fcntl = None

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

@contextmanager
def _promote_lock():
    """Acquire an exclusive file lock around model promotion.

    On POSIX systems this uses fcntl.flock for cross-process mutual exclusion.
    On Windows fcntl is unavailable, so locking is a no-op (single-process safe).
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = ARTIFACTS_DIR / ".promote.lock"
    handle = open(lock_path, "a+")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

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
    with _promote_lock():
        MODEL_STORE.mkdir(parents=True, exist_ok=True)
        final_dir = MODEL_STORE / metadata["model_version"]
        if final_dir.exists():
            raise FileExistsError(f"Model version already exists: {final_dir.name}")
        shutil.copytree(candidate_dir, final_dir)
        pointer = {
            "model_version": metadata["model_version"],
            "model_path": (final_dir / "model.joblib").relative_to(ACTIVE_MODEL_POINTER.parent).as_posix(),
            "metadata_path": (final_dir / "metadata.json").relative_to(ACTIVE_MODEL_POINTER.parent).as_posix(),
            "baseline_path": (final_dir / "baseline.json").relative_to(ACTIVE_MODEL_POINTER.parent).as_posix(),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "promotion_policy": {"metric": metric_name, "minimum": minimum, "observed": metric},
        }
        ACTIVE_MODEL_POINTER.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publish: write to a sibling temp file then os.replace into place.
        tmp_path = ACTIVE_MODEL_POINTER.with_name(ACTIVE_MODEL_POINTER.name + ".tmp")
        tmp_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        os.replace(tmp_path, ACTIVE_MODEL_POINTER)
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
