import json
import time
from threading import Lock
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from sentinelml.artifacts import active_paths
from sentinelml.ai_analyst import analyze_incident
from sentinelml.config import FEATURE_NAMES
from sentinelml.monitoring import monitor_frame
from sentinelml.schemas import DriftRequest, PredictionRequest

app = FastAPI(title="SentinelML", description="Quality-gated ML delivery and drift intelligence API.", version="1.0.0")
_metrics = {"requests": 0, "prediction_requests": 0, "errors": 0, "total_latency_ms": 0.0}
_lock = Lock()

@app.on_event("startup")
def load_active_model():
    paths = active_paths()
    app.state.model = joblib.load(paths["model"])
    app.state.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    app.state.baseline_path = paths["baseline"]

@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    started = time.perf_counter()
    with _lock:
        _metrics["requests"] += 1
    try:
        return await call_next(request)
    except Exception:
        with _lock:
            _metrics["errors"] += 1
        raise
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        with _lock:
            _metrics["total_latency_ms"] += elapsed

@app.get("/health")
def health():
    loaded = hasattr(app.state, "model")
    return {"status": "ok" if loaded else "not_ready", "model_loaded": loaded}

@app.get("/model-info")
def model_info():
    if not hasattr(app.state, "metadata"):
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return app.state.metadata

@app.post("/predict")
def predict(payload: PredictionRequest):
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    frame = pd.DataFrame([x.ordered_values() for x in payload.instances], columns=FEATURE_NAMES)
    with _lock:
        _metrics["prediction_requests"] += 1
    try:
        model = app.state.model
        labels = model.predict(frame).astype(int).tolist()
        probs = model.predict_proba(frame)[:, 1].tolist()
        return {"model_version": app.state.metadata["model_version"], "predictions": [
            {"prediction": label, "probability_class_1": round(float(prob), 6)}
            for label, prob in zip(labels, probs)]}
    except Exception as exc:
        with _lock:
            _metrics["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}") from exc

@app.post("/monitor/drift")
def drift(payload: DriftRequest):
    if not hasattr(app.state, "baseline_path"):
        raise HTTPException(status_code=503, detail="Model baseline is not loaded.")
    frame = pd.DataFrame([x.ordered_values() for x in payload.instances], columns=FEATURE_NAMES)
    try:
        return monitor_frame(frame, app.state.baseline_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/ai/analyze")
def ai_analyze(payload: DriftRequest):
    """Ask an LLM to analyze drift evidence; advisory only, never acts on models."""
    if not hasattr(app.state, "metadata") or not hasattr(app.state, "baseline_path"):
        raise HTTPException(status_code=503, detail="Model metadata is not loaded.")
    frame = pd.DataFrame([x.ordered_values() for x in payload.instances], columns=FEATURE_NAMES)
    try:
        drift_report = monitor_frame(frame, app.state.baseline_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    evidence = {
        "active_model_metadata": app.state.metadata,
        "feature_drift_report": drift_report,
        "batch_size": len(frame),
    }
    try:
        return analyze_incident(evidence)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI analysis provider failed: {type(exc).__name__}") from exc

@app.get("/metrics")
def metrics():
    with _lock:
        result = dict(_metrics)
    result["average_latency_ms"] = round(result["total_latency_ms"] / result["requests"], 3) if result["requests"] else 0.0
    result["total_latency_ms"] = round(result["total_latency_ms"], 3)
    return result
