from sentinelml.config import FEATURE_NAMES
from sentinelml.data import generate_dataset
from sentinelml.monitoring import monitor_frame
from sentinelml.training import train_model

def test_monitor_report(tmp_path, monkeypatch):
    import sentinelml.artifacts as artifacts
    monkeypatch.setattr(artifacts, "MODEL_STORE", tmp_path / "models")
    monkeypatch.setattr(artifacts, "ACTIVE_MODEL_POINTER", tmp_path / "active_model.json")
    metadata = train_model(generate_dataset(n_samples=400), output_dir=tmp_path / "candidate",
                           minimum_roc_auc=0.5)
    baseline = tmp_path / "models" / metadata["model_version"] / "baseline.json"
    report = monitor_frame(generate_dataset(n_samples=100)[FEATURE_NAMES], baseline)
    assert set(report["features"]) == set(FEATURE_NAMES)
    assert "recommendation" in report["summary"]
