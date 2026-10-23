import json
from sentinelml.data import generate_dataset
from sentinelml.training import train_model

def test_train_candidate_and_promote(tmp_path, monkeypatch):
    import sentinelml.artifacts as artifacts
    monkeypatch.setattr(artifacts, "MODEL_STORE", tmp_path / "models")
    monkeypatch.setattr(artifacts, "ACTIVE_MODEL_POINTER", tmp_path / "active_model.json")
    metadata = train_model(generate_dataset(n_samples=400), output_dir=tmp_path / "candidate",
                           minimum_roc_auc=0.5)
    assert metadata["metrics"]["roc_auc"] >= 0.5
    assert (tmp_path / "active_model.json").exists()
    pointer = json.loads((tmp_path / "active_model.json").read_text())
    assert pointer["model_version"] == metadata["model_version"]

def test_quality_gate_rejects(tmp_path, monkeypatch):
    import sentinelml.artifacts as artifacts
    monkeypatch.setattr(artifacts, "MODEL_STORE", tmp_path / "models")
    monkeypatch.setattr(artifacts, "ACTIVE_MODEL_POINTER", tmp_path / "active_model.json")
    try:
        train_model(generate_dataset(n_samples=400), output_dir=tmp_path / "candidate",
                    minimum_roc_auc=1.01)
    except ValueError as exc:
        assert "Promotion denied" in str(exc)
    else:
        raise AssertionError("Expected quality gate to reject candidate")
    assert not (tmp_path / "active_model.json").exists()
