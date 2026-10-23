from fastapi.testclient import TestClient
from sentinelml.api import app
from sentinelml.data import generate_dataset
from sentinelml.training import train_model

def test_api_prediction(monkeypatch, tmp_path):
    import sentinelml.artifacts as artifacts
    monkeypatch.setattr(artifacts, "MODEL_STORE", tmp_path / "models")
    monkeypatch.setattr(artifacts, "ACTIVE_MODEL_POINTER", tmp_path / "active_model.json")
    # api imports active_paths by reference; it uses the artifacts module globals.
    train_model(generate_dataset(n_samples=400), output_dir=tmp_path / "candidate",
                minimum_roc_auc=0.5)
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        from sentinelml.config import FEATURE_NAMES
        response = client.post("/predict", json={"instances": [{k: 0.1 for k in FEATURE_NAMES}]})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 1
