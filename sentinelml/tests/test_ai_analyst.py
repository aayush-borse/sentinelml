import pytest
from sentinelml.ai_analyst import analyze_incident

def test_ai_analyst_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        analyze_incident({"example": True})
