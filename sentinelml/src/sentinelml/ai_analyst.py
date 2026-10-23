import json
import os
from typing import Any


def analyze_incident(evidence: dict[str, Any]) -> dict[str, Any]:
    """Generate an evidence-grounded ML operations incident analysis.

    The LLM can summarize and recommend investigation steps only. It cannot
    change artifacts, promote models, trigger retraining, or deploy anything.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Copy .env.example values into your environment; never commit the real key.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    instructions = """You are SentinelML's ML operations incident analyst.
Analyze ONLY the supplied JSON evidence. Do not invent metrics, causes, logs, or facts.
Separate observed facts from hypotheses. If evidence is insufficient, say so.
Return concise Markdown with these headings: Summary, Evidence, Plausible Causes,
Recommended Checks, Retraining Decision. Every factual statement must refer to a
field/value in the supplied evidence. Do not recommend automatic deployment. Feature
drift alone is not proof of model quality degradation; request labelled outcomes before
claiming model performance has declined. Recommendations must be read-only and safe."""
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(evidence, ensure_ascii=False),
        max_output_tokens=900,
    )
    return {
        "model": model,
        "analysis": response.output_text,
        "policy": "Advisory only; no model promotion, retraining, or deployment actions are executed.",
    }
