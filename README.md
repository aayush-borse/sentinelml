# SentinelML — Reliable ML Delivery and Drift Intelligence

**A portfolio-grade MLOps system for reproducible training, model promotion, API serving, and evidence-based monitoring.**

SentinelML is a research-and-engineering project demonstrating how an ML model can be validated, trained, evaluated against a baseline, packaged, served through an API, and monitored for feature drift. It uses a synthetic dataset by default so the project can be run locally without external data access.

> **Scope honesty:** this is an engineering portfolio project, not a certified production platform. The default model is trained on synthetic data and must not be used for real decisions. The monitoring uses heuristic feature-drift signals; drift is not proof of model quality degradation.

## Why this project exists

A trained model alone is not a dependable ML service. A service also needs data contracts, reproducible training, evaluation gates, controlled artifact promotion, health endpoints, observability, and a process for investigating changes in production data.

SentinelML brings these concerns into one coherent codebase.

## Capabilities

- **Data contracts:** required feature/target columns, numeric checks, missing-value checks, finite-value checks, and useful validation errors.
- **Reproducible training:** deterministic dataset generation, fixed random seeds, train/test split, scikit-learn pipeline, and recorded evaluation metrics.
- **Quality gate:** a candidate model is compared with a configured minimum ROC-AUC before it can be promoted to the active artifact.
- **Model versioning:** immutable, timestamped model packages with metadata and a pointer to the active model.
- **API inference:** FastAPI endpoints with schema validation, bounded batch size, model information, and readiness checks.
- **Monitoring:** request/latency counters and PSI-based feature drift reports with transparent recommendations.
- **Operational workflow:** CLI commands, unit/integration tests, Docker, and GitHub Actions CI.
- **Carbon-inspired web console:** React + TypeScript UI using IBM Carbon React components, responsive navigation, dark/light themes, model metrics, inference playground, PSI drift report, model metadata, and optional AI incident analysis. The console reads live backend responses rather than inventing telemetry.
- **Audit trail:** each training run produces metadata containing its metrics, dataset size, timestamp, version, and validation report.

## Architecture

```text
                         SENTINELML
                             |
       +---------------------v----------------------+
       | Dataset contract + finite/numeric checks  |
       +---------------------+----------------------+
                             |
       +---------------------v----------------------+
       | Reproducible training and held-out testing |
       +---------------------+----------------------+
                             |
       +---------------------v----------------------+
       | Quality gate: candidate metrics vs policy  |
       +---------------------+----------------------+
                             |
       +---------------------v----------------------+
       | Versioned package + active-model pointer   |
       +---------------------+----------------------+
                             |
                     +-------v--------+
                     | FastAPI serving|
                     +-------+--------+
                             |
             +---------------+----------------+
             |                                |
      Request/latency metrics          PSI drift checks
             |                                |
             +---------------+----------------+
                             |
                  Evidence-based report
```

## Technology stack

- Frontend: React 18, TypeScript, Vite, IBM Carbon Design System, Recharts, Lucide icons
- Python 3.10+
- pandas and NumPy
- scikit-learn
- FastAPI and Pydantic
- joblib
- pytest and HTTPX
- Docker / Docker Compose
- GitHub Actions

## Quick start — Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run validation, train a candidate, evaluate it, and promote if it passes.
python -m sentinelml.cli train

# Run tests.
pytest -q

# Start API.
uvicorn sentinelml.api:app --reload
```

If `py -3.10` is not installed, use Python 3.11 with `py -3.11 -m venv .venv`.

Open `http://127.0.0.1:8000/docs`.


## Web console — Windows PowerShell

Run the backend and frontend in **two separate terminals**. The Vite development server proxies `/api/*` requests to `http://127.0.0.1:8000`, so local development does not require permissive CORS settings.

**Terminal 1 — backend** (from the repository root):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m sentinelml.cli train
uvicorn sentinelml.api:app --reload
```

**Terminal 2 — frontend**:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The console supports a persistent dark/light preference, live service status, auto-refreshing API counters, model-quality metrics, single-record predictions, a deterministic 30-row drift demo, model metadata, and optional AI analysis. AI analysis requires `OPENAI_API_KEY` in the backend terminal.

**Production build check:** from `frontend/`, run `npm run build`. This emits static assets to `frontend/dist/`. For a deployed frontend, configure `VITE_API_BASE_URL` to a reachable API URL and configure the API's CORS allowlist or route both applications through the same origin; the local Vite proxy is development-only.

### Frontend interaction and data notes

- IBM Carbon React components and Carbon spacing/typography conventions are used as the UI foundation; the app adds a restrained operations-console layout and theme-specific styling.
- Theme preference is stored in browser local storage. Reduced-motion preferences are respected.
- API counters are process-local and reset when the backend restarts. Session activity is browser-local and is not presented as a durable audit log.
- The drift page labels its generated demo batch; it does not imply that real production traffic is being monitored.
- The model's default training dataset is synthetic. Do not use the demo prediction for real-world decisions.

## CLI

```powershell
python -m sentinelml.cli validate
python -m sentinelml.cli train
python -m sentinelml.cli model-info
python -m sentinelml.cli monitor --input data/sample_inference.csv
```

`train` generates deterministic synthetic data, validates it, trains and evaluates a candidate, applies the quality gate, and records a versioned artifact. A candidate that fails the configured quality gate is not made active.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and model readiness |
| `GET /model-info` | Active model metadata and evaluation metrics |
| `POST /predict` | Batched predictions and probabilities |
| `POST /monitor/drift` | Drift report for a batch |
| `GET /metrics` | Basic process-local request/latency counters |

### Example prediction payload

```json
{
  "instances": [
    {
      "feature_0": 0.25,
      "feature_1": -0.4,
      "feature_2": 1.1,
      "feature_3": 0.7,
      "feature_4": -0.2,
      "feature_5": 0.9,
      "feature_6": 0.1,
      "feature_7": -0.6,
      "feature_8": 0.3,
      "feature_9": 1.2
    }
  ]
}
```

## Model promotion policy

A candidate is evaluated on a held-out test split. Promotion is denied when:
- validation fails;
- the evaluation metric is not finite;
- ROC-AUC is below `SENTINELML_MIN_ROC_AUC`.

Set a threshold before running training, for example:

```powershell
$env:SENTINELML_MIN_ROC_AUC = "0.70"
python -m sentinelml.cli train
```

This gate is a minimal example. Real promotion policies should consider confidence intervals, data leakage, segment-level metrics, fairness, calibration, and performance on temporally appropriate validation data.

## Drift monitoring

The service compares a current batch with a stored reference distribution using Population Stability Index (PSI). It reports per-feature scores and a recommendation. The reference summary is deliberately transparent.

PSI thresholds are heuristics:
- `< 0.10`: low
- `0.10–0.20`: moderate
- `>= 0.20`: significant

Feature drift is not equivalent to a loss in predictive performance. When labels become available, evaluate actual outcome metrics before deciding to retrain.

## Docker

```powershell
docker compose up --build
```

Then visit `http://127.0.0.1:8000/docs`.

## Repository layout

```text
sentinelml-final/
├── frontend/                 # Carbon-inspired React + TypeScript console
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── src/sentinelml/
│   ├── api.py
│   ├── artifacts.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── monitoring.py
│   ├── schemas.py
│   ├── training.py
│   └── validation.py
├── tests/
├── docs/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Limitations and next research questions

- Synthetic dataset only; replace it with a licensed, documented real dataset for a research-grade evaluation.
- Process-local metrics reset on restart and are not shared across replicas.
- Local artifacts are not a substitute for a governed registry or durable object storage.
- No authentication, rate limiting, or secrets management is included.
- PSI is a heuristic and does not estimate causal change or model quality loss.
- No LLM is used in the core path. A future AI-assisted incident analyst should be constrained to evidence from metrics/logs, return cited findings, and never promote or deploy a model without an explicit deterministic policy and human approval.

## Research/engineering extensions

1. Add MLflow for experiment tracking and registry-backed model promotion.
2. Add DVC for data versioning and dataset provenance.
3. Export metrics to Prometheus and build Grafana dashboards.
4. Add temporal and segment-level validation.
5. Add drift simulation experiments and measure detection false positives.
6. Add model rollback and artifact integrity checks.
7. Add cloud deployment with infrastructure as code.
8. Add a constrained LLM incident analyst that summarizes verified telemetry, links every claim to evidence, and recommends (but does not autonomously execute) actions.

## License

MIT. Review the licenses of any future dataset and third-party integration separately.
