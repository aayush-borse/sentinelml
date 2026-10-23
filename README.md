<div align="center">

<img src="docs/assets/sentinelml-banner.png" alt="SentinelML — Build, Validate, Deploy, Monitor" width="100%" />

Reliable ML delivery. Observable model behavior. Evidence-first operations.

A full-stack machine-learning engineering project for reproducible training, quality-gated model promotion, API inference, feature-drift analysis, and advisory incident investigation.

<br />








</div>

Overview

A model that performs well during training is not, by itself, a dependable ML service. A useful delivery workflow also needs input validation, repeatable experiments, evaluation gates, versioned artifacts, a serving interface, and a way to investigate changes in incoming data.

SentinelML brings those lifecycle concerns into one locally runnable codebase. It includes a Python ML pipeline and FastAPI service alongside a React and TypeScript operations console. The default workflow uses a deterministic synthetic classification dataset so the system can be explored without downloading a private or third-party dataset.

Scope and data note
SentinelML is an engineering and learning project, not a certified production platform. The default model is trained on synthetic data and is not suitable for real-world decisions. Feature drift is a signal to investigate—not proof that model performance has degraded.

Contents

Design goals

Capabilities

System architecture

Model lifecycle

Technology stack

Getting started

Web console

CLI reference

HTTP API

Configuration

Drift interpretation

Testing and quality checks

Docker

Repository layout

Current limitations

Roadmap

Contributing

License

Design goals

SentinelML is built around a few explicit engineering principles:

Validate before training. Reject malformed or invalid data early and explain what failed.

Make training repeatable. Use deterministic data generation and fixed random seeds for comparable local runs.

Separate evaluation from promotion. A candidate must pass a configured metric threshold before it becomes active.

Keep operational signals interpretable. Expose metrics and per-feature drift results rather than hiding the decision behind a single score.

Keep AI advisory-only. The incident analyst can summarize supplied evidence and suggest checks; it cannot promote, retrain, or deploy a model.

Be explicit about limitations. Synthetic data, local artifact storage, and process-local metrics are documented as constraints rather than presented as production infrastructure.

Capabilities

Area

Implemented behavior

Data validation

Required-column checks, numeric-value checks, missing-value checks, and finite-value validation

Training

Reproducible synthetic binary-classification dataset, train/test split, and scikit-learn pipeline

Evaluation

Accuracy, precision, recall, F1, and ROC-AUC metrics

Quality gate

Candidate promotion is blocked when validation fails or ROC-AUC is below SENTINELML_MIN_ROC_AUC

Artifact management

Versioned candidate/model artifacts, metadata, baseline information, and active-model pointer

Prediction service

FastAPI endpoint with request-schema validation and batch inference

Drift monitoring

Population Stability Index (PSI) scores for incoming feature distributions against a reference distribution

Operational metrics

Basic process-local request and latency counters

AI incident analysis

Optional OpenAI-powered, evidence-bounded analysis with read-only recommendations

Web console

React + TypeScript interface for overview metrics, predictions, drift reports, model metadata, and incident analysis

Engineering workflow

CLI, automated tests, Docker, Docker Compose, and GitHub Actions CI configuration

System architecture

                          SENTINELML
                              │
                 ┌────────────▼────────────┐
                 │ Dataset contract checks │
                 │ numeric / finite values  │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │ Reproducible training    │
                 │ held-out evaluation      │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │ Configurable quality gate│
                 │ ROC-AUC threshold        │
                 └────────────┬────────────┘
                              │ pass
                 ┌────────────▼────────────┐
                 │ Versioned model artifact │
                 │ metadata + active pointer│
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │ FastAPI inference service│
                 └──────┬───────────┬──────┘
                        │           │
              ┌─────────▼───┐  ┌────▼───────────┐
              │ Predictions │  │ PSI drift report│
              └─────────────┘  └────┬───────────┘
                                    │
                         ┌──────────▼──────────┐
                         │ Evidence-based report│
                         │ + optional AI review │
                         └─────────────────────┘

             React + TypeScript operations console
                   communicates with the API

The browser console is a presentation and interaction layer; model training, prediction, and drift calculations are performed by the Python service. The optional AI analyst receives evidence from the service and returns recommendations. It is not part of the model-promotion path.

For more detail, see docs/ARCHITECTURE.md and docs/ROADMAP.md.

Model lifecycle

Generate the reproducible sample dataset.

Validate the dataset contract and values.

Train a candidate model and evaluate it on a held-out split.

Compare the recorded ROC-AUC against the configured minimum.

Promote the candidate only if the validation and quality checks pass.

Persist model metadata and the active-model pointer.

Serve predictions and compare incoming feature distributions with the stored reference.

A failed quality gate should not replace the active-model pointer. The current implementation demonstrates this control, but it is not a fully transactional registry: promotion locking, durable remote storage, and automated rollback are not implemented.

Technology stack

Layer

Technologies

Language

Python 3.10+

ML and data

scikit-learn, pandas, NumPy, joblib

API

FastAPI, Pydantic, Uvicorn

Optional AI analysis

OpenAI Python SDK and Responses API

Frontend

React, TypeScript, Vite

UI and charts

IBM Carbon React components, Recharts, Lucide icons

Tests

pytest, HTTPX / FastAPI TestClient

Packaging and delivery

pyproject.toml, Docker, Docker Compose, GitHub Actions

Getting started

Prerequisites

Python 3.10 or newer (Python 3.10 or 3.11 is a good choice for a predictable local setup).

Node.js and npm for the web console.

Git to clone the repository.

Docker Desktop only if you want to use the container workflow.

1. Get the source

git clone <your-repository-url>
cd sentinelml

Replace <your-repository-url> with the URL of your GitHub repository.

2. Create the Python environment

Windows — Command Prompt

py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

Windows — PowerShell

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

The editable installation reads the package configuration in pyproject.toml. Installing the dev extra also installs the test dependencies.

3. Train and inspect a model

python -m sentinelml.cli validate
python -m sentinelml.cli train
python -m sentinelml.cli model-info

Training creates local artifacts under artifacts/. These generated files are intentionally excluded from Git by default.

4. Run the test suite

python -m pytest -q

5. Start the API

uvicorn sentinelml.api:app --reload

Open the interactive API documentation at http://127.0.0.1:8000/docs. Keep this terminal running while using the web console.

Web console

The frontend lives in frontend/ and uses Vite during local development. Open a second terminal from the repository root:

cd frontend
npm install
npm run dev

Open http://127.0.0.1:5173.

The local Vite configuration proxies /api requests to the backend at http://127.0.0.1:8000. Start the backend before testing live console features.

The console includes:

an overview of model/service information;

a prediction playground;

a drift-monitoring view using a clearly labelled demo batch;

model metadata;

optional AI-assisted incident analysis;

persistent dark/light theme preference.

Production frontend build

From frontend/:

npm run build

The local Vite proxy is for development. For a deployed frontend, configure VITE_API_BASE_URL to point to the reachable API and configure the API's CORS allowlist—or route both applications through a common origin.

CLI reference

Run commands from the repository root with the virtual environment activated.

Command

Purpose

python -m sentinelml.cli validate

Validate the configured sample data workflow

python -m sentinelml.cli train

Generate data, validate, train/evaluate a candidate, apply the quality gate, and write artifacts

python -m sentinelml.cli model-info

Display active model metadata and metrics

python -m sentinelml.cli monitor --input data/sample_inference.csv

Run drift monitoring against a CSV input, if that sample file is present

The monitor command requires a compatible CSV containing the expected feature columns. Use the CLI's validation output and the source schemas to understand the required input shape.

HTTP API

Start the service with uvicorn sentinelml.api:app --reload.

Method

Endpoint

Purpose

GET

/health

Service health and model readiness

GET

/model-info

Active model metadata and evaluation metrics

POST

/predict

Generate predictions and probabilities for a batch

POST

/monitor/drift

Compare submitted feature data with the reference distribution

POST

/ai/analyze

Generate an optional advisory incident analysis from supplied monitoring evidence

GET

/metrics

Return basic process-local request and latency counters

Use the interactive API documentation at /docs to inspect the exact request and response schemas for your current version.

Example prediction request

The default model expects ten numeric features named feature_0 through feature_9.

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

Submit this payload to POST /predict. Request and response details are available in /docs.

Configuration

Variable

Purpose

Default / behavior

SENTINELML_MIN_ROC_AUC

Minimum ROC-AUC required for model promotion

0.60

OPENAI_API_KEY

Enables the optional AI incident analyst

Required only for /ai/analyze; keep secret and out of Git

OPENAI_MODEL

OpenAI model used for incident analysis

gpt-5-mini

VITE_API_BASE_URL

API base URL for a deployed frontend

Configure for the target deployment; local development uses Vite proxying

Configure the promotion threshold

PowerShell

$env:SENTINELML_MIN_ROC_AUC = "0.70"
python -m sentinelml.cli train

Command Prompt

set SENTINELML_MIN_ROC_AUC=0.70
python -m sentinelml.cli train

The threshold is an example policy, not a universally correct quality target. Choose it based on the task, validation design, and consequences of errors.

Enable AI incident analysis (optional)

Set the key in the backend terminal before starting the API.

PowerShell

$env:OPENAI_API_KEY = "your_api_key_here"
$env:OPENAI_MODEL = "gpt-5-mini"
uvicorn sentinelml.api:app --reload

Do not paste a real key into source files, commit it to Git, or publish it in logs or screenshots. The analyst is intended to summarize supplied evidence, separate observations from hypotheses, and recommend read-only checks. It does not trigger retraining, promotion, or deployment.

Drift interpretation

SentinelML uses Population Stability Index (PSI) to compare the reference feature distribution with a submitted batch. The current thresholds are heuristic indicators:

PSI

Interpretation in this project

< 0.10

Low distribution shift

0.10–0.20

Moderate distribution shift

>= 0.20

Significant distribution shift

PSI depends on binning and the reference data. It does not establish causality or prove that prediction quality has fallen. Investigate the affected features and, when labelled outcomes become available, measure actual model performance before deciding whether retraining is justified.

Testing and quality checks

Run the tests with:

python -m pytest -q

The repository also includes a GitHub Actions workflow under .github/workflows/. Review the workflow file for the exact checks configured for CI. For a frontend build check, run npm run build from frontend/.

Docker

With Docker and the Compose plugin installed, run from the repository root:

docker compose up --build

Then open http://127.0.0.1:8000/docs. Docker is a local packaging path; it does not by itself provide cloud deployment, production security, or persistent multi-replica observability.

Repository layout

sentinelml/
├── .github/workflows/ci.yml    # CI workflow
├── docs/
│   ├── ARCHITECTURE.md         # Architecture and design boundaries
│   └── ROADMAP.md              # Planned engineering work
├── frontend/                   # React + TypeScript operations console
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── src/sentinelml/
│   ├── ai_analyst.py           # Advisory-only LLM analysis
│   ├── api.py                  # FastAPI routes and request metrics
│   ├── artifacts.py            # Versioned artifact and promotion logic
│   ├── cli.py                  # Command-line interface
│   ├── config.py               # Configuration
│   ├── data.py                 # Deterministic sample dataset
│   ├── monitoring.py           # Feature drift analysis
│   ├── schemas.py              # API request/response schemas
│   ├── training.py             # Training and evaluation
│   └── validation.py           # Data contract checks
├── tests/                      # Automated tests
├── .env.example                # Example environment configuration
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md

Current limitations

SentinelML intentionally keeps the local demo easy to run. The following are not currently provided as production guarantees:

Synthetic training data: results demonstrate the pipeline, not real-world predictive value.

Local artifact storage: no managed registry or remote object storage is configured.

Non-transactional promotion: no concurrent promotion lock, signed artifact verification, or automated rollback workflow.

Process-local metrics: counters reset when the service restarts and are not shared across replicas.

No API authentication or rate limiting: do not expose the demo API publicly without adding appropriate controls.

Heuristic drift alerts: feature distribution changes are not equivalent to model quality loss.

Optional external AI dependency: incident analysis requires an OpenAI API key and may incur API usage costs; it is not required for core training or inference.

No cloud deployment included: Docker support is for packaging and local execution, not evidence of a deployed cloud service.

Roadmap

The next engineering steps are tracked in docs/ROADMAP.md. Areas for further work include:

MLflow experiment tracking and model registry integration;

DVC-backed data versioning and lineage;

Prometheus metrics and Grafana dashboards;

atomic promotion, artifact signing, and rollback;

authentication, rate limiting, and structured logging;

temporal and segment-level evaluation, calibration, and fairness checks;

drift evaluation against labelled outcomes, including false-positive analysis;

cloud deployment and infrastructure as code;

structured, evaluated AI incident analysis with evidence references and latency/cost tracking.

These are roadmap items, not claims about current functionality.

Contributing

Contributions, bug reports, and design discussions are welcome. A useful contribution should make its behavior testable and document any new operational assumptions.

Open an issue describing the problem or proposed change.

Keep changes focused and preserve the separation between training, evaluation, promotion, serving, and monitoring.

Add or update tests for behavior changes.

Run python -m pytest -q and, for frontend changes, npm run build from frontend/.

Update the relevant documentation and call out limitations or migration steps.

Please do not submit secrets, private datasets, or generated virtual-environment files.

License

SentinelML is distributed under the MIT License. See LICENSE for details.

<div align="center">

Build models you can inspect. Ship changes you can explain.

</div>