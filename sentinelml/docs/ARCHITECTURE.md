# Architecture and design decisions

## Quality-gated model lifecycle
Training produces a candidate package in `artifacts/candidates/<version>`. The package is evaluated on a held-out split and includes model, metadata, and baseline. `artifacts.py` checks the recorded ROC-AUC against a configurable minimum before copying the candidate into the version store and updating `active_model.json`.

A failed quality gate does not update the active pointer. The current implementation demonstrates the invariant, but it is not a fully transactional registry and does not include rollback commands or concurrent promotion locking.

## Drift monitoring
The service compares current feature distributions to a training reference using PSI. Scores are heuristic and should be interpreted alongside newly labelled outcomes.

## Deployment boundaries
Local files and in-process metrics keep the starter easy to run. They are not suitable for multi-replica production. A real deployment should use durable artifact storage, a governed registry, authentication, shared telemetry, and secrets management.
