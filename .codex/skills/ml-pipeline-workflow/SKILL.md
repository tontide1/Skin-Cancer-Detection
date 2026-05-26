---
name: ml-pipeline-workflow
description: Design and automate end-to-end ML pipelines from data ingestion through training, validation, deployment, and retraining. Use when creating DAGs, reproducible training workflows, scheduled model jobs, or MLOps automation across multiple stages.
---

# ML Pipeline Workflow

## Quick Start

1. Define the pipeline stages, owners, triggers, and promotion gates.
2. Make every stage explicit about inputs, outputs, and artifact versions.
3. Keep orchestration separate from feature, training, and serving logic.
4. Add data quality checks and rollback paths before automating deployment.
5. Validate that the pipeline can be rerun, backfilled, and debugged stage by stage.

## Workflow

### Model the pipeline

- Split the flow into ingestion, validation, feature building, training, evaluation, packaging, deployment, and monitoring.
- Keep steps idempotent where possible so retries are safe.
- Persist metadata needed to trace a model back to code, config, and data.

### Add control points

- Fail fast on schema mismatches and stale or incomplete data.
- Gate promotion on metric thresholds, smoke tests, and serving readiness.
- Reserve manual approval for risky or user-visible promotions.

### Operate the pipeline

- Design for partial re-runs and backfills.
- Capture logs and stage artifacts for every failure.
- Separate transient retries from deterministic failures that need investigation.

## Deliverables

- A stage-by-stage workflow definition.
- Artifact, lineage, and promotion rules.
- An operating plan for retries, rollback, monitoring, and retraining.
