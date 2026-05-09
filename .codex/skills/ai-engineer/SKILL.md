---
name: ai-engineer
description: Build or review LLM features, RAG systems, agent workflows, multimodal AI flows, and production generative AI integrations. Use when the task involves model selection, retrieval design, tool use, evaluation, safety controls, or shipping AI features to production.
---

# AI Engineer

## Quick Start

1. Clarify the product goal, latency budget, cost limits, and failure tolerance.
2. Decide whether the problem needs prompting, retrieval, tools, fine-tuning, or a simpler non-LLM path.
3. Design the data flow, guardrails, and evaluation plan before coding.
4. Implement with observable inputs and outputs, typed boundaries, and rollback paths.
5. Validate with task-level tests, adversarial cases, and cost checks.

## Workflow

### Scope the system

- Define the user action, required context, and expected output format.
- Identify where hallucination, prompt injection, stale retrieval, or tool misuse can happen.
- Reject AI-first designs when deterministic logic is cheaper and safer.

### Design the architecture

- Choose the smallest model stack that meets quality targets.
- Separate orchestration, retrieval, prompting, tool calls, and post-processing.
- Prefer structured outputs and explicit schemas over free-form text whenever possible.
- Keep prompts, evaluation datasets, and feature flags versioned.

### Implement safely

- Redact secrets and sensitive data before external model calls.
- Add retries, timeouts, and fallback behavior around model and tool boundaries.
- Record prompts, model settings, and request metadata needed for debugging.
- Keep retrieval quality visible with top-k samples and grounding traces.

### Validate before shipping

- Test happy path, empty context, adversarial input, and malformed tool output.
- Measure latency, token cost, output quality, and regression risk together.
- Document operating limits, monitoring signals, and rollback conditions.

## Deliverables

- A concrete architecture recommendation or implementation plan.
- Production-ready code or diffs where appropriate.
- An evaluation and monitoring checklist tied to the feature.
