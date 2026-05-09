---
name: debugging-toolkit-smart-debug
description: Triage complex bugs with logs, traces, metrics, profilers, and targeted experiments. Use when a failure spans multiple components, needs observability data, or requires a structured debugging report instead of ad-hoc guesses.
---

# Smart Debug Toolkit

## Quick Start

1. Capture the exact symptom, scope, and impact.
2. Collect the smallest set of logs, traces, metrics, and recent changes that can separate hypotheses.
3. Rank likely causes by evidence, not intuition.
4. Run targeted experiments that falsify one hypothesis at a time.
5. Propose fixes only after the failure mechanism is clear.

## Workflow

### Triage the incident

- Record the failing path, environment, first-seen time, and blast radius.
- Preserve exact error messages, stack traces, and correlation identifiers.
- Note whether the issue is deterministic, intermittent, load-related, or config-specific.

### Gather observability evidence

- Pull structured logs around the failing request or job.
- Inspect traces for latency cliffs, fan-out, retries, and timeouts.
- Check metrics for regressions tied to deployments, feature flags, or traffic shifts.
- Compare healthy and failing executions whenever a working example exists.

### Drive debugging with hypotheses

- Write down 3 or fewer plausible causes.
- For each cause, state what evidence would prove or disprove it.
- Prefer temporary instrumentation and reversible probes over broad code churn.

### Report findings clearly

- Summarize the symptom, root cause, fix, validation, and prevention steps.
- Call out residual risk and what remains unproven.

## Guardrails

- Do not prescribe production changes without a rollback path.
- Do not hide uncertainty; say what is known, inferred, and missing.
- Do not replace evidence with generic observability checklists.
