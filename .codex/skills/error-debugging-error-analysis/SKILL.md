---
name: error-debugging-error-analysis
description: Investigate recurring errors, production incidents, and cross-service failures with structured root-cause analysis. Use when the task centers on stack traces, log correlation, observability gaps, incident reports, or reliability improvements.
---

# Error Analysis and Resolution

## Quick Start

1. Capture the failing symptom, affected systems, and timeline.
2. Group related errors so the investigation targets the incident, not a single log line.
3. Reproduce or narrow the failure with targeted experiments.
4. Prove the root cause with an evidence chain from symptom to trigger.
5. Recommend the fix, validation, and prevention work separately.

## Workflow

### Establish scope

- Identify user impact, error frequency, and environment boundaries.
- Separate primary failures from secondary noise and cascading symptoms.
- Check whether the incident aligns with a deployment, config change, or dependency event.

### Analyze the failure

- Start from the origin point in the stack trace or trace span, not the top-level symptom alone.
- Compare healthy and unhealthy executions for timing, inputs, and dependency behavior.
- Trace invalid state back to the earliest component that could have created it.

### Close the loop

- Propose the minimum safe fix for recovery.
- Add tests, instrumentation, or alerts that would catch the issue earlier next time.
- Document what evidence confirmed the diagnosis and what remains assumed.

## Open References When Needed

- `references/implementation-playbook.md` for deeper checklists, observability patterns, and RCA techniques.

## Guardrails

- Redact secrets and PII in shared logs or traces.
- Separate confirmed facts from inferred causes in the final report.
