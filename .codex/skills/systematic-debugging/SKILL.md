---
name: systematic-debugging
description: Investigate bugs, test failures, flaky behavior, and regressions by proving root cause before fixing anything. Use when a failure is not fully understood, quick patches are tempting, or previous fixes have not held.
---

# Systematic Debugging

## Iron Law

Do not propose or implement fixes until the failure mechanism is understood well enough to explain why the bug happens.

## Workflow

### Phase 1: Establish the facts

1. Read the exact error, stack trace, and failing output.
2. Reproduce the problem reliably or narrow the conditions until the uncertainty is explicit.
3. Check recent code, config, dependency, and environment changes.
4. Trace the bad state or wrong output back toward its source.

### Phase 2: Compare against a working path

1. Find a passing test, healthy request, or known-good code path.
2. Compare inputs, state transitions, timing, and side effects.
3. Write down the smallest set of differences that could explain the failure.

### Phase 3: Test one hypothesis at a time

1. State one hypothesis in plain language.
2. Run the smallest experiment that can disprove it.
3. If it fails, stop and form a new hypothesis from the new evidence.

### Phase 4: Fix and verify

1. Add or identify a reproducible test.
2. Fix the source of the issue, not the downstream symptom.
3. Re-run the relevant tests and verification steps.
4. Add validation or instrumentation when the bug can recur silently.

## Open References When Needed

- `references/root-cause-tracing.md` for tracing failures backward through the call chain.
- `references/defense-in-depth.md` for adding validation after the root cause is known.
- `references/condition-based-waiting.md` for replacing arbitrary sleeps with condition polling.
- `scripts/find-polluter.sh` for isolating test pollution.
- `scripts/condition-based-waiting-example.ts` for a waiting helper example.

## Guardrails

- Do not stack multiple speculative fixes together.
- Do not keep a failed fix in place "just in case."
- Do not treat time pressure as evidence.
