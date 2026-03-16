---
name: python-testing-patterns
description: Write and improve Python tests with pytest, fixtures, mocks, parametrization, and integration patterns. Use when adding coverage, fixing flaky tests, structuring test suites, or making Python changes safer through testing.
---

# Python Testing Patterns

## Quick Start

1. Identify the behavior under test and the failure you want the test to catch.
2. Choose the narrowest useful test level: unit, integration, or end-to-end.
3. Arrange inputs explicitly, keep assertions specific, and avoid incidental setup.
4. Mock only true boundaries such as networks, clocks, filesystems, or external services.
5. Run the relevant tests before and after the code change.

## Workflow

### Design the test

- Prefer behavior-focused tests over implementation snapshots.
- Use parametrization when the same behavior must hold across multiple cases.
- Keep fixtures small and composable so failures stay local.

### Avoid common failure modes

- Do not share mutable state across tests.
- Do not over-mock internal code paths that should be exercised for real.
- Make async tests, filesystem tests, and database tests explicit about cleanup.

### Finish with confidence

- Cover success cases, edge cases, and expected failures.
- Add regression tests for fixed bugs before closing the task.
- Keep the suite fast enough that developers will actually run it.

## Open References When Needed

- `references/implementation-playbook.md` for pytest patterns, examples, and fixture design guidance.
