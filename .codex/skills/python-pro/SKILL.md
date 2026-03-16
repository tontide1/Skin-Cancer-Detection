---
name: python-pro
description: Write, review, and modernize Python codebases using current packaging, typing, async, API, and tooling patterns. Use when designing Python services, refactoring Python architecture, or applying modern Python best practices to production code.
---

# Python Pro

## Quick Start

1. Confirm the Python version, runtime constraints, and existing tooling.
2. Prefer standard-library and simple designs before introducing framework-heavy abstractions.
3. Keep interfaces typed, module boundaries explicit, and failure modes obvious.
4. Implement tests and linting alongside code changes, not afterward.
5. Optimize only after the code is correct and measured.

## Workflow

### Shape the design

- Choose synchronous code by default; justify async with concrete concurrency needs.
- Use modern packaging and project configuration that matches the current repo.
- Keep public APIs stable and data models explicit.

### Implement cleanly

- Write code that is easy to test, profile, and reason about.
- Use type hints on public boundaries and clear exception handling.
- Favor composable functions and small modules over deeply coupled classes.

### Finish like production code

- Add tests for happy path, edge cases, and error handling.
- Check linting, formatting, and static analysis expectations used by the repo.
- Document assumptions, env vars, and operational constraints.

## Deliverables

- Production-ready Python code that matches the repo's style.
- The smallest set of tooling or architecture changes needed to solve the task.
