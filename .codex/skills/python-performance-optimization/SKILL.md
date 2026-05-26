---
name: python-performance-optimization
description: Profile and optimize Python code using measurement-first workflows for CPU, memory, I/O, and concurrency bottlenecks. Use when a Python service, script, pipeline, or test suite is too slow or too memory-hungry and needs targeted optimization.
---

# Python Performance Optimization

## Quick Start

1. State the latency, throughput, or memory target before changing code.
2. Measure the current behavior with the right profiler for the bottleneck.
3. Fix the dominant cost first instead of micro-optimizing everything.
4. Re-measure after each change to confirm the win.
5. Keep readability unless the measured gain justifies extra complexity.

## Workflow

### Measure correctly

- Distinguish CPU-bound, I/O-bound, allocation-heavy, and lock-contention problems.
- Use realistic inputs and representative workloads.
- Compare wall-clock time, call counts, allocation hotspots, and query counts together.

### Optimize deliberately

- Prefer better algorithms and data movement reduction before local code tweaks.
- Remove redundant work, repeated queries, and unnecessary serialization.
- Use vectorization, batching, caching, or concurrency only when profiling says they help.

### Verify the result

- Confirm the improvement with the same benchmark setup.
- Check correctness, tail latency, memory growth, and operational side effects.

## Open References When Needed

- `references/implementation-playbook.md` for profiler choices, examples, and optimization patterns.
