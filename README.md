# ast-matrix-scheduler

An AST-guided test impact analyzer and worker load-balancing pipeline. Analyzes patch hunks against target source code to identify affected test suites and balances test execution across multi-worker pools using constrained bin-packing heuristics.

## Overview

- **Statement-Level AST Diffing**: Locates enclosing functions, classes, and statements for changed lines across unified diffs.
- **Coverage & Dependency Tracing**: Maps execution spans, class attribute accesses, and fixture opens to candidate test identifiers.
- **Load Balancing**: Distributes candidate test workloads across 1 to 8 worker slots using an adaptive Longest Processing Time (LPT) scheduler to minimize the maximum worker load and startup cost.

## Project Structure

- `Dockerfile`: Standalone container configuration for reproducible execution.
- `selector.py`: Core impact analysis engine and multi-worker scheduler.
- `map.json.gz`: Compact precomputed dependency and coverage index.
- `Makefile`: Build and verification helpers.

## Configuration

The runner is configured via environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `DATA_DIR` | Directory containing `tests.json` and patches in `commits/` | `/data` |
| `SRC_DIR` | Pinned source tree for statement resolution | `/app/src` |
| `MAP_FILE` | Path to compressed coverage index | `/app/map.json.gz` |
