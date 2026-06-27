"""Multi-seed benchmark orchestrator (placeholder).

Multi-seed mode is not part of this refactor's scope. This stub exists
so benchmark.py can import without NameError. Multi-seed sweep runs
should use --workers N in single-model mode for now.
"""
import sys


def benchmark_multi_seed(*args, **kwargs) -> int:
    print("Error: --model-dir (multi-seed) mode is not yet implemented in this refactor.", file=sys.stderr)
    print("Use single-model mode with --workers N for parallelism, or implement multi-seed separately.", file=sys.stderr)
    return 1