"""Multi-seed benchmark orchestrator (placeholder).

Multi-seed mode is not part of this refactor's scope. This stub exists
so benchmark.py can import without NameError. Multi-seed sweep runs
should use --workers N in single-model mode for now.
"""


def benchmark_multi_seed(*args, **kwargs):
    print("benchmark_multi_seed: not yet implemented in refactor; "
          "use single-model --workers N mode instead.")
    return 0