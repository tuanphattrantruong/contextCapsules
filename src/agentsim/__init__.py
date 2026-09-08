"""agentsim — Context Capsules workload simulator (package 2).

Trace-driven multi-tenant agent-swarm simulator that replays workloads against
three KV-state managers (no-share, vLLM-style prefix cache, capsules) and drives
the real libcapsule C library via ctypes for validation and security studies.

All modules here are Python 3.14 + numpy (stdlib otherwise). They code only against
the FROZEN C ABI in DESIGN.md section 2.7 and the dylib at
src/libcapsule/libcapsule.dylib; no C sources are read.

Honesty labeling rule (DESIGN.md section 8): every numeric leaf in a results file
is grouped under "measured" (from timers / footprint APIs / real counters) or
"modeled" (token counts multiplied by the documented KV-geometry constants).
"""

__all__ = [
    "kvmodel",
    "traces",
    "managers",
    "capsule_ffi",
    "metrics",
    "common",
]

__version__ = "1.0.0"
