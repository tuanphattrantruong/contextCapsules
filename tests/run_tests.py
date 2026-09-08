#!/usr/bin/env python3
"""run_tests.py — integration test runner for the Context Capsules prototype.

Discovers every `test_*` function in the tests/test_*.py modules, runs each,
and reports a pass/fail summary. Exit status is nonzero if any test fails, so it
can gate CI / run_all.sh. No third-party test framework required (plain asserts).

Requires the dylib built first:  make -C src/libcapsule
Usage:  python3 tests/run_tests.py
"""

from __future__ import annotations

import importlib.util
import shutil
import traceback
from pathlib import Path

HERE = Path(__file__).resolve()
TEST_FILES = ["test_capsule_ffi.py", "test_managers.py", "test_paged_adapter.py"]
# test scratch stores live here (inside the project); cleaned before and after.
_ITEST_DIR = HERE.parents[1] / "experiments" / "tmp_store_py" / "itests"


def _cleanup() -> None:
    if _ITEST_DIR.exists():
        shutil.rmtree(_ITEST_DIR, ignore_errors=True)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    _cleanup()
    total = passed = failed = 0
    failures = []
    for fname in TEST_FILES:
        mod = _load(HERE.parent / fname)
        tests = sorted(n for n in dir(mod) if n.startswith("test_")
                       and callable(getattr(mod, n)))
        print(f"\n=== {fname} ({len(tests)} tests) ===")
        for name in tests:
            total += 1
            try:
                getattr(mod, name)()
                passed += 1
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append((fname, name, exc))
                print(f"  FAIL  {name}: {exc!r}")
                traceback.print_exc()

    _cleanup()
    print(f"\n=== {passed}/{total} passed, {failed} failed ===")
    if failures:
        print("failures:")
        for f, n, e in failures:
            print(f"  {f}::{n}: {e!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
