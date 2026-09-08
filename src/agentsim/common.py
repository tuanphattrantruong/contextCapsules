"""common.py — run metadata capture and results-file conventions (DESIGN.md 8).

Every results file written by agentsim experiments has the shape:

    {"meta": {...run metadata...}, "data": {...}}

where meta embeds the LITERAL output of `date -u` and `uname -a` (captured via
subprocess on this machine, never fabricated) plus seeds/iterations/arch/timer,
and every numeric leaf under data is grouped as "measured" or "modeled".

Failures are recorded honestly: on an unhandled exception an experiment writes
{"status": "failed", "error": <traceback>} into its result file and exits nonzero.
No plausible number is ever substituted for a failed measurement.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable


def _cmd_output(argv: list[str]) -> str:
    """Return literal stdout of a subprocess, stripped; or an explicit error marker.

    Never raises — a failed metadata command is recorded as its error text so the
    honesty contract (real output or an explicit note) holds.
    """
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False
        )
        if out.returncode != 0:
            return f"<{argv[0]} exit {out.returncode}: {out.stderr.strip()}>"
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001 - metadata capture must not crash the run
        return f"<{argv[0]} failed: {exc!r}>"


def run_meta(experiment: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the run-metadata dict for a results file.

    Captures literal `date -u` and `uname -a` output on this machine, plus the
    Python version, machine arch, timer identity, and whatever seeds/iterations
    the caller passes in `extra`.
    """
    meta: dict[str, Any] = {
        "experiment": experiment,
        "date_u": _cmd_output(["date", "-u"]),
        "uname_a": _cmd_output(["uname", "-a"]),
        "arch": platform.machine(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "timer": "time.perf_counter_ns (Python monotonic ns)",
        "cmd": " ".join([sys.executable.split("/")[-1]] + sys.argv),
        "content_note": (
            "no git; content bytes for real-C runs derived from "
            "hashlib.sha256-seeded numpy RNG (stable across runs)"
        ),
    }
    if extra:
        meta.update(extra)
    return meta


def write_result(path: str | os.PathLike[str], meta: dict[str, Any],
                 data: dict[str, Any]) -> None:
    """Write {"meta":meta, "data":data} to `path` atomically (tmp + rename).

    Enforces the results-file convention. `data` is expected to carry its numeric
    leaves under "measured"/"modeled" groups (the caller's responsibility; this
    function does not reshape data, it only validates the envelope shape).
    Refuses to silently corrupt: writes to a temp file in the same directory and
    atomically renames over the target.
    """
    if not isinstance(meta, dict):
        raise TypeError("meta must be a dict")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "data": data}
    _atomic_write_json(target, payload)


def write_failure(path: str | os.PathLike[str], experiment: str,
                  error: str | BaseException,
                  meta: dict[str, Any] | None = None) -> None:
    """Write a failure record to `path` (atomic). NEVER fabricates numbers.

    If `error` is an exception the full traceback is captured; if it is a string
    it is stored verbatim.
    """
    if isinstance(error, BaseException):
        err_text = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    else:
        err_text = str(error)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "status": "failed",
        "experiment": experiment,
        "error": err_text,
    }
    if meta is not None:
        record["meta"] = meta
    _atomic_write_json(target, record)


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    """Serialize `payload` to JSON and atomically place it at `target`."""
    text = json.dumps(payload, indent=2, sort_keys=False, default=_json_default)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)  # atomic rename on same filesystem
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy scalars/arrays and bytes so results serialize cleanly."""
    # numpy scalar/array support without importing numpy at module top (keep light).
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def guard_main(experiment: str, result_path: str | os.PathLike[str],
               fn: Callable[[], None]) -> None:
    """Run `fn`; on any exception write a failure record and exit nonzero.

    This wires the honesty failure convention into every experiment's __main__:
    a crash produces {"status":"failed","error":traceback} at result_path and a
    nonzero process exit, never a partially-fabricated results file.
    """
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - deliberate: record then re-exit
        try:
            write_failure(result_path, experiment, exc)
        finally:
            traceback.print_exc()
            sys.exit(1)
