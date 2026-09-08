"""Lifecycle, isolation, concurrency, and reuse tests for the paged adapter."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
from agentsim.paged_adapter import AdapterError, BlockArena, PagedCapsuleAdapter  # noqa: E402


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except AdapterError:
        return True


def test_domain_and_revocation_fail_closed():
    a = PagedCapsuleAdapter(); cap = a.create("tenant:a", [b"secret"])
    assert _raises(lambda: a.attach(cap, "tenant:b"))
    a.revoke(cap, "tenant:a")
    assert _raises(lambda: a.attach(cap, "tenant:a"))
    a.destroy(cap, "tenant:a")
    assert a.arena.stats()["live_blocks"] == 0


def test_fork_copy_on_write_preserves_parent():
    a = PagedCapsuleAdapter(); parent = a.create("d", [b"a", b"b"])
    child = a.fork(parent, "d")
    shared = a.object_handles(parent, "d")[0]
    assert a.arena.refcount(shared) == 2
    a.write_cow(child, "d", 0, b"child")
    with a.attach(parent, "d") as pv, a.attach(child, "d") as cv:
        assert pv.read(0) == b"a" and cv.read(0) == b"child"
    a.destroy(parent, "d"); a.destroy(child, "d")
    assert a.arena.stats()["live_blocks"] == 0


def test_generation_rejects_reused_slot():
    arena = BlockArena(); old = arena.allocate(b"old")
    arena.release(old); new = arena.allocate(b"new")
    assert old.slot == new.slot and old.generation != new.generation
    assert _raises(lambda: arena.read(old))


def test_concurrent_attach_revoke_and_release():
    a = PagedCapsuleAdapter(); cap = a.create("d", [bytes([i]) for i in range(32)])
    failures: list[Exception] = []
    barrier = threading.Barrier(9)

    def reader() -> None:
        try:
            barrier.wait()
            for _ in range(200):
                try:
                    with a.attach(cap, "d") as view:
                        assert view.read(7) == b"\x07"
                except AdapterError as exc:
                    assert "revoked" in str(exc)
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for thread in threads: thread.start()
    barrier.wait(); a.revoke(cap, "d")
    for thread in threads: thread.join()
    assert not failures
    a.destroy(cap, "d")
    assert a.arena.stats()["references"] == 0
