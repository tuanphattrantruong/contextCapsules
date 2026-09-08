"""Executable paged-KV block-handle adapter for context capsules.

This CPU metadata prototype mirrors the allocator contract required by a
continuous-batching engine: generation-tagged handles, atomic refcounts,
capability/domain checks before attach, fork sharing, copy-on-write mutation,
revocation, destruction, and stale-handle rejection.  It intentionally does
not claim GPU DMA or scheduler integration.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass


class AdapterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BlockHandle:
    slot: int
    generation: int


@dataclass(slots=True)
class _Slot:
    generation: int = 0
    refs: int = 0
    data: bytes | None = None


class BlockArena:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._slots: list[_Slot] = []
        self._free: list[int] = []

    def allocate(self, data: bytes) -> BlockHandle:
        with self._lock:
            if self._free:
                index = self._free.pop()
                slot = self._slots[index]
                slot.generation += 1
            else:
                index = len(self._slots)
                slot = _Slot(generation=1)
                self._slots.append(slot)
            slot.data = bytes(data)
            slot.refs = 1
            return BlockHandle(index, slot.generation)

    def _get(self, handle: BlockHandle) -> _Slot:
        if handle.slot < 0 or handle.slot >= len(self._slots):
            raise AdapterError("invalid block slot")
        slot = self._slots[handle.slot]
        if slot.generation != handle.generation or slot.refs == 0:
            raise AdapterError("stale block handle")
        return slot

    def retain(self, handle: BlockHandle) -> None:
        with self._lock:
            self._get(handle).refs += 1

    def release(self, handle: BlockHandle) -> None:
        with self._lock:
            slot = self._get(handle)
            slot.refs -= 1
            if slot.refs == 0:
                slot.data = None
                self._free.append(handle.slot)

    def read(self, handle: BlockHandle) -> bytes:
        with self._lock:
            data = self._get(handle).data
            assert data is not None
            return data

    def refcount(self, handle: BlockHandle) -> int:
        with self._lock:
            return self._get(handle).refs

    def stats(self) -> dict[str, int]:
        with self._lock:
            live = sum(1 for x in self._slots if x.refs)
            return {"slots": len(self._slots), "live_blocks": live,
                    "free_slots": len(self._free),
                    "references": sum(x.refs for x in self._slots)}


@dataclass(slots=True)
class _Object:
    domain: str
    handles: list[BlockHandle]
    revoked: bool = False


class AttachedView:
    def __init__(self, arena: BlockArena, handles: tuple[BlockHandle, ...]):
        self._arena = arena
        self.handles = handles
        self._closed = False

    def read(self, index: int) -> bytes:
        if self._closed:
            raise AdapterError("view closed")
        return self._arena.read(self.handles[index])

    def close(self) -> None:
        if not self._closed:
            for handle in self.handles:
                self._arena.release(handle)
            self._closed = True

    def __enter__(self) -> "AttachedView":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class PagedCapsuleAdapter:
    """Capability-checked vectors over generation-tagged allocator handles."""

    def __init__(self, arena: BlockArena | None = None) -> None:
        self.arena = arena or BlockArena()
        self._lock = threading.RLock()
        self._objects: dict[str, _Object] = {}
        self._serial = 0

    def _token(self, domain: str) -> str:
        self._serial += 1
        return hashlib.sha256(f"{domain}:{self._serial}".encode()).hexdigest()

    def create(self, domain: str, blocks: list[bytes]) -> str:
        with self._lock:
            token = self._token(domain)
            self._objects[token] = _Object(
                domain, [self.arena.allocate(block) for block in blocks])
            return token

    def _authorize(self, token: str, domain: str) -> _Object:
        obj = self._objects.get(token)
        if obj is None:
            raise AdapterError("unknown capability")
        if obj.domain != domain:
            raise AdapterError("domain denied")
        if obj.revoked:
            raise AdapterError("capability revoked")
        return obj

    def attach(self, token: str, domain: str) -> AttachedView:
        with self._lock:
            obj = self._authorize(token, domain)
            retained: list[BlockHandle] = []
            try:
                for handle in obj.handles:
                    self.arena.retain(handle)
                    retained.append(handle)
            except Exception:
                for handle in retained:
                    self.arena.release(handle)
                raise
            return AttachedView(self.arena, tuple(retained))

    def fork(self, token: str, domain: str) -> str:
        with self._lock:
            parent = self._authorize(token, domain)
            for handle in parent.handles:
                self.arena.retain(handle)
            child = self._token(domain)
            self._objects[child] = _Object(domain, list(parent.handles))
            return child

    def extend(self, token: str, domain: str, blocks: list[bytes]) -> None:
        with self._lock:
            obj = self._authorize(token, domain)
            obj.handles.extend(self.arena.allocate(block) for block in blocks)

    def write_cow(self, token: str, domain: str, index: int,
                  data: bytes) -> None:
        with self._lock:
            obj = self._authorize(token, domain)
            old = obj.handles[index]
            replacement = self.arena.allocate(data)
            obj.handles[index] = replacement
            self.arena.release(old)

    def revoke(self, token: str, domain: str) -> None:
        with self._lock:
            self._authorize(token, domain).revoked = True

    def destroy(self, token: str, domain: str, allow_revoked: bool = True) -> None:
        with self._lock:
            obj = self._objects.get(token)
            if obj is None or obj.domain != domain:
                raise AdapterError("domain denied")
            if obj.revoked and not allow_revoked:
                raise AdapterError("capability revoked")
            del self._objects[token]
            for handle in obj.handles:
                self.arena.release(handle)

    def object_handles(self, token: str, domain: str) -> tuple[BlockHandle, ...]:
        with self._lock:
            return tuple(self._authorize(token, domain).handles)
