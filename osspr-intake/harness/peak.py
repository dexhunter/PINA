"""Live heap high-water mark for one call.

At every Python line and return inside the call, read glibc ``mallinfo2``.
Large tensor blocks are mmap'd, so ``uordblks + hblkhd`` tracks them and falls
when they are freed. The result is bytes above the heap size at entry.
"""

from __future__ import annotations

import ctypes
import gc
import sys


class _Mallinfo2(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


_libc = ctypes.CDLL("libc.so.6")
_libc.mallinfo2.restype = _Mallinfo2


def heap_bytes() -> int:
    info = _libc.mallinfo2()
    return int(info.uordblks + info.hblkhd)


def peak_bytes(fn):
    """Return ``(peak_delta_bytes, result)`` for one call of ``fn``."""
    gc.collect()
    gc.disable()
    base = heap_bytes()
    best = base

    def tracer(frame, event, arg):
        nonlocal best
        if event in ("line", "return"):
            current = heap_bytes()
            if current > best:
                best = current
        return tracer

    sys.settrace(tracer)
    try:
        result = fn()
    finally:
        sys.settrace(None)
        gc.enable()
    return best - base, result
