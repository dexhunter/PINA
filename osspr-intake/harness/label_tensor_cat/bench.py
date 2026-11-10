"""Allocation count for LabelTensor.cat along the labelled dimension.

PYTHONMALLOC=malloc makes each Python object a libc allocation. A small
preload counts malloc, calloc, realloc, and aligned allocation calls while
400 tensors of 4 labels are concatenated on dim 1. The tensors are built
before the count starts.
"""

from __future__ import annotations

import ctypes
import gc
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HARNESS = Path(__file__).resolve().parents[1]
SO = Path("/tmp/osspr-countalloc.so")
SRC = HARNESS / "countalloc.c"


def _ensure_counter() -> None:
    if os.environ.get("OSSPR_CAT_READY") == "1":
        return
    if not SO.exists() or SO.stat().st_mtime < SRC.stat().st_mtime:
        subprocess.check_call(
            ["gcc", "-shared", "-fPIC", "-O2", "-o", str(SO), str(SRC), "-ldl"]
        )
    env = os.environ.copy()
    env["PYTHONMALLOC"] = "malloc"
    env["LD_PRELOAD"] = str(SO)
    env["OSSPR_CAT_READY"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], env)


_ensure_counter()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(ROOT))

import torch

torch.set_num_threads(1)

from pina import LabelTensor

_lib = ctypes.CDLL(str(SO))
_lib.osspr_count.restype = ctypes.c_size_t

N_TENSORS = 400
ROWS = 30
WARMUP = 2
REPEATS = 7


def build():
    pieces = []
    for i in range(N_TENSORS):
        values = torch.arange(ROWS * 4, dtype=torch.float32).reshape(ROWS, 4) + i
        pieces.append(
            LabelTensor(values, [f"a{i}", f"b{i}", f"c{i}", f"d{i}"])
        )
    return pieces


def once(pieces):
    merged = LabelTensor.cat(pieces, dim=1)
    if merged.shape != (ROWS, N_TENSORS * 4) or len(merged.labels) != N_TENSORS * 4:
        raise RuntimeError(
            f"unexpected cat {tuple(merged.shape)} n_labels={len(merged.labels)}"
        )
    return merged


def count_once(pieces):
    gc.collect()
    _lib.osspr_reset()
    merged = once(pieces)
    _lib.osspr_stop()
    return int(_lib.osspr_count()), merged


def main():
    pieces = build()
    for _ in range(WARMUP):
        _, result = count_once(pieces)
        del result
    samples = []
    checksum = 0.0
    for _ in range(REPEATS):
        count, result = count_once(pieces)
        checksum = float(result.sum())
        del result
        samples.append(count)
    median = statistics.median(samples)
    spread = (max(samples) - min(samples)) / median * 100.0
    print(f"checksum={checksum:.8f}")
    print("samples_allocs=" + ",".join(str(sample) for sample in samples))
    print(f"median_allocs={median:.1f}")
    print(f"spread_pct={spread:.4f}")


if __name__ == "__main__":
    main()
