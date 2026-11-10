"""Peak live heap of LabelTensor.append(..., mode="cross").

The two inputs stay allocated. Each repeat crosses a 500x4 tensor with a
500x3 tensor. The traced peak is the extra bytes above the heap at entry.
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    os.sched_setaffinity(0, {0})
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "osspr-intake" / "harness"))

import torch

torch.set_num_threads(1)

from peak import peak_bytes
from pina import LabelTensor

N1 = 500
N2 = 500
WARMUP = 2
REPEATS = 7


def build():
    left = LabelTensor(
        torch.arange(N1 * 4, dtype=torch.float32).reshape(N1, 4),
        ["a", "b", "c", "d"],
    )
    right = LabelTensor(
        torch.arange(N2 * 3, dtype=torch.float32).reshape(N2, 3),
        ["e", "f", "g"],
    )
    return left, right


def once(left, right):
    points = left.append(right, mode="cross")
    if points.shape != (N1 * N2, 7) or list(points.labels) != [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
    ]:
        raise RuntimeError(
            f"unexpected append {tuple(points.shape)} labels={list(points.labels)}"
        )
    return points


def main():
    left, right = build()
    for _ in range(WARMUP):
        _, result = peak_bytes(lambda: once(left, right))
        del result
    samples = []
    checksum = 0.0
    for _ in range(REPEATS):
        delta, result = peak_bytes(lambda: once(left, right))
        checksum = float(result.sum())
        del result
        samples.append(delta)
    median = statistics.median(samples)
    spread = (max(samples) - min(samples)) / median * 100.0
    print(f"checksum={checksum:.8f}")
    print("samples_bytes=" + ",".join(str(sample) for sample in samples))
    print(f"median_bytes={median:.1f}")
    print(f"spread_pct={spread:.4f}")


if __name__ == "__main__":
    main()
