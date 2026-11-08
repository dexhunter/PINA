"""Time Exclusion.sample on two overlapping rectangles.

The inputs are fixed. Each repeat reseeds torch and random, then asks for the
same 1500 points, so every repeat does the same work. Exclusion keeps points
that belong to exactly one of the two rectangles.
"""

from __future__ import annotations

import gc
import os
import random
import statistics
import sys
import time
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

import torch

torch.set_num_threads(1)

from pina.domain import CartesianDomain, Exclusion

N = 1500
WARMUP = 2
REPEATS = 9
INNER = 1


def build():
    left = CartesianDomain({"x": [0, 1], "y": [0, 1]})
    right = CartesianDomain({"x": [0.35, 1.2], "y": [0.15, 0.85]})
    return Exclusion([left, right])


def once(domain):
    torch.manual_seed(0)
    random.seed(0)
    points = domain.sample(N, mode="random")
    if points.shape != (N, 2) or list(points.labels) != ["x", "y"]:
        raise RuntimeError(
            f"unexpected sample {tuple(points.shape)} labels={list(points.labels)}"
        )
    return points


def main():
    domain = build()
    checksum = 0.0
    for _ in range(WARMUP):
        for _ in range(INNER):
            checksum = float(once(domain).sum())
    samples = []
    for _ in range(REPEATS):
        gc.collect()
        gc.disable()
        try:
            started = time.perf_counter()
            for _ in range(INNER):
                checksum = float(once(domain).sum())
            samples.append((time.perf_counter() - started) * 1000.0)
        finally:
            gc.enable()
    median_ms = statistics.median(samples)
    spread = (max(samples) - min(samples)) / median_ms * 100.0
    print(f"checksum={checksum:.8f}")
    print("samples_ms=" + ",".join(f"{sample:.4f}" for sample in samples))
    print(f"median_ms={median_ms:.4f}")
    print(f"spread_pct={spread:.4f}")


if __name__ == "__main__":
    main()
