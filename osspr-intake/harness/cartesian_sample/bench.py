"""Peak live heap of CartesianDomain.sample on a 6-D box.

Each repeat reseeds torch and draws the same 200,000 points. The traced peak
is bytes above the heap at entry, from glibc mallinfo2.
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
from pina.domain import CartesianDomain

N = 200_000
WARMUP = 2
REPEATS = 7


def build():
    return CartesianDomain({f"v{i}": [0.0, 1.0] for i in range(6)})


def once(domain):
    torch.manual_seed(0)
    points = domain.sample(N, mode="random")
    if points.shape != (N, 6) or list(points.labels) != [f"v{i}" for i in range(6)]:
        raise RuntimeError(
            f"unexpected sample {tuple(points.shape)} labels={list(points.labels)}"
        )
    return points


def main():
    domain = build()
    for _ in range(WARMUP):
        _, result = peak_bytes(lambda: once(domain))
        del result
    samples = []
    checksum = 0.0
    for _ in range(REPEATS):
        delta, result = peak_bytes(lambda: once(domain))
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
