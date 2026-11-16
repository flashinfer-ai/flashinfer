# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import csv
import heapq
import math
import os
import warnings
from collections import Counter
from dataclasses import dataclass

from .launch import BACKEND_OF, LaunchShape, coef_rows, launch_config

#: Kernel -> the modes that price it -> the column measure.py times it in. The
#: membership is what INFER_KERNELS / TRAIN_KERNELS are; the columns are fit-only.
KERNEL_SOURCES = {
    "prepare_h": {"infer": "prepare_h", "train": "prepare_h_bidi"},
    "correct_h0": {"infer": "correct_h0", "train": "correct_h0"},
    "correct_dht": {"train": "correct_dht"},
    "fused_fwd": {"infer": "fused_fwd", "train": "fused_fwd"},
    "prepare_dh": {"train": "prepare_dh"},
    "recompute_h": {"train": "recompute_h"},
    "fused_bwd": {"train": "fused_bwd"},
}

KERNELS = tuple(KERNEL_SOURCES)
INFER_KERNELS = tuple(k for k, modes in KERNEL_SOURCES.items() if "infer" in modes)
TRAIN_KERNELS = tuple(k for k, modes in KERNEL_SOURCES.items() if "train" in modes)

CP_ONLY_KERNELS = ("prepare_h", "correct_h0", "correct_dht", "prepare_dh")

#: Kernel -> which run list its schedule depth is a makespan of. prep/bwd launch a
#: plain batch*H grid, fb/co are DV-tiled so their CTA count is ways * H.
KERNEL_FAMILY = {
    "prepare_h": "prep",
    "prepare_dh": "bwd",
    "correct_h0": "co",
    "correct_dht": "co",
    "fused_fwd": "fb",
    "recompute_h": "fb",
    "fused_bwd": "fb",
}

MIN_LCP = 4
WARMUP_TOL = -10.0

COEFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coefs")

#: Every fitted row obeys the same law: t = a * depth + b * u + c.
COEF_COLUMNS = ("a", "b", "c")


def predict_kernel(coefs: dict, feat: "StructFeatures", kernel: str) -> float:
    row = launch_config(coefs["backend"], kernel, feat.shape).row
    k = coefs["kernels"][row]
    return k["a"] * feat.depth[kernel] + k["b"] * feat.u + k["c"]


class AutocpInfo(UserWarning):
    """Non-fatal autocp heads-up (e.g. borrowing a same-backend sibling's
    coefficients). Subclasses UserWarning so a plain filter still silences it."""


def coefs_path_for(arch: str) -> str | None:
    if arch not in BACKEND_OF:
        raise ValueError(f"unknown arch {arch!r}, expected one of {sorted(BACKEND_OF)}")
    path = os.path.join(COEFS_DIR, f"{arch}.csv")
    if os.path.exists(path):
        return path
    # A sibling on the same backend launches identically, so its rows apply; only
    # the numbers are off. Across backends they would be meaningless.
    for other, backend in BACKEND_OF.items():
        if other == arch or backend != BACKEND_OF[arch]:
            continue
        alt = os.path.join(COEFS_DIR, f"{other}.csv")
        if os.path.exists(alt):
            warnings.warn(
                f"autocp: no {arch}.csv; using {other}'s coefficients "
                f"({backend} launches the same kernels, so the decision stays "
                f"valid though tuned on different silicon). Fit {arch}.csv for "
                f"exact timings.",
                AutocpInfo,
                stacklevel=2,
            )
            return alt
    return None


def is_calibrated(arch: str) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AutocpInfo)
        return coefs_path_for(arch) is not None


def load_coefs(path: str | None = None, *, arch: str) -> dict:
    explicit = path is not None
    if path is None:
        path = coefs_path_for(arch)
        if path is None:
            raise ValueError(f"autocp has no coefficients for {arch}")
    backend = BACKEND_OF[arch]
    kernels: dict[str, dict[str, float]] = {}
    P = chunk = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["kernel"].strip()
            if name in kernels:
                raise ValueError(f"coefficient CSV defines row {name!r} twice: {path}")
            tagged = row.get("arch", "").strip()
            if tagged and tagged != arch:
                # A borrowed same-backend sibling (auto-resolved by
                # coefs_path_for) legitimately carries another arch tag; an
                # explicit path or a cross-backend file is a real mixup.
                if explicit or BACKEND_OF.get(tagged) != backend:
                    raise ValueError(
                        f"coefficient CSV is fitted for {tagged!r} but was "
                        f"loaded as {arch!r}: {path}"
                    )
            kernels[name] = {
                c: float(row[c]) if row.get(c) else 0.0 for c in COEF_COLUMNS
            }
            P, chunk = int(row["P"]), int(row["chunk"])
    if P is None:
        raise ValueError(f"coefficient CSV has no rows: {path}")

    missing = [r for r in coef_rows(backend, INFER_KERNELS) if r not in kernels]
    if missing:
        raise ValueError(
            f"coefficient CSV is missing rows {missing} needed on {arch} "
            f"({backend}): {path}"
        )
    absent = [r for r in coef_rows(backend, KERNELS) if r not in kernels]
    if absent:
        warnings.warn(
            f"autocp: {path} has no rows {absent}; training-mode decisions on "
            f"{arch} will fall back to the heuristic path if they are priced.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "kernels": kernels,
        "P": P,
        "chunk": chunk,
        "arch": arch,
        "backend": backend,
    }


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------
def seq_chunks(cu_seqlens, chunk: int) -> list[int]:
    cu = list(cu_seqlens)
    return [math.ceil((cu[i + 1] - cu[i]) / chunk) for i in range(len(cu) - 1)]


def schedule_depth(runs, copies: int, P: int) -> float:
    P, copies = int(P), int(copies)
    if P < 1:
        raise ValueError(f"P (SM count) must be >= 1, got {P}")

    free = {0.0: P}  # free time -> how many SMs come free at it
    times = [0.0]  # min-heap over the distinct free times
    end = 0.0
    for d, count in runs:
        need = count * copies
        while need:
            t = times[0]
            n = free[t]
            k = n if n <= need else need
            if k == n:
                del free[t]
                heapq.heappop(times)
            else:
                free[t] = n - k
            t1 = t + d
            if t1 in free:
                free[t1] += k
            else:
                free[t1] = k
                heapq.heappush(times, t1)
            if t1 > end:
                end = t1
            need -= k
    return end


@dataclass(frozen=True)
class StructFeatures:
    S: int
    H: int
    P: int
    num_raw: int  # raw sequences
    num_partitions: int  # CP partitions (== cp_batch_size)
    u: float  # G = H * num_partitions, total CTAs
    depth: dict  # kernel -> schedule depth under the launch it takes

    @property
    def enable_cp(self) -> bool:
        return self.num_partitions > self.num_raw

    @property
    def shape(self) -> LaunchShape:
        return LaunchShape(H=self.H, n_part=self.num_partitions, P=self.P)


def warmup_from_gate_avg(
    avg_per_head, chunk: int, tol: float = WARMUP_TOL
) -> list[float]:
    out: list[float] = []
    for a in avg_per_head:
        if a >= -1e-12:
            out.append(math.inf)
        else:
            out.append(max(1, math.ceil(tol / (a * chunk))))
    return out


def struct_features(
    chunks,
    S: int,
    H: int,
    P: int,
    warmup_per_head,
    is_train: bool = False,
    *,
    backend: str,
) -> StructFeatures:
    if len(warmup_per_head) != H:
        raise ValueError(
            f"warmup_per_head has {len(warmup_per_head)} entries, expected H={H}"
        )

    fb, co = [], []
    for c in chunks:
        n = max(1, math.ceil(c / S))
        if n > 1:
            fb.append((S, n - 1))
        fb.append((c - S * (n - 1), 1))
        co.append((n, 1))
    n_part = sum(n for n, _ in co)  # corr depth *is* the sequence's partition count

    prep, bwd = [], []
    for depth, copies in Counter(min(k, S) for k in warmup_per_head).items():
        w = min(S, depth)
        for c in chunks:
            n = max(1, math.ceil(c / S))
            if not is_train:
                if n > 1:
                    prep.append((w, (n - 1) * copies))
                prep.append((0, copies))
                continue
            if n == 1:
                prep.append((0, copies))
                bwd.append((0, copies))
                continue
            rem = min(c - S * (n - 1), w)  # ragged partition can't warm past its len
            prep.append((w, (n - 1) * copies))
            prep.append((rem, copies))
            if n > 2:
                bwd.append((w, (n - 2) * copies))
            bwd.append((rem, copies))
            bwd.append((0, copies))

    runs = {"fb": fb, "co": co, "prep": prep, "bwd": bwd}
    shape = LaunchShape(H=H, n_part=n_part, P=P)
    memo: dict[tuple[str, int], float] = {}
    depths: dict[str, float] = {}
    for kernel, family in KERNEL_FAMILY.items():
        if family == "bwd" and not is_train:
            depths[kernel] = 0.0
            continue
        copies = (
            1
            if family in ("prep", "bwd")
            else launch_config(backend, kernel, shape).ways * H
        )
        key = (family, copies)
        if key not in memo:
            memo[key] = schedule_depth(runs[family], copies, P)
        depths[kernel] = memo[key]

    return StructFeatures(
        S=S,
        H=H,
        P=P,
        num_raw=len(chunks),
        num_partitions=n_part,
        u=H * n_part,
        depth=depths,
    )
