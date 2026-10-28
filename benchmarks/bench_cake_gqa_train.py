"""Benchmark of the Cake packed-varlen HD256 GQA training attention against FlashAttention CuTe.

One training step of one attention layer under non-reentrant activation checkpointing is the forward, the
checkpoint replay (a second forward) and the backward, ``2 x forward + backward``.  The dK / dV of the step go to
caller-owned FP32 destination rows through a row map: the Cake arm adds them in its backward kernel; the
FlashAttention arm (``flash_attn.cute``, when installed) returns BF16 window gradients and the step then zeroes the
destinations and adds them with ``index_add_`` (the issue's packed-CP adapter), so initialization counts on both
sides.  Both arms see identical generated inputs; the arms alternate within each repetition and the median of the
repetitions is reported, per stage and per step, with the workspace bytes.

Layouts are context-parallel zigzag packs: each document is cut into ``2 CP`` chunks; rank ``r`` holds chunks ``r``
and ``2 CP - 1 - r``, and a chunk's key window is the document's prefix up to the chunk's end (bottom-right causal).
``issue1``-``issue3`` reproduce the issue's (T_q, T_k, segments, max Q, max K) rows exactly.

    python benchmarks/bench_cake_gqa_train.py --layouts issue1,issue2,issue3 --out results.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from typing import Any, Callable

import torch

from flashinfer.gqa_train import (
    gqa_train_workspace,
    gqa_train_workspace_bytes,
    varlen_gqa_attention_bwd,
)
from flashinfer.gqa_train import varlen_gqa_attention_fwd

H_Q, H_KV, D = 24, 4, 256
SCALE = 0.0625
ISSUE_ROWS = {
    "issue1": (30193, 256933, 12, 3270, 52310),
    "issue2": (32737, 278362, 6, 5534, 88534),
    "issue3": (32388, 275373, 4, 8933, 142924),
}
# (CP, rank, documents, tokens) of the generated zigzag packs (lengths no multiple of any tile).
ZIGZAG_ROWS = {
    "zz_cp8_r3_120000": (8, 3, 5, 120_000),
    "zz_cp4_r1_96000": (4, 1, 3, 96_000),
    "zz_cp2_r0_40000": (2, 0, 2, 40_000),
    "zz_cp8_r7_300000": (8, 7, 8, 300_000),
    "zz_cp16_r0_420000": (16, 0, 3, 420_000),
    "zz_cp1_r0_16461": (1, 0, 4, 16_384 + 77),
}


# ----------------------------------------------------------------------------------------------------------------
# Layouts: (q_lens, k_lens, window destination starts, destination rows)


def _split(n: int, parts: int) -> list[int]:
    base, rem = divmod(n, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def zigzag(
    doc_lens: list[int], cp: int, rank: int
) -> tuple[list[int], list[int], list[int], int]:
    q_lens, k_lens, starts, dst = [], [], [], 0
    for length in doc_lens:
        chunks = _split(length, 2 * cp)
        ends = [sum(chunks[: i + 1]) for i in range(2 * cp)]
        for c in (rank, 2 * cp - 1 - rank):
            q_lens.append(chunks[c])
            k_lens.append(ends[c])
            starts.append(dst)
        dst += length
    return q_lens, k_lens, starts, dst


def issue_case(t_q: int, t_k: int, segments: int, max_q: int, max_k: int, seed: int):
    """A CP8 rank-0 zigzag layout whose (T_q, T_k, segments, max Q, max K) equal the arguments: document ``d``
    gives the segments ``(q = a_d, k = a_d)`` and ``(q = b_d, k = L_d)``."""
    n_docs = segments // 2
    rng = random.Random(seed)
    for _ in range(10000):
        weights = [1.0] + [rng.uniform(0.25, 0.95) for _ in range(n_docs - 1)]
        rest_q = t_q - 2 * max_q
        w = sum(weights[1:]) or 1.0
        qd = [2 * max_q] + [
            max(2, round(rest_q * weights[i] / w)) for i in range(1, n_docs)
        ]
        qd[-1 if n_docs > 1 else 0] += t_q - sum(qd)
        a, b, ok = [], [], True
        for i, s in enumerate(qd):
            bi = max_q if i == 0 else (s + 1) // 2
            ai = s - bi
            if ai < 1 or bi < 1 or ai > max_q or bi > max_q:
                ok = False
                break
            a.append(ai)
            b.append(bi)
        if not ok:
            continue
        lens = [max_k] + [
            16 * max(a[i], b[i]) + rng.randint(0, 15) for i in range(1, n_docs)
        ]
        diff = t_k - sum(a) - sum(lens)
        for i in range(1, n_docs):
            if diff == 0:
                break
            lo = 16 * max(a[i], b[i]) - 15
            room = (max_k - lens[i]) if diff > 0 else (lo - lens[i])
            step = max(min(diff, room), room) if diff < 0 else min(diff, room)
            lens[i] += step
            diff -= step
        if diff != 0 or any(lens[i] < a[i] + b[i] for i in range(n_docs)):
            continue
        q_lens, k_lens, starts, dst = [], [], [], 0
        for i in range(n_docs):
            q_lens += [a[i], b[i]]
            k_lens += [a[i], lens[i]]
            starts += [dst, dst]
            dst += lens[i]
        got = (sum(q_lens), sum(k_lens), len(q_lens), max(q_lens), max(k_lens))
        if got == (t_q, t_k, segments, max_q, max_k):
            return q_lens, k_lens, starts, dst
    raise ValueError(f"no zigzag layout for {(t_q, t_k, segments, max_q, max_k)}")


def layout(name: str):
    if name in ISSUE_ROWS:
        return issue_case(*ISSUE_ROWS[name], seed=list(ISSUE_ROWS).index(name))
    cp, rank, n_docs, total = ZIGZAG_ROWS[name]
    rng = random.Random(7073)
    for earlier in ZIGZAG_ROWS:  # the generator's draws follow the table order
        e_cp, e_rank, e_docs, e_total = ZIGZAG_ROWS[earlier]
        cuts = sorted(rng.sample(range(1, e_total), e_docs - 1))
        if earlier == name:
            break
    docs = [b - a for a, b in zip([0] + cuts, cuts + [total], strict=True)]
    return zigzag(docs, cp, rank)


# ----------------------------------------------------------------------------------------------------------------
# Arms


def make_inputs(lay, device: torch.device, seed: int = 0) -> dict[str, Any]:
    q_lens, k_lens, starts, n_dst = lay
    g = torch.Generator(device=device).manual_seed(seed)
    t_q, t_k = sum(q_lens), sum(k_lens)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device).to(torch.bfloat16)

    def cu(lens):
        out = [0]
        for n in lens:
            out.append(out[-1] + n)
        return torch.tensor(out, dtype=torch.int32, device=device)

    rows = [s + j for s, k in zip(starts, k_lens, strict=True) for j in range(k)]
    return {
        "q": randn(t_q, H_Q, D),
        "k": randn(t_k, H_KV, D),
        "v": randn(t_k, H_KV, D),
        "dout": randn(t_q, H_Q, D),
        "cu_q": cu(q_lens),
        "cu_k": cu(k_lens),
        "row_map": torch.tensor(rows, dtype=torch.int32, device=device),
        "dk_dst": torch.empty(n_dst, H_KV, D, device=device),
        "dv_dst": torch.empty(n_dst, H_KV, D, device=device),
        "max_q": max(q_lens),
        "max_k": max(k_lens),
        "segments": len(q_lens),
    }


class CakeArm:
    name = "cake"

    def __init__(self, x: dict[str, Any]) -> None:
        self.ws = gqa_train_workspace(x["q"].device)

    def stages(self, x: dict[str, Any]) -> list[tuple[str, Callable[[], Any]]]:
        state: dict[str, Any] = {}

        def fwd():
            state["out"], state["lse"] = varlen_gqa_attention_fwd(
                x["q"],
                x["k"],
                x["v"],
                x["cu_q"],
                x["cu_k"],
                softmax_scale=SCALE,
                workspace=self.ws,
            )

        def bwd():
            x["dk_dst"].zero_()
            x["dv_dst"].zero_()
            varlen_gqa_attention_bwd(
                x["q"],
                x["k"],
                x["v"],
                state["out"],
                state["lse"],
                x["dout"],
                x["cu_q"],
                x["cu_k"],
                softmax_scale=SCALE,
                dk_acc=x["dk_dst"],
                dv_acc=x["dv_dst"],
                row_map=x["row_map"],
                max_seqlen_k=x["max_k"],
                workspace=self.ws,
            )

        return [("fwd", fwd), ("fwd_replay", fwd), ("bwd", bwd)]


class FlashAttentionArm:
    name = "fa4"

    def __init__(self, x: dict[str, Any]) -> None:
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

        self._fwd, self._bwd = _flash_attn_fwd, _flash_attn_bwd

    def stages(self, x: dict[str, Any]) -> list[tuple[str, Callable[[], Any]]]:
        kw = dict(
            cu_seqlens_q=x["cu_q"],
            cu_seqlens_k=x["cu_k"],
            max_seqlen_q=x["max_q"],
            max_seqlen_k=x["max_k"],
            softmax_scale=SCALE,
            causal=True,
        )
        state: dict[str, Any] = {}
        rows = x["row_map"].long()

        def fwd():
            state["out"], state["lse"] = self._fwd(
                x["q"], x["k"], x["v"], return_lse=True, **kw
            )[:2]

        def bwd():
            _dq, dk, dv = self._bwd(
                x["q"], x["k"], x["v"], state["out"], x["dout"], state["lse"], **kw
            )[:3]
            x["dk_dst"].zero_()
            x["dv_dst"].zero_()
            x["dk_dst"].index_add_(0, rows, dk.float())
            x["dv_dst"].index_add_(0, rows, dv.float())

        return [("fwd", fwd), ("fwd_replay", fwd), ("bwd", bwd)]


def time_step(stages: list[tuple[str, Callable[[], Any]]]) -> dict[str, float]:
    events = [torch.cuda.Event(enable_timing=True) for _ in range(len(stages) + 1)]
    torch.cuda.synchronize()
    events[0].record()
    for i, (_name, call) in enumerate(stages):
        call()
        events[i + 1].record()
    torch.cuda.synchronize()
    times = {
        name: events[i].elapsed_time(events[i + 1])
        for i, (name, _call) in enumerate(stages)
    }
    times["step"] = events[0].elapsed_time(events[-1])
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layouts", default=",".join([*ISSUE_ROWS, *ZIGZAG_ROWS]))
    parser.add_argument("--arms", default="cake,fa4")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    rows = []
    for name in args.layouts.split(","):
        lay = layout(name)
        x = make_inputs(lay, device)
        arms = []
        for arm in args.arms.split(","):
            try:
                arms.append(CakeArm(x) if arm == "cake" else FlashAttentionArm(x))
            except ImportError as exc:
                print(f"{name}: arm {arm} unavailable ({exc})", file=sys.stderr)
        samples: dict[str, list[dict[str, float]]] = {arm.name: [] for arm in arms}
        plans = {arm.name: arm.stages(x) for arm in arms}
        for _ in range(args.warmup):
            for arm in arms:
                time_step(plans[arm.name])
        for rep in range(args.reps):
            for arm in arms if rep % 2 == 0 else arms[::-1]:
                samples[arm.name].append(time_step(plans[arm.name]))
        row = {
            "layout": name,
            "T_q": sum(lay[0]),
            "T_k": sum(lay[1]),
            "segments": len(lay[0]),
            "max_q": max(lay[0]),
            "max_k": max(lay[1]),
            "workspace_bytes": gqa_train_workspace_bytes(sum(lay[0]), H_Q, len(lay[0])),
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
        }
        for arm_name, runs in samples.items():
            for key in runs[0]:
                row[f"{arm_name}_{key}_ms"] = statistics.median(r[key] for r in runs)
        if "cake_step_ms" in row and "fa4_step_ms" in row:
            row["speedup"] = row["fa4_step_ms"] / row["cake_step_ms"]
        print(json.dumps(row), flush=True)
        rows.append(row)
        del x
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
