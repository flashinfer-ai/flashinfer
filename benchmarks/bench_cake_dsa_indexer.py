"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Benchmark the experimental Cake DSA indexer top-k backend (SM100 / SM103 / SM107).

Times the *complete operator* -- scoring, exact selection, ascending-id
ordering, aligned scores and padding, including the workspace traffic -- of
``flashinfer.experimental.cake_dsa_indexer`` against two torch + FlashInfer
compositions of the same contract on representative rows (32 heads, head
dimension 128, BF16 ``q`` / ``k``, FP32 ``w``, iid Gaussian inputs with a
peaked variant):

* ``P1`` the recorded packed training call (``T = 16231``, ``Tkv = 268757``,
  eight segments, top-k 2048), ``P1_peaked`` its peaked variant, ``P2`` the
  second recorded token count (``T = 16172`` / ``Tkv = 267520``; boundaries
  derived from the ``P1`` pattern), ``P3`` = ``P1`` with top-k 4096;
* ``S1`` .. ``S8`` one causal document with ``Q = N`` in 8k .. 1M keys,
  ``S4_peaked``; ``C1`` / ``C2`` context-parallel tails (4096 queries over 64k /
  128k keys); ``C3`` eight packed 2048 x 32768 segments; ``V1`` (``ratio = 2``),
  ``V2`` (top-k 256), ``V3`` (top-k 4096) at ``N = 65536``.

Arms (``--arms``): ``cake`` (this backend, a prepared runner over caller-owned
outputs and workspace), ``chunked_pipeline`` (a reconstruction of the current
training-stack path: chunked BF16 coarse scoring, coarse top-``(K + 256)`` with
``flashinfer.top_k``, candidate gather, FP32 rescoring, sorting, padding) and
``materialized_fp32`` (exact FP32 scores materialized per memory-bounded query
chunk, ``flashinfer.top_k``, ascending-id ordering).  Both compositions use
``flashinfer.top_k(deterministic=True, tie_break=LARGE)`` up to ``k = 2048``
and the CUB backend with ``tie_break=LARGE`` above (the deterministic path is
limited to 2048); they are speed references, not contract-exact oracles (their
tie handling at signed zeros and their candidate margin differ from the
contract).

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and a
cold L2 between iterations (per-iteration GPU span of the whole operator).
Every row is measured in three paired phases -- the arms in the given order
(``AB``), in reverse order (``BA``) and interleaved one call at a time
(``IL``) -- and the medians of each phase and of all samples are reported
together with the wrapper-inclusive wall time, the peak extra device memory
beyond inputs / outputs / workspace, and the explicit workspace bytes of the
``cake`` arm.  ``--verify`` judges every arm against the FP64 / FP32
reference of ``tests/test_helpers/cake_dsa_indexer_reference.py`` (small rows
only; the reference scores every visible pair).

Usage::

    python benchmarks/bench_cake_dsa_indexer.py [--rows P1 S2 ...] [--arms cake,chunked_pipeline,materialized_fp32]
        [--reps 20] [--strided-k] [--verify] [--json out.json]
"""

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashinfer  # noqa: E402
from flashinfer.api_logging import ExperimentalWarning  # noqa: E402
from flashinfer.experimental.cake_dsa_indexer import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from flashinfer.topk import TopKTieBreak  # noqa: E402
from tests.test_helpers import cake_dsa_indexer_reference as ref  # noqa: E402

P1_CU_SEQLENS_Q = (0, 1777, 3532, 5655, 7765, 9888, 11998, 14121, 16231)
P1_CU_SEQLENS_K = (0, 1777, 58619, 60742, 128665, 130788, 198711, 200834, 268757)
P2_T, P2_TKV = 16172, 267520
# Coarse-only GB200 reference latencies quoted by the request (single causal document, top-k 2048).
REFERENCE_GB200_MS = {
    8192: 2.57,
    16384: 4.58,
    32768: 11.6,
    65536: 37.6,
    131072: 139.0,
    262144: 557.0,
    524288: 2331.0,
    1048576: 11156.0,
}
DETERMINISTIC_TOPK_MAX_K = 2048
COARSE_MARGIN = 256


def _lengths(cu):
    return [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)]


def _scale_lengths(lengths, total):
    out = [max(1, round(v * total / sum(lengths))) for v in lengths]
    out[-1] += total - sum(out)
    return out


def derive_p2_lengths(t_total=P2_T, tkv_total=P2_TKV):
    """P1 pattern at the second recorded totals: even segments keep Lk = Lq, odd (tail) segments share the rest."""
    q_len = _scale_lengths(_lengths(P1_CU_SEQLENS_Q), t_total)
    k_len = list(q_len)
    k_len[1::2] = _scale_lengths(
        _lengths(P1_CU_SEQLENS_K)[1::2], tkv_total - sum(q_len[0::2])
    )
    return q_len, k_len


def _row(
    seg_q,
    seg_k,
    *,
    top_k=2048,
    ratio=1,
    offsets=None,
    peaked=False,
    seed=7570,
    reps=20,
    note="",
):
    return dict(
        seg_q=list(seg_q),
        seg_k=list(seg_k),
        top_k=top_k,
        ratio=ratio,
        offsets=offsets,
        peaked=peaked,
        seed=seed,
        reps=reps,
        note=note,
    )


ROWS: dict[str, dict] = {}
ROWS["P1"] = _row(
    _lengths(P1_CU_SEQLENS_Q),
    _lengths(P1_CU_SEQLENS_K),
    seed=7571,
    note="recorded packed call",
)
ROWS["P1_peaked"] = _row(
    _lengths(P1_CU_SEQLENS_Q),
    _lengths(P1_CU_SEQLENS_K),
    peaked=True,
    seed=7571,
    note="P1 with dominating keys",
)
ROWS["P2"] = _row(
    *derive_p2_lengths(),
    seed=7572,
    note="second recorded token count, boundaries derived",
)
ROWS["P3"] = _row(
    _lengths(P1_CU_SEQLENS_Q),
    _lengths(P1_CU_SEQLENS_K),
    top_k=4096,
    seed=7573,
    note="P1 with top-k 4096",
)
for _i, _n in enumerate(
    (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576), start=1
):
    ROWS[f"S{_i}"] = _row(
        [_n],
        [_n],
        seed=7580 + _i,
        reps=20 if _n < 524288 else 7,
        note=f"single causal document N = {_n}",
    )
ROWS["S4_peaked"] = _row(
    [65536], [65536], peaked=True, seed=7584, note="S4 with dominating keys"
)
for _i, _lk in enumerate((65536, 131072), start=1):
    ROWS[f"C{_i}"] = _row([4096], [_lk], seed=7590 + _i, note=f"CP tail 4096 x {_lk}")
ROWS["C3"] = _row([2048] * 8, [32768] * 8, seed=7593, note="packed 8 x (2048 x 32768)")
ROWS["V1"] = _row([65536], [32768], ratio=2, seed=7601, note="N = 65536, ratio 2")
ROWS["V2"] = _row([65536], [65536], top_k=256, seed=7602, note="N = 65536, top-k 256")
ROWS["V3"] = _row([65536], [65536], top_k=4096, seed=7603, note="N = 65536, top-k 4096")
ROWS["smoke"] = _row(
    [200, 300, 500],
    [200, 3000, 1000],
    top_k=128,
    seed=7700,
    reps=5,
    note="three mixed segments (quick check)",
)
ROWS["smoke_p1"] = _row(
    [max(1, v // 16) for v in _lengths(P1_CU_SEQLENS_Q)],
    [max(1, v // 16) for v in _lengths(P1_CU_SEQLENS_K)],
    top_k=256,
    seed=7701,
    reps=5,
    note="P1 pattern / 16",
)
DEFAULT_ROWS = [n for n in ROWS if not n.startswith("smoke")]


def build_inputs(name: str, device, strided_k: bool):
    row = ROWS[name]
    return ref.make_random_inputs(
        row["seg_q"],
        row["seg_k"],
        top_k=row["top_k"],
        seed=row["seed"],
        device=device,
        peaked=row["peaked"],
        ratio=row["ratio"],
        q_causal_offsets=row["offsets"],
        k_row_stride=ref.TRAINER_KEY_ROW_STRIDE if strided_k else None,
        label=name,
    )


# ---------------------------------------------------------------------------
# Composition helpers shared by the two reference arms
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _topk_algo(algo):
    previous = os.environ.pop("FLASHINFER_TOPK_ALGO", None)
    if algo is not None:
        os.environ["FLASHINFER_TOPK_ALGO"] = algo
    try:
        yield
    finally:
        os.environ.pop("FLASHINFER_TOPK_ALGO", None)
        if previous is not None:
            os.environ["FLASHINFER_TOPK_ALGO"] = previous


def select_topk(x, k):
    """Row-wise top-k (value desc, id desc): the deterministic FlashInfer path up to 2048, CUB above."""
    if k <= DETERMINISTIC_TOPK_MAX_K:
        return flashinfer.top_k(
            x, k, sorted=False, deterministic=True, tie_break=TopKTieBreak.LARGE
        )
    with _topk_algo("cub"):
        return flashinfer.top_k(
            x, k, sorted=False, deterministic=False, tie_break=TopKTieBreak.LARGE
        )


def ascending_with_padding(ids, scores, valid):
    big = torch.full_like(ids, 2**62)
    key, perm = torch.sort(torch.where(valid, ids, big), dim=1)
    pad = key == 2**62
    out_ids = torch.where(pad, torch.full_like(key, -1), key).to(torch.int32)
    out_scores = torch.where(
        pad, torch.full_like(scores, float("-inf")), scores.gather(1, perm)
    )
    return out_ids, out_scores


def _fp32_mm_supported(device) -> bool:
    try:
        a = torch.ones((16, 32), dtype=torch.bfloat16, device=device)
        b = torch.ones((32, 8), dtype=torch.bfloat16, device=device)
        return torch.mm(a, b, out_dtype=torch.float32).dtype == torch.float32
    except (TypeError, RuntimeError):
        return False


def _segments(inputs):
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    visible = ref.visible_rows(inputs)
    visible_cpu = visible.cpu()
    for s in range(inputs.num_segments):
        lq, lk = inputs.seg_q_len[s], inputs.seg_k_len[s]
        if lq and lk:
            yield q_starts[s], lq, k_starts[s], lk, visible, visible_cpu


def chunked_pipeline(
    inputs, *, query_chunk=2048, rescoring_chunk=512, coarse_bytes=2 << 30
):
    """Coarse BF16 scoring -> top-(K + 256) -> FP32 rescoring of the candidates -> sorts -> padding."""
    dev = inputs.device
    T, K = inputs.num_queries, inputs.top_k
    scale = inputs.softmax_scale
    out_ids = torch.full((T, K), -1, dtype=torch.int32, device=dev)
    out_scores = torch.full((T, K), float("-inf"), dtype=torch.float32, device=dev)
    k32 = inputs.k.to(torch.float32)
    for q0, lq, k0, lk, visible, visible_cpu in _segments(inputs):
        kT = inputs.k[k0 : k0 + lk].t()
        for t0 in range(q0, q0 + lq, query_chunk):
            t1 = min(q0 + lq, t0 + query_chunk)
            c = t1 - t0
            jmax = int(visible_cpu[t1 - 1])
            if jmax == 0:
                continue
            coarse = torch.empty((c, jmax), dtype=torch.float32, device=dev)
            sub = max(1, min(c, coarse_bytes // (ref.NUM_HEADS * jmax * 2)))
            for r0 in range(0, c, sub):
                r1 = min(c, r0 + sub)
                m = r1 - r0
                logits = torch.relu_(
                    torch.mm(
                        inputs.q[t0 + r0 : t0 + r1].reshape(
                            m * ref.NUM_HEADS, ref.HEAD_DIM
                        ),
                        kT[:, :jmax],
                    )
                )
                w_bf16 = inputs.w[t0 + r0 : t0 + r1].to(torch.bfloat16).unsqueeze(1)
                coarse[r0:r1] = (
                    torch.bmm(w_bf16, logits.view(m, ref.NUM_HEADS, jmax))
                    .float()
                    .squeeze(1)
                )
                del logits
            ar = torch.arange(jmax, dtype=torch.int64, device=dev).unsqueeze(0)
            coarse.masked_fill_(ar >= visible[t0:t1].unsqueeze(1), float("-inf"))
            kc = min(K + COARSE_MARGIN, jmax)
            _, cand = select_topk(coarse, kc)
            del coarse
            kk = min(K, kc)
            for r0 in range(0, c, rescoring_chunk):
                r1 = min(c, r0 + rescoring_chunk)
                m = r1 - r0
                ids = cand[r0:r1].to(torch.int64)
                gathered = k32.index_select(0, (ids + k0).reshape(-1)).view(
                    m, kc, ref.HEAD_DIM
                )
                with ref.fp32_ieee_matmul():
                    logits = torch.relu_(
                        torch.bmm(
                            inputs.q[t0 + r0 : t0 + r1].float(),
                            gathered.transpose(1, 2),
                        ).mul_(scale)
                    )
                    sc = torch.bmm(
                        inputs.w[t0 + r0 : t0 + r1].unsqueeze(1), logits
                    ).squeeze(1)
                del gathered, logits
                valid = ids < visible[t0 + r0 : t0 + r1].unsqueeze(1)
                sc = sc.masked_fill(~valid, float("-inf"))
                ids_d, p1 = torch.sort(ids, dim=1, descending=True, stable=True)
                _, p2 = torch.sort(
                    sc.gather(1, p1) + 0.0, dim=1, descending=True, stable=True
                )
                p2 = p2[:, :kk]
                o_ids, o_sc = ascending_with_padding(
                    ids_d.gather(1, p2),
                    sc.gather(1, p1).gather(1, p2),
                    valid.gather(1, p1).gather(1, p2),
                )
                out_ids[t0 + r0 : t0 + r1, :kk] = o_ids
                out_scores[t0 + r0 : t0 + r1, :kk] = o_sc
    return out_ids, out_scores


def materialized_fp32(
    inputs, *, bytes_budget=2 << 30, min_rows=64, min_rows_budget=16 << 30
):
    """Exact FP32 scores per memory-bounded query chunk -> flashinfer top-k -> ascending-id ordering -> padding."""
    dev = inputs.device
    T, K = inputs.num_queries, inputs.top_k
    scale = inputs.softmax_scale
    out_ids = torch.full((T, K), -1, dtype=torch.int32, device=dev)
    out_scores = torch.full((T, K), float("-inf"), dtype=torch.float32, device=dev)
    fp32_out = _fp32_mm_supported(dev)
    for q0, lq, k0, lk, visible, visible_cpu in _segments(inputs):
        jmax_seg = int(visible_cpu[q0 + lq - 1])
        if jmax_seg == 0:
            continue
        per_row = ref.NUM_HEADS * jmax_seg * 4 + jmax_seg * 8
        rows = max(
            1,
            min(
                lq,
                max(bytes_budget // per_row, min(min_rows, min_rows_budget // per_row)),
            ),
        )
        kT = inputs.k[k0 : k0 + lk].t()
        for t0 in range(q0, q0 + lq, rows):
            t1 = min(q0 + lq, t0 + rows)
            m = t1 - t0
            jmax = int(visible_cpu[t1 - 1])
            if jmax == 0:
                continue
            q2d = inputs.q[t0:t1].reshape(m * ref.NUM_HEADS, ref.HEAD_DIM)
            if fp32_out:
                logits = torch.mm(q2d, kT[:, :jmax], out_dtype=torch.float32)
            else:
                with ref.fp32_ieee_matmul():
                    logits = torch.mm(q2d.float(), kT[:, :jmax].float())
            logits = torch.relu_(logits.mul_(scale))
            with ref.fp32_ieee_matmul():
                scores = torch.bmm(
                    inputs.w[t0:t1].unsqueeze(1), logits.view(m, ref.NUM_HEADS, jmax)
                ).squeeze(1)
            del logits
            ar = torch.arange(jmax, dtype=torch.int64, device=dev).unsqueeze(0)
            vis_rows = visible[t0:t1].unsqueeze(1)
            scores.masked_fill_(ar >= vis_rows, float("-inf"))
            kk = min(K, jmax)
            vals, ids = select_topk(scores, kk)
            del scores
            ids = ids.to(torch.int64)
            o_ids, o_sc = ascending_with_padding(ids, vals, ids < vis_rows)
            out_ids[t0:t1, :kk] = o_ids
            out_scores[t0:t1, :kk] = o_sc
    return out_ids, out_scores


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


class Arm:
    name = ""
    workspace_bytes = None

    def __init__(self, inputs):
        self.inputs = inputs

    def __call__(self):
        raise NotImplementedError


class CakeArm(Arm):
    name = "cake"

    def __init__(self, inputs):
        super().__init__(inputs)
        dev = inputs.device
        self.workspace_bytes = cake_backend.dsa_indexer_workspace_size(
            inputs.top_k, dev
        )
        self.workspace = torch.empty(
            self.workspace_bytes, dtype=torch.uint8, device=dev
        )
        self.indices = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=dev
        )
        self.scores = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=dev
        )
        self.runner = cake_backend.prepare_dsa_indexer_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            workspace_buffer=self.workspace,
            indices=self.indices,
            scores=self.scores,
        )

    def __call__(self):
        return self.runner()


class ChunkedPipelineArm(Arm):
    name = "chunked_pipeline"

    def __call__(self):
        return chunked_pipeline(self.inputs)


class MaterializedArm(Arm):
    name = "materialized_fp32"

    def __call__(self):
        return materialized_fp32(self.inputs)


ARMS = {cls.name: cls for cls in (CakeArm, ChunkedPipelineArm, MaterializedArm)}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def gpu_times(fn, reps, warmup):
    return [
        float(t)
        for t in bench_gpu_time(
            fn,
            dry_run_iters=warmup,
            repeat_iters=reps,
            enable_cupti=True,
            cold_l2_cache=True,
        )
    ]


def wall_ms(fn, reps):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def peak_extra_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    return extra - sum(t.numel() * t.element_size() for t in out)


def measure_row(name, inputs, arms, args):
    row = ROWS[name]
    reps = args.reps if args.reps is not None else row["reps"]
    prepared = {}
    for arm_name in arms:
        try:
            prepared[arm_name] = ARMS[arm_name](inputs)
        except Exception as exc:  # noqa: BLE001 - report the arm as unavailable, keep the others
            print(
                f"  [{name}] {arm_name}: unavailable ({type(exc).__name__}: {exc})",
                flush=True,
            )
    result = dict(
        row=name,
        T=inputs.num_queries,
        Tkv=inputs.num_keys,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        reps=reps,
        arms={},
    )
    samples = {a: {"AB": [], "BA": [], "IL": []} for a in prepared}
    order = list(prepared)
    for arm_name in order:
        samples[arm_name]["AB"] = gpu_times(prepared[arm_name], reps, args.warmup)
    for arm_name in reversed(order):
        samples[arm_name]["BA"] = gpu_times(prepared[arm_name], reps, args.warmup)
    for _ in range(reps):
        for arm_name in order:
            samples[arm_name]["IL"].extend(gpu_times(prepared[arm_name], 1, 0))
    outputs = {}
    for arm_name, arm in prepared.items():
        out = arm()
        torch.cuda.synchronize()
        outputs[arm_name] = (out[0].clone(), out[1].clone())
        every = (
            samples[arm_name]["AB"] + samples[arm_name]["BA"] + samples[arm_name]["IL"]
        )
        record = dict(
            gpu_ms={
                phase: statistics.median(v) for phase, v in samples[arm_name].items()
            },
            wall_ms=wall_ms(arm, min(reps, 5)),
            peak_extra_bytes=peak_extra_bytes(arm),
            workspace_bytes=arm.workspace_bytes,
            structure_failures=ref.structure_failures(inputs, *outputs[arm_name])[:3],
        )
        record["gpu_ms"]["all"] = statistics.median(every)
        result["arms"][arm_name] = record
    if "cake" in outputs:
        for arm_name, out in outputs.items():
            if arm_name != "cake":
                same = (out[0] == outputs["cake"][0]).all(dim=1)
                result["arms"][arm_name]["rows_identical_to_cake"] = int(same.sum())
                result["arms"][arm_name]["speedup_of_cake"] = (
                    result["arms"][arm_name]["gpu_ms"]["all"]
                    / result["arms"]["cake"]["gpu_ms"]["all"]
                )
    if args.verify:
        reference = ref.select_reference(inputs)
        for arm_name, out in outputs.items():
            verdict = ref.judge(out, reference)
            result["arms"][arm_name]["verdict"] = str(verdict).splitlines()[0]
    if (
        len(row["seg_q"]) == 1
        and row["seg_q"][0] == row["seg_k"][0]
        and row["seg_q"][0] in REFERENCE_GB200_MS
        and inputs.top_k == 2048
        and not row["peaked"]
    ):
        result["reference_gb200_ms"] = REFERENCE_GB200_MS[row["seg_q"][0]]
    return result


def print_row(result):
    for arm_name, rec in result["arms"].items():
        gpu = rec["gpu_ms"]
        extra = (
            ""
            if rec["workspace_bytes"] is None
            else f" workspace {rec['workspace_bytes'] / 2**20:.1f} MiB"
        )
        speed = (
            f" x{rec['speedup_of_cake']:.2f} vs cake"
            if "speedup_of_cake" in rec
            else ""
        )
        bad = (
            f" STRUCTURE {rec['structure_failures'][0]}"
            if rec["structure_failures"]
            else ""
        )
        verdict = f" {rec['verdict']}" if "verdict" in rec else ""
        print(
            f"  [{result['row']}] {arm_name:<18} gpu {gpu['all']:10.3f} ms (AB {gpu['AB']:.3f} / BA {gpu['BA']:.3f} / IL {gpu['IL']:.3f})"
            f" wall {rec['wall_ms']:.3f} ms extra {rec['peak_extra_bytes'] / 2**20:.1f} MiB{extra}{speed}{bad}{verdict}",
            flush=True,
        )
    if "reference_gb200_ms" in result:
        print(
            f"  [{result['row']}] coarse-only GB200 reference latency quoted by the request: {result['reference_gb200_ms']} ms"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows",
        nargs="*",
        default=DEFAULT_ROWS,
        help=f"rows to measure (default: all regression rows; also {', '.join(n for n in ROWS if n.startswith('smoke'))})",
    )
    parser.add_argument(
        "--arms",
        default="cake,chunked_pipeline,materialized_fp32",
        help="comma-separated arms in AB order",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="measured iterations per phase (default: the row's count)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--strided-k",
        action="store_true",
        help="hand k as the [:, :128] view of a [Tkv, 704] packed tensor",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="judge every arm against the FP64 / FP32 reference (small rows)",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore", ExperimentalWarning)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(ARMS)}")
    print(
        f"{props.name} ({props.multi_processor_count} SMs, cc {props.major}.{props.minor}), rows {args.rows}, arms {arms}"
    )
    if not cake_backend.generated_program_available(device) and "cake" in arms:
        print(
            "generated program not registered for this device: the cake arm is unavailable",
            flush=True,
        )
    results = []
    for name in args.rows:
        inputs = build_inputs(name, device, args.strided_k)
        print(
            f"[{name}] T={inputs.num_queries} Tkv={inputs.num_keys} segments={inputs.num_segments} top_k={inputs.top_k} ratio={inputs.ratio}: {ROWS[name]['note']}",
            flush=True,
        )
        result = measure_row(name, inputs, arms, args)
        print_row(result)
        results.append(result)
        del inputs
        torch.cuda.empty_cache()
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                dict(
                    device=props.name, sms=props.multi_processor_count, results=results
                ),
                indent=1,
            )
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
