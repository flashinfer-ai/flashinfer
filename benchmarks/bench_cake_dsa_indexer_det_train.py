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
``flashinfer.experimental.cake.cake_dsa_indexer_det_train`` against two torch + FlashInfer
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
  ``V2`` (top-k 256), ``V3`` (top-k 4096) at ``N = 65536``;
* the top-512 rows of flashinfer-ai/flashinfer#7549 (its ratio-4 pooled
  selection: top-512 pools over a 4-token pool layout, ``ratio = 4`` with the
  keys pooled 4:1): ``V4`` the packed 4k .. 64k documents of ``Z3cp`` (zigzag
  rank 0 of 16) pooled, ``V5`` the packed documents of ``Z3`` (full causal)
  pooled, ``V6`` one 262144-token document pooled, and ``V7`` one 65536-token
  document at top-k 512 without pooling; ``--rows pool`` names the group;
* the attention-sink + local-band rows of flashinfer-ai/flashinfer#7042
  (``tests/experimental/test_helpers/cake_dsa_indexer_reference.make_sink_inputs``, the
  issue's generator): ``Z1`` one 262144-token document, context-parallel
  zigzag rank 0 of 16 (queries ``[0, 8192)`` and ``[253952, 262144)``, the
  issue's call), ``Z1k4096`` / ``Z1k256`` its top-k variants, ``Z2`` eight 32k
  documents (full causal) and ``Z2cp`` their zigzag rank-0 halves, ``Z3`` a
  packed batch of 4k .. 64k documents and ``Z3cp`` its zigzag halves,
  ``P1sink`` the recorded ``P1`` geometry, ``S4sink`` / ``S6sink`` one 64k /
  262k causal document.  ``--rows iid`` / ``--rows sink`` name the two groups.

Arms (``--arms``): ``cake`` (this backend, a prepared runner over caller-owned
outputs and workspace), ``cake_nosample`` (the same runner with
``sample_tiles_max=0``: the sampled first threshold off, same programs, same
bits), ``cake_far`` (the same runner with ``far_sample=1``: the sampled first
threshold ranks the far sample tiles only, same programs, same bits),
``cake_floor16k`` (forced dispatch: the record's 16384-slot
capacity floor on every ``top_k``, ``cand_cap_floor_rule`` None -- the plan
of the package before the top_k-keyed floor on ``sm_100a``, in this build --
the record itself on ``sm_103a``; its ``_aa`` twin
``cake_floor16k_aa``), ``cake_wide`` / ``cake_l6`` / ``cake_pair_l6`` / ``cake_wide_c1`` /
``cake_narrow`` and their ``_nosample`` twins (forced
dispatch: the local package under the device's registry record with a rule
replaced -- ``cake_wide`` drops ``l6_rule`` and ``pair_rule`` (the
single-CTA wide program on every architecture: the record's ``pair_units``
gate keeps ``pair_for`` off the wide unit on ``sm_100a`` / ``sm_103a``, and
the drop keeps the arm off the registered ``pair_wide`` on ``sm_107a`` too),
``cake_l6`` drops ``pair_rule`` and the wide-first rule (``wide_first_rule``
None), ``cake_pair_l6`` widens both unit rules to ``[1024.0, 512]`` /
``[128.0, 1024.0, 512]`` with no wide-first rule, ``cake_wide_c1`` drops
both and takes the wide program from one round, ``wide_rule [1.0, 0.0]``,
``cake_narrow`` drops both and never qualifies for wide (the narrow program;
only where ``narrow`` is registered for the row's fit); the recorded
``plan`` and workspace bytes are the forced policy's, see
``forced_policy``), ``cake_main`` / ``cake_main_nosample`` (a
second copy of the package loaded from ``--main-package DIR``, e.g. the
``flashinfer/experimental/cake/cake_dsa_indexer_det_train``
directory of a FlashInfer main checkout (``flashinfer/experimental/cake_dsa_indexer_det``
in a checkout before flashinfer-ai/flashinfer#8928), in the same process: the package is
self-contained -- relative imports, ``Path(__file__)`` manifest, JIT names
salted by its own closure digests -- so both versions run on the same input
tensors), ``cake_aa`` / ``cake_far_aa`` / ``cake_main_aa`` (the same arm prepared a second time:
the in-process A/A band), ``chunked_pipeline`` (a reconstruction of the current
training-stack path: chunked BF16 coarse scoring, coarse top-``(K + 256)`` with
``flashinfer.top_k``, candidate gather, FP32 rescoring, sorting, padding) and
``materialized_fp32`` (exact FP32 scores materialized per memory-bounded query
chunk, ``flashinfer.top_k``, ascending-id ordering).  Both compositions use
``flashinfer.top_k(deterministic=True, tie_break=LARGE)`` up to ``k = 2048``
and the CUB backend with ``tie_break=LARGE`` above (the deterministic path is
limited to 2048); they are speed references, not contract-exact oracles (their
tie handling at signed zeros and their candidate margin differ from the
contract).  Every pair of arms is compared bitwise (ids and score bits) on
every row and the verdicts are recorded (``same_bits``), together with a
SHA-256 of each arm's ids and score bits for cross-process comparisons.

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and a
cold L2 between iterations (per-iteration GPU span of the whole operator).
Without ``cupti-python >= 13`` that timer silently falls back to CUDA events;
``--cupti-required`` turns the fallback into a failure and the JSON records
which timer ran (``timer``).
Every row is measured in three paired phases -- the arms in the given order
(``AB``), in reverse order (``BA``) and interleaved one call at a time
(``IL``) -- and the medians of each phase and of all samples are reported
together with the wrapper-inclusive wall time, the peak extra device memory
beyond inputs / outputs / workspace, and the explicit workspace bytes of the
``cake`` arm.  ``--verify`` judges every arm against the FP64 / FP32
reference of ``tests/experimental/test_helpers/cake_dsa_indexer_reference.py`` (small rows
only; the reference scores every visible pair).

FP8 mode (``--form fp8``): the inputs are float8_e4m3fn ``q`` / ``k`` with
per-key FP32 ``k_scale`` (``ref.make_fp8_inputs``: amax scaling per (query,
head) and per key, the query and softmax scales folded into ``w``) on the iid
rows; ``--rows fp8`` (the default of the mode) names ``P1``, ``glm2k_16k``
(2048 queries over 16384 keys), ``s1_8k_on_64k`` (8192 over 73728),
``s2_4x2k_packed`` (four segments of 2048 queries over 18432, 43008, 67584
and 100352 keys) and ``s3_cold16k`` (16384 over 16384), top-k 2048, causal.
Arms: ``cake_fp8`` (this backend's FP8 form; ``cake_fp8_aa`` /
``cake_fp8_nosample`` as above; ``cake_fp8_w6`` the FP8 dispatch before the
FP8 wide program, which ran a wide decision on the l6 program with the l6
kind's sampled first threshold and check period (the FP8 wide programs
hidden: a twin of ``cake_fp8`` where none is registered); ``cake_fp8_wn``
the FP8 dispatch before that, which ran a wide decision on the narrow
program with the wide decision's sampled first threshold and check period;
``cake_fp8_l6`` /
``cake_fp8_narrow`` on the l6 / narrow program on every row, and
``cake_l6_all`` the BF16 form on the l6 program where the architecture
registers the l6 key the row composes), ``cake`` (the
BF16 form on the same data dequantized to BF16: ``q`` exactly, ``bf16(k * k_scale)``, the same ``w``,
``softmax_scale = 1``), ``deepgemm_topk`` (stock DeepGEMM FP8 MQA logits with
FP32 per-key scales and FP32 logits: ``deep_gemm.fp8_mqa_logits``, or
``fp8_fp4_mqa_logits(..., logits_dtype=torch.float32)`` in builds without it;
``clean_logits=True``, then ``flashinfer.top_k(deterministic=True,
tie_break=LARGE)``) and ``deepgemm_ragged_topk`` (``fp8_mqa_logits`` with
``clean_logits=False``, then ``flashinfer.top_k_ragged_transform(
deterministic=True, tie_break=LARGE)`` over each row's visible keys); the
DeepGEMM arms run one logits call and one top-k call per segment and are
reported unavailable where ``deep_gemm`` does not import or the call fails.
Timing boundary: a Cake arm's span is the complete operator (scan, merge,
finalize into caller-owned outputs and workspace); a DeepGEMM arm's span is
its logits allocation and kernel plus its top-k kernel per segment (the
``deepgemm_topk`` arm also copies the row-padded logits to contiguous rows for
``flashinfer.top_k``, recorded as ``logits_copied``; the ragged arm reads the
padded rows in place), and the
conversion of their output to the contract's ascending segment-local ids
with aligned scores and ``-1`` / ``-inf`` padding (the score gather of the
ragged arm included) is outside it.  ``same_bits`` / ``all_cake_arms_identical``
compare the Cake arms of one form; the speed ratios are against ``cake_fp8``.

Usage::

    python benchmarks/bench_cake_dsa_indexer_det_train.py [--rows P1 S2 ...] [--arms cake,chunked_pipeline,materialized_fp32]
        [--reps 20] [--strided-k] [--verify] [--json out.json]
    python benchmarks/bench_cake_dsa_indexer_det_train.py --rows sink --arms cake,cake_nosample,cake_main,cake_main_nosample
        --main-package /path/to/main/flashinfer/experimental/cake/cake_dsa_indexer_det_train --cupti-required --json out.json
        (a main checkout before flashinfer-ai/flashinfer#8928: --main-package /path/to/main/flashinfer/experimental/cake_dsa_indexer_det)
    python benchmarks/bench_cake_dsa_indexer_det_train.py --rows sink --arms cake,cake_far,cake_aa,cake_far_aa --cupti-required --json out.json
    python benchmarks/bench_cake_dsa_indexer_det_train.py --rows sink --arms cake,cake_wide,cake_l6,cake_pair_l6,cake_wide_c1 --cupti-required --json out.json
    python benchmarks/bench_cake_dsa_indexer_det_train.py --form fp8 --rows fp8 --arms cake_fp8,cake,deepgemm_topk,deepgemm_ragged_topk
        --cupti-required --json out.json
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import itertools
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
from flashinfer.experimental.cake.cake_dsa_indexer_det_train import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from flashinfer.topk import TopKTieBreak  # noqa: E402
from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref  # noqa: E402

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
    dist="iid",
    bound_seqlen=False,
):
    """``dist``: ``"iid"`` (``make_random_inputs``) or ``"sink"`` (``make_sink_inputs``); ``bound_seqlen`` passes
    ``max_seqlen_k = max(seg_k)`` to the backend as the issue's call does (it sizes the rank finalize's bitmap pool
    only; bits are identical with and without it)."""
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
        dist=dist,
        bound_seqlen=bound_seqlen,
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
IID_ROWS = list(ROWS)


def cp_zigzag_rank0(length, parts=16):
    """Rank 0 of a ``parts``-way context-parallel zigzag split of one ``length``-token document: the query chunks
    ``[0, c)`` and ``[length - c, length)`` with ``c = length // (2 parts)``, each attending its causal key prefix
    (``seg_q = [c, c]``, ``seg_k = [c, length]``: the second segment's offset ``length - c`` by the ``Lk - Lq`` rule)."""
    c = length // (2 * parts)
    return [c, c], [c, length]


def _sink(seg_q, seg_k, **kw):
    kw.setdefault("seed", ref.SINK_SEED)
    kw.setdefault("bound_seqlen", True)
    return _row(seg_q, seg_k, dist="sink", **kw)


_Z1_Q, _Z1_K = cp_zigzag_rank0(262144)
ROWS["Z1"] = _sink(_Z1_Q, _Z1_K, note="#7042 repro: 262k document, CP16 zigzag rank 0")
ROWS["Z1k4096"] = _sink(_Z1_Q, _Z1_K, top_k=4096, note="Z1 with top-k 4096")
ROWS["Z1k256"] = _sink(_Z1_Q, _Z1_K, top_k=256, note="Z1 with top-k 256")
ROWS["Z2"] = _sink(
    [32768] * 8, [32768] * 8, reps=7, note="8 x 32k documents, full causal"
)
ROWS["Z2cp"] = _sink(
    [v for _ in range(8) for v in cp_zigzag_rank0(32768)[0]],
    [v for _ in range(8) for v in cp_zigzag_rank0(32768)[1]],
    note="8 x 32k documents, CP16 zigzag rank 0",
)
_PACK = (4096, 6144, 8192, 12288, 16384, 24576, 32768, 65536)
ROWS["Z3"] = _sink(
    list(_PACK), list(_PACK), note="packed 4k-64k documents, full causal"
)
ROWS["Z3cp"] = _sink(
    [v for n in _PACK for v in cp_zigzag_rank0(n)[0]],
    [v for n in _PACK for v in cp_zigzag_rank0(n)[1]],
    note="packed 4k-64k documents, CP16 zigzag rank 0",
)
ROWS["P1sink"] = _sink(
    _lengths(P1_CU_SEQLENS_Q),
    _lengths(P1_CU_SEQLENS_K),
    note="recorded P1 geometry with sink statistics",
)
ROWS["S4sink"] = _sink(
    [65536], [65536], note="one 64k causal document, sink statistics"
)
ROWS["S6sink"] = _sink(
    [262144], [262144], note="one 262k causal document, sink statistics"
)
SINK_ROWS = [n for n in ROWS if n not in IID_ROWS]


def pooled(seg_q, seg_k, offsets=None, ratio=4):
    """The ``ratio``:1 pooled key layout of a geometry (flashinfer-ai/flashinfer#7549's ratio-4 pooled selection): every
    key segment shrinks to its pool count (``ceil(Lk / ratio)``) and the segment offsets stay in query tokens (query ``u``
    of a segment with offset ``o`` sees the pools ``j < floor((o + u + 1) / ratio)``); ``ratio > 1`` makes the default
    offset 0, so a context-parallel chunk passes its token offset explicitly."""
    return [-(-v // ratio) for v in seg_k], offsets


def cp_zigzag_rank0_offsets(length, parts=16):
    """The token offsets of ``cp_zigzag_rank0``'s two chunks: 0 for the head chunk, ``length - c`` for the tail chunk."""
    return [0, length - length // (2 * parts)]


_POOL_RATIO = 4
_Z3CP_POOL_K, _Z3CP_POOL_OFF = pooled(
    [v for n in _PACK for v in cp_zigzag_rank0(n)[0]],
    [v for n in _PACK for v in cp_zigzag_rank0(n)[1]],
    offsets=[o for n in _PACK for o in cp_zigzag_rank0_offsets(n)],
)
ROWS["V4"] = _row(
    [v for n in _PACK for v in cp_zigzag_rank0(n)[0]],
    _Z3CP_POOL_K,
    top_k=512,
    ratio=_POOL_RATIO,
    offsets=_Z3CP_POOL_OFF,
    seed=7604,
    reps=40,
    note="packed 4k-64k documents, CP16 zigzag rank 0, 4-token pools, top-k 512 (#7549 pooled selection)",
)
ROWS["V5"] = _row(
    list(_PACK),
    pooled(list(_PACK), list(_PACK))[0],
    top_k=512,
    ratio=_POOL_RATIO,
    seed=7605,
    note="packed 4k-64k documents, full causal, 4-token pools, top-k 512",
)
ROWS["V6"] = _row(
    [262144],
    pooled([262144], [262144])[0],
    top_k=512,
    ratio=_POOL_RATIO,
    seed=7606,
    note="one 262k document, 4-token pools, top-k 512",
)
ROWS["V7"] = _row([65536], [65536], top_k=512, seed=7607, note="N = 65536, top-k 512")
POOL_ROWS = ["V4", "V5", "V6", "V7"]
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
# the FP8 form's shape set (``--form fp8``; the BF16 default rows above stay as they are)
ROWS["glm2k_16k"] = _row(
    [2048], [16384], seed=7801, note="2048 queries over 16384 keys"
)
ROWS["s1_8k_on_64k"] = _row(
    [8192], [73728], seed=7802, note="8192 queries over 73728 keys"
)
ROWS["s2_4x2k_packed"] = _row(
    [2048] * 4,
    [18432, 43008, 67584, 100352],
    seed=7803,
    note="packed 4 x 2048 queries over 18432 / 43008 / 67584 / 100352 keys",
)
ROWS["s3_cold16k"] = _row(
    [16384], [16384], seed=7804, note="16384 queries over 16384 keys"
)
FP8_ROWS = ["P1", "glm2k_16k", "s1_8k_on_64k", "s2_4x2k_packed", "s3_cold16k"]
ROW_GROUPS = {
    "iid": IID_ROWS,
    "sink": SINK_ROWS,
    "pool": POOL_ROWS,
    "all": DEFAULT_ROWS,
    "fp8": FP8_ROWS,
}
FORMS = ("bf16", "fp8")


def expand_rows(names):
    """``--rows`` with the group names ``iid`` / ``sink`` / ``pool`` / ``all`` / ``fp8`` expanded, in order, without
    repeats."""
    out = []
    for name in names:
        for row in ROW_GROUPS.get(name, [name]):
            if row not in out:
                out.append(row)
    return out


def build_inputs(name: str, device, strided_k: bool):
    row = ROWS[name]
    if row["dist"] == "sink":
        if strided_k:
            raise SystemExit(f"--strided-k is not supported for the sink row {name}")
        return ref.make_sink_inputs(
            row["seg_q"],
            row["seg_k"],
            top_k=row["top_k"],
            seed=row["seed"],
            device=device,
            q_causal_offsets=row["offsets"],
            label=name,
        )
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


def build_fp8_inputs(name: str, device, strided_k: bool):
    """The FP8 form's inputs of an iid row (``ref.make_fp8_inputs``: its geometry, ``top_k``, ``ratio``, offsets and
    seed)."""
    row = ROWS[name]
    if row["dist"] != "iid" or row["peaked"]:
        raise SystemExit(f"--form fp8 measures the plain iid rows; {name} is not one")
    return ref.make_fp8_inputs(
        row["seg_q"],
        row["seg_k"],
        top_k=row["top_k"],
        seed=row["seed"],
        device=device,
        ratio=row["ratio"],
        q_causal_offsets=row["offsets"],
        k_row_stride=ref.TRAINER_KEY_ROW_STRIDE if strided_k else None,
        label=name,
    )


# ---------------------------------------------------------------------------
# A second copy of the package (the FI-main baseline) in the same process
# ---------------------------------------------------------------------------

MAIN_PACKAGE_NAME = "cake_dsa_indexer_main"
_MAIN = {"backend": None, "path": None}


def load_package_copy(directory: Path, name: str = MAIN_PACKAGE_NAME):
    """Import the package rooted at ``directory`` (its ``__init__.py``) under the module name ``name``.

    The package imports only relatively (``from . import cake_build``, ``from .cake_jit import ...``), resolves its
    manifest and sources through ``Path(__file__)`` and names its JIT libraries ``<entry>_<closure_sha256[:20]>``,
    so a copy under another module name coexists with the installed package: each copy JIT-compiles its own
    sources (digest-verified against its own ``build.json``) through the local FlashInfer toolchain.  Two entries
    with the same closure digest and identical sources share one cached library (harmless); the same closure with
    different sources would reuse a stale library, which :func:`check_manifest_pair` refuses up front.
    """
    directory = Path(directory).resolve()
    init = directory / "__init__.py"
    if not init.is_file():
        raise SystemExit(f"--main-package {directory} has no __init__.py")
    if not (directory / "build.json").is_file():
        raise SystemExit(f"--main-package {directory} has no build.json")
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(directory)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.cake_backend")


def _manifest_entries(directory: Path) -> tuple[dict, dict]:
    """``(name -> entry, generator)`` of the ``build.json`` under ``directory``."""
    document = json.loads((Path(directory) / "build.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in document["libraries"]}
    return entries, document.get("generator", {})


def check_manifest_pair(candidate: Path, main: Path) -> dict:
    """Entries whose closure digest is equal in both manifests but whose source digests differ: such a pair would
    reuse one JIT library for two programs.  Returns the summary (shared / changed / collisions)."""
    cand, cand_gen = _manifest_entries(candidate)
    base, base_gen = _manifest_entries(main)
    shared = sorted(set(cand) & set(base))
    same_closure = [
        n for n in shared if cand[n]["closure_sha256"] == base[n]["closure_sha256"]
    ]
    collisions = [n for n in same_closure if cand[n]["sha256"] != base[n]["sha256"]]
    changed = [n for n in shared if n not in same_closure]
    return dict(
        candidate=str(candidate),
        main=str(main),
        candidate_generator=cand_gen,
        main_generator=base_gen,
        entries_candidate=len(cand),
        entries_main=len(base),
        shared=len(shared),
        unchanged_closure=len(same_closure),
        changed_closure=changed,
        closure_collisions=collisions,
    )


def main_backend():
    if _MAIN["backend"] is None:
        raise RuntimeError("the cake_main arms need --main-package DIR")
    return _MAIN["backend"]


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

    def finish(self, out):
        """The contract's ``(indices, scores)`` of one timed call's output (outside the timed span)."""
        return out

    def outputs(self, out):
        """The tensors of one timed call's output that are the arm's results (``peak_extra_bytes`` counts the rest)."""
        return out


def row_extra(inputs) -> dict:
    """Backend keywords a row adds beyond ``inputs.kwargs()`` (``max_seqlen_k`` for the bounded sink rows)."""
    row = ROWS.get(inputs.label)
    if row is not None and row["bound_seqlen"]:
        return {"max_seqlen_k": max(row["seg_k"])}
    return {}


def backend_api(backend, name: str, *previous: str):
    """The callable ``name`` of ``backend``; a baseline package copy that predates the current names (see
    ``--main-package``) provides the same callable under a previous name: one of ``previous`` (for example the
    workspace bound's ``dsa_indexer_det_workspace_size`` before flashinfer-ai/flashinfer#8928), else the name
    without its first ``_det`` (``plan_dsa_indexer_det_topk`` -> ``plan_dsa_indexer_topk``)."""
    for candidate in (name, *previous):
        fn = getattr(backend, candidate, None)
        if fn is not None:
            return fn
    return getattr(backend, name.replace("_det", "", 1))


def form_extra(inputs) -> dict:
    """``dtype`` for the FP8 form's plan and workspace bound; nothing for BF16 (a main package copy that predates the
    keyword serves the BF16 arms unchanged)."""
    return {"dtype": inputs.q.dtype} if inputs.fp8 else {}


def plan_record(backend, inputs, knobs: dict) -> dict:
    """The host plan of ``backend`` for ``inputs`` under ``knobs`` (the record of truth of what each arm ran)."""
    plan = backend_api(backend, "plan_dsa_indexer_det_topk")(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=inputs.device,
        **row_extra(inputs),
        **form_extra(inputs),
        **knobs,
    )
    return dict(
        scan_key=plan.scan_key,
        merge_key=plan.merge_key,
        finalize_key=plan.finalize_key,
        kind=plan.kind,
        block_q=int(plan.block_q),
        n_split=int(plan.n_split),
        grid_ctas=int(plan.grid_ctas),
        cand_cap=int(plan.cand_cap),
        check_period=int(plan.check_period),
        sample_tiles_max=int(plan.sample_tiles_max),
        sample_shift_permille=int(plan.sample_shift_permille),
        # None: a main package without the knob
        far_sample=getattr(plan, "far_sample", None),
        sample_fit=bool(plan.sample_fit),
    )


@contextlib.contextmanager
def forced_policy(backend, overrides: dict):
    """Rebind ``backend.policy_for`` to the architecture's registry record with ``overrides`` applied.

    The forced-dispatch arms measure the same package under another host dispatch rule without touching it:
    ``plan_dsa_indexer_det_topk`` and ``prepare_dsa_indexer_det_topk`` look ``policy_for`` up on the module at every call, so
    the workspace bound, the plan record and the prepared launches of an arm built inside this scope all follow the
    forced policy (``DispatchPolicy.from_record`` validates the record as it does the registry's); the original
    function is restored on exit.  The FP8 form's record (``fp8_policy_record`` of ``POLICY`` and ``POLICY_FP8``) takes
    the same overrides.
    """
    original = backend.policy_for
    forced = {}

    def policy_for(arch, form="bf16"):
        if (arch, form) not in forced:
            record = backend.POLICY[arch]
            if form != "bf16":
                record = backend.fp8_policy_record(record, backend.POLICY_FP8[arch])
            forced[(arch, form)] = backend.DispatchPolicy.from_record(
                {**record, **overrides}
            )
        return forced[(arch, form)]

    backend.policy_for = policy_for
    try:
        yield
    finally:
        backend.policy_for = original


@contextlib.contextmanager
def forced_fp8_fallback(backend, overrides: dict):
    """The previous FP8 dispatch for an arm built inside this scope: entries of the FP8 kind fallback
    (``cake_policy.FP8_KIND_FALLBACK``) replaced, and the fallback program run with the decided kind's sampled first
    threshold and check period (``cake_policy.fallback_knobs`` made the identity).  ``select_fp8_program`` reads both
    module globals at every call; the originals are restored on exit."""
    module = sys.modules[backend.select_fp8_program.__module__]
    table = module.FP8_KIND_FALLBACK
    saved, saved_knobs = dict(table), module.fallback_knobs
    table.update(overrides)
    module.fallback_knobs = lambda policy, choice, physical, call: choice
    try:
        yield
    finally:
        table.clear()
        table.update(saved)
        module.fallback_knobs = saved_knobs


@contextlib.contextmanager
def hidden_fp8_kinds(backend, kinds):
    """The FP8 dispatch without the FP8 scan programs of the physical ``kinds`` for an arm built inside this scope:
    ``plan_dsa_indexer_det_topk`` looks ``fp8_programs_for`` up on the module at every call, so a decision of a hidden
    kind runs the program of its ``cake_policy.FP8_KIND_FALLBACK`` kind with that kind's sampled first threshold and
    check period (``cake_policy.fallback_knobs``), as on an architecture that registers none.  The process cache of
    ``fp8_programs_for`` is read, not changed; the original function is restored on exit."""
    original = backend.fp8_programs_for

    def fp8_programs_for(arch):
        return {
            (kind, lsf): unrolls
            for (kind, lsf), unrolls in original(arch).items()
            if kind not in kinds
        }

    backend.fp8_programs_for = fp8_programs_for
    try:
        yield
    finally:
        backend.fp8_programs_for = original


class CakeArm(Arm):
    name = "cake"

    def __init__(
        self,
        inputs,
        *,
        backend=None,
        knobs=None,
        name=None,
        policy=None,
        fp8_fallback=None,
        fp8_hidden=None,
    ):
        super().__init__(inputs)
        self.backend = cake_backend if backend is None else backend
        self.knobs = dict(knobs or {})
        # forced dispatch: registry fields replaced for this arm (empty: the package's own policy), FP8 kind
        # fallback entries replaced (empty: the package's own) and the FP8 programs of these physical kinds hidden
        # (empty: every registered one)
        self.policy = dict(policy or {})
        self.fp8_fallback = dict(fp8_fallback or {})
        self.fp8_hidden = list(fp8_hidden or ())
        if name is not None:
            self.name = name
        dev = inputs.device
        scope = contextlib.ExitStack()
        if self.policy:
            scope.enter_context(forced_policy(self.backend, self.policy))
        if self.fp8_fallback:
            scope.enter_context(forced_fp8_fallback(self.backend, self.fp8_fallback))
        if self.fp8_hidden:
            scope.enter_context(hidden_fp8_kinds(self.backend, self.fp8_hidden))
        with scope:  # one scope: the workspace bound, the plan record and prepare see the same policy
            self.workspace_bytes = backend_api(
                self.backend,
                "cake_dsa_indexer_det_train_topk_workspace_size",
                "dsa_indexer_det_workspace_size",
                "dsa_indexer_workspace_size",
            )(
                inputs.num_queries,
                inputs.num_keys,
                inputs.num_segments,
                top_k=inputs.top_k,
                ratio=inputs.ratio,
                device=dev,
                **form_extra(inputs),
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
            self.plan = plan_record(self.backend, inputs, self.knobs)
            self.runner = backend_api(self.backend, "prepare_dsa_indexer_det_topk")(
                inputs.q,
                inputs.k,
                inputs.w,
                inputs.cu_seqlens_q,
                inputs.cu_seqlens_k,
                **inputs.kwargs(),
                **row_extra(inputs),
                **self.knobs,
                workspace_buffer=self.workspace,
                indices=self.indices,
                scores=self.scores,
            )

    def __call__(self):
        return self.runner()

    def outputs(self, out):
        """None: the runner writes into ``indices`` / ``scores`` allocated before the measured call."""
        return ()


class ChunkedPipelineArm(Arm):
    name = "chunked_pipeline"

    def __call__(self):
        return chunked_pipeline(self.inputs)


class MaterializedArm(Arm):
    name = "materialized_fp32"

    def __call__(self):
        return materialized_fp32(self.inputs)


def _aligned(t):
    """``t`` contiguous at a 16-byte aligned address (DeepGEMM asserts it of its operands): a segment slice of a packed
    call that starts off that alignment is copied once, when the arm is built."""
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def deep_gemm_fp8_mqa_logits(deep_gemm):
    """DeepGEMM's FP8 MQA logits with FP32 per-key scales and FP32 logits as ``f(q, (k, k_scale), w, starts, ends, *,
    clean_logits)``: ``fp8_mqa_logits`` where the build has it, else ``fp8_fp4_mqa_logits`` with an FP8 ``q`` (no
    scale factors) and ``logits_dtype=torch.float32``.  A build whose call does not take these arguments (one that needs
    block scale factors for FP8) fails at the arm's first call, which reports the arm unavailable."""
    if hasattr(deep_gemm, "fp8_mqa_logits"):
        return deep_gemm.fp8_mqa_logits
    if not hasattr(deep_gemm, "fp8_fp4_mqa_logits"):
        raise AttributeError(
            "deep_gemm has neither fp8_mqa_logits nor fp8_fp4_mqa_logits"
        )

    def logits(q, kv, w, starts, ends, *, clean_logits):
        return deep_gemm.fp8_fp4_mqa_logits(
            q=(q, None),
            kv=kv,
            weights=w,
            cu_seq_len_k_start=starts,
            cu_seq_len_k_end=ends,
            clean_logits=clean_logits,
            max_seqlen_k=0,
            logits_dtype=torch.float32,
        )

    return logits


class DeepGemmArm(Arm):
    """Stock DeepGEMM FP8 MQA logits (:func:`deep_gemm_fp8_mqa_logits`) per segment, then a FlashInfer exact
    deterministic top-k (FP8 inputs only).

    The timed call returns the raw per-segment outputs; :meth:`finish` converts them to the contract's ascending
    segment-local ids with aligned scores and padding outside the timed span.  A first call in the constructor makes a
    missing ``deep_gemm`` or an unsupported call an unavailable arm."""

    name = "deepgemm_topk"
    clean_logits = True
    # True when DeepGEMM returned a row-padded logits view and the arm copied it to a contiguous [Lq, Lk] tensor
    # inside the timed span (recorded per arm)
    logits_copied = False

    def __init__(self, inputs):
        super().__init__(inputs)
        if not inputs.fp8:
            raise ValueError(
                "the DeepGEMM arms take the FP8 form's inputs (--form fp8)"
            )
        if inputs.top_k > DETERMINISTIC_TOPK_MAX_K:
            raise ValueError(
                f"FlashInfer's deterministic top-k serves k <= {DETERMINISTIC_TOPK_MAX_K}"
            )
        import deep_gemm

        self.logits = deep_gemm_fp8_mqa_logits(deep_gemm)
        visible = ref.visible_rows(inputs).to(torch.int32)
        self.segments = []
        for q0, lq, k0, lk, _visible, _visible_cpu in _segments(inputs):
            ends = _aligned(visible[q0 : q0 + lq])
            self.segments.append(
                dict(
                    q0=q0,
                    lq=lq,
                    lk=lk,
                    q=_aligned(inputs.q[q0 : q0 + lq]),
                    kv=(
                        _aligned(inputs.k[k0 : k0 + lk]),
                        _aligned(inputs.k_scale[k0 : k0 + lk]),
                    ),
                    w=_aligned(inputs.w[q0 : q0 + lq]),
                    starts=torch.zeros_like(ends),
                    ends=ends,
                )
            )
        self.finish(self())
        torch.cuda.synchronize()

    def _segment_logits(self, seg):
        logits = self.logits(
            seg["q"],
            seg["kv"],
            seg["w"],
            seg["starts"],
            seg["ends"],
            clean_logits=self.clean_logits,
        )[:, : seg["lk"]]
        if not logits.is_contiguous():
            self.logits_copied = True
            logits = logits.contiguous()
        return logits

    def _segment_topk(self, seg, logits):
        return select_topk(logits, min(self.inputs.top_k, seg["lk"]))

    def __call__(self):
        out = []
        for seg in self.segments:
            logits = self._segment_logits(seg)
            out.extend((logits, *self._segment_topk(seg, logits)))
        return tuple(out)

    def _segment_result(self, seg, logits, first, second):
        values, ids = first, second.to(torch.int64)
        return ids, values, ids < seg["ends"].unsqueeze(1).to(torch.int64)

    def outputs(self, out):
        """The top-k results of every segment, each tensor once; the logits are the arm's working memory."""
        unique = {}
        for i in range(0, len(out), 3):
            for t in out[i + 1 : i + 3]:
                unique.setdefault(t.data_ptr(), t)
        return list(unique.values())

    def finish(self, out):
        inputs = self.inputs
        T, K, dev = inputs.num_queries, inputs.top_k, inputs.device
        out_ids = torch.full((T, K), -1, dtype=torch.int32, device=dev)
        out_scores = torch.full((T, K), float("-inf"), dtype=torch.float32, device=dev)
        for i, seg in enumerate(self.segments):
            logits, first, second = out[3 * i : 3 * i + 3]
            ids, values, valid = self._segment_result(seg, logits, first, second)
            o_ids, o_scores = ascending_with_padding(ids, values.float(), valid)
            n = o_ids.shape[1]
            out_ids[seg["q0"] : seg["q0"] + seg["lq"], :n] = o_ids
            out_scores[seg["q0"] : seg["q0"] + seg["lq"], :n] = o_scores
        return out_ids, out_scores


class DeepGemmRaggedArm(DeepGemmArm):
    """The DeepGEMM logits without the logits clean, then ``top_k_ragged_transform`` over each row's visible keys (ids
    only: the scores are gathered from the logits in :meth:`finish`)."""

    name = "deepgemm_ragged_topk"
    clean_logits = False

    def _segment_logits(self, seg):
        """The logits buffer as DeepGEMM allocated it: ``top_k_ragged_transform`` reads ``[0, ends)`` of each row of a
        contiguous ``(Lq, row_stride)`` input, so the row-padded buffer serves it without a copy."""
        logits = self.logits(
            seg["q"],
            seg["kv"],
            seg["w"],
            seg["starts"],
            seg["ends"],
            clean_logits=self.clean_logits,
        )
        lq, stride = logits.shape[0], logits.stride(0)
        needed = (logits.storage_offset() + lq * stride) * logits.element_size()
        if logits.stride(1) == 1 and logits.untyped_storage().nbytes() >= needed:
            return logits.as_strided((lq, stride), (stride, 1))
        self.logits_copied = True
        return logits[:, : seg["lk"]].contiguous()

    def _segment_topk(self, seg, logits):
        ids = flashinfer.top_k_ragged_transform(
            logits,
            seg["starts"],
            seg["ends"],
            self.inputs.top_k,
            deterministic=True,
            tie_break=TopKTieBreak.LARGE,
        )
        return ids, ids

    def _segment_result(self, seg, logits, first, second):
        ids = first.to(torch.int64)
        valid = ids >= 0
        values = logits.float().gather(1, ids.clamp_min(0))
        return ids, values, valid


NOSAMPLE = {"sample_tiles_max": 0}
FAR = {"far_sample": 1}
# forced dispatch records: registry fields replaced through ``forced_policy`` (these arms run the local package only).
# ``pair_for`` is gated by the record's ``pair_units`` (the units whose pair program the architecture registers), so a
# record that drops ``l6_rule`` cannot compose the unregistered ``pair_wide`` on sm_100a / sm_103a; ``WIDE`` keeps dropping
# ``pair_rule`` so the arm is the single-CTA wide program on sm_107a (which registers ``pair_wide``) as well.  The plain-l6
# and reverse pair arms pin ``wide_first_rule`` to None: without the pin they would inherit an architecture's wide-first
# precedence and run the wide program on every eligible row at or above its floor.  Every forced unit arm pins
# ``wide_wave_rule`` to None: a record that sets it (the sm_100a and sm_103a FP8 records) decides narrow / l6 against
# wide by waves of the persistent grid instead of the rules these arms replace.
WIDE = {"l6_rule": None, "pair_rule": None, "wide_wave_rule": None}
L6 = {"pair_rule": None, "wide_first_rule": None, "wide_wave_rule": None}
PAIR_L6 = {
    "l6_rule": [1024.0, 512],
    "pair_rule": [128.0, 1024.0, 512],
    "wide_first_rule": None,
    "wide_wave_rule": None,
}
WIDE_C1 = {
    "l6_rule": None,
    "pair_rule": None,
    "wide_rule": [1.0, 0.0],
    "wide_wave_rule": None,
}
# forced narrow: no l6, no pair, and a wide rule no call can meet (the single-warpgroup narrow program on every row)
NARROW = {
    "l6_rule": None,
    "pair_rule": None,
    "wide_rule": [1e18, 1e18],
    "wide_wave_rule": None,
}
# the 16384-slot capacity floor on every top_k (no top_k-keyed floor rule): the plan of the package before the rule, so on
# sm_100a a top_k <= 512 row runs the same programs with twice the candidate buffers (the same bits) in this build; the
# sm_103a record carries this floor already (there the arm is a second A/A twin)
FLOOR16K = {"cand_cap_floor": 16384, "cand_cap_floor_rule": None}
# FP8 dispatch measurements: every row on the single-CTA l6 program (no pair, no wide-first, a wide rule no call meets)
# where the architecture registers the l6 key the row composes (sm_107a has no BF16 snake-order l6 program in the plain
# fit form: a causal single-segment BF16 row above 512 mean unit tiles has none), and every FP8 row on the narrow
# program (NARROW without the unpaired-l6 preference, which sm_107a's record applies to packed and short-unit calls)
L6_ALL = {
    "l6_rule": [1e9, 0],
    "pair_rule": None,
    "wide_first_rule": None,
    "wide_rule": [1e18, 1e18],
    "wide_wave_rule": None,
}
FP8_NARROW = {**NARROW, "l6_solo_rule": None, "l6_solo_unsampled_max_queries": None}
# the FP8 kind fallback before a wide decision ran the FP8 l6 program: the narrow program
FP8_FALLBACK_NARROW = {"wide": "narrow", "pair_wide": "narrow"}
# the FP8 dispatch before the FP8 wide program: its programs hidden (``hidden_fp8_kinds``), a wide decision runs the
# program of its FP8 kind fallback (the l6 program, or with FP8_FALLBACK_NARROW the narrow one)
FP8_NO_WIDE = ("wide",)


def _cake(
    name, backend=None, knobs=None, policy=None, fp8_fallback=None, fp8_hidden=None
):
    def make(inputs):
        return CakeArm(
            inputs,
            backend=main_backend() if backend == "main" else None,
            knobs=knobs,
            name=name,
            policy=policy,
            fp8_fallback=fp8_fallback,
            fp8_hidden=fp8_hidden,
        )

    return make


# name -> factory(inputs) -> Arm.  ``cake_aa`` / ``cake_far_aa`` / ``cake_main_aa`` prepare the same arm a second time (the
# in-process A/A band); the ``_nosample`` arms run the same programs with the sampled first threshold off, ``cake_far`` with
# the far-sample first threshold (``far_sample=1``; the main package has no such knob, so the arm has no main twin).  The
# forced-dispatch arms (``cake_wide`` / ``cake_l6`` / ``cake_pair_l6`` / ``cake_wide_c1`` / ``cake_narrow`` and their
# ``_nosample`` twins) run the local package under a replaced dispatch rule -- ``cake_wide`` without the three-warpgroup
# and CTA-pair rules (the single-CTA wide program), ``cake_l6`` without the CTA-pair and wide-first rules,
# ``cake_pair_l6`` with both unit rules widened to 1024 mean unit tiles and no wide-first rule, ``cake_wide_c1`` the wide
# program from one round, ``cake_narrow`` the narrow program -- so the same bits are expected against ``cake`` /
# ``cake_nosample`` while the programs (``plan``) differ.  ``cake_floor16k`` / ``cake_floor16k_aa`` run the local package
# under the 16384-slot floor on every top_k (the record before ``cand_cap_floor_rule``): the same programs, twice the
# candidate buffers on sm_100a's top_k <= 512 rows (the sm_103a record keeps that floor), the same bits.  The FP8 arms
# ``cake_fp8_w6`` / ``cake_fp8_wn`` run the local package under a previous FP8 dispatch, the FP8 wide programs hidden
# (a wide decision on the l6 / the narrow program): the same bits as ``cake_fp8``.
ARMS = {
    "cake": _cake("cake"),
    "cake_nosample": _cake("cake_nosample", knobs=NOSAMPLE),
    "cake_far": _cake("cake_far", knobs=FAR),
    "cake_aa": _cake("cake_aa"),
    "cake_far_aa": _cake("cake_far_aa", knobs=FAR),
    "cake_main": _cake("cake_main", backend="main"),
    "cake_main_nosample": _cake("cake_main_nosample", backend="main", knobs=NOSAMPLE),
    "cake_main_aa": _cake("cake_main_aa", backend="main"),
    "cake_wide": _cake("cake_wide", policy=WIDE),
    "cake_wide_nosample": _cake("cake_wide_nosample", knobs=NOSAMPLE, policy=WIDE),
    "cake_l6": _cake("cake_l6", policy=L6),
    "cake_l6_nosample": _cake("cake_l6_nosample", knobs=NOSAMPLE, policy=L6),
    "cake_pair_l6": _cake("cake_pair_l6", policy=PAIR_L6),
    "cake_pair_l6_nosample": _cake(
        "cake_pair_l6_nosample", knobs=NOSAMPLE, policy=PAIR_L6
    ),
    "cake_wide_c1": _cake("cake_wide_c1", policy=WIDE_C1),
    "cake_wide_c1_nosample": _cake(
        "cake_wide_c1_nosample", knobs=NOSAMPLE, policy=WIDE_C1
    ),
    "cake_narrow": _cake("cake_narrow", policy=NARROW),
    "cake_narrow_nosample": _cake(
        "cake_narrow_nosample", knobs=NOSAMPLE, policy=NARROW
    ),
    "cake_floor16k": _cake("cake_floor16k", policy=FLOOR16K),
    "cake_floor16k_aa": _cake("cake_floor16k_aa", policy=FLOOR16K),
    ChunkedPipelineArm.name: ChunkedPipelineArm,
    MaterializedArm.name: MaterializedArm,
    # the FP8 form (``--form fp8``): the same runner on the FP8 inputs, and the DeepGEMM compositions
    "cake_fp8": _cake("cake_fp8"),
    "cake_fp8_aa": _cake("cake_fp8_aa"),
    "cake_fp8_nosample": _cake("cake_fp8_nosample", knobs=NOSAMPLE),
    "cake_fp8_l6": _cake("cake_fp8_l6", policy=L6_ALL),
    "cake_fp8_narrow": _cake("cake_fp8_narrow", policy=FP8_NARROW),
    "cake_fp8_w6": _cake("cake_fp8_w6", fp8_hidden=FP8_NO_WIDE),
    "cake_fp8_wn": _cake(
        "cake_fp8_wn", fp8_fallback=FP8_FALLBACK_NARROW, fp8_hidden=FP8_NO_WIDE
    ),
    "cake_l6_all": _cake("cake_l6_all", policy=L6_ALL),
    DeepGemmArm.name: DeepGemmArm,
    DeepGemmRaggedArm.name: DeepGemmRaggedArm,
}
MAIN_ARMS = ("cake_main", "cake_main_nosample", "cake_main_aa")
# arms that take the FP8 inputs in ``--form fp8``; every other arm takes the same data dequantized to BF16
FP8_ARMS = (
    "cake_fp8",
    "cake_fp8_aa",
    "cake_fp8_nosample",
    "cake_fp8_l6",
    "cake_fp8_narrow",
    "cake_fp8_w6",
    "cake_fp8_wn",
    DeepGemmArm.name,
    DeepGemmRaggedArm.name,
)
DEFAULT_ARMS = {
    "bf16": "cake,chunked_pipeline,materialized_fp32",
    "fp8": "cake_fp8,cake,deepgemm_topk,deepgemm_ragged_topk",
}
TIMING_BOUNDARY = (
    "cake arms: the complete operator (scan, merge, finalize) of a prepared runner into caller-owned outputs and "
    "workspace; deepgemm arms: per segment the fp8_mqa_logits call (its logits allocation included, and a contiguous "
    "copy of row-padded logits where an arm records logits_copied: the dense top-k arm; the ragged arm reads the padded "
    "rows in place) and the top-k call, without the conversion of their "
    "output to ascending segment-local ids with aligned scores and padding; "
    "CUPTI GPU span of every launched kernel per iteration, cold L2"
)


def arm_form(arm_name: str) -> str:
    return "fp8" if arm_name in FP8_ARMS else "bf16"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


FALLBACK_TEXT = "Falling back to CUDA events"  # flashinfer/testing/utils.py, bench_gpu_time_with_cupti
TIMER = {"required": False, "fallbacks": 0, "calls": 0}


def gpu_times(fn, reps, warmup):
    """Per-iteration GPU spans (ms); records whether CUPTI or the CUDA-event fallback timed them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        times = bench_gpu_time(
            fn,
            dry_run_iters=warmup,
            repeat_iters=reps,
            enable_cupti=True,
            cold_l2_cache=True,
        )
    TIMER["calls"] += 1
    fallback = [w for w in caught if FALLBACK_TEXT in str(w.message)]
    if fallback:
        TIMER["fallbacks"] += 1
        message = str(fallback[0].message)
        if TIMER["required"]:
            raise SystemExit(f"--cupti-required: {message}")
        if TIMER["fallbacks"] == 1:
            print(f"WARNING: {message}", flush=True)
    for w in caught:
        if FALLBACK_TEXT not in str(w.message) and not issubclass(
            w.category, ExperimentalWarning
        ):
            warnings.warn_explicit(
                w.message, w.category, w.filename, w.lineno, source=w.source
            )
    return [float(t) for t in times]


def timer_record() -> dict:
    return dict(
        name="cupti" if TIMER["fallbacks"] == 0 else "cuda_events_fallback",
        fallback_calls=TIMER["fallbacks"],
        calls=TIMER["calls"],
        required=TIMER["required"],
    )


def result_digest(out) -> dict:
    """SHA-256 of the ids and of the score bits of one ``(indices, scores)`` result (cross-process identity)."""
    ids, scores = out
    ids_bytes = ids.contiguous().cpu().numpy().tobytes()
    bits_bytes = scores.contiguous().view(torch.int32).cpu().numpy().tobytes()
    return dict(
        ids_sha256=hashlib.sha256(ids_bytes).hexdigest(),
        score_bits_sha256=hashlib.sha256(bits_bytes).hexdigest(),
    )


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
    return extra - sum(t.numel() * t.element_size() for t in fn.outputs(out))


def measure_row(name, inputs, arms, args, bf16_inputs=None):
    """``bf16_inputs``: in ``--form fp8``, the FP8 ``inputs`` dequantized to BF16, which every arm outside
    ``FP8_ARMS`` takes."""
    row = ROWS[name]
    reps = args.reps if args.reps is not None else row["reps"]
    fp8 = bf16_inputs is not None
    arm_inputs = {
        a: bf16_inputs if fp8 and arm_form(a) == "bf16" else inputs for a in arms
    }
    prepared = {}
    unavailable = {}
    for arm_name in arms:
        try:
            prepared[arm_name] = ARMS[arm_name](arm_inputs[arm_name])
        except Exception as exc:  # noqa: BLE001 - report the arm as unavailable, keep the others
            if arm_name.startswith("cake"):
                # a comparison without one of its cake arms is not the A/B the row set exists for
                raise SystemExit(
                    f"[{name}] {arm_name}: {type(exc).__name__}: {exc}"
                ) from exc
            unavailable[arm_name] = f"{type(exc).__name__}: {exc}"
            print(
                f"  [{name}] {arm_name}: unavailable ({type(exc).__name__}: {exc})",
                flush=True,
            )
    anchor = "cake_fp8" if fp8 else "cake"
    result = dict(
        row=name,
        T=inputs.num_queries,
        Tkv=inputs.num_keys,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        reps=reps,
        form="fp8" if fp8 else "bf16",
        anchor=anchor,
        unavailable=unavailable,
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
        out = arm.finish(arm())
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
            structure_failures=ref.structure_failures(
                arm_inputs[arm_name], *outputs[arm_name]
            )[:3],
        )
        record["gpu_ms"]["all"] = statistics.median(every)
        record["digest"] = result_digest(outputs[arm_name])
        if isinstance(arm, DeepGemmArm):
            record["logits_copied"] = arm.logits_copied
        if isinstance(arm, CakeArm):
            record["plan"] = arm.plan
            record["knobs"] = arm.knobs
            record["policy_overrides"] = arm.policy
            record["fp8_kind_fallback_overrides"] = arm.fp8_fallback
            record["fp8_hidden_kinds"] = arm.fp8_hidden
            record["package"] = (
                "main" if arm.backend is not cake_backend else "candidate"
            )
        result["arms"][arm_name] = record
    # bitwise identity of every pair of arms (ids and score bits): the contract says every cake arm agrees
    result["same_bits"] = {
        f"{a}|{b}": ref.same_bits(outputs[a], outputs[b])
        for a, b in itertools.combinations(outputs, 2)
    }
    # the cake arms of one form agree bitwise; the FP8 form's bits are not the BF16 form's
    cake_pairs = [key for key in result["same_bits"] if _cake_pair(key)]
    result["all_cake_arms_identical"] = (
        all(result["same_bits"][key] for key in cake_pairs) if cake_pairs else None
    )
    if anchor in outputs:
        for arm_name, out in outputs.items():
            if arm_name != anchor:
                same = (out[0] == outputs[anchor][0]).all(dim=1)
                result["arms"][arm_name]["rows_identical_to_cake"] = int(same.sum())
                result["arms"][arm_name]["speedup_of_cake"] = (
                    result["arms"][arm_name]["gpu_ms"]["all"]
                    / result["arms"][anchor]["gpu_ms"]["all"]
                )
    if args.verify:
        references = {}
        for arm_name, out in outputs.items():
            data = arm_inputs[arm_name]
            if id(data) not in references:
                references[id(data)] = ref.select_reference(data)
            verdict = ref.judge(out, references[id(data)])
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


def _cake_pair(key: str) -> bool:
    """``a|b`` names two cake arms of one form."""
    a, b = key.split("|")
    return a.startswith("cake") and b.startswith("cake") and arm_form(a) == arm_form(b)


def print_row(result):
    for arm_name, rec in result["arms"].items():
        gpu = rec["gpu_ms"]
        extra = (
            ""
            if rec["workspace_bytes"] is None
            else f" workspace {rec['workspace_bytes'] / 2**20:.1f} MiB"
        )
        speed = (
            f" x{rec['speedup_of_cake']:.2f} vs {result.get('anchor', 'cake')}"
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
            f"  [{result['row']}] {arm_name:<22} gpu {gpu['all']:10.3f} ms (AB {gpu['AB']:.3f} / BA {gpu['BA']:.3f} / IL {gpu['IL']:.3f})"
            f" wall {rec['wall_ms']:.3f} ms extra {rec['peak_extra_bytes'] / 2**20:.1f} MiB{extra}{speed}{bad}{verdict}",
            flush=True,
        )
    if "reference_gb200_ms" in result:
        print(
            f"  [{result['row']}] coarse-only GB200 reference latency quoted by the request: {result['reference_gb200_ms']} ms"
        )
    same = result.get("same_bits", {})
    cake_pairs = [key for key in same if _cake_pair(key)]
    if cake_pairs:
        identical = sum(same[key] for key in cake_pairs)
        differing = [key for key, v in same.items() if not v]
        tail = f"; DIFFER {differing}" if differing else ""
        print(
            f"  [{result['row']}] same_bits: {identical}/{len(cake_pairs)} cake pairs identical{tail}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--form",
        choices=FORMS,
        default="bf16",
        help="input form: bf16 (default) or fp8 (float8_e4m3fn q / k with per-key scales; see the module docstring)",
    )
    parser.add_argument(
        "--rows",
        nargs="*",
        default=None,
        help=f"rows to measure (default: all regression rows, iid and sink, with --form bf16; the fp8 group with --form fp8; the group names iid / sink / pool / all / fp8 expand; also {', '.join(n for n in ROWS if n.startswith('smoke'))})",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help=f"comma-separated arms in AB order (default: {DEFAULT_ARMS['bf16']} with --form bf16, {DEFAULT_ARMS['fp8']} with --form fp8)",
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
    parser.add_argument(
        "--main-package",
        default=None,
        help=(
            "directory of a second copy of the package (e.g. a FlashInfer main checkout's "
            "flashinfer/experimental/cake/cake_dsa_indexer_det_train, or "
            "flashinfer/experimental/cake_dsa_indexer_det in a checkout before #8928) for the cake_main arms"
        ),
    )
    parser.add_argument(
        "--cupti-required",
        action="store_true",
        help="fail when flashinfer.testing.bench_gpu_time falls back from CUPTI to CUDA events",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore", ExperimentalWarning)
    TIMER["required"] = bool(args.cupti_required)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    fp8 = args.form == "fp8"
    if fp8 and args.strided_k:
        raise SystemExit(
            "--strided-k measures the BF16 rows: the FP8 mode's BF16 and DeepGEMM arms take contiguous keys"
        )
    if args.rows is None:
        args.rows = FP8_ROWS if fp8 else DEFAULT_ROWS
    if args.arms is None:
        args.arms = DEFAULT_ARMS[args.form]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(ARMS)}")
    if len(set(arms)) != len(arms):
        raise SystemExit(
            f"an arm may be named once; use cake_aa / cake_far_aa / cake_main_aa / cake_floor16k_aa / cake_fp8_aa for an A/A pair: {arms}"
        )
    if not fp8 and any(arm_form(a) == "fp8" for a in arms):
        raise SystemExit(f"the arms {list(FP8_ARMS)} need --form fp8")
    args.rows = expand_rows(args.rows)
    unknown_rows = [r for r in args.rows if r not in ROWS]
    if unknown_rows:
        raise SystemExit(
            f"unknown rows {unknown_rows}; known: {sorted(ROWS)} and the groups {sorted(ROW_GROUPS)}"
        )
    candidate_dir = Path(cake_backend.__file__).resolve().parent
    packages = {
        "candidate": dict(
            path=str(candidate_dir), generator=_manifest_entries(candidate_dir)[1]
        )
    }
    if args.main_package is not None:
        main_dir = Path(args.main_package).resolve()
        if main_dir == candidate_dir:
            raise SystemExit(
                f"--main-package {main_dir} is the installed package itself"
            )
        pair = check_manifest_pair(candidate_dir, main_dir)
        if pair["closure_collisions"]:
            raise SystemExit(
                f"--main-package: {len(pair['closure_collisions'])} entries keep the main closure digest with changed sources (a stale JIT library would be shared): {pair['closure_collisions'][:5]}"
            )
        _MAIN["backend"] = load_package_copy(main_dir)
        _MAIN["path"] = str(main_dir)
        packages["main"] = dict(path=str(main_dir), generator=pair["main_generator"])
        packages["manifest_pair"] = pair
        print(
            f"main package {main_dir}: {pair['shared']} shared entries, {len(pair['changed_closure'])} with a changed closure",
            flush=True,
        )
    elif any(a in MAIN_ARMS for a in arms):
        raise SystemExit("the cake_main arms need --main-package DIR")
    print(
        f"{props.name} ({props.multi_processor_count} SMs, cc {props.major}.{props.minor}), rows {args.rows}, arms {arms}"
    )
    if not cake_backend.generated_program_available(device) and "cake" in arms:
        print(
            "generated program not registered for this device: the cake arm is unavailable",
            flush=True,
        )
    if (
        fp8
        and not cake_backend.generated_program_available(
            device, dtype=torch.float8_e4m3fn
        )
        and any(a.startswith("cake_fp8") for a in arms)
    ):
        print(
            "generated FP8 programs not registered for this device: the cake_fp8 arms are unavailable",
            flush=True,
        )
    print(f"timing boundary: {TIMING_BOUNDARY}", flush=True)
    results = []
    for name in args.rows:
        if fp8:
            inputs = build_fp8_inputs(name, device, args.strided_k)
            bf16_inputs = ref.dequantized_bf16(inputs)
            bf16_inputs.label = name
        else:
            inputs, bf16_inputs = build_inputs(name, device, args.strided_k), None
        print(
            f"[{name}] T={inputs.num_queries} Tkv={inputs.num_keys} segments={inputs.num_segments} top_k={inputs.top_k} ratio={inputs.ratio} form={args.form}: {ROWS[name]['note']}",
            flush=True,
        )
        result = measure_row(name, inputs, arms, args, bf16_inputs=bf16_inputs)
        print_row(result)
        results.append(result)
        del inputs, bf16_inputs
        torch.cuda.empty_cache()
    print(f"timer: {timer_record()}", flush=True)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                dict(
                    device=props.name,
                    sms=props.multi_processor_count,
                    results=results,
                    timer=timer_record(),
                    packages=packages,
                    torch=torch.__version__,
                    flashinfer=str(Path(flashinfer.__file__).resolve().parent),
                    arms=arms,
                    warmup=args.warmup,
                    reps=args.reps,
                    form=args.form,
                    timing_boundary=TIMING_BOUNDARY,
                ),
                indent=1,
            )
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
