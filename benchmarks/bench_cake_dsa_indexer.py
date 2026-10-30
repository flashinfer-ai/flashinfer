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
  ``V2`` (top-k 256), ``V3`` (top-k 4096) at ``N = 65536``;
* the top-512 rows of flashinfer-ai/flashinfer#7549 (its ratio-4 pooled
  selection: top-512 pools over a 4-token pool layout, ``ratio = 4`` with the
  keys pooled 4:1): ``V4`` the packed 4k .. 64k documents of ``Z3cp`` (zigzag
  rank 0 of 16) pooled, ``V5`` the packed documents of ``Z3`` (full causal)
  pooled, ``V6`` one 262144-token document pooled, and ``V7`` one 65536-token
  document at top-k 512 without pooling; ``--rows pool`` names the group;
* the attention-sink + local-band rows of flashinfer-ai/flashinfer#7042
  (``tests/test_helpers/cake_dsa_indexer_reference.make_sink_inputs``, the
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
``flashinfer/experimental/cake_dsa_indexer``
directory of a FlashInfer main checkout, in the same process: the package is
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
reference of ``tests/test_helpers/cake_dsa_indexer_reference.py`` (small rows
only; the reference scores every visible pair).

Usage::

    python benchmarks/bench_cake_dsa_indexer.py [--rows P1 S2 ...] [--arms cake,chunked_pipeline,materialized_fp32]
        [--reps 20] [--strided-k] [--verify] [--json out.json]
    python benchmarks/bench_cake_dsa_indexer.py --rows sink --arms cake,cake_nosample,cake_main,cake_main_nosample
        --main-package /path/to/main/flashinfer/experimental/cake_dsa_indexer --cupti-required --json out.json
    python benchmarks/bench_cake_dsa_indexer.py --rows sink --arms cake,cake_far,cake_aa,cake_far_aa --cupti-required --json out.json
    python benchmarks/bench_cake_dsa_indexer.py --rows sink --arms cake,cake_wide,cake_l6,cake_pair_l6,cake_wide_c1 --cupti-required --json out.json
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
ROW_GROUPS = {
    "iid": IID_ROWS,
    "sink": SINK_ROWS,
    "pool": POOL_ROWS,
    "all": DEFAULT_ROWS,
}


def expand_rows(names):
    """``--rows`` with the group names ``iid`` / ``sink`` / ``pool`` / ``all`` expanded, in order, without repeats."""
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


def row_extra(inputs) -> dict:
    """Backend keywords a row adds beyond ``inputs.kwargs()`` (``max_seqlen_k`` for the bounded sink rows)."""
    row = ROWS.get(inputs.label)
    if row is not None and row["bound_seqlen"]:
        return {"max_seqlen_k": max(row["seg_k"])}
    return {}


def plan_record(backend, inputs, knobs: dict) -> dict:
    """The host plan of ``backend`` for ``inputs`` under ``knobs`` (the record of truth of what each arm ran)."""
    plan = backend.plan_dsa_indexer_topk(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=inputs.device,
        **row_extra(inputs),
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
    ``plan_dsa_indexer_topk`` and ``prepare_dsa_indexer_topk`` look ``policy_for`` up on the module at every call, so
    the workspace bound, the plan record and the prepared launches of an arm built inside this scope all follow the
    forced policy (``DispatchPolicy.from_record`` validates the record as it does the registry's); the original
    ``functools.cache`` function is restored on exit.
    """
    original = backend.policy_for
    forced = {}

    def policy_for(arch):
        if arch not in forced:
            forced[arch] = backend.DispatchPolicy.from_record(
                {**backend.POLICY[arch], **overrides}
            )
        return forced[arch]

    backend.policy_for = policy_for
    try:
        yield
    finally:
        backend.policy_for = original


class CakeArm(Arm):
    name = "cake"

    def __init__(self, inputs, *, backend=None, knobs=None, name=None, policy=None):
        super().__init__(inputs)
        self.backend = cake_backend if backend is None else backend
        self.knobs = dict(knobs or {})
        # forced dispatch: registry fields replaced for this arm (empty: the package's own policy)
        self.policy = dict(policy or {})
        if name is not None:
            self.name = name
        dev = inputs.device
        scope = (
            forced_policy(self.backend, self.policy)
            if self.policy
            else contextlib.nullcontext()
        )
        with scope:  # one scope: the workspace bound, the plan record and prepare see the same policy
            self.workspace_bytes = self.backend.dsa_indexer_workspace_size(
                inputs.num_queries,
                inputs.num_keys,
                inputs.num_segments,
                top_k=inputs.top_k,
                ratio=inputs.ratio,
                device=dev,
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
            self.runner = self.backend.prepare_dsa_indexer_topk(
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


class ChunkedPipelineArm(Arm):
    name = "chunked_pipeline"

    def __call__(self):
        return chunked_pipeline(self.inputs)


class MaterializedArm(Arm):
    name = "materialized_fp32"

    def __call__(self):
        return materialized_fp32(self.inputs)


NOSAMPLE = {"sample_tiles_max": 0}
FAR = {"far_sample": 1}
# forced dispatch records: registry fields replaced through ``forced_policy`` (these arms run the local package only).
# ``pair_for`` is gated by the record's ``pair_units`` (the units whose pair program the architecture registers), so a
# record that drops ``l6_rule`` cannot compose the unregistered ``pair_wide`` on sm_100a / sm_103a; ``WIDE`` keeps dropping
# ``pair_rule`` so the arm is the single-CTA wide program on sm_107a (which registers ``pair_wide``) as well.  The plain-l6
# and reverse pair arms pin ``wide_first_rule`` to None: without the pin they would inherit an architecture's wide-first
# precedence and run the wide program on every eligible row at or above its floor.
WIDE = {"l6_rule": None, "pair_rule": None}
L6 = {"pair_rule": None, "wide_first_rule": None}
PAIR_L6 = {
    "l6_rule": [1024.0, 512],
    "pair_rule": [128.0, 1024.0, 512],
    "wide_first_rule": None,
}
WIDE_C1 = {"l6_rule": None, "pair_rule": None, "wide_rule": [1.0, 0.0]}
# forced narrow: no l6, no pair, and a wide rule no call can meet (the single-warpgroup narrow program on every row)
NARROW = {"l6_rule": None, "pair_rule": None, "wide_rule": [1e18, 1e18]}
# the 16384-slot capacity floor on every top_k (no top_k-keyed floor rule): the plan of the package before the rule, so on
# sm_100a a top_k <= 512 row runs the same programs with twice the candidate buffers (the same bits) in this build; the
# sm_103a record carries this floor already (there the arm is a second A/A twin)
FLOOR16K = {"cand_cap_floor": 16384, "cand_cap_floor_rule": None}


def _cake(name, backend=None, knobs=None, policy=None):
    def make(inputs):
        return CakeArm(
            inputs,
            backend=main_backend() if backend == "main" else None,
            knobs=knobs,
            name=name,
            policy=policy,
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
# candidate buffers on sm_100a's top_k <= 512 rows (the sm_103a record keeps that floor), the same bits.
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
}
MAIN_ARMS = ("cake_main", "cake_main_nosample", "cake_main_aa")


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
    return extra - sum(t.numel() * t.element_size() for t in out)


def measure_row(name, inputs, arms, args):
    row = ROWS[name]
    reps = args.reps if args.reps is not None else row["reps"]
    prepared = {}
    for arm_name in arms:
        try:
            prepared[arm_name] = ARMS[arm_name](inputs)
        except Exception as exc:  # noqa: BLE001 - report the arm as unavailable, keep the others
            if arm_name.startswith("cake"):
                # a comparison without one of its cake arms is not the A/B the row set exists for
                raise SystemExit(
                    f"[{name}] {arm_name}: {type(exc).__name__}: {exc}"
                ) from exc
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
        record["digest"] = result_digest(outputs[arm_name])
        if isinstance(arm, CakeArm):
            record["plan"] = arm.plan
            record["knobs"] = arm.knobs
            record["policy_overrides"] = arm.policy
            record["package"] = (
                "main" if arm.backend is not cake_backend else "candidate"
            )
        result["arms"][arm_name] = record
    # bitwise identity of every pair of arms (ids and score bits): the contract says every cake arm agrees
    result["same_bits"] = {
        f"{a}|{b}": ref.same_bits(outputs[a], outputs[b])
        for a, b in itertools.combinations(outputs, 2)
    }
    cake_pairs = [
        key
        for key in result["same_bits"]
        if all(part.startswith("cake") for part in key.split("|"))
    ]
    result["all_cake_arms_identical"] = (
        all(result["same_bits"][key] for key in cake_pairs) if cake_pairs else None
    )
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
            f"  [{result['row']}] {arm_name:<22} gpu {gpu['all']:10.3f} ms (AB {gpu['AB']:.3f} / BA {gpu['BA']:.3f} / IL {gpu['IL']:.3f})"
            f" wall {rec['wall_ms']:.3f} ms extra {rec['peak_extra_bytes'] / 2**20:.1f} MiB{extra}{speed}{bad}{verdict}",
            flush=True,
        )
    if "reference_gb200_ms" in result:
        print(
            f"  [{result['row']}] coarse-only GB200 reference latency quoted by the request: {result['reference_gb200_ms']} ms"
        )
    same = result.get("same_bits", {})
    cake_pairs = [
        key for key in same if all(p.startswith("cake") for p in key.split("|"))
    ]
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
        "--rows",
        nargs="*",
        default=DEFAULT_ROWS,
        help=f"rows to measure (default: all regression rows, iid and sink; the group names iid / sink / all expand; also {', '.join(n for n in ROWS if n.startswith('smoke'))})",
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
    parser.add_argument(
        "--main-package",
        default=None,
        help="directory of a second copy of the package (e.g. a FlashInfer main checkout's flashinfer/experimental/cake_dsa_indexer) for the cake_main arms",
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
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(ARMS)}")
    if len(set(arms)) != len(arms):
        raise SystemExit(
            f"an arm may be named once; use cake_aa / cake_far_aa / cake_main_aa / cake_floor16k_aa for an A/A pair: {arms}"
        )
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
                ),
                indent=1,
            )
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
