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

Fused radix top-k -> sparse top-p -> sampling for Hopper and newer (compute capability 9.x,
10.x, 11.x, 12.x; one frozen source compiled once into a fatbin for the CompilationContext
targets).

Two frozen kernels per row of probabilities:

1. ``radix_topk``: one thread-block cluster per row selects the exact top-k by a three-pass
   radix select (11/11/10 bits of the float32 key), reducing the 2048-bucket histograms across
   the cluster through distributed shared memory, and writes a ``[batch, 1024]`` slab of
   (value, index) pairs.
2. ``sparse_topp_sample``: one CTA per row sorts the slab (descending probability, ascending
   index), keeps the shortest prefix whose exclusive mass is below ``top_p * mass(top-k)``,
   renormalizes, draws one token by inverse CDF from ``curand_init(seed, row, offset)``, and
   rewrites the slab in sorted order.  It is launched with programmatic dependent launch so its
   prologue overlaps the tail of stage 1.

Semantics follow :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs` with
``filter_apply_order="top_k_first"`` (same support, same Philox stream advancement) with these
guarantees on top:

* **Strict determinism.** Ties at the top-k boundary are resolved toward the lower vocabulary
  index (the support is exactly the first ``k`` entries of ``lexsort(-prob, index)``), every
  top-p / sampling decision uses exact 64-bit fixed-point prefix sums, and no atomic decides an
  output.  Identical inputs give bitwise identical samples, ``renorm_out`` and slab for every
  kernel variant, stream launch or CUDA-graph replay.
* **NaN / Inf.** NaN, negative and ``-0.0`` probabilities are treated as ``+0`` and never
  sampled.  A row containing ``+inf`` keeps exactly its ``+inf`` entries, samples uniformly among
  them and renormalizes them to ``1/m`` (``top_p`` is ignored for that row).  A row whose top-k
  mass is zero returns its smallest slab index with an all-zero ``renorm_out``.  Entries below
  ``2**-53`` times the row maximum carry zero mass.  Every output is finite.
* ``top_p`` is clamped to ``(0, 1]`` per row (``p <= 0`` selects the argmax, ``p >= 1`` or NaN
  keeps every entry with mass); ``top_k`` is clamped to ``[1, min(vocab, 1024)]`` per row.

Requests the kernels cannot serve (top-k disabled, ``k > 1024``, non-float32 rows, or an
unsupported GPU) are dispatched to :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs`
(``deterministic=True``); large ``batch * vocab`` launches run on the streaming stage-1 variants.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, TypeAlias, Union

import torch

from .api_logging import flashinfer_api
from .jit.cake_sampling import (
    load_cake_sampling_module,
    load_manifest,
    supported_capability,
)
from .sampling import get_seed_and_offset

_THREADS = 512
_MAX_CLUSTER = 8
_TOPK_SCALAR, _TOPK_PER_ROW = 1, 2
_TOPP_SCALAR, _TOPP_PER_ROW = 1, 2
# Clusters are co-scheduled inside one GPC, so the number of stage-1 CTAs that run in a single
# wave depends on the cluster size and on the device's SM / GPC layout.  Measured single-wave CTA
# capacity per cluster size, keyed by SM count: 148 = B200 / GB300 (B300 tracks it), 132 = H100
# SXM (64 cluster-4 CTAs run in one wave and 128 take two, so the GPC layout bounds the capacity
# to 112-127; 120 is used), 212 = Rubin R200 (cuOccupancyMaxActiveClusters: 106 two-CTA, 46
# four-CTA and 22 eight-CTA clusters for the one-CTA-per-SM variants; the measured wave step of
# the streaming variants sits between 212 and 216 CTAs for clusters of 1-2 and between 176 and
# 192 for clusters of 4-8).  A device with another SM count uses the table of the nearest SM count.
_WAVE_CTAS_BY_SM_COUNT: dict[int, dict[int, int]] = {
    148: {1: 148, 2: 144, 4: 128, 8: 64},
    132: {1: 132, 2: 132, 4: 120, 8: 64},
    212: {1: 212, 2: 212, 4: 184, 8: 176},
}
_DEFAULT_SM_COUNT = 148
_PREFERRED_MIN_EPT = 16
# Stage-1 cost model, per wave table (same keys as _WAVE_CTAS_BY_SM_COUNT).  A register-resident wave
# costs resident_base_us + resident_per_ept_us per register entry; a streaming wave costs
# stream_wave_base_us plus stream_chunk_us per 512 x 16-entry chunk each CTA walks plus
# stream_cluster_cta_us per CTA beyond the first in the cluster; both forms add launch_cta_us per
# launched CTA normalised by the single-CTA wave capacity.  Re-fitted for the round-4 stage-1 kernels
# (streaming template with a gathered candidate list, (8, 16) streaming variant) on per-variant sweeps
# of B200 + B300 (148), H100 (132) and R200 (212), 25 (vocab, batch) cells each: zero regret against the
# measured-best variant on every cell of every table (one shared constant set cannot do that: the
# (8, 32) resident beats the (8, 16) stream at V = 128256, B <= 8 on H100 and up to B = 16 on R200 while
# the stream wins on B200).  A launch whose largest top-k exceeds _STAGE1_LARGE_K (the fused-tail cap)
# adds stream_large_k_us per streaming wave: the streaming template then takes its per-bucket list path,
# which is slower than the register-resident candidate path at the same shape (k = 1000 sweeps: (8, 32)
# beats the (8, 16) stream at V = 128256, B <= 8 on every architecture; the (8, 48) resident beats the
# stream at V = 151936 on H100 and R200 but not on B200 / B300, hence the smaller term for 148).  The
# constants only rank the frozen variants.
_STAGE1_LARGE_K = 64
_Stage1CostRow: TypeAlias = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]
# Round 6 (CAKE-776) re-fitted the k <= 64 stream constants of the 148 and 212 tables on policy-aware sweeps
# (coarse-sample twins at k <= 64; B200 / GB300 / R200, k = 50) under two constraints -- no k > 64 pick changes and no
# cell's pick gets slower: 148 stream_chunk_us 0.25 with stream_large_k_chunk_us 0.35 (large-k per-chunk sum unchanged;
# V = 151936 B <= 8 -> the (8, 32) stream, worst regret 4.5 -> 2.8 %); 212 stream_chunk_us 0.15, stream_chunk32_us 0.15,
# stream_cluster_us 0.25, stream_cluster_cta_us 0.05, stream_large_k_chunk32_us 0.05 (V = 128256 / 151936 B <= 16 -> the
# (4, 32) stream, worst 4.1 -> 1.7 %); 132 unchanged.  Kernels and bundle untouched.
_STAGE1_COST_BY_SM_COUNT: dict[int, _Stage1CostRow] = {
    # (resident_base_us, resident_per_ept_us, stream_wave_base_us, stream_chunk_us,
    #  stream_cluster_cta_us, launch_cta_us, stream_large_k_us, stream_chunk32_us, stream_cluster_us,
    #  stream_wide_wave_us, stream_wide_two_launch_us, notail_two_launch_us, chain_over_tail_us,
    #  stream_wide_large_k_us, resident_two_launch_us, fused_tail_cluster_us, fused_tail_wide_cluster_us,
    #  stream_large_k_chunk_us, stream_large_k_chunk32_us, fused_tail_stream_us, stream_large_k_cluster_cta_us,
    #  stream_wide_short_row_large_k_us)
    # stream_chunk32_us: per-chunk cost of the ept-32 streaming variants (0 = not ranked by this table);
    # stream_cluster_us: fixed cost of a clustered streaming wave (cluster barrier + DSM exchange latency);
    # stream_wide_wave_us: fixed cost per ept-32 streaming wave (its prologue / register footprint; H100 only);
    # stream_wide_two_launch_us: per-launch cost of an ept-32 stream on a row of at most
    # _STREAM_WIDE_TWO_LAUNCH_CHUNKS 512 x 32 chunks (V < 65536) on the two-launch path (top-k above
    # _STAGE1_LARGE_K): with the stage-2/3 launch queued behind it the ept-32 kernel's span grows 3-5 us
    # (measured on B200 / H100 / GB300 / R200, PDL on or off); the ept-16 form and ept-32 rows of >= 4 chunks do not.
    # On a device that runs the whole-CTA tail (_block_tail_for_capability) a candidate whose one-wave stage-1
    # grid fuses (_fuse_block_tail) finishes in one kernel and pays none of the two-launch terms; the candidates
    # that still need the stage-2/3 kernel pay, besides the ept-32 term above, notail_two_launch_us (the (8, 48)
    # resident, built without the tails: its stage-2/3 kernel plus the cold-code cost of a second launch) or
    # chain_over_tail_us (a multi-wave grid: the stage-2/3 kernel over the in-kernel tail of its one-wave peers).
    # stream_wide_large_k_us: per streaming wave of the ept-32 form at large top-k (both regimes): its per-bucket
    # list path and, fused, its in-kernel tail cost more than the ept-16 form's.
    # resident_two_launch_us: charged to a register-resident candidate at large top-k on a device that keeps the
    # two-launch chain (GB300): the stage-2/3 launch queued behind a resident grid costs more than behind a
    # streaming grid; 0 on the devices that fuse the whole-CTA tail.
    # fused_tail_cluster_us: per CTA beyond the first of the cluster when the candidate fuses the whole-CTA tail
    # (rank 0 sorts after the cluster barrier while its peers idle; the gathered list grows with the cluster);
    # fused_tail_wide_cluster_us: the same per-CTA term for the ept-32 streaming form only, whose fused tail grows
    # faster with the cluster than the ept-16 form's.
    148: (
        2.0,
        0.15,
        4.0,
        0.25,
        0.0,
        1.0,
        1.0,
        0.3,
        0.5,
        0.0,
        0.0,
        8.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.35,
        0.6,
        0.0,
        0.2,
        1.0,
    ),
    132: (
        2.0,
        0.15,
        3.0,
        0.5,
        0.1,
        1.5,
        1.0,
        0.8,
        2.0,
        1.5,
        0.0,
        8.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.0,
        0.0,
        0.0,
    ),
    212: (
        1.5,
        0.15,
        4.0,
        0.15,
        0.05,
        0.0,
        2.0,
        0.15,
        0.25,
        0.0,
        3.0,
        8.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.0,
        0.0,
        0.0,
    ),
}
_STREAM_WIDE_TWO_LAUNCH_CHUNKS = 3

_WORKSPACES: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _capability(device: torch.device) -> Optional[tuple[int, int]]:
    return supported_capability(torch.cuda.get_device_capability(device))


def _stage1_variants(smem_limit: Optional[int] = None) -> list[tuple[int, int, bool]]:
    """Frozen stage-1 variants whose dynamic shared memory fits ``smem_limit`` (all when None).

    The frozen variants were sized for the 227 KB opt-in limit of 9.x-11.x devices; 12.x devices
    opt in to 99 KB, so the larger register-resident variants are not candidates there."""
    return [
        (v["cluster"], v["ept"], bool(v["stream"]))
        for v in load_manifest()["stage1"]
        if smem_limit is None or int(v["dynamic_smem_bytes"]) <= smem_limit
    ]


@functools.cache
def _stage23_variants() -> tuple[tuple[int, int], ...]:
    """Distinct frozen ``(threads, items)`` slabs (each is exported once per static form, see
    ``stage23_variant_flags``), smallest slab first.  Cached: the eager k > 64 path picks a slab per launch."""
    seen: list[tuple[int, int]] = []
    for v in load_manifest()["stage23"]:
        ti = (v["threads"], v["items"])
        if ti not in seen:
            seen.append(ti)
    return tuple(sorted(seen, key=lambda ti: ti[0] * ti[1]))


# Static per-architecture forms of the stage-2/3 kernel (manifest ``features`` -> bit): every form is bit-identical;
# the dispatched form is the one that measured fastest on that architecture (round 5, (256,4) kernel, same run):
#   int_tests (bit 0)       integer forms of the f64 top-p cut / sample tests: B300 -7..-10 % (slow FP64 there),
#                           +1..+3 % on B200 / H100 / R200 (the integer precompute chain adds latency).
#   one_cmp_select (bit 1)  one 64-bit compare + select per bitonic exchange: B200 / H100 / B300 -16..-19 % of the
#                           kernel (sort stage 4.0 -> 2.7 us), R200 +0..+2 %.
STAGE23_FEATURE_BITS = {"int_tests": 1, "one_cmp_select": 2}
_STAGE23_FLAGS_BY_CAPABILITY = {
    (9, 0): STAGE23_FEATURE_BITS["one_cmp_select"],
    (10, 0): STAGE23_FEATURE_BITS["one_cmp_select"],
    (10, 3): STAGE23_FEATURE_BITS["int_tests"] | STAGE23_FEATURE_BITS["one_cmp_select"],
}


@functools.cache
def _stage23_variant_flags(device_index: int) -> int:
    cc = tuple(torch.cuda.get_device_capability(device_index))
    flags = _STAGE23_FLAGS_BY_CAPABILITY.get(cc, 0)
    if flags not in {v["variant_flags"] for v in load_manifest()["stage23"]}:
        raise RuntimeError(
            f"frozen bundle lacks stage-2/3 variant_flags={flags} for compute capability {cc}"
        )
    return flags


def stage23_variant_flags(device: torch.device | int | None = None) -> int:
    """Bit mask of the static stage-2/3 features dispatched on ``device`` (``STAGE23_FEATURE_BITS``); 0 = base
    form (f64 tests, max / min exchange), the form every unlisted architecture runs.  Cached per device index:
    the eager k > 64 path calls this on every launch."""
    if isinstance(device, int):
        index = device
    elif device is None or device.index is None:
        index = torch.cuda.current_device()
    else:
        index = device.index
    return _stage23_variant_flags(index)


def _slab() -> int:
    return int(load_manifest()["slab_entries"])


def _fused_tail_kcap() -> int:
    """Largest top-k whose stage 2/3 runs inside the stage-1 kernel (one launch)."""
    return int(load_manifest()["fused_tail_kcap"])


@functools.cache
def _stage1_has_fused_tail(cluster: int, ept: int, stream: bool) -> bool:
    """Whether the frozen variant is built with the fused stage-2/3 tail (manifest ``fused_tail``).

    The (8, 48) resident is a large-k pick and ships without the tail (inlining it cost that variant
    2-7 % of stage-1 time on Blackwell / Rubin), so a launch that lands there takes two kernels."""
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            return bool(v["fused_tail"])
    raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")


def _fused_block_tail_kcap() -> int:
    """Largest top-k the whole-CTA tail serves inside the stage-1 kernel (the slab row)."""
    return int(load_manifest()["fused_block_tail_kcap"])


@functools.cache
def _stage1_has_fused_block_tail(cluster: int, ept: int, stream: bool) -> bool:
    """Whether the frozen variant ships a whole-CTA-tail build (a manifest entry with ``fused_block_tail``): the
    kernel taken by launch_flags bit 3.  The default build of every variant carries only the two-warp tail --
    compiling the whole-CTA tail into it slowed every k <= 64 launch on sm_103a / sm_107a by 13-33 %."""
    found = False
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            found = True
            if v["fused_block_tail"]:
                return True
    if not found:
        raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")
    return False


# Compute capabilities whose launches with _STAGE1_LARGE_K < max top-k <= fused_block_tail_kcap run the whole-CTA
# tail inside the stage-1 kernel (launch_flags bit 3) instead of the stage-2/3 kernel.  Both forms give the same
# outputs; the choice is timing only.  Measured chain -> fused ratios at k = 1000 (cold L2, chain = sum of both
# kernels' spans) on the integrated build: R200 (10, 7) 1.05-1.32 (0.96-0.99 only at V = 32768 B <= 8), B200 (10, 0)
# 1.19 on V = 151936 B = 4, H100 (9, 0) 1.07-1.13 at V = 32768 / 128256 B <= 16 against the chain's own 0.77-0.92,
# winning only V = 151936 B <= 8 (0.87); GB300 (10, 3) resident cells 1.06-1.09.  No capability fuses k > 64 and the
# frozen bundle ships no whole-CTA-tail build; the mechanism (bit 3, manifest fused_block_tail) is kept for a build
# that does.
_BLOCK_TAIL_CAPABILITIES: frozenset[tuple[int, int]] = frozenset()


def _block_tail_for_capability(capability: tuple[int, int]) -> bool:
    return (int(capability[0]), int(capability[1])) in _BLOCK_TAIL_CAPABILITIES


@functools.cache
def _block_tail_enabled(device_index: int) -> bool:
    return _block_tail_for_capability(torch.cuda.get_device_capability(device_index))


def _fuse_block_tail(
    batch: int,
    cluster: int,
    ept: int,
    stream: bool,
    sm_count: int,
    top_k_max: Optional[int],
    two_launch: bool,
) -> bool:
    """Whether a launch of ``batch`` rows on the variant runs the whole-CTA tail (launch_flags bit 3): the
    largest top-k lies in (_STAGE1_LARGE_K, fused_block_tail_kcap], the variant carries the tail, the device
    policy is on (``two_launch`` false) and the stage-1 grid is a single wave.  Rank 0 of every cluster runs the
    tail after its row, so a multi-wave grid would serialize one tail per wave (R200 k = 1000: cluster-8 grids of
    4-6 waves run 1.3-1.6x slower than the chain) while the chain's stage-2/3 kernel costs one launch for all
    rows; one-wave grids gain 4-16 % on every measured cell of B200 / H100 / R200.  Rows of such a launch with
    k <= CAKE_SAMPLING_FUSED_TAIL_KCAP take the two-warp tail inside the kernel (bit 3 enables it for them too)."""
    if two_launch or top_k_max is None:
        return False
    if not (_STAGE1_LARGE_K < top_k_max <= _fused_block_tail_kcap()):
        return False
    if not _stage1_has_fused_block_tail(cluster, ept, stream):
        return False
    wave_ctas = _wave_ctas(int(sm_count))
    return batch * cluster <= wave_ctas[cluster]


def _launch_is_two_kernels(
    top_k_max: Optional[int], device_index: Optional[int] = None
) -> bool:
    """Whether a launch whose largest top-k is ``top_k_max`` (None: small) needs the stage-2/3 kernel: above
    ``_STAGE1_LARGE_K`` unless the device runs the whole-CTA tail and the top-k fits its slab (no CUDA: yes)."""
    if top_k_max is None or top_k_max <= _STAGE1_LARGE_K:
        return False
    if top_k_max > _fused_block_tail_kcap():
        return True
    if not torch.cuda.is_available():
        return True
    index = torch.cuda.current_device() if device_index is None else int(device_index)
    return not _block_tail_enabled(index)


_FLAG_FUSE_TAIL = 1
_FLAG_EARLY_TRIGGER = 2
_FLAG_STREAM_PREPASS = 4
_FLAG_FUSE_BLOCK_TAIL = 8
_FLAG_COARSE_SAMPLE = 16
_FLAG_ROW_SPAN_DIET = 32
_ROW_SPAN_DIET_MIN_CLUSTER = 8
_ROW_SPAN_DIET_CAPABILITIES = ((9, 0), (10, 0), (10, 3))


@functools.cache
def _stage1_has_coarse_sample(cluster: int, ept: int, stream: bool) -> bool:
    """Whether the frozen variant ships a coarse-sample build (a manifest entry with ``coarse_sample``): the kernel taken
    by launch_flags bit 4, whose sampled first pass reads 1/8 of the row instead of 1/4 (every streaming variant;
    residents take no sample).  The exact passes are the same and every output is bit-identical."""
    found = False
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            found = True
            if v["coarse_sample"]:
                return True
    if not found:
        raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")
    return False


def _coarse_sample_flag(cluster: int, ept: int, stream: bool, top_k_max: int) -> int:
    """Stage-1 ``launch_flags`` bit that selects the coarse-sample build: a streaming variant that ships it, for a
    launch whose largest top-k fits the two-warp fused tail (``fused_tail_kcap``).  Above that the candidate count
    of a k ~ 1000 row sits at the gather capacity and the coarser estimate sends 3-12 % of the rows down the
    slower path (round 6, lever S4: +5-14 % on the k = 1000 streams), so those launches keep the 1/4 sample."""
    if not stream or int(top_k_max) > _fused_tail_kcap():
        return 0
    return _FLAG_COARSE_SAMPLE if _stage1_has_coarse_sample(cluster, ept, True) else 0


@functools.cache
def _row_span_diet_capability(device_index: int) -> bool:
    return tuple(torch.cuda.get_device_capability(device_index)) in _ROW_SPAN_DIET_CAPABILITIES


def _row_span_diet_flag(
    cluster: int, stream: bool, top_k_max: int, device_index: int
) -> int:
    """Stage-1 ``launch_flags`` bit that makes a streaming variant's filter pass choose its arm from the
    whole row's expected candidate density instead of one CTA's span (identical candidate segments,
    identical outputs).  Set for cluster >= 8 streams whose largest top-k exceeds the two-warp tail
    on Hopper, B200 and GB300 (round 6, lever FD5: those k ~ 1000 cells run 2-5 % faster); a
    cluster-1 row that takes the arm can lose 5 % and Rubin measures neutral, so nothing else."""
    if (
        not stream
        or int(cluster) < _ROW_SPAN_DIET_MIN_CLUSTER
        or int(top_k_max) <= _fused_tail_kcap()
    ):
        return 0
    return _FLAG_ROW_SPAN_DIET if _row_span_diet_capability(device_index) else 0


def _early_trigger_flag(batch: int, cluster: int, sm_count: int) -> int:
    """Stage-1 ``launch_flags`` bit that lets the PDL-chained stage-2/3 launch start before stage 1
    ends.  Its ``batch`` CTAs are placed on the SMs free at trigger time and keep that placement; one
    stage-1 CTA occupies an SM, so the dependent must fit next to the last stage-1 wave or it gets
    packed onto a few SMs and its tail grows 1.5-2.5x.  Otherwise stage 1 triggers at CTA exit."""
    grid = int(batch) * int(cluster)
    last_wave = grid % int(sm_count) or int(sm_count)
    return _FLAG_EARLY_TRIGGER if int(batch) <= int(sm_count) - last_wave else 0


@functools.cache
def _stream_prepass_flag(device_index: int) -> int:
    """Stage-1 ``launch_flags`` bit that moves a streaming variant's early trigger from after its
    filter pass (the whole row read once) to before its first pass.  The earlier point saves the
    one-wave large-k cells 0.5-1 us of dependent launch latency on Blackwell and Rubin but costs
    Hopper 1.4-1.9 us (the dependent's launch contends with the row read), so it is set for compute
    capability >= 10 only.  Ignored by the register-resident variants and without the early trigger."""
    major, _minor = torch.cuda.get_device_capability(device_index)
    return _FLAG_STREAM_PREPASS if major >= 10 else 0


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@functools.cache
def _smem_optin(device_index: int) -> int:
    return int(
        torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin
    )


def _nearest_table(sm_count: int) -> int:
    return min(_WAVE_CTAS_BY_SM_COUNT, key=lambda n: (abs(n - sm_count), -n))


def _wave_ctas(sm_count: int) -> dict[int, int]:
    return _WAVE_CTAS_BY_SM_COUNT[_nearest_table(sm_count)]


def _stage1_cost(sm_count: int) -> _Stage1CostRow:
    return _STAGE1_COST_BY_SM_COUNT[_nearest_table(sm_count)]


def choose_stage1(
    batch: int,
    vocab: int,
    sm_count: Optional[int] = None,
    smem_limit: Optional[int] = None,
    top_k_max: Optional[int] = None,
    two_launch: Optional[bool] = None,
) -> tuple[int, int, bool]:
    """``(cluster, ept, stream)`` for ``batch`` rows of ``vocab`` entries (largest top-k ``top_k_max``).

    Register-resident candidates: fewest waves, then a register chunk of at least 16 entries,
    then the larger cluster.  That resident choice is compared with every streaming variant
    through the fitted cost model of the device's wave table (``_STAGE1_COST_BY_SM_COUNT``:
    resident ``waves * (base + per_ept * ept)`` against streaming ``waves * (base + chunk *
    chunks + cluster_cta * (cluster - 1))``, both plus ``launch_cta * ctas / wave_ctas[1]``); the
    resident variant wins ties and streaming ties prefer the smaller cluster.  The wave table and
    its constants are selected by ``sm_count`` (the current device's SM count when omitted; B200
    148, H100 132 and Rubin R200 212 are measured, other counts use the nearest).  ``top_k_max``
    above ``_STAGE1_LARGE_K`` adds the table's large-k streaming term (None ranks as a small
    top-k); ``two_launch`` says whether such a launch takes the stage-2/3 kernel separately (None:
    ``_launch_is_two_kernels`` on the current device, i.e. its whole-CTA tail policy) and selects
    the two-launch terms.  Variants needing more dynamic shared memory than ``smem_limit`` (the
    current device's opt-in limit when omitted) are not candidates."""
    if sm_count is None:
        sm_count = (
            _sm_count(torch.cuda.current_device())
            if torch.cuda.is_available()
            else _DEFAULT_SM_COUNT
        )
    if smem_limit is None and torch.cuda.is_available():
        smem_limit = _smem_optin(torch.cuda.current_device())
    wave_ctas = _wave_ctas(int(sm_count))
    (
        resident_base,
        resident_per_ept,
        stream_base,
        stream_chunk,
        stream_cluster_cta,
        launch_cta,
        stream_large_k,
        stream_chunk32,
        stream_cluster,
        stream_wide_wave,
        stream_wide_two_launch,
        notail_two_launch,
        chain_over_tail,
        stream_wide_large_k,
        resident_two_launch,
        fused_tail_cluster,
        fused_tail_wide_cluster,
        stream_large_k_chunk,
        stream_large_k_chunk32,
        fused_tail_stream,
        stream_large_k_cluster_cta,
        stream_wide_short_row_large_k,
    ) = _stage1_cost(int(sm_count))
    large_k = (
        stream_large_k if top_k_max is not None and top_k_max > _STAGE1_LARGE_K else 0.0
    )
    if two_launch is None:
        two_launch = _launch_is_two_kernels(top_k_max)
    regime_two_launch = bool(two_launch)

    def takes_two_kernels(ce: tuple[int, int], stream: bool) -> bool:
        return large_k > 0.0 and not _fuse_block_tail(
            batch, ce[0], ce[1], stream, int(sm_count), top_k_max, regime_two_launch
        )

    variants = _stage1_variants(smem_limit)
    epts = sorted({e for _, e, st in variants if not st})
    available = {(c, e) for c, e, st in variants if not st}
    streaming = [(c, e) for c, e, st in variants if st]
    candidates = []
    for cluster in (1, 2, 4, 8):
        need = math.ceil(vocab / (_THREADS * cluster))
        ept = next((e for e in epts if e >= need), None)
        if ept is None or (cluster, ept) not in available:
            continue
        if cluster > 1 and _THREADS * ept * (cluster // 2) >= vocab:
            continue
        candidates.append((cluster, ept))

    def waves(c: int) -> int:
        return -(-(batch * c) // wave_ctas[c])

    resident = None
    if candidates:
        resident = min(
            candidates,
            key=lambda ce: (
                waves(ce[0]),
                0 if ce[1] >= _PREFERRED_MIN_EPT else 1,
                -ce[0],
            ),
        )
    if not streaming:
        if resident is None:
            raise ValueError(f"vocab={vocab} exceeds the frozen stage-1 capacity")
        return resident[0], resident[1], False

    def launch_cost(c: int) -> float:
        return launch_cta * batch * c / wave_ctas[1]

    def stream_cost(ce: tuple[int, int]) -> float:
        chunks = math.ceil(vocab / (ce[0] * _THREADS * ce[1]))
        per_chunk = stream_chunk if ce[1] <= 16 else stream_chunk32
        if per_chunk <= 0.0:
            return math.inf  # variant not ranked by this table
        return (
            waves(ce[0])
            * (
                stream_base
                + (stream_wide_wave if ce[1] > 16 else 0.0)
                + per_chunk * chunks
                + (stream_cluster if ce[0] > 1 else 0.0)
                + stream_cluster_cta * (ce[0] - 1)
                + large_k
                + (stream_wide_large_k if ce[1] > 16 and large_k > 0.0 else 0.0)
                + (
                    (stream_large_k_chunk if ce[1] <= 16 else stream_large_k_chunk32)
                    * chunks
                    if large_k > 0.0
                    else 0.0
                )
                + (stream_large_k_cluster_cta * (ce[0] - 1) if large_k > 0.0 else 0.0)
                + (
                    stream_wide_short_row_large_k
                    if large_k > 0.0
                    and ce[1] > 16
                    and chunks <= _STREAM_WIDE_TWO_LAUNCH_CHUNKS
                    else 0.0
                )
            )
            + launch_cost(ce[0])
            + (stream_two_launch_cost(ce))
        )

    def stream_two_launch_cost(ce: tuple[int, int]) -> float:
        if not takes_two_kernels(ce, True):
            if large_k <= 0.0:
                return 0.0
            return fused_tail_stream + (
                fused_tail_cluster + (fused_tail_wide_cluster if ce[1] > 16 else 0.0)
            ) * (ce[0] - 1)
        cost = 0.0
        if (
            ce[1] > 16
            and math.ceil(vocab / (_THREADS * ce[1])) <= _STREAM_WIDE_TWO_LAUNCH_CHUNKS
        ):
            cost += stream_wide_two_launch
        if not regime_two_launch:
            cost += chain_over_tail  # multi-wave grid: the chain against its one-wave fused peers
        return cost

    best = min(streaming, key=lambda ce: (stream_cost(ce), ce[0]))
    if resident is not None:
        # On a fusing device a resident that still needs the stage-2/3 kernel pays for it: the (8, 48) has no
        # tails at all, a multi-wave grid keeps the chain.
        extra = 0.0
        if large_k > 0.0 and not takes_two_kernels(resident, False):
            extra = fused_tail_cluster * (resident[0] - 1)
        if takes_two_kernels(resident, False):
            if regime_two_launch:
                extra = resident_two_launch
            else:
                extra = (
                    chain_over_tail
                    if _stage1_has_fused_tail(resident[0], resident[1], False)
                    else notail_two_launch
                )
        resident_cost = (
            waves(resident[0]) * (resident_base + resident_per_ept * resident[1])
            + launch_cost(resident[0])
            + extra
        )
        if resident_cost <= stream_cost(best):
            return resident[0], resident[1], False
    return best[0], best[1], True


@functools.cache
def choose_stage23(top_k_max: int) -> tuple[int, int]:
    """Smallest frozen ``(threads, items)`` slab that holds ``top_k_max`` entries."""
    for threads, items in _stage23_variants():
        if threads * items >= top_k_max:
            return threads, items
    raise ValueError(f"top_k_max={top_k_max} exceeds the frozen stage-2/3 capacity")


def cake_sampling_route(
    probs: torch.Tensor,
    top_k: Optional[Union[int, torch.Tensor]],
    top_k_max: Optional[int] = None,
) -> str:
    """``"pipeline"`` when the frozen kernels serve this request, else ``"fallback:<reason>"``.

    ``top_k_max`` avoids a device sync when ``top_k`` is a tensor.
    """
    if not probs.is_cuda:
        return "fallback:device"
    if probs.dtype != torch.float32:
        return "fallback:dtype"
    if probs.dim() != 2 or probs.stride(1) != 1 or probs.stride(0) != probs.size(1):
        return "fallback:layout"
    if _capability(probs.device) is None:
        return "fallback:arch"
    batch, vocab = probs.shape
    if top_k is None:
        return "fallback:no_top_k"
    if isinstance(top_k, int):
        kmax = top_k
    else:
        kmax = int(top_k_max) if top_k_max is not None else int(top_k.max().item())
    if kmax <= 0 or kmax >= vocab:
        return "fallback:top_k_disabled"
    if kmax > _slab():
        return "fallback:top_k_gt_slab"
    try:
        choose_stage1(batch, vocab, top_k_max=kmax)
    except ValueError:
        return "fallback:vocab_too_large"
    return "pipeline"


def _per_row_param(
    value: torch.Tensor, batch: int, dtype: torch.dtype, name: str
) -> torch.Tensor:
    """Per-request sampling parameter as a contiguous ``[batch]`` tensor of the kernel dtype.

    Same contract as the tensor form accepted by :mod:`flashinfer.sampling` (any integer or
    floating dtype, one entry per row); the kernels read ``int32`` / ``float32`` only.
    """
    if value.dim() != 1 or value.shape[0] != batch:
        raise ValueError(
            f"{name}: expected a 1D tensor of shape (batch_size,), got {tuple(value.shape)}"
        )
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value.to(dtype=dtype).contiguous()


def _workspace(batch: int, slab: int, device: torch.device):
    key = (batch, slab, device.index or 0)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = (
            torch.empty(batch, slab, device=device, dtype=torch.float32),
            torch.empty(batch, slab, device=device, dtype=torch.int32),
            torch.empty(batch, device=device, dtype=torch.int32),
        )
        _WORKSPACES[key] = ws
    return ws


@flashinfer_api
def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    renorm_out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    r"""Fused top-k-then-top-p sampling from probabilities (thread-block-cluster radix pipeline).

    Parameters
    ----------
    probs: torch.Tensor
        ``float32 [batch, vocab]`` probabilities, contiguous.
    top_k: Union[int, torch.Tensor]
        Number of candidates kept per row (``int`` or ``int32 [batch]``), ``1 <= k <= 1024``.
    top_p: Union[float, torch.Tensor]
        Nucleus threshold in ``(0, 1]`` (``float`` or ``float32 [batch]``).
    top_k_max: Optional[int]
        Upper bound of ``top_k`` when it is a tensor (avoids a device synchronization).
    generator: Optional[torch.Generator]
        Source of the Philox seed/offset (default CUDA generator when omitted), advanced exactly
        like :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs`.
    philox_seed, philox_offset: Optional[int]
        Explicit Philox parameters (both required together); ``generator`` is then not touched.
    out: Optional[torch.Tensor]
        ``int32 [batch]`` output buffer (allocated when omitted).
    renorm_out: Optional[torch.Tensor]
        Optional ``float32 [batch, 1024]`` buffer that receives the renormalized kept
        probabilities in sorted slab order (descending probability, ascending index; zeros for
        dropped entries; only the first ``count`` entries of a row are written).  Served by the
        pipeline route only.
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        Stage-1 slab buffers ``(values float32 [batch, 1024], indices int32 [batch, 1024],
        counts int32 [batch])``. When given, the sorted top-k slab of this call is left in them
        (aligned with ``renorm_out``); otherwise a cached per-shape workspace is used.
    enable_pdl: bool
        Launch stage 2/3 with programmatic dependent launch.

    Returns
    -------
    samples: torch.Tensor
        ``int32 [batch]`` sampled token ids.
    """
    if (philox_seed is None) != (philox_offset is None):
        raise ValueError("philox_seed and philox_offset must be given together")
    route = cake_sampling_route(probs, top_k, top_k_max)
    if route != "pipeline":
        if renorm_out is not None:
            raise ValueError(
                f"renorm_out is served by the pipeline route only; this request falls back ({route})"
            )
        from .sampling import top_k_top_p_sampling_from_probs as _fallback

        res = _fallback(
            probs,
            top_k,
            top_p,
            filter_apply_order="top_k_first",
            deterministic=True,
            generator=generator,
            seed=philox_seed,
            offset=philox_offset,
        )
        if out is not None:
            out.copy_(res.to(torch.int32))
            return out
        return res.to(torch.int32)

    batch, vocab = probs.shape
    if isinstance(top_k, torch.Tensor):
        top_k = _per_row_param(top_k, batch, torch.int32, "top_k")
    if isinstance(top_p, torch.Tensor):
        top_p = _per_row_param(top_p, batch, torch.float32, "top_p")
    kmax = (
        top_k
        if isinstance(top_k, int)
        else (int(top_k_max) if top_k_max is not None else int(top_k.max().item()))
    )
    cluster, ept, stream_variant = choose_stage1(batch, vocab, top_k_max=kmax)
    threads, items = choose_stage23(kmax)
    slab = _slab()
    vals, idxs, cnt = (
        workspace if workspace is not None else _workspace(batch, slab, probs.device)
    )
    if out is None:
        out = torch.empty(batch, device=probs.device, dtype=torch.int32)
    if philox_seed is None:
        # Same stride as top_p_sampling_from_probs (32 reserved draws per row): a generator shared with
        # the top_k_first route stays in lockstep.
        philox_seed, philox_offset = get_seed_and_offset(
            batch * 32, generator, probs.device
        )
    if isinstance(top_k, int):
        k_arr, k_scalar, k_kind = cnt, int(top_k), _TOPK_SCALAR
    else:
        k_arr, k_scalar, k_kind = top_k, 0, _TOPK_PER_ROW
    if isinstance(top_p, (int, float)):
        p_arr, p_scalar, p_kind = probs, float(top_p), _TOPP_SCALAR
    else:
        p_arr, p_scalar, p_kind = top_p, 0.0, _TOPP_PER_ROW
    renorm = renorm_out if renorm_out is not None else vals
    module = load_cake_sampling_module()
    stream = torch.cuda.current_stream(device=probs.device).cuda_stream
    # Small top-k: stage 2/3 runs inside the stage-1 kernel (same outputs, one launch).
    fused = kmax <= _fused_tail_kcap() and _stage1_has_fused_tail(
        cluster, ept, bool(stream_variant)
    )
    # Larger top-k up to the slab: the whole-CTA tail, on the capabilities where it beats the chain, for a
    # one-wave stage-1 grid.
    fused_block = not fused and _fuse_block_tail(
        batch,
        cluster,
        ept,
        bool(stream_variant),
        _sm_count(probs.device.index or 0),
        kmax,
        not _block_tail_enabled(probs.device.index or 0),
    )
    # Two launches: stage 2/3 may start early only when its CTAs fit beside the last stage-1 wave; a
    # streaming variant triggers before its first pass on Blackwell / Rubin, after its filter pass on Hopper.
    if fused:
        launch_flags = _FLAG_FUSE_TAIL | _coarse_sample_flag(
            cluster, ept, bool(stream_variant), kmax
        )
    elif fused_block:
        launch_flags = _FLAG_FUSE_BLOCK_TAIL
    else:
        launch_flags = _early_trigger_flag(
            batch, cluster, _sm_count(probs.device.index)
        )
        if stream_variant:
            launch_flags |= _stream_prepass_flag(probs.device.index)
            launch_flags |= _row_span_diet_flag(
                cluster, True, kmax, probs.device.index or 0
            )
    module.radix_topk(
        probs,
        k_arr,
        k_scalar,
        k_kind,
        vals,
        idxs,
        cnt,
        cluster,
        ept,
        1 if stream_variant else 0,
        p_arr,
        p_scalar,
        p_kind,
        out,
        renorm,
        int(philox_seed) & 0xFFFFFFFFFFFFFFFF,
        int(philox_offset) & 0xFFFFFFFFFFFFFFFF,
        1 if renorm_out is not None else 0,
        launch_flags,
        stream,
    )
    if fused or fused_block:
        return out
    module.sparse_topp_sample(
        vals,
        idxs,
        cnt,
        p_arr,
        p_scalar,
        p_kind,
        out,
        renorm,
        int(philox_seed) & 0xFFFFFFFFFFFFFFFF,
        int(philox_offset) & 0xFFFFFFFFFFFFFFFF,
        1 if renorm_out is not None else 0,
        threads,
        items,
        stage23_variant_flags(probs.device),
        1 if enable_pdl else 0,
        stream,
    )
    return out


@flashinfer_api
def top_k_probs_to_slab(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    out_vals: Optional[torch.Tensor] = None,
    out_idx: Optional[torch.Tensor] = None,
    out_count: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Stage 1 alone: exact per-row top-k into a ``[batch, 1024]`` slab.

    The slab holds the first ``k`` entries of ``lexsort(-prob, index)`` (NaN/negative
    probabilities sanitized to ``+0``).  Its layout is deterministic (a pure function of the
    input) but *not* sorted; entries beyond ``count`` are undefined.  Same dispatch conditions
    as :func:`top_k_top_p_sampling_from_probs`; requests the frozen kernels cannot serve raise
    ``ValueError`` with the :func:`cake_sampling_route` reason.

    Parameters
    ----------
    probs: torch.Tensor
        ``float32 [batch, vocab]`` probabilities, contiguous.
    top_k: Union[int, torch.Tensor]
        Number of candidates kept per row (``int`` or ``int32 [batch]``), ``1 <= k <= 1024``.
    top_k_max: Optional[int]
        Upper bound of ``top_k`` when it is a tensor (avoids a device synchronization).
    out_vals: Optional[torch.Tensor]
        ``float32 [batch, 1024]`` slab values buffer (allocated when omitted).
    out_idx: Optional[torch.Tensor]
        ``int32 [batch, 1024]`` slab indices buffer (allocated when omitted).
    out_count: Optional[torch.Tensor]
        ``int32 [batch]`` per-row kept counts buffer (allocated when omitted).

    Returns
    -------
    values, indices, counts: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The slab ``(values [batch, 1024], indices [batch, 1024], counts [batch])``.
    """
    route = cake_sampling_route(probs, top_k, top_k_max)
    if route != "pipeline":
        raise ValueError(f"frozen radix top-k cannot serve this request ({route})")
    batch, vocab = probs.shape
    slab = _slab()
    kmax = (
        top_k
        if isinstance(top_k, int)
        else (int(top_k_max) if top_k_max is not None else int(top_k.max().item()))
    )
    cluster, ept, stream_variant = choose_stage1(batch, vocab, top_k_max=kmax)
    vals = (
        out_vals
        if out_vals is not None
        else torch.empty(batch, slab, device=probs.device, dtype=torch.float32)
    )
    idxs = (
        out_idx
        if out_idx is not None
        else torch.empty(batch, slab, device=probs.device, dtype=torch.int32)
    )
    cnt = (
        out_count
        if out_count is not None
        else torch.empty(batch, device=probs.device, dtype=torch.int32)
    )
    if isinstance(top_k, int):
        k_arr, k_scalar, k_kind = cnt, int(top_k), _TOPK_SCALAR
    else:
        k_arr, k_scalar, k_kind = top_k, 0, _TOPK_PER_ROW
    stream = torch.cuda.current_stream(device=probs.device).cuda_stream
    load_cake_sampling_module().radix_topk(
        probs,
        k_arr,
        k_scalar,
        k_kind,
        vals,
        idxs,
        cnt,
        cluster,
        ept,
        1 if stream_variant else 0,
        vals,  # stage-2/3 tensors: unused without launch_flags bit 0
        1.0,
        _TOPP_SCALAR,
        cnt,
        vals,
        0,
        0,
        0,
        # no fused tail, no PDL dependent follows; the coarse-sample build for a small top-k on a stream,
        # the row-span filter arm for a large one on a cluster-8 stream
        _coarse_sample_flag(cluster, ept, bool(stream_variant), kmax)
        | _row_span_diet_flag(cluster, bool(stream_variant), kmax, probs.device.index or 0),
        stream,
    )
    return vals, idxs, cnt


__all__ = [
    "cake_sampling_route",
    "choose_stage1",
    "choose_stage23",
    "stage23_variant_flags",
    "top_k_probs_to_slab",
    "top_k_top_p_sampling_from_probs",
]
