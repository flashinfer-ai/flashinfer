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
``filter_apply_order="top_k_first"`` (same support, same Philox stream advancement outside
CUDA-graph capture) with these guarantees on top:

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
from typing import NamedTuple, Optional, TypeAlias, Union

import torch

from .api_logging import flashinfer_api
from .jit.cake_sampling import (
    load_cake_sampling_module,
    load_manifest,
    stage23_flags_by_capability,
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
    #  stream_wide_short_row_large_k_us, stream_wide_ragged_large_k_us)
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
    # resident_two_launch_us: charged to a register-resident candidate at large top-k whenever it takes the
    # stage-2/3 kernel (round 7: no capability fuses a resident): the stage-2/3 launch queued behind a resident
    # grid costs more than behind a streaming grid (GB300 k = 1000, V = 32768: the (4, 16) resident chain
    # 19.3-22.7 us vs the cluster-1 streams 18.5-21.4 while the resident's own kernel is the faster one).
    # stream_wide_ragged_large_k_us: per streaming wave of the ept-32 form at large top-k when its last 512 x 32
    # chunk is less than _RAGGED_CHUNK_FILL full (V = 151936: 4.64 chunks on cluster 2, 2.32 on cluster 4, 1.16 on
    # cluster 8).  Round-7 e2e graph-replay sweeps (k = 1000): at V = 151936 the ept-32 stream loses to the ept-16
    # form on every cluster on B200, GB300 and R200 (cluster 2 B = 64: 34.4 vs 28.4 us B200, 33.3 vs 27.5 GB300,
    # 28.0 vs 22.7 R200) although stage 1 alone measures equal, while at V = 128256 / 262144 (chunks within 10 % of
    # full) the ept-32 form keeps its lead; the term keeps the dispatcher off those cells.
    # fused_tail_cluster_us: per CTA beyond the first of the cluster when the candidate fuses the whole-CTA tail
    # (rank 0 sorts after the cluster barrier while its peers idle; the gathered list grows with the cluster);
    # fused_tail_wide_cluster_us: the same per-CTA term for the ept-32 streaming form only, whose fused tail grows
    # faster with the cluster than the ept-16 form's.
    148: (
        2.0,
        0.15,
        3.6,
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
        0.25,
        0.0,
        0.0,
        0.35,
        0.6,
        0.0,
        0.2,
        1.0,
        2.0,
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
        0.0,
    ),
}
_STREAM_WIDE_TWO_LAUNCH_CHUNKS = 3
# Fill of the last per-CTA 512 x 32 chunk below which an ept-32 stream pays stream_wide_ragged_large_k_us.
_RAGGED_CHUNK_FILL = 0.75

_WORKSPACES: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}
# (device index, id(generator)) -> next Philox offset handed to a call captured into a CUDA graph.
_CAPTURE_PHILOX_OFFSETS: dict[tuple[int, int], int] = {}


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _philox_params(
    batch: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    philox_seed: Optional[int],
    philox_offset: Optional[int],
) -> tuple[int, int]:
    """Philox ``(seed, offset)`` for one call (32 reserved draws per row, the ``top_k_first`` stride).

    Outside CUDA-graph capture the generator is read and advanced exactly like
    :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs`, so a generator shared with the
    ``top_k_first`` route stays in lockstep.  Inside capture the generator is never touched: reading
    its state there makes PyTorch register it with the graph, and every replay then runs two
    ``FillFunctor`` kernels that refresh device-side seed/offset words the Cake kernels do not read
    (both routes take the Philox parameters by value, so a replay reproduces the captured draws
    either way; measured +46 us per replay on GB300).  The capture path uses the generator's
    initial seed and a host-side per-(device, generator) offset counter, so distinct captures draw
    from distinct Philox streams while replays stay bitwise reproducible.
    """
    if philox_seed is not None:
        return int(philox_seed), int(philox_offset)
    if not torch.cuda.is_current_stream_capturing():
        return get_seed_and_offset(batch * 32, generator, device)
    index = _device_index(device)
    gen = generator if generator is not None else torch.cuda.default_generators[index]
    key = (index, id(gen))
    offset = _CAPTURE_PHILOX_OFFSETS.get(key, 0)
    _CAPTURE_PHILOX_OFFSETS[key] = offset + (batch * 32 + 3) // 4 * 4
    return int(gen.initial_seed()), offset


def _capability(device: torch.device) -> Optional[tuple[int, int]]:
    return supported_capability(_device_capability(_device_index(device)))


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
# The capability -> form table is the manifest's (``capabilities`` of each stage-2/3 row; the frozen source compiles a
# form only for the targets that dispatch it), read once per process.
STAGE23_FEATURE_BITS = {"int_tests": 1, "one_cmp_select": 2}


@functools.cache
def _stage23_variant_flags(device_index: int) -> int:
    table = stage23_flags_by_capability()
    return table.get(_device_capability(device_index), table[None])


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


# Capabilities whose host may run the whole-CTA tail (launch_flags bit 3; round 7, lever (b)).  Under CUDA-graph replay
# (the eager span of the two-launch chain includes the host gap between its launches and is not the measure) the in-CTA
# tail loses to the chain on every resident (B200 1.20-1.54, GB300 1.40-1.84) and on every stream at k <= 750; the
# cluster-8 streams at k = 1000 win on both (B200 0.89-0.94, GB300 0.89-0.97), so ``_fuse_block_tail`` takes the twin
# only for a cluster-8 stream with top_k_max > _BLOCK_TAIL_MIN_K on a one-wave grid.  Round 8 adds H100: under graph
# replay the twin runs 0.93-0.97 of the chain at V = 151936 and 0.96-0.99 at V = 262144 (B = 1-8, k = 800 / 1000, bitwise
# identical; eager 0.93-0.98).  R200 keeps the chain (no graph-replay A/B recorded for it).
_BLOCK_TAIL_CAPABILITIES: frozenset[tuple[int, int]] = frozenset(
    {(9, 0), (10, 0), (10, 3)}
)
_BLOCK_TAIL_MIN_CLUSTER = 8
_BLOCK_TAIL_MIN_K = 768
# Largest row the whole-CTA tail serves per capability (None: any).  GB300 round-7 matrices: the cluster-8 streams at
# V = 151936 run 3-11 % faster fused, at V = 262144 3-7 % slower eager (a cold-L2 stage 1 is hidden by the chain's
# host-bound launch gap there) and even under graph replay, so GB300 keeps the chain above this vocabulary.
_BLOCK_TAIL_MAX_VOCAB_BY_CAPABILITY: dict[tuple[int, int], Optional[int]] = {
    (9, 0): None,
    (10, 0): None,
    (10, 3): 196608,
}


def _block_tail_for_capability(capability: tuple[int, int]) -> bool:
    return (int(capability[0]), int(capability[1])) in _BLOCK_TAIL_CAPABILITIES


@functools.cache
def _device_capability(device_index: int) -> tuple[int, int]:
    cc = torch.cuda.get_device_capability(device_index)
    return int(cc[0]), int(cc[1])


@functools.cache
def _block_tail_enabled(device_index: int) -> bool:
    return _block_tail_for_capability(_device_capability(device_index))


def _fuse_block_tail(
    batch: int,
    cluster: int,
    ept: int,
    stream: bool,
    sm_count: int,
    top_k_max: Optional[int],
    two_launch: bool,
    vocab: Optional[int] = None,
    capability: Optional[tuple[int, int]] = None,
) -> bool:
    """Whether a launch of ``batch`` rows on the variant runs the whole-CTA tail (launch_flags bit 3): the
    largest top-k lies in (_BLOCK_TAIL_MIN_K, fused_block_tail_kcap], the variant is a stream on a cluster of at
    least _BLOCK_TAIL_MIN_CLUSTER CTAs and carries the tail, the device policy is on (``two_launch`` false) and the
    stage-1 grid is a single wave.  Rank 0 of every cluster runs the tail after its row, so a multi-wave grid would
    serialize one tail per wave while the chain's stage-2/3 kernel costs one launch for all rows; under CUDA-graph
    replay the in-CTA tail also loses to the chain on every resident and on every stream at k <= 750 (round 7), and
    wins 3-11 % only for the cluster-8 streams at k = 1000 on B200 / GB300.  Rows of such a launch with
    k <= CAKE_SAMPLING_FUSED_TAIL_KCAP take the two-warp tail inside the kernel (bit 3 enables it for them too)."""
    if two_launch or top_k_max is None:
        return False
    if not (_BLOCK_TAIL_MIN_K < top_k_max <= _fused_block_tail_kcap()):
        return False
    if not (stream and int(cluster) >= _BLOCK_TAIL_MIN_CLUSTER):
        return False
    if not _stage1_has_fused_block_tail(cluster, ept, stream):
        return False
    if capability is not None:
        if vocab is None:
            raise ValueError(
                "the whole-CTA tail policy of this capability needs the vocabulary"
            )
        bound = _BLOCK_TAIL_MAX_VOCAB_BY_CAPABILITY.get(
            (int(capability[0]), int(capability[1]))
        )
        if bound is not None and int(vocab) > bound:
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
_FLAG_SPEC_SAMPLE = 64
# Launch flag bit 7 (round 8, lever L-B): the slab-tail form of the speculative-sample build.  The selected pairs of a
# fused launch are pushed into rank 0's shared-memory slab (DSM stores) and the two-warp tail reads them there instead
# of re-reading the global slab row; every output is bit-identical.  The coarse-sample build carries the slab tail
# without a bit (it only serves fused launches); the default build never does (the k > 64 chains keep binaries without
# the slab code, which moved them by 1-14 % when compiled in).  Taken on the capabilities where the paired
# perturbed-process A/Bs won on every served fused cell (B200 0.948-0.993, GB300 0.959-0.992, 4/4 processes); Hopper
# measured 1.003-1.047 with it and keeps the plain speculative build.  Only rows of at most `_SLAB_TAIL_MAX_CHUNKS`
# register chunks per CTA take it: the round-8 matrices and same-node in-process A/Bs on B200 / GB300 put the 8-chunk
# `_cs` rows at 1.000-1.015, the 10-chunk rows at 1.012-1.020 and the 16-chunk `_sp` row at 0.997-1.014 (the slab's
# fixed saving does not grow with the row while its per-chunk bookkeeping does), and every served 1-5-chunk row at
# 0.93-0.99.
_FLAG_SLAB_TAIL = 128
_SLAB_TAIL_SPEC_CAPABILITIES: frozenset[tuple[int, int]] = frozenset({(10, 0), (10, 3)})
_SLAB_TAIL_MAX_CHUNKS = 5
# Stage-1 launch_flags bit 8: the pushed-coarse-sums form of the default build (`_lg` twin; round 8 lever L-G(d)): each
# CTA stores its coarse histogram sums into every CTA's shared memory so the two-level cluster select reads them locally.
# A two-launch chain on a streaming variant with the two-level select takes it on these capabilities (GB300 cluster-8
# chains 0.985-0.987, H100 0.992-0.994, medians over four perturbed processes); B200 keeps the plain build (+0.9..+1.7 %).
# Only the ept-32 build (`_COARSE_PUSH_MIN_EPT`) takes it: those gates ran the two-chunk ept-32 cluster-8 chains, and the
# GB300 round-8 matrix put the served four-chunk ept-16 chains (V = 262144, B = 1-8, k = 1000) at 1.014-1.052 in both
# tree orders with same-node in-process A/Bs of 1.022-1.056 (the pushed stores grow with the chunk count, the two
# dependent select round trips they replace do not).
_FLAG_COARSE_PUSH = 256
# Stage-1 launch_flags bit 9 (round 9, lever L-P): the leader-push exchange form of the default or sample build of a
# multi-CTA streaming variant (`_lp` / `_cs_lp` / `_sp_lp` twins).  Every CTA stores its compacted candidate list straight
# into rank 0's receive buffer (one st.shared::cluster.v2.b32 per pair) and its length into every CTA before the single
# exchange barrier; rank 0 gathers locally, so the pull form's DSM read rounds and exit rendezvous disappear.  Taken on
# the capabilities in _LEADER_PUSH_CAPABILITIES for any ept-16 stream and for an ept-32 (_LEADER_PUSH_WIDE_EPT) stream only
# when the row is at most _LEADER_PUSH_WIDE_MAX_CHUNKS register chunks per CTA (round-9 ledger: R200 ept-16 `_cs` cells
# 0.88-0.94, H100 c8 e16 `_cs` 0.82-0.86 / `_sp` 0.92-0.96, B200 `_sp` 0.89-0.97, GB300 ept-16 `_sp` cells 0.96-0.99 against the slab form; the two-chunk and longer ept-32 rows 1.02-1.31 on
# R200).  It displaces the slab tail (bit 7) and the pushed coarse sums (bit 8) where both would apply; never with bit 3.
_FLAG_LEADER_PUSH = 512
_LEADER_PUSH_CAPABILITIES: frozenset[tuple[int, int]] = frozenset(
    {(9, 0), (10, 0), (10, 3), (10, 7)}
)
_LEADER_PUSH_WIDE_EPT = 32
_LEADER_PUSH_WIDE_MAX_CHUNKS = 1
# Round 11 (lever M4-lb): on B200 the two-chunk speculative-sample row (`_sp` on cluster 8 ept 32: V151936) at a small
# batch is faster with the leader push (plain `_sp_lp`, no local select) than with the served slab-tail pull `_sp_lb`
# once both carry E1: k50 V151936 B1 0.976, B2 0.995 (6 perturbed processes each, bit-identical), B4 1.000, B8 1.011
# -> (max batch, max top-k) per capability; rows above either bound and the coarse-sample build keep the slab form.
_LEADER_PUSH_TWO_CHUNK_SMALL_BATCH_BY_CAPABILITY: dict[
    tuple[int, int], tuple[int, int]
] = {
    (10, 0): (2, 50),
}
# Stage-1 launch_flags bit 10 (round 10, lever L1): the CTA-local select form of a leader-push sample build (`_cs_lp_l1` /
# `_sp_lp_l1` twins).  Each CTA picks its filter bucket from its own sample (sampled mass >= k) instead of the cluster-wide
# coarse histogram, so that DSM round disappears and the leader push is the only cluster round; the row's exact top-k lies
# in the union of the per-CTA lists and every output is bit-identical.  The per-CTA lists grow with k, so a fused
# leader-push launch takes it only when its largest top-k is at most the capability's cap (round-10 ledger, perturbed-
# process medians at the served cluster-8 ept-32 picks: GB300 k = 10 / 20 / 32 0.90-0.97 on every served cell, k = 50
# mixed; B200 k = 10 / 20 0.93-0.98, k = 32 mixed; H100 served c8 e16 `_cs_lp` cells k = 10 0.943-0.965, whose k = 50 rows
# overflow the coarse-sample lists into the exact fallback -> cap 10).  R200 (mixed by vocabulary) and RTX PRO keep the plain
# leader-push / pull form.
_FLAG_LOCAL_SELECT = 1024
_LOCAL_SELECT_MAX_K_BY_CAPABILITY: dict[tuple[int, int], int] = {
    (9, 0): 10,
    (10, 0): 20,
    (10, 3): 32,
    (10, 7): 20,
}
# Round 11 (lever M4): a one-chunk row (an ept-32 stream whose rows fit one register chunk per CTA, V128256 on cluster 8)
# keeps its per-CTA lists small enough that the local select still wins at k 50 / 64 (GB300 `_sp_lp` -> `_sp_lp_l1` k50
# V128256 B1/B2/B4/B8 0.976/0.970/0.981/0.984, k64 0.985/0.982, 8 perturbed processes each; B200 k50 B1/B2/B4/B8
# 0.978-0.981/0.935-0.997/0.966-0.970/0.973-0.976, k64 B1/B8 0.951-0.991/0.982-0.985 over 6 histories) while the two-chunk
# rows read 0.998-1.018 at k50 and keep the capability cap.  A capability absent from the table keeps the capability cap.
_LOCAL_SELECT_ONE_CHUNK_MAX_K_BY_CAPABILITY: dict[tuple[int, int], int] = {
    (10, 0): 64,
    (10, 3): 64,
}
# Reach of the CTA-local select onto the rows the leader-push chunk rule excludes (an ept-32 stream whose rows take two
# register chunks per CTA): with the local select the pushed lists stay small, so the one-round form beats the served
# slab-tail pull form there too -- when the second chunk is full or the batch is at least the capability's minimum in
# _LOCAL_SELECT_WIDE_MIN_BATCH_BY_CAPABILITY (10.3: a 16-75 % filled last chunk loses at B <= 2 -> 4; 10.0: wins at every
# batch -> 1).  A capability absent from the table keeps the chunk rule.  Measured per capability (round-10 ledger).
_LOCAL_SELECT_WIDE_MIN_BATCH_BY_CAPABILITY: dict[tuple[int, int], int] = {
    (10, 0): 1,
    (10, 3): 4,
}
_LOCAL_SELECT_WIDE_MAX_CHUNKS = 2
# Stage-1 launch_flags bit 11 (round 10, lever L3-T): the integer-tested form of a streaming variant's whole-CTA tail
# build (`_bt_tia` twin).  The tail's f64 target product, cut tests and sample tests run as the stage-2/3 `int_tests`
# integer emulation, bit-identical; it pays only where the FP64 pipe is slow (GB300 served cluster-8 ept-16 k = 1000 / 200
# cells 0.916-0.955); B200's full-rate FP64 loses 3-7 % with it and keeps the f64 `_bt` twin.
_FLAG_INT_TAIL = 2048
_INT_TAIL_CAPABILITIES: frozenset[tuple[int, int]] = frozenset({(10, 3)})
_COARSE_PUSH_CAPABILITIES: frozenset[tuple[int, int]] = frozenset({(9, 0), (10, 3)})
_COARSE_PUSH_MIN_EPT = 32
_SPEC_SAMPLE_WIDE_EPT = 32
_SPEC_SAMPLE_WIDE_TWO_CHUNK_CLUSTER = 8
_SPEC_SAMPLE_WIDE_MIN_CHUNKS = 16
# A two-chunk ept-32 stream whose second chunk is at least this full keeps the coarse-sample build on the listed
# capability: on B200 the strided histogram of a full second chunk costs more than the sampled read it replaces
# (round 7, V = 262144 on the cluster-8 stream: `_sp` kernel +2.1 % in two interleaved profiler passes, +0.2..+0.8 %
# in the interleaved CUPTI harness, +0.8 / +2.4 % in two idle-node matrices at k = 10 / 50), while the 16 %-full
# second chunk of V = 151936 stays neutral-to-faster there and GB300 keeps the win on both.
_SPEC_SAMPLE_WIDE_TWO_CHUNK_MAX_FILL_BY_CAPABILITY: dict[tuple[int, int], float] = {
    (10, 0): 0.5
}
# Capabilities whose every eligible stream launch takes the speculative-sample build (B200 / GB300: the paired stage-1
# A/B and the round-7 FlashInfer matrices).  Elsewhere (Hopper, Rubin: round-7 matrices only) the build is taken on
# clusters up to _SPEC_SAMPLE_NARROW_MAX_CLUSTER with at least _SPEC_SAMPLE_NARROW_MIN_CTAS CTAs: on a small-top-k launch
# (one kernel; the alternative is the coarse-sample build) only for rows of at least _SPEC_SAMPLE_NARROW_FUSED_MIN_CHUNKS
# chunks per CTA, on the two-launch chain for any row, where a row of _SPEC_SAMPLE_NARROW_DEEP_CHUNKS or more chunks
# needs _SPEC_SAMPLE_NARROW_DEEP_MIN_CTAS CTAs.  The saving is the sampled read, which only a wide, bandwidth-bound grid
# pays for: the cluster-8 streams there lose 3-8 % with it (H100 / R200 k = 1000, V >= 151936, B <= 16), the 4-chunk
# rows at k <= 64 1-2 % against the coarse build, the 32-CTA grid and the 16-chunk rows on 64 CTAs 1-3 %.  On every
# capability a cluster-8 stream above fused_tail_kcap (the 1/2-rate sample, two-launch chain) keeps the default build:
# GB300 V = 262144 B = 1-8 measured +11..+17 % eager / +3..+7 % graph with it (round-7 matrix r7m7_gb300).
_SPEC_SAMPLE_ALL_STREAMS_CAPABILITIES: frozenset[tuple[int, int]] = frozenset(
    {(10, 0), (10, 3)}
)
_SPEC_SAMPLE_CHAIN_MAX_CLUSTER = 4
_SPEC_SAMPLE_NARROW_MAX_CLUSTER = 4
_SPEC_SAMPLE_NARROW_MIN_CTAS = 64
_SPEC_SAMPLE_NARROW_FUSED_MIN_CHUNKS = 5
_SPEC_SAMPLE_NARROW_DEEP_CHUNKS = 16
_SPEC_SAMPLE_NARROW_DEEP_MIN_CTAS = 128


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
def _stage1_has_spec_sample(cluster: int, ept: int, stream: bool) -> bool:
    """Whether the frozen variant ships a speculative-sample build (a manifest entry with ``spec_sample``): the kernel
    taken by launch_flags bit 6, whose first register chunk doubles as the sample (every streaming variant; residents
    take no sample).  The exact passes are the same and every output is bit-identical."""
    found = False
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            found = True
            if v["spec_sample"]:
                return True
    if not found:
        raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")
    return False


def _spec_sample_flag(
    cluster: int,
    ept: int,
    stream: bool,
    vocab: int,
    top_k_max: Optional[int] = None,
    batch: Optional[int] = None,
    capability: Optional[tuple[int, int]] = None,
) -> int:
    """Stage-1 ``launch_flags`` bit that selects the speculative-sample build.  On B200 / GB300
    (``_SPEC_SAMPLE_ALL_STREAMS_CAPABILITIES``; ``capability`` None): every ept-16 streaming launch, and an ept-32
    (``_SPEC_SAMPLE_WIDE_EPT``) streaming launch whose row is one register chunk per CTA, two chunks on a cluster of at
    least ``_SPEC_SAMPLE_WIDE_TWO_CHUNK_CLUSTER`` CTAs, or at least ``_SPEC_SAMPLE_WIDE_MIN_CHUNKS`` chunks.  There the
    strided histogram of the chunk already in registers is cheaper than the separate sampled read it replaces (round 7,
    lever SP: ept-16 streams 2-13 % faster at 1-16 chunks, the cluster-8 ept-32 stream at one chunk 13-18 % and at two
    chunks 0-6 %, the cluster-1 ept-32 stream at 16 chunks 2-15 %); ept-32 streams at 2-10 chunks per CTA on clusters
    1-4 measured 0-6 % slower with it and keep their round-6 build, and on a capability listed in
    ``_SPEC_SAMPLE_WIDE_TWO_CHUNK_MAX_FILL_BY_CAPABILITY`` so does a two-chunk row whose second chunk is at least that
    full (B200 V = 262144 on the cluster-8 stream: 0.2-2.4 % slower with it; V = 151936 keeps the twin).  A cluster-8 stream above ``fused_tail_kcap`` never
    takes it on any capability (the 1/2-rate register histogram on the two-launch chain: GB300 V = 262144 +11..+17 %
    eager, +3..+7 % graph; H100 / R200 +3..+8 %).  On any other capability the same chunk rule applies
    only on a cluster of at most ``_SPEC_SAMPLE_NARROW_MAX_CLUSTER`` CTAs whose grid has at least
    ``_SPEC_SAMPLE_NARROW_MIN_CTAS`` CTAs, for a small-top-k launch (at most ``fused_tail_kcap``) only to rows of at
    least ``_SPEC_SAMPLE_NARROW_FUSED_MIN_CHUNKS`` chunks per CTA, and on the two-launch chain with
    ``_SPEC_SAMPLE_NARROW_DEEP_MIN_CTAS`` CTAs for rows of ``_SPEC_SAMPLE_NARROW_DEEP_CHUNKS`` or more chunks: the
    round-7 H100 / R200 matrices show the cluster-8 streams 1-8 % slower with the build, the 4-chunk rows at k <= 64
    1-2 % slower than their coarse-sample build, and the wide cluster <= 4 streams 2-20 % faster."""
    if not stream:
        return 0
    chunks = math.ceil(int(vocab) / (_THREADS * int(ept) * int(cluster)))
    if (
        top_k_max is not None
        and int(top_k_max) > _fused_tail_kcap()
        and int(cluster) > _SPEC_SAMPLE_CHAIN_MAX_CLUSTER
    ):
        return 0
    if (
        capability is not None
        and tuple(int(x) for x in capability)
        not in _SPEC_SAMPLE_ALL_STREAMS_CAPABILITIES
    ):
        if batch is None:
            raise ValueError(
                "the speculative-sample policy of this capability needs the batch"
            )
        if int(cluster) > _SPEC_SAMPLE_NARROW_MAX_CLUSTER:
            return 0
        ctas = int(batch) * int(cluster)
        if ctas < _SPEC_SAMPLE_NARROW_MIN_CTAS:
            return 0
        if top_k_max is None or int(top_k_max) <= _fused_tail_kcap():
            if chunks < _SPEC_SAMPLE_NARROW_FUSED_MIN_CHUNKS:
                return 0
        elif (
            chunks >= _SPEC_SAMPLE_NARROW_DEEP_CHUNKS
            and ctas < _SPEC_SAMPLE_NARROW_DEEP_MIN_CTAS
        ):
            return 0
    if int(ept) < _SPEC_SAMPLE_WIDE_EPT:
        return _FLAG_SPEC_SAMPLE if _stage1_has_spec_sample(cluster, ept, True) else 0
    if chunks == 2:
        if int(cluster) < _SPEC_SAMPLE_WIDE_TWO_CHUNK_CLUSTER:
            return 0
        max_fill = (
            None
            if capability is None
            else _SPEC_SAMPLE_WIDE_TWO_CHUNK_MAX_FILL_BY_CAPABILITY.get(
                (int(capability[0]), int(capability[1]))
            )
        )
        if (
            max_fill is not None
            and int(vocab) / (_THREADS * int(ept) * int(cluster)) - 1.0 >= max_fill
        ):
            return 0
    elif chunks != 1 and chunks < _SPEC_SAMPLE_WIDE_MIN_CHUNKS:
        return 0
    return _FLAG_SPEC_SAMPLE if _stage1_has_spec_sample(cluster, ept, True) else 0


@functools.cache
def _stage1_has_slab_tail(cluster: int, ept: int, stream: bool, coarse: bool) -> bool:
    """Whether the frozen variant ships the slab-tail form of its coarse-sample (``coarse``) or speculative-sample build
    (a manifest entry with ``slab_tail`` and the matching sample flag): the kernel taken by launch_flags bit 7 together
    with bit 0 and bit 4 / 6.  Every output is bit-identical to the plain sample build."""
    found = False
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            found = True
            if v["slab_tail"] and (v["coarse_sample"] if coarse else v["spec_sample"]):
                return True
    if not found:
        raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")
    return False


def _slab_tail_flag(
    cluster: int,
    ept: int,
    stream: bool,
    sample_flag: int,
    vocab: int,
    capability: Optional[tuple[int, int]] = None,
) -> int:
    """Stage-1 ``launch_flags`` bit 7 for a fused launch that takes a sample build (bit 4 or bit 6 in ``sample_flag``):
    the slab-tail twin of that build on the capabilities in ``_SLAB_TAIL_SPEC_CAPABILITIES`` when the variant ships it
    and the row is at most ``_SLAB_TAIL_MAX_CHUNKS`` register chunks per CTA.  Default-build and non-fused launches
    never take it; Hopper / Rubin and the long cluster-1 / cluster-2 rows keep the plain sample builds."""
    flag = int(sample_flag) & (_FLAG_SPEC_SAMPLE | _FLAG_COARSE_SAMPLE)
    if not stream or not flag:
        return 0
    if (
        capability is None
        or (int(capability[0]), int(capability[1])) not in _SLAB_TAIL_SPEC_CAPABILITIES
    ):
        return 0
    if (
        math.ceil(int(vocab) / (_THREADS * int(ept) * int(cluster)))
        > _SLAB_TAIL_MAX_CHUNKS
    ):
        return 0
    coarse = bool(flag & _FLAG_COARSE_SAMPLE)
    return _FLAG_SLAB_TAIL if _stage1_has_slab_tail(cluster, ept, True, coarse) else 0


@functools.cache
def _stage1_has_coarse_push(cluster: int, ept: int, stream: bool) -> bool:
    """Whether the frozen variant ships the pushed-coarse-sums form of its default build (a manifest entry with
    ``coarse_push``): the kernel taken by launch_flags bit 8 on a two-launch chain.  Every output is bit-identical to the
    default build."""
    found = False
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            found = True
            if v["coarse_push"]:
                return True
    if not found:
        raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")
    return False


def _coarse_push_flag(
    cluster: int,
    ept: int,
    stream: bool,
    chain_flags: int,
    capability: Optional[tuple[int, int]] = None,
) -> int:
    """Stage-1 ``launch_flags`` bit 8 for a two-launch chain (``chain_flags``: the chain's flags so far; a sample build in
    them, bit 4 / 6, keeps the plain build): the pushed-coarse-sums twin of the default build on the capabilities in
    ``_COARSE_PUSH_CAPABILITIES`` when the variant ships it and the build is at least ``_COARSE_PUSH_MIN_EPT`` elements
    per thread.  Fused launches never take it; B200 / Rubin and the ept-16 chains keep the plain default build."""
    if (
        not stream
        or int(ept) < _COARSE_PUSH_MIN_EPT
        or int(chain_flags)
        & (
            _FLAG_FUSE_TAIL
            | _FLAG_FUSE_BLOCK_TAIL
            | _FLAG_COARSE_SAMPLE
            | _FLAG_SPEC_SAMPLE
            | _FLAG_SLAB_TAIL
        )
    ):
        return 0
    if (
        capability is None
        or (int(capability[0]), int(capability[1])) not in _COARSE_PUSH_CAPABILITIES
    ):
        return 0
    return _FLAG_COARSE_PUSH if _stage1_has_coarse_push(cluster, ept, True) else 0


@functools.cache
def _stage1_has_leader_push(
    cluster: int, ept: int, stream: bool, coarse: bool, spec: bool
) -> bool:
    """Whether the frozen variant ships the leader-push exchange form of its default (neither flag), coarse-sample
    (``coarse``) or speculative-sample (``spec``) build (a manifest entry with ``leader_push`` and the matching sample
    flags): the kernel taken by launch_flags bit 9.  Every output is bit-identical to the pull form."""
    found = False
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            found = True
            if (
                v.get("leader_push", False)
                and bool(v["coarse_sample"]) == bool(coarse)
                and bool(v["spec_sample"]) == bool(spec)
            ):
                return True
    if not found:
        raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")
    return False


def _leader_push_flag(
    cluster: int,
    ept: int,
    stream: bool,
    launch_flags: int,
    vocab: int,
    capability: Optional[tuple[int, int]] = None,
    local_select_reach: bool = False,
    small_batch_reach: bool = False,
) -> int:
    """Stage-1 ``launch_flags`` bit 9 for a launch whose other bits are ``launch_flags``: the leader-push exchange form
    of the build the sample bits select, on a multi-CTA streaming variant that ships it, on the capabilities in
    ``_LEADER_PUSH_CAPABILITIES`` (``capability`` None: every capability), for any row on an ept-16 stream and for a
    row of at most ``_LEADER_PUSH_WIDE_MAX_CHUNKS`` register chunks per CTA on an ept-``_LEADER_PUSH_WIDE_EPT`` stream.
    Never with the whole-CTA tail (bit 3); the caller drops the slab tail (bit 7) and the pushed coarse sums (bit 8)
    when this bit is taken.  ``local_select_reach`` (round 10) and ``small_batch_reach`` (round 11,
    ``_leader_push_small_batch_reach``) extend it to a two-chunk ept-32 row."""
    if not stream or int(cluster) < 2 or (launch_flags & _FLAG_FUSE_BLOCK_TAIL) != 0:
        return 0
    if (
        capability is not None
        and (int(capability[0]), int(capability[1])) not in _LEADER_PUSH_CAPABILITIES
    ):
        return 0
    if int(ept) >= _LEADER_PUSH_WIDE_EPT:
        chunks = -(-int(vocab) // (_THREADS * int(ept) * int(cluster)))
        if chunks > _LEADER_PUSH_WIDE_MAX_CHUNKS and not (
            (local_select_reach or small_batch_reach)
            and chunks <= _LOCAL_SELECT_WIDE_MAX_CHUNKS
        ):
            return 0
    coarse = (launch_flags & _FLAG_COARSE_SAMPLE) != 0
    spec = (launch_flags & _FLAG_SPEC_SAMPLE) != 0
    return (
        _FLAG_LEADER_PUSH
        if _stage1_has_leader_push(cluster, ept, True, coarse, spec)
        else 0
    )


def _stage1_has_local_select(
    cluster: int, ept: int, stream: bool, coarse: bool, spec: bool
) -> bool:
    """Whether the frozen variant ships the CTA-local select form of its leader-push coarse-sample (``coarse``) or
    speculative-sample (``spec``) build (a manifest entry with ``local_select`` and the matching sample flag): the kernel
    taken by launch_flags bit 10.  Every output is bit-identical to the cluster-wide select."""
    found = False
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            found = True
            if (
                v.get("local_select", False)
                and bool(v["coarse_sample"]) == bool(coarse)
                and bool(v["spec_sample"]) == bool(spec)
            ):
                return True
    if not found:
        raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")
    return False


def _leader_push_small_batch_reach(
    cluster: int,
    ept: int,
    stream: bool,
    launch_flags: int,
    top_k_max: Optional[int],
    vocab: int,
    batch: int,
    capability: Optional[tuple[int, int]] = None,
) -> bool:
    """Whether a fused speculative-sample launch (``launch_flags`` carry bits 0 and 6) on a row of two register chunks
    per CTA at a small batch reaches the leader push (round 11, lever M4-lb): an ept-``_LEADER_PUSH_WIDE_EPT`` stream
    whose row takes more than ``_LEADER_PUSH_WIDE_MAX_CHUNKS`` and at most ``_LOCAL_SELECT_WIDE_MAX_CHUNKS`` chunks,
    ``batch`` and ``top_k_max`` within the capability's bounds in ``_LEADER_PUSH_TWO_CHUNK_SMALL_BATCH_BY_CAPABILITY``
    (``capability`` None: the loosest bounds of the table)."""
    if not stream or top_k_max is None or int(ept) < _LEADER_PUSH_WIDE_EPT:
        return False
    if (launch_flags & _FLAG_FUSE_TAIL) == 0 or (launch_flags & _FLAG_SPEC_SAMPLE) == 0:
        return False
    if capability is None:
        max_batch = max(
            b for b, _ in _LEADER_PUSH_TWO_CHUNK_SMALL_BATCH_BY_CAPABILITY.values()
        )
        max_k = max(
            k for _, k in _LEADER_PUSH_TWO_CHUNK_SMALL_BATCH_BY_CAPABILITY.values()
        )
    else:
        cc = (int(capability[0]), int(capability[1]))
        if cc not in _LEADER_PUSH_TWO_CHUNK_SMALL_BATCH_BY_CAPABILITY:
            return False
        max_batch, max_k = _LEADER_PUSH_TWO_CHUNK_SMALL_BATCH_BY_CAPABILITY[cc]
    if int(batch) > max_batch or int(top_k_max) > max_k:
        return False
    chunks = -(-int(vocab) // (_THREADS * int(ept) * int(cluster)))
    return _LEADER_PUSH_WIDE_MAX_CHUNKS < chunks <= _LOCAL_SELECT_WIDE_MAX_CHUNKS


def _local_select_reach(
    cluster: int,
    ept: int,
    stream: bool,
    launch_flags: int,
    top_k_max: Optional[int],
    vocab: int,
    batch: int,
    capability: Optional[tuple[int, int]] = None,
) -> bool:
    """Whether a fused sample-build launch (``launch_flags`` with bit 0 and bit 4 or 6) reaches past the leader-push chunk
    rule because the CTA-local select form (bit 10) would ride on it: a variant that ships the form, on a capability in
    ``_LOCAL_SELECT_WIDE_MIN_BATCH_BY_CAPABILITY`` (None: taken at any batch), ``top_k_max`` at most that capability's
    local-select cap, an ept-``_LEADER_PUSH_WIDE_EPT`` stream whose rows take exactly ``_LOCAL_SELECT_WIDE_MAX_CHUNKS``
    register chunks per CTA, and either a full last chunk or ``batch`` at least the capability's minimum batch.  The
    caller then passes it to ``_leader_push_flag`` and ``_local_select_flag`` follows."""
    if not stream or top_k_max is None or (launch_flags & _FLAG_FUSE_TAIL) == 0:
        return False
    coarse = (launch_flags & _FLAG_COARSE_SAMPLE) != 0
    spec = (launch_flags & _FLAG_SPEC_SAMPLE) != 0
    if not (coarse or spec) or not _stage1_has_local_select(
        cluster, ept, True, coarse, spec
    ):
        return False
    if int(ept) < _LEADER_PUSH_WIDE_EPT:
        return False
    if capability is None:
        cap = max(_LOCAL_SELECT_MAX_K_BY_CAPABILITY.values())
        min_batch = min(_LOCAL_SELECT_WIDE_MIN_BATCH_BY_CAPABILITY.values())
    else:
        cc = (int(capability[0]), int(capability[1]))
        if cc not in _LOCAL_SELECT_WIDE_MIN_BATCH_BY_CAPABILITY:
            return False
        cap = _LOCAL_SELECT_MAX_K_BY_CAPABILITY.get(cc, 0)
        min_batch = _LOCAL_SELECT_WIDE_MIN_BATCH_BY_CAPABILITY[cc]
    if int(top_k_max) > cap:
        return False
    span = _THREADS * int(ept) * int(cluster)
    chunks = -(-int(vocab) // span)
    if chunks <= _LEADER_PUSH_WIDE_MAX_CHUNKS or chunks > _LOCAL_SELECT_WIDE_MAX_CHUNKS:
        return False
    return int(vocab) % span == 0 or int(batch) >= min_batch


def _local_select_flag(
    cluster: int,
    ept: int,
    stream: bool,
    launch_flags: int,
    top_k_max: Optional[int],
    capability: Optional[tuple[int, int]] = None,
    vocab: Optional[int] = None,
) -> int:
    """Stage-1 ``launch_flags`` bit 10 for a fused launch whose other bits are ``launch_flags``: the CTA-local select
    form of the leader-push sample build those bits select (bit 0 with bit 9 and bit 4 or 6), on a variant that ships it,
    when the largest top-k is at most ``_LOCAL_SELECT_MAX_K_BY_CAPABILITY`` for ``capability`` (None: the largest cap of
    the table) -- or, for a row of one register chunk per CTA (``vocab`` given), at most the capability's
    ``_LOCAL_SELECT_ONE_CHUNK_MAX_K_BY_CAPABILITY`` cap when that is larger.  Never on a chain, the whole-CTA tail or the
    pull form."""
    if not stream or top_k_max is None:
        return 0
    if (launch_flags & _FLAG_FUSE_TAIL) == 0 or (launch_flags & _FLAG_LEADER_PUSH) == 0:
        return 0
    coarse = (launch_flags & _FLAG_COARSE_SAMPLE) != 0
    spec = (launch_flags & _FLAG_SPEC_SAMPLE) != 0
    if not (coarse or spec):
        return 0
    one_chunk = (
        vocab is not None
        and -(-int(vocab) // (_THREADS * int(ept) * int(cluster))) == 1
    )
    if capability is None:
        cap = max(_LOCAL_SELECT_MAX_K_BY_CAPABILITY.values())
        if one_chunk:
            cap = max(cap, max(_LOCAL_SELECT_ONE_CHUNK_MAX_K_BY_CAPABILITY.values()))
    else:
        cc = (int(capability[0]), int(capability[1]))
        cap = _LOCAL_SELECT_MAX_K_BY_CAPABILITY.get(cc, 0)
        if one_chunk:
            cap = max(cap, _LOCAL_SELECT_ONE_CHUNK_MAX_K_BY_CAPABILITY.get(cc, 0))
    if int(top_k_max) > cap:
        return 0
    return (
        _FLAG_LOCAL_SELECT
        if _stage1_has_local_select(cluster, ept, True, coarse, spec)
        else 0
    )


def _stage1_has_int_tail(cluster: int, ept: int, stream: bool) -> bool:
    """Whether the frozen variant ships the integer-tested form of its whole-CTA tail build (a manifest entry with
    ``int_tail``): the kernel taken by launch_flags bit 11 with bit 3.  Every output is bit-identical to the f64 tail."""
    found = False
    for v in load_manifest()["stage1"]:
        if (v["cluster"], v["ept"], bool(v["stream"])) == (cluster, ept, stream):
            found = True
            if v.get("int_tail", False):
                return True
    if not found:
        raise ValueError(f"no frozen stage-1 variant ({cluster}, {ept}, {stream})")
    return False


def _int_tail_flag(
    cluster: int,
    ept: int,
    stream: bool,
    launch_flags: int,
    capability: Optional[tuple[int, int]] = None,
) -> int:
    """Stage-1 ``launch_flags`` bit 11 for a whole-CTA-tail launch (``launch_flags`` carries bit 3): the integer-tested
    tail of a streaming variant that ships it, on the capabilities in ``_INT_TAIL_CAPABILITIES`` (``capability`` None:
    taken)."""
    if not stream or (launch_flags & _FLAG_FUSE_BLOCK_TAIL) == 0:
        return 0
    if (
        capability is not None
        and (int(capability[0]), int(capability[1])) not in _INT_TAIL_CAPABILITIES
    ):
        return 0
    return _FLAG_INT_TAIL if _stage1_has_int_tail(cluster, ept, True) else 0


def _sample_build_flag(
    cluster: int,
    ept: int,
    stream: bool,
    top_k_max: int,
    vocab: int,
    batch: Optional[int] = None,
    capability: Optional[tuple[int, int]] = None,
) -> int:
    """The stage-1 sample-build selection for one launch: bit 6 (speculative sample) where its policy applies,
    otherwise bit 4 (coarse sample) where that policy applies, otherwise the default build.  The two twins are
    exclusive."""
    spec = _spec_sample_flag(cluster, ept, stream, vocab, top_k_max, batch, capability)
    return spec if spec else _coarse_sample_flag(cluster, ept, stream, top_k_max)


@functools.cache
@functools.cache
def _row_span_diet_capability(device_index: int) -> bool:
    return _device_capability(device_index) in _ROW_SPAN_DIET_CAPABILITIES


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
    return _FLAG_STREAM_PREPASS if _device_capability(device_index)[0] >= 10 else 0


@functools.cache
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
    The device-dependent defaults are resolved here and the ranking itself is memoised per resolved argument
    tuple (``_choose_stage1_resolved``): the served route asks twice per call and the round-7 fused-tail terms
    doubled the ranking's cost (B200 host: 102 -> 162 us per call), which a host-bound two-launch chain pays
    inside its launch gap.

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
    if two_launch is None:
        two_launch = _launch_is_two_kernels(top_k_max)
    return _choose_stage1_resolved(
        int(batch),
        int(vocab),
        int(sm_count),
        None if smem_limit is None else int(smem_limit),
        None if top_k_max is None else int(top_k_max),
        bool(two_launch),
    )


def _ragged_last_chunk(vocab: int, cluster: int, ept: int) -> bool:
    """True when the last per-CTA ``512 x ept`` chunk of a ``vocab`` row on ``cluster`` CTAs is less than
    ``_RAGGED_CHUNK_FILL`` full (and not exactly full)."""
    exact = vocab / (cluster * _THREADS * ept)
    frac = exact - math.floor(exact)
    return frac > 0.0 and frac < _RAGGED_CHUNK_FILL


# Round-10 lever L6 on R200 (sm 212, compute capability 10.7): with the leader-push builds the cluster-8 ept-16 stream
# (`_cs_lp`, launch flags 529) runs the small-k rows the cost table hands to a cluster-4 / cluster-8 ept-32 stream 8-15 %
# faster while both grids fit one wave (V128256 / V151936 B1 / B8 / B16 k10 / k50 0.85-0.91, V262144 0.914-0.919; two
# perturbed processes, eager and graph, bitwise) and 1.5x slower once the cluster-8 grid needs two waves (B32).  The 212
# table cannot express that ordering without flipping unmeasured or k > 64 picks, so the measured rule is applied after the
# ranking (round-10 ledger; mirrored by cake `one_wave_e16_repick`).
_ONE_WAVE_E16_REPICK_SM_COUNTS: frozenset[int] = frozenset({212})


def _one_wave_e16_repick(
    batch: int,
    best: tuple[int, int],
    streaming: list[tuple[int, int]],
    sm_count: int,
    large_k: bool,
) -> tuple[int, int]:
    """``best`` is the ranked streaming ``(cluster, ept)``; the cluster-8 ept-16 stream replaces a small-k ept-32 pick
    when both grids run in one wave of the ``sm_count`` table (see ``_ONE_WAVE_E16_REPICK_SM_COUNTS``)."""
    if (
        _nearest_table(int(sm_count)) not in _ONE_WAVE_E16_REPICK_SM_COUNTS
        or large_k
        or best[1] < 32
    ):
        return best
    if (8, 16) not in streaming:
        return best
    wave_ctas = _wave_ctas(int(sm_count))
    if (
        -(-(batch * best[0]) // wave_ctas[best[0]]) != 1
        or -(-(batch * 8) // wave_ctas[8]) != 1
    ):
        return best
    return (8, 16)


# Round-11 lever M4 on H100 (sm 132, compute capability 9.0): at V262144 the 132 table ranks the cluster-2 ept-32 stream
# (128 one-wave CTAs, 8 chunks each) ahead of the cluster-1 ept-32 stream (64 CTAs, 16 chunks) by the per-chunk constant,
# but from B64 on the single-CTA rows already saturate the read and skip the cluster exchange (measured 0.981 / 0.986 at
# k10 / k50, perturbed processes, eager and graph, bitwise; the round-10 regret row).  A constant refit cannot keep the
# B <= 32 ranking while flipping B64, so the measured crossover is applied after the ranking (mirrored by cake
# `one_wave_c1_repick`): a small-k one-wave cluster-2 ept-32 pick on rows of at least _ONE_WAVE_C1_REPICK_MIN_CHUNKS
# single-CTA chunks yields to the cluster-1 ept-32 stream from the table's measured batch on, while that grid runs in one wave.
_ONE_WAVE_C1_REPICK_MIN_BATCH_BY_SM_COUNT: dict[int, int] = {132: 64}
_ONE_WAVE_C1_REPICK_MIN_CHUNKS = 16


def _one_wave_c1_repick(
    batch: int,
    vocab: int,
    best: tuple[int, int],
    streaming: list[tuple[int, int]],
    sm_count: int,
    large_k: bool,
) -> tuple[int, int]:
    """``best`` is the ranked streaming ``(cluster, ept)``; the cluster-1 ept-32 stream replaces a small-k cluster-2 ept-32
    pick on long rows from the measured batch on when both grids run in one wave (see
    ``_ONE_WAVE_C1_REPICK_MIN_BATCH_BY_SM_COUNT``)."""
    min_batch = _ONE_WAVE_C1_REPICK_MIN_BATCH_BY_SM_COUNT.get(
        _nearest_table(int(sm_count))
    )
    if min_batch is None or large_k or best != (2, 32) or int(batch) < min_batch:
        return best
    if (
        -(-int(vocab) // (_THREADS * 32)) < _ONE_WAVE_C1_REPICK_MIN_CHUNKS
        or (1, 32) not in streaming
    ):
        return best
    wave_ctas = _wave_ctas(int(sm_count))
    if -(-(batch * 2) // wave_ctas[2]) != 1 or -(-batch // wave_ctas[1]) != 1:
        return best
    return (1, 32)


@functools.lru_cache(maxsize=8192)
def _choose_stage1_resolved(
    batch: int,
    vocab: int,
    sm_count: int,
    smem_limit: Optional[int],
    top_k_max: Optional[int],
    two_launch: bool,
) -> tuple[int, int, bool]:
    """The ranking behind ``choose_stage1`` for fully resolved arguments (pure in its arguments and the frozen
    manifest, hence memoised)."""
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
        stream_wide_ragged_large_k,
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
                + (
                    stream_wide_ragged_large_k
                    if large_k > 0.0
                    and ce[1] > 16
                    and _ragged_last_chunk(vocab, ce[0], ce[1])
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
            # the stage-2/3 kernel behind a resident grid (no capability fuses a resident); on a device that fuses
            # streams the chain additionally pays the tail-vs-chain terms against its one-launch peers
            extra = resident_two_launch
            if not regime_two_launch:
                extra += (
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
    best = _one_wave_e16_repick(batch, best, streaming, int(sm_count), large_k > 0.0)
    best = _one_wave_c1_repick(
        batch, vocab, best, streaming, int(sm_count), large_k > 0.0
    )
    return best[0], best[1], True


@functools.cache
def choose_stage23(top_k_max: int) -> tuple[int, int]:
    """Smallest frozen ``(threads, items)`` slab that holds ``top_k_max`` entries."""
    for threads, items in _stage23_variants():
        if threads * items >= top_k_max:
            return threads, items
    raise ValueError(f"top_k_max={top_k_max} exceeds the frozen stage-2/3 capacity")


class _LaunchPlan(NamedTuple):
    """Every dispatch decision of one pipeline call: the stage-1 variant, the stage-2/3 slab, the stage-1
    ``launch_flags`` and whether stage 2/3 runs inside the stage-1 kernel (one launch)."""

    cluster: int
    ept: int
    stream: bool
    threads: int
    items: int
    launch_flags: int
    one_launch: bool
    stage23_flags: int


@functools.lru_cache(maxsize=8192)
def _launch_plan(
    device_index: int, batch: int, vocab: int, top_k_max: int
) -> _LaunchPlan:
    """The dispatch of ``top_k_top_p_sampling_from_probs`` for ``batch`` rows of ``vocab`` entries with largest
    top-k ``top_k_max`` on device ``device_index``: a pure function of these four values and the frozen bundle, so
    it is computed once per distinct call shape (the route check and the launch share it).  Raises ``ValueError``
    when no frozen stage-1 variant covers ``vocab``."""
    sm_count = _sm_count(device_index)
    capability = _device_capability(device_index)
    two_launch = _launch_is_two_kernels(top_k_max, device_index)
    cluster, ept, stream = _choose_stage1_resolved(
        batch, vocab, sm_count, _smem_optin(device_index), top_k_max, two_launch
    )
    threads, items = choose_stage23(top_k_max)
    # Small top-k: stage 2/3 runs inside the stage-1 kernel (same outputs, one launch).
    fused = top_k_max <= _fused_tail_kcap() and _stage1_has_fused_tail(
        cluster, ept, stream
    )
    # Larger top-k up to the slab: the whole-CTA tail, on the capabilities where it beats the chain, for a
    # one-wave stage-1 grid.
    fused_block = not fused and _fuse_block_tail(
        batch,
        cluster,
        ept,
        stream,
        sm_count,
        top_k_max,
        not _block_tail_enabled(device_index),
        vocab,
        capability,
    )
    if fused:
        sample_flag = _sample_build_flag(
            cluster, ept, stream, top_k_max, vocab, batch, capability
        )
        launch_flags = _FLAG_FUSE_TAIL | sample_flag
        leader = _leader_push_flag(
            cluster,
            ept,
            stream,
            launch_flags,
            vocab,
            capability,
            _local_select_reach(
                cluster, ept, stream, launch_flags, top_k_max, vocab, batch, capability
            ),
            _leader_push_small_batch_reach(
                cluster, ept, stream, launch_flags, top_k_max, vocab, batch, capability
            ),
        )
        launch_flags |= leader or _slab_tail_flag(
            cluster, ept, stream, sample_flag, vocab, capability
        )
        launch_flags |= _local_select_flag(
            cluster, ept, stream, launch_flags, top_k_max, capability, vocab
        )
    elif fused_block:
        launch_flags = _FLAG_FUSE_BLOCK_TAIL
        launch_flags |= _int_tail_flag(cluster, ept, stream, launch_flags, capability)
    else:
        # Two launches: stage 2/3 may start early only when its CTAs fit beside the last stage-1 wave; a
        # streaming variant triggers before its first pass on Blackwell / Rubin, after its filter pass on Hopper.
        launch_flags = _early_trigger_flag(batch, cluster, sm_count)
        if stream:
            launch_flags |= _stream_prepass_flag(device_index)
            launch_flags |= _row_span_diet_flag(cluster, True, top_k_max, device_index)
            launch_flags |= _spec_sample_flag(
                cluster, ept, True, vocab, top_k_max, batch, capability
            )
            leader = _leader_push_flag(
                cluster, ept, True, launch_flags, vocab, capability
            )
            launch_flags |= leader or _coarse_push_flag(
                cluster, ept, True, launch_flags, capability
            )
    return _LaunchPlan(
        cluster,
        ept,
        stream,
        threads,
        items,
        launch_flags,
        fused or fused_block,
        _stage23_variant_flags(device_index),
    )


@functools.lru_cache(maxsize=8192)
def _slab_plan(
    device_index: int, batch: int, vocab: int, top_k_max: int
) -> tuple[int, int, bool, int]:
    """Stage-1 variant and ``launch_flags`` of :func:`top_k_probs_to_slab` (no tail, no PDL dependent): the
    speculative- or coarse-sample build on a stream, the row-span filter arm for a large top-k on a cluster-8
    stream."""
    cluster, ept, stream = _choose_stage1_resolved(
        batch,
        vocab,
        _sm_count(device_index),
        _smem_optin(device_index),
        top_k_max,
        _launch_is_two_kernels(top_k_max, device_index),
    )
    capability = _device_capability(device_index)
    flags = _sample_build_flag(
        cluster, ept, stream, top_k_max, vocab, batch, capability
    ) | _row_span_diet_flag(cluster, stream, top_k_max, device_index)
    flags |= _leader_push_flag(cluster, ept, stream, flags, vocab, capability)
    return cluster, ept, stream, flags


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
        _launch_plan(_device_index(probs.device), int(batch), int(vocab), int(kmax))
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
        like :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs`.  While a CUDA graph is
        being captured the generator is not read or advanced (that would register it with the
        graph and add two fill kernels to every replay); the captured launch uses the generator's
        initial seed with a host-side offset that differs between captures.
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
    plan = _launch_plan(_device_index(probs.device), int(batch), int(vocab), int(kmax))
    slab = _slab()
    vals, idxs, cnt = (
        workspace if workspace is not None else _workspace(batch, slab, probs.device)
    )
    if out is None:
        out = torch.empty(batch, device=probs.device, dtype=torch.int32)
    philox_seed, philox_offset = _philox_params(
        batch, generator, probs.device, philox_seed, philox_offset
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
    stage1_args = (
        probs,
        k_arr,
        k_scalar,
        k_kind,
        vals,
        idxs,
        cnt,
        plan.cluster,
        plan.ept,
        1 if plan.stream else 0,
        p_arr,
        p_scalar,
        p_kind,
        out,
        renorm,
        int(philox_seed) & 0xFFFFFFFFFFFFFFFF,
        int(philox_offset) & 0xFFFFFFFFFFFFFFFF,
        1 if renorm_out is not None else 0,
        plan.launch_flags,
    )
    module.radix_topk(*stage1_args, stream)
    if plan.one_launch:
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
        plan.threads,
        plan.items,
        plan.stage23_flags,
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
    cluster, ept, stream_variant, launch_flags = _slab_plan(
        _device_index(probs.device), int(batch), int(vocab), int(kmax)
    )
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
        launch_flags,  # no fused tail, no PDL dependent follows (see _slab_plan)
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
