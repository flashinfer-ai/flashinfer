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

"""Host side of the experimental Cake Kimi-K3 AttnRes backend (SM100 / SM103).

The operator (Kimi-K3 attention-residual mixing, ``_apply_attn_res`` of the
model) on caller-owned BF16 tensors with ``H = 7168`` and ``K = num_blocks``
in ``[0, 8]``::

    prefix[M, H]       += delta[M, H]            (BF16 rounded, when delta is given)
    blocks[:, w, :]     = prefix                 (exact snapshot, when block_write_idx = w >= 0)
    c_i                 = blocks[:, i, :] (i < K), c_K = prefix
    s_i                 = sum_h rmsnorm(c_i, eps)[h] * norm_weight[h] * qk_weight[h]   (FP32)
    p                   = softmax(s)                                                  (FP32)
    mixed               = sum_i p_i * c_i                                             (FP32)
    out                 = BF16(rmsnorm(mixed, output_norm_eps) * output_norm_weight)  (or BF16(mixed))

``plan_route`` selects the generated program from host-known scalars and tensor
metadata only (architecture, SM count, ``M``, ``K``, the PDL launch policy and
the semantic flags), exactly as the Cake production dispatcher does; the export
verifies key, grid and schedule parity for every row of its denominator.  Dense
token-major inputs with the residual add and the fused output norm take the
persistent TMEM path (or one of the installed native ports, the pair kernel or
the exact K = 0 path); the block-boundary snapshot write (``block_write_idx`` ==
K) runs the write variants of the dense write programs; the other semantic
variants (no delta, a snapshot write below K, no output norm, row-padded
layouts) take the one-CTA-per-token bootstrap program.

The route tables are the registry's ``ROUTES`` literals (``cake_jit.py``, written
by the exporter together with ``MODULES`` and ``KERNELS``): the legacy tables of
the Cake dispatcher plus the round-2 cell tables (``R2_*``: per architecture and
SM count the cells whose program changed -- a persistent key, a native port, the
pair kernel, a small-M cluster or fast-math choice).  Persistent flag bit 4 is
``register_carry_only`` (the K1 / K2 register-carry bodies).

``prepare_kimi_k3_attn_res`` binds one call; its ``launch()`` performs no CUDA
allocation and no host synchronisation and is CUDA-graph capturable (capture
belongs to the caller).  ``prefix`` (and ``blocks`` when a snapshot is written)
are mutated in place on every launch, as in the model.
"""

import functools
import re
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional

import torch
import tvm_ffi

from .cake_jit import KERNELS, MODULES, ROUTES, load, record_of

HIDDEN_SIZE = 7168
MAX_BLOCKS = int(ROUTES["MAX_BLOCKS"])
DIRECT_THREADS = 256
PERSISTENT_THREADS = 288
PAIR_THREADS = 320
#: The pair kernel's ``consumer_regs`` builds, three whole warpgroups each: the ``_cr232`` keys hold
#: the two consumer groups at 232 registers each warp and the producer warpgroup at 40; the
#: ``_cr200_pk1`` keys hold them at 200 and 104 and park each consumer thread's accumulators in
#: tensor memory while the prefix chunk streams (``park_acc`` 1).
PAIR_CONSUMER_REGS = 232
PAIR_PARK_ACC_CONSUMER_REGS = 200
PAIR_CONSUMER_REGS_THREADS = 384
#: Physical SM counts the native ports and the pair kernel are qualified on: 148 (B200 / B300) and
#: 152 (GB300).  They launch one CTA per SM and grid-stride over the tokens, so their grid is the
#: device's SM count; any other count keeps the persistent / small-M families, whose grids derive
#: from ``num_sms`` as well.
NATIVE_SM_COUNTS = tuple(int(n) for n in ROUTES["NATIVE_SM_COUNTS"])
PERSISTENT_BALANCED_GRID = int(ROUTES["PERSISTENT_BALANCED_GRID"])
PERSISTENT_BALANCED_GRID_MIN_M = int(ROUTES["PERSISTENT_BALANCED_GRID_MIN_M"])
PERSISTENT_BALANCED_GRID_MAX_M = int(ROUTES["PERSISTENT_BALANCED_GRID_MAX_M"])
PERSISTENT_BALANCED_GRID_MIN_SMS = int(ROUTES["PERSISTENT_BALANCED_GRID_MIN_SMS"])
PERSISTENT_BALANCED_NC = frozenset(int(n) for n in ROUTES["PERSISTENT_BALANCED_NC"])
PERSISTENT_CHUNK_DEPTH = int(ROUTES["PERSISTENT_CHUNK_DEPTH"])
PERSISTENT_SOURCES_PER_CHUNK = int(ROUTES["PERSISTENT_SOURCES_PER_CHUNK"])
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
ARCHES = tuple(sorted(set(SUPPORTED_COMPUTE_CAPABILITIES.values())))


# ---------------------------------------------------------------------------
# Production route policy (Cake ``kimi_k3_attn_res.select_route``), ported.
# Every table below is read from the registry's ``ROUTES`` literals: a set is
# spelled as a sorted list, a table keyed by (M, K) cells as [[M, K], value]
# pairs (an empty one as {}), the round-2 tables as architecture -> SM count ->
# cell pairs.  The export checks this port against the policy row by row.
# ---------------------------------------------------------------------------


def _cells(table) -> dict[tuple[int, int], Any]:
    """``[[[M, K], value], ...]`` -> ``{(M, K): value}`` (an empty table is ``{}``)."""
    if isinstance(table, dict):
        return {}
    return {(int(cell[0]), int(cell[1])): value for cell, value in table}


def _cell_set(table) -> frozenset[tuple[int, int]]:
    return frozenset((int(cell[0]), int(cell[1])) for cell in table)


def _bands(table) -> dict[int, tuple[tuple[int, int], ...]]:
    """``{K: [[a, b], ...]}`` -> ``{K: ((a, b), ...)}``."""
    return {
        int(k): tuple((int(a), int(b)) for a, b in bands) for k, bands in table.items()
    }


def _by_arch(table, convert) -> dict[str, Any]:
    return {arch: convert(value) for arch, value in table.items()}


def _by_arch_sm(table, convert) -> dict[str, dict[int, Any]]:
    return {
        arch: {int(sm): convert(value) for sm, value in by_sm.items()}
        for arch, by_sm in table.items()
    }


_SM103_EARLY_CONSUMED_RELEASE_CELLS = _cell_set(
    ROUTES["SM103_EARLY_CONSUMED_RELEASE_CELLS"]
)
_SM100_K1_NC2_M = frozenset(int(m) for m in ROUTES["SM100_K1_NC2_M"])
_SM103_K1_NC2_M = frozenset(int(m) for m in ROUTES["SM103_K1_NC2_M"])
_SM103_K5_NC4_D3_MIN_M = int(ROUTES["SM103_K5_NC4_D3_MIN_M"])
_SM100_K5_NC4_D3_MIN_M = int(ROUTES["SM100_K5_NC4_D3_MIN_M"])
_NC3_D3_CELLS = _by_arch(ROUTES["NC3_D3_CELLS"], _cell_set)
_SM100_K4_NC3_D3_MIN_M = int(ROUTES["SM100_K4_NC3_D3_MIN_M"])
_SM100_HELD_CONSUMED_RELEASE_CELLS = _cell_set(
    ROUTES["SM100_HELD_CONSUMED_RELEASE_CELLS"]
)
_SM100_HELD_CONSUMED_RELEASE_MAX_M = {
    int(k): int(m) for k, m in ROUTES["SM100_HELD_CONSUMED_RELEASE_MAX_M"].items()
}
_SM103_EARLY_CONSUMED_RELEASE_MIN_M = {
    int(k): int(m) for k, m in ROUTES["SM103_EARLY_CONSUMED_RELEASE_MIN_M"].items()
}
_SM103_EARLY_CONSUMED_RELEASE_MAX_M = {
    int(k): int(m) for k, m in ROUTES["SM103_EARLY_CONSUMED_RELEASE_MAX_M"].items()
}
_R5_PERSISTENT_SCHEDULE_CELLS = _by_arch(
    ROUTES["R5_PERSISTENT_SCHEDULE_CELLS"],
    lambda t: {cell: (int(v[0]), int(v[1])) for cell, v in _cells(t).items()},
)
_R5_WRITE_PERSISTENT_SCHEDULE_CELLS = _by_arch(
    ROUTES["R5_WRITE_PERSISTENT_SCHEDULE_CELLS"],
    lambda t: {cell: (int(v[0]), int(v[1])) for cell, v in _cells(t).items()},
)
_R5_PREFIX_BF16_ADD_CELLS = _by_arch(ROUTES["R5_PREFIX_BF16_ADD_CELLS"], _cell_set)
_SM100_R5_EARLY_CONSUMED_RELEASE_CELLS = _cell_set(
    ROUTES["SM100_R5_EARLY_CONSUMED_RELEASE_CELLS"]
)
_SM103_R5_HELD_CONSUMED_RELEASE_CELLS = _cell_set(
    ROUTES["SM103_R5_HELD_CONSUMED_RELEASE_CELLS"]
)
_R5_WRITE_EARLY_CONSUMED_RELEASE_CELLS = _by_arch(
    ROUTES["R5_WRITE_EARLY_CONSUMED_RELEASE_CELLS"], _cell_set
)
_R5_SMALL_M_EXCLUDED_CELLS = _by_arch(ROUTES["R5_SMALL_M_EXCLUDED_CELLS"], _cell_set)
_R5_SMALL_M_INCLUDED_CELLS = _by_arch(ROUTES["R5_SMALL_M_INCLUDED_CELLS"], _cell_set)
_R5_SMALL_M_CHUNK_CELLS = _by_arch(ROUTES["R5_SMALL_M_CHUNK_CELLS"], _cells)
_R5_SMALL_M_CLUSTER_CELLS = _by_arch(ROUTES["R5_SMALL_M_CLUSTER_CELLS"], _cells)
_WRITE_HELD_CONSUMED_RELEASE_MIN_M = _by_arch(
    ROUTES["WRITE_HELD_CONSUMED_RELEASE_MIN_M"],
    lambda t: {int(k): int(m) for k, m in t.items()},
)
_K0_TMA_GRID_MULTIPLIER_M = _by_arch(
    ROUTES["K0_TMA_GRID_MULTIPLIER_M"], lambda t: {int(m): int(x) for m, x in t.items()}
)
_SM103_RELAXED_PRODUCER_WAIT_CELLS = _cell_set(
    ROUTES["SM103_RELAXED_PRODUCER_WAIT_CELLS"]
)
# Native ports: name, {arch: routed M}, num_blocks.  Only the cells above the small-M table are
# listed: every other cell the ports were measured on resolves to a small-M program first.
_NATIVE_ROUTES = tuple(
    (
        str(name),
        {arch: frozenset(int(m) for m in ms) for arch, ms in routed.items()},
        int(k),
    )
    for name, routed, k in ROUTES["NATIVE_ROUTES"]
)
# Small-M direct kernel: one token per 256-thread CTA, every source register-resident,
# selected ahead of every other dense program for M <= max_m (per architecture and K).
_SMALL_M_DIRECT_MAX_M = _by_arch(
    ROUTES["SMALL_M_DIRECT_MAX_M"], lambda t: {int(k): int(m) for k, m in t.items()}
)
# Cluster split of the small-M kernel: K -> ((max_m, cluster), ...) bands in ascending max_m; the
# first band with max_m >= M gives the CTAs per token (2 or 4). K absent or M above the last band
# -> one CTA per token.
_SMALL_M_CLUSTER = _by_arch(ROUTES["SMALL_M_CLUSTER"], _bands)
# Online-softmax chunk size of the small-M programs: K -> ((max_m, sources_per_chunk), ...) bands in
# ascending max_m; the first band with max_m >= M applies, K absent or M above the last band -> the
# program default (nc4 for K <= 3, nc3 for K >= 4).
_SMALL_M_CHUNK_BANDS = _by_arch(ROUTES["SMALL_M_CHUNK_BANDS"], _bands)
# Round r5 (dense coverage D): a structural cell also serves the off-grid M just below it: (lo, hi)
# -> the r5 cell tables are consulted with M = hi for lo <= M <= hi (dense programs only).
_R5_CELL_RANGES = _by_arch(ROUTES["R5_CELL_RANGES"], _bands)
# Per program class ("small_m" / "persistent" and their snapshot-write twins), K -> inclusive M bands
# whose program compiles with --use_fast_math (kernel key ``_fm``, schedule id ``_fastmath``).
_FAST_MATH_BANDS = _by_arch(
    ROUTES["FAST_MATH_BANDS"],
    lambda t: {program: _bands(bands) for program, bands in t.items()},
)
# Round-2 cell tables (architecture -> SM count -> (M, K) cell), consulted before the legacy
# derivation; a cell absent here keeps the legacy program.  The export spells the round-3 cells
# into the same tables (the native K5 cells of both architectures and the sm_100a ``_crfnone``
# persistent twins).
_R2_PERSISTENT_KEY_CELLS = _by_arch_sm(
    ROUTES["R2_PERSISTENT_KEY_CELLS"],
    lambda t: {
        cell: (tuple(v) if isinstance(v, (list, tuple)) else str(v))
        for cell, v in _cells(t).items()
    },
)
_R2_NATIVE_CELLS = _by_arch_sm(
    ROUTES["R2_NATIVE_CELLS"], lambda t: {c: str(v) for c, v in _cells(t).items()}
)
_R2_PAIR_CELLS = _by_arch_sm(
    ROUTES["R2_PAIR_CELLS"], lambda t: {c: str(v) for c, v in _cells(t).items()}
)
_R2_SMALL_M_CLUSTER_CELLS = _by_arch_sm(
    ROUTES["R2_SMALL_M_CLUSTER_CELLS"],
    lambda t: {c: int(v) for c, v in _cells(t).items()},
)
_R2_SMALL_M_WRITE_CLUSTER_CELLS = _by_arch_sm(
    ROUTES["R2_SMALL_M_WRITE_CLUSTER_CELLS"],
    lambda t: {c: int(v) for c, v in _cells(t).items()},
)
_R2_SMALL_M_FAST_MATH_CELLS = _by_arch_sm(
    ROUTES["R2_SMALL_M_FAST_MATH_CELLS"], _cell_set
)
_R2_SMALL_M_WRITE_FAST_MATH_CELLS = _by_arch_sm(
    ROUTES["R2_SMALL_M_WRITE_FAST_MATH_CELLS"], _cell_set
)
# Round-5 write cells: the dense key of the program a snapshot-write cell's variant derives from,
# consulted before the derivation (the derived write program holds the release and adds in bf16
# without the early release); ``_persistent_plan_exact`` appends ``_write``.  A cell names one key
# or one key per PDL mode (off, on).
_WRITE_PERSISTENT_KEY_CELLS = _by_arch_sm(
    ROUTES["WRITE_PERSISTENT_KEY_CELLS"],
    lambda t: {
        cell: (tuple(v) if isinstance(v, (list, tuple)) else str(v))
        for cell, v in _cells(t).items()
    },
)


def _r2(table: dict[str, dict[int, Any]], arch: str, num_sms: Optional[int]) -> Any:
    """The round-2 cells of ``table`` at one configuration (empty when there are none)."""
    if num_sms is None:
        return {}
    return table.get(arch, {}).get(int(num_sms), {})


def _small_m_cluster(
    arch: str, M: int, K: int, *, num_sms: Optional[int] = None, write: bool = False
) -> int:
    r2 = _r2(
        _R2_SMALL_M_WRITE_CLUSTER_CELLS if write else _R2_SMALL_M_CLUSTER_CELLS,
        arch,
        num_sms,
    ).get((M, K))
    if r2 is not None:
        return int(r2)
    cell = _R5_SMALL_M_CLUSTER_CELLS.get(arch, {}).get((M, K))
    if cell is not None:
        return int(cell)
    for max_m, cluster in _SMALL_M_CLUSTER[arch].get(K, ()):
        if max_m >= M:
            return int(cluster)
    return 1


def _cell_m(arch: str, M: int, K: int) -> int:
    """The structural-cell M whose round-r5 tables apply to a dense row (``M`` itself outside the ranges)."""

    for lo, hi in _R5_CELL_RANGES[arch].get(K, ()):
        if lo <= M <= hi:
            return hi
    return M


def _fast_math(gpu_arch, M, num_blocks, program, num_sms: Optional[int] = None):
    if program == "small_m" and (M, num_blocks) in _r2(
        _R2_SMALL_M_FAST_MATH_CELLS, gpu_arch, num_sms
    ):
        return True
    if program == "small_m_write" and (M, num_blocks) in _r2(
        _R2_SMALL_M_WRITE_FAST_MATH_CELLS, gpu_arch, num_sms
    ):
        return True
    return any(
        min_m <= M <= max_m
        for min_m, max_m in _FAST_MATH_BANDS[gpu_arch][program].get(num_blocks, ())
    )


def _small_m_sources_per_chunk(arch: str, M: int, K: int) -> int | None:
    default = 4 if K <= 3 else 3
    cell = _R5_SMALL_M_CHUNK_CELLS.get(arch, {}).get((M, K))
    if cell is not None:
        return None if int(cell) == default else int(cell)
    for max_m, sources_per_chunk in _SMALL_M_CHUNK_BANDS[arch].get(K, ()):
        if max_m >= M:
            return None if int(sources_per_chunk) == default else int(sources_per_chunk)
    return None


class RoutePlan(NamedTuple):
    """The generated program one call launches and its launch geometry."""

    kind: str  # "direct" | "small_m" | "native" | "pair" | "k0_tma" | "persistent"
    kernel_key: str
    grid_x: int
    threads: int
    schedule_id: str
    route_id: str
    arch: Optional[str]
    use_pdl: bool
    # Exact (table) key when this plan runs a registered variant of the same program family
    # instead, because the checkout does not register the exact variant; None for exact plans.
    fallback_from: Optional[str] = None


@functools.lru_cache(maxsize=None)
def _device_capability(index: int) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(index))


@functools.lru_cache(maxsize=None)
def _device_sm_count(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def arch_for(device: torch.device) -> str:
    capability = _device_capability(_device_index(device))
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 AttnRes kernels require compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def sm_count_for(device: torch.device) -> int:
    return _device_sm_count(_device_index(device))


def _wait_policy(arch: str) -> bool:
    return arch == "sm_100a"


def _schedule(arch: str, M: int, K: int, write: bool = False) -> tuple[int, int]:
    """``(sources_per_chunk, chunk_depth)`` of the persistent common path (``write``: the program the
    snapshot-write twin is derived from; round r5 cells may give it its own schedule)."""
    r5_cells = (
        _R5_WRITE_PERSISTENT_SCHEDULE_CELLS if write else _R5_PERSISTENT_SCHEDULE_CELLS
    ).get(arch, {})
    cell = (M if write else _cell_m(arch, M, K), K)
    if cell in r5_cells:
        return r5_cells[cell]
    if (M, K) in _NC3_D3_CELLS[arch] or (
        arch == "sm_100a" and K == 4 and M >= _SM100_K4_NC3_D3_MIN_M
    ):
        return 3, 3
    if arch == "sm_100a":
        if K == 1 and M in _SM100_K1_NC2_M:
            return 2, PERSISTENT_CHUNK_DEPTH
        if K == 5 and M >= _SM100_K5_NC4_D3_MIN_M:
            return PERSISTENT_SOURCES_PER_CHUNK, 3
        if K == 4:
            return (
                5 if M in {1024, 2048, 4096, 8192, 16384} else 3
            ), PERSISTENT_CHUNK_DEPTH
    else:
        if K == 1 and M in _SM103_K1_NC2_M:
            return 2, PERSISTENT_CHUNK_DEPTH
        if K == 5 and M >= _SM103_K5_NC4_D3_MIN_M:
            return PERSISTENT_SOURCES_PER_CHUNK, 3
        if K == 4:
            return (3 if M <= 512 else 5), PERSISTENT_CHUNK_DEPTH
    if K == 8:
        return (3 if M <= 256 else 5), PERSISTENT_CHUNK_DEPTH
    return PERSISTENT_SOURCES_PER_CHUNK, PERSISTENT_CHUNK_DEPTH


def _grid(M: int, num_sms: int, sources_per_chunk: int) -> tuple[int, str]:
    if (
        sources_per_chunk in PERSISTENT_BALANCED_NC
        and PERSISTENT_BALANCED_GRID_MIN_M <= M <= PERSISTENT_BALANCED_GRID_MAX_M
        and num_sms >= PERSISTENT_BALANCED_GRID_MIN_SMS
    ):
        return PERSISTENT_BALANCED_GRID, "balanced128_m256_m512"
    return min(M, num_sms), "token_or_full_sm"


def _early_consumed_release(
    arch: str, M: int, K: int, grid_x: int, write: bool = False
) -> bool:
    if (
        write
        and _WRITE_HELD_CONSUMED_RELEASE_MIN_M[arch].get(K, M + 1) <= M
        and (M, K) not in _R5_WRITE_EARLY_CONSUMED_RELEASE_CELLS.get(arch, ())
    ):
        return False
    if arch == "sm_100a":
        return (
            K > 0
            and grid_x < M
            and (M, K) not in _SM100_HELD_CONSUMED_RELEASE_CELLS
            and (
                (
                    not write
                    and (_cell_m(arch, M, K), K)
                    in _SM100_R5_EARLY_CONSUMED_RELEASE_CELLS
                )
                or _SM100_HELD_CONSUMED_RELEASE_MAX_M.get(K, 0) < M
            )
        )
    return (
        K > 0
        and grid_x < M
        and (write or (M, K) not in _SM103_R5_HELD_CONSUMED_RELEASE_CELLS)
        and (
            (M, K) in _SM103_EARLY_CONSUMED_RELEASE_CELLS
            or (
                _SM103_EARLY_CONSUMED_RELEASE_MIN_M.get(K, M + 1)
                <= M
                <= _SM103_EARLY_CONSUMED_RELEASE_MAX_M.get(K, M)
            )
        )
    )


def _producer_wait_acquire(arch: str, M: int, K: int) -> bool:
    return not (arch == "sm_103a" and (M, K) in _SM103_RELAXED_PRODUCER_WAIT_CELLS)


def _native_route(arch: str, num_sms: int, M: int, K: int) -> Optional[str]:
    if num_sms not in NATIVE_SM_COUNTS:
        return None
    port = _r2(_R2_NATIVE_CELLS, arch, num_sms).get((M, K))
    if port is not None:
        return port
    for name, routed, k in _NATIVE_ROUTES:
        if k == K and M in routed[arch]:
            return name
    return None


# Flag positions of the persistent key's ``f<bits>`` field (see ``_plan_route_exact``).  Bit 4 is
# ``register_carry_only`` (round-2 decision D3): the K1 / K2 register-carry bodies, which need bit 2.
_PERSISTENT_FLAG_ECR = 0
_PERSISTENT_FLAG_PWA = 1
_PERSISTENT_FLAG_PREFIX_BF16_ADD = 2
_PERSISTENT_FLAG_PREFIX_ROUND_ONCE = 3
_PERSISTENT_FLAG_REGISTER_CARRY_ONLY = 4
# ``persistent:k<K>_nc<nc>_d<depth>_f<15 bits>[_fm][_crfnone][_crheld][_rcp][_write][_cta2]``:
# ``_fm`` is the fast-math twin, ``_crfnone`` (round 3) the fence-authored twin -- the same program built
# with ``consumed_release_fence_mode="none"`` (the kernel states its own tcgen05 fences), a
# registered program of its own whose schedule label ends in ``_authored_fences`` --, ``_crheld``
# (round 19) the held-release program -- the program of the same key with flag bit 0 set and
# without the suffix, built without the early consumed release; the key spells bit 0 as 0 and
# the schedule label adds ``_held_release`` after ``_authored_fences`` --, ``_rcp`` (round 20) the
# packed register carry of a K = 3 register-carry program -- the program of the same key without
# the suffix keeping each token's bf16 source words in registers instead of TMEM; the schedule
# label adds ``_packed_register_carry`` after ``_held_release`` --, ``_write`` the snapshot-write
# variant and ``_cta2`` the program built for two CTAs per SM, launched on ``min(M, 2 x SMs)``
# CTAs.  A fallback never crosses the ``_fm`` / ``_crfnone`` / ``_crheld`` / ``_rcp`` /
# ``_write`` / ``_cta2`` lines.
_PERSISTENT_KEY = re.compile(
    r"^persistent:k(?P<K>\d+)_nc(?P<nc>\d+)_d(?P<depth>\d+)_f(?P<bits>[01]+)"
    r"(?P<fm>_fm)?(?P<crfnone>_crfnone)?(?P<crheld>_crheld)?(?P<rcp>_rcp)?(?P<write>_write)?"
    r"(?P<cta2>_cta2)?$"
)
_SMALL_M_KEY = re.compile(
    r"^small_m_(direct|cluster(\d+)):k(\d+)(?:_nc(\d+))?(_fm)?(_write)?$"
)
# ``pair:k<K>_nc<nc>[_pbf16a][_fm][_cr232|_cr200_pk1][_write]``: ``_cr232`` is the pair kernel
# built at ``consumer_regs`` = ``PAIR_CONSUMER_REGS``, ``_cr200_pk1`` the one built at
# ``consumer_regs`` = ``PAIR_PARK_ACC_CONSUMER_REGS`` with ``park_acc`` 1 (both
# ``PAIR_CONSUMER_REGS_THREADS`` threads; the schedule label adds ``_consumer_regs232`` /
# ``_consumer_regs200_park_acc1``).
_PAIR_KEY = re.compile(
    r"^pair:k(?P<K>\d+)_nc(?P<nc>\d+)(?P<pbf16a>_pbf16a)?(?P<fm>_fm)?"
    rf"(?P<cr>_cr{PAIR_CONSUMER_REGS}|_cr{PAIR_PARK_ACC_CONSUMER_REGS}(?P<park>_pk1))?"
    r"(?P<write>_write)?$"
)


def _persistent_schedule_id(
    nc: int,
    depth: int,
    bits: str,
    authored_fences: bool = False,
    held_release: bool = False,
    packed_register_carry: bool = False,
) -> str:
    schedule_id = f"trtllm_persistent_ws288_nc{nc}_d{depth}_vec128_fp32x2"
    if bits[_PERSISTENT_FLAG_ECR] == "1":
        schedule_id += "_early_consume"
    if bits[_PERSISTENT_FLAG_PWA] == "0":
        schedule_id += "_relaxed_producer_wait"
    if bits[_PERSISTENT_FLAG_PREFIX_BF16_ADD] == "1":
        schedule_id += "_prefix_bf16_add_packed_inputs_prefix_shared_addr"
    if bits[_PERSISTENT_FLAG_PREFIX_ROUND_ONCE] == "1":
        schedule_id += "_prefix_round_once"
    if bits[_PERSISTENT_FLAG_REGISTER_CARRY_ONLY] == "1":
        schedule_id += "_register_carry_only"
    if authored_fences:
        schedule_id += "_authored_fences"
    if held_release:
        schedule_id += "_held_release"
    if packed_register_carry:
        schedule_id += "_packed_register_carry"
    return schedule_id


def _small_m_plan(
    arch: str,
    M: int,
    K: int,
    use_pdl: bool,
    cluster: int,
    exact_key: Optional[str],
    *,
    sources_per_chunk: Optional[int] = None,
    write_block: bool = False,
    num_sms: Optional[int] = None,
) -> RoutePlan:
    """A small-M family plan: ``cluster`` CTAs per token, chunk ``sources_per_chunk`` (``None`` =
    the module default of K), ``write_block`` = the snapshot-write variant (round r4), ``num_sms``
    = the configuration whose round-2 fast-math cells apply.  ``exact_key`` marks a
    registered-variant fallback of that exact plan."""
    threads = DIRECT_THREADS // cluster
    fast_math = _fast_math(
        arch, M, K, "small_m_write" if write_block else "small_m", num_sms
    )
    nc_suffix = "" if sources_per_chunk is None else f"_nc{int(sources_per_chunk)}"
    fm_suffix = "_fastmath" if fast_math else ""
    fm_key = "_fm" if fast_math else ""
    write_suffix = "_write" if write_block else ""
    if cluster == 1:
        schedule_id = (
            f"small_m_direct_cta256_regres_fp32x2{nc_suffix}{fm_suffix}{write_suffix}"
        )
        grid_policy = "one_token_per_cta"
        kernel_key = f"small_m_direct:k{K}{nc_suffix}{fm_key}{write_suffix}"
    else:
        schedule_id = f"small_m_cluster{cluster}_cta{threads}_regres_fp32x2{nc_suffix}{fm_suffix}{write_suffix}"
        grid_policy = f"one_token_per_cluster{cluster}"
        kernel_key = f"small_m_cluster{cluster}:k{K}{nc_suffix}{fm_key}{write_suffix}"
    route_id = (
        f"{schedule_id}.{arch}.none_k{K}.k{K}.delta1.write{int(write_block)}.norm1."
        f"pdl{int(use_pdl)}.{grid_policy}"
    )
    if exact_key is not None:
        route_id += ".registered_fallback"
    return RoutePlan(
        "small_m",
        kernel_key,
        M * cluster,
        threads,
        schedule_id,
        route_id,
        arch,
        use_pdl,
        exact_key,
    )


def _cell_pinned_write_keys(arch: Optional[str]) -> frozenset[str]:
    """The write programs that ``WRITE_PERSISTENT_KEY_CELLS`` pins to cells on ``arch``.  They
    serve those cells only: an unregistered derived write key never resolves to one of them, so
    pinning a write cell changes no other route."""
    return frozenset(
        f"{key}_write"
        for by_cell in (_WRITE_PERSISTENT_KEY_CELLS.get(arch) or {}).values()
        for value in by_cell.values()
        for key in (value if isinstance(value, tuple) else (value,))
    )


def _persistent_fallback(
    plan: RoutePlan, table: dict[str, str], num_sms: int, M: int, K: int, exact_key: str
) -> Optional[RoutePlan]:
    """The registered persistent variant of block count ``K`` closest to the exact plan
    (never a cell-pinned write program)."""
    exact = (
        _PERSISTENT_KEY.match(plan.kernel_key) if plan.kind == "persistent" else None
    )
    want_nc, want_depth, want_bits = (
        (int(exact["nc"]), int(exact["depth"]), exact["bits"])
        if exact
        else (None, None, None)
    )
    want_write = bool(exact and exact["write"]) or plan.kernel_key.endswith("_write")
    want_fm = bool(exact and exact["fm"]) or "_fm" in plan.kernel_key
    want_authored = bool(exact and exact["crfnone"]) or "_crfnone" in plan.kernel_key
    want_held = bool(exact and exact["crheld"]) or "_crheld" in plan.kernel_key
    want_packed = bool(exact and exact["rcp"]) or "_rcp" in plan.kernel_key
    want_two_ctas = bool(exact and exact["cta2"]) or plan.kernel_key.endswith("_cta2")
    pinned = _cell_pinned_write_keys(plan.arch)
    candidates = []
    for key in table:
        match = _PERSISTENT_KEY.match(key)
        if match is None or int(match["K"]) != K:
            continue
        if key in pinned:
            # a write-cell program serves its table cells only
            continue
        if bool(match["write"]) != want_write:
            # the snapshot store exists only in the write variants: never cross that line
            continue
        if bool(match["fm"]) != want_fm:
            # fast-math programs round differently (one BF16 ulp class): never substitute across it
            continue
        if bool(match["crfnone"]) != want_authored:
            # the fence-authored twin states its own tcgen05 fences: never substitute across it
            continue
        if bool(match["crheld"]) != want_held:
            # a held-release program keeps the early-release levers: never substitute across it
            continue
        if bool(match["rcp"]) != want_packed:
            # a packed-carry program keeps its sources in registers: never substitute across it
            continue
        if bool(match["cta2"]) != want_two_ctas:
            # a two-CTA program launches on another grid: never substitute across it
            continue
        nc, depth, bits = int(match["nc"]), int(match["depth"]), match["bits"]
        score = (
            nc != want_nc,
            depth != want_depth,
            want_bits is None
            or bits[_PERSISTENT_FLAG_ECR] != want_bits[_PERSISTENT_FLAG_ECR],
            want_bits is None
            or bits[_PERSISTENT_FLAG_PWA] != want_bits[_PERSISTENT_FLAG_PWA],
            want_bits is None
            or sum(a != b for a, b in zip(bits, want_bits, strict=False)),
            key,
        )
        candidates.append((score, key, nc, depth, bits))
    if not candidates:
        return None
    _score, key, nc, depth, bits = min(candidates)
    if want_two_ctas:
        grid_x, grid_policy = min(M, 2 * num_sms), "two_ctas_per_sm"
    else:
        grid_x, grid_policy = _grid(M, num_sms, nc)
    schedule_id = _persistent_schedule_id(
        nc,
        depth,
        bits,
        want_authored,
        held_release=want_held,
        packed_register_carry=want_packed,
    )
    if want_fm:
        schedule_id += "_fastmath"
    if want_two_ctas:
        schedule_id += "_two_ctas_per_sm"
    if want_write:
        schedule_id += "_write"
    wait_policy = "deferred_wait_st" if _wait_policy(plan.arch) else "control_wait_st"
    route_id = (
        f"{schedule_id}.{plan.arch}.{wait_policy}.k{K}.delta1.write{int(want_write)}.norm1."
        f"pdl{int(plan.use_pdl)}.{grid_policy}.registered_fallback"
    )
    return RoutePlan(
        "persistent",
        key,
        grid_x,
        PERSISTENT_THREADS,
        schedule_id,
        route_id,
        plan.arch,
        plan.use_pdl,
        exact_key,
    )


def _resolve_registered(plan: RoutePlan, num_sms: int, M: int) -> RoutePlan:
    """``plan`` itself when its program is registered (or no registry exists for the
    architecture); otherwise the closest registered variant of the same family and block
    count.  Bootstrap ``direct`` variants encode semantics in their key and never substitute;
    neither do the native ports and the pair kernel (their cells are measured)."""
    if plan.arch is None or plan.kind == "direct":
        return plan
    table = KERNELS.get(plan.arch)
    if not table or plan.kernel_key in table:
        return plan
    exact_key = plan.kernel_key
    if plan.kind == "small_m":
        match = _SMALL_M_KEY.match(exact_key)
        if match is None:
            return plan
        K = int(match.group(3))
        same_cluster = int(match.group(2)) if match.group(2) else 1
        fm_key = match.group(5) or ""
        write_block = match.group(6) is not None
        write_suffix = "_write" if write_block else ""
        for cluster in (same_cluster, 2, 4, 1):
            family = "direct" if cluster == 1 else f"cluster{cluster}"
            if f"small_m_{family}:k{K}{fm_key}{write_suffix}" in table:
                return _small_m_plan(
                    plan.arch,
                    M,
                    K,
                    plan.use_pdl,
                    cluster,
                    exact_key,
                    write_block=write_block,
                    num_sms=num_sms,
                )
        if write_block:
            # the snapshot store exists only in the small-M write programs: never substitute
            # a program of other semantics
            return plan
        fallback = _persistent_fallback(plan, table, num_sms, M, K, exact_key)
        return fallback if fallback is not None else plan
    if plan.kind == "persistent":
        match = _PERSISTENT_KEY.match(exact_key)
        if match is None:
            return plan
        fallback = _persistent_fallback(
            plan, table, num_sms, M, int(match["K"]), exact_key
        )
        return fallback if fallback is not None else plan
    return plan


def plan_route(
    arch: Optional[str],
    num_sms: int,
    M: int,
    num_blocks: int,
    use_pdl: bool,
    *,
    has_delta: bool = True,
    block_write_idx: int = -1,
    apply_output_norm: bool = True,
    common_path: bool = True,
) -> RoutePlan:
    """Select the generated program for one call from host-known facts only.

    The route tables name the schedule variant measured for every cell of the evaluation
    grid.  Cells outside that grid (token counts that are not powers of two, block counts
    5-7 at mid token counts, ...) may name a variant this checkout does not register; such a
    plan is resolved to a registered variant of the same program family and block count
    (``fallback_from`` records the exact key), so every dense call has a program.  Measured
    cells always run their exact program.
    """
    plan = _plan_route_exact(
        arch,
        num_sms,
        M,
        num_blocks,
        use_pdl,
        has_delta=has_delta,
        block_write_idx=block_write_idx,
        apply_output_norm=apply_output_norm,
        common_path=common_path,
    )
    return _resolve_registered(plan, num_sms, M)


def _plan_route_exact(
    arch: Optional[str],
    num_sms: int,
    M: int,
    num_blocks: int,
    use_pdl: bool,
    *,
    has_delta: bool = True,
    block_write_idx: int = -1,
    apply_output_norm: bool = True,
    common_path: bool = True,
) -> RoutePlan:
    """The table-exact plan (the schedule variant named by the route tables)."""
    M = int(M)
    K = int(num_blocks)
    use_pdl = bool(use_pdl)
    if M <= 0:
        raise ValueError("AttnRes requires M > 0")
    if not 0 <= K <= MAX_BLOCKS:
        raise ValueError(f"num_blocks must be in [0, {MAX_BLOCKS}]")
    if not common_path:
        write_block = int(block_write_idx) >= 0
        schedule_id = "direct_cta256_bootstrap"
        route_id = (
            f"{schedule_id}.k{K}.d{int(bool(has_delta))}.w{int(write_block)}."
            f"n{int(bool(apply_output_norm))}.pdl{int(use_pdl)}.m_all"
        )
        key = f"direct:k{K}_d{int(bool(has_delta))}_w{int(write_block)}_n{int(bool(apply_output_norm))}"
        return RoutePlan(
            "direct", key, M, DIRECT_THREADS, schedule_id, route_id, arch, use_pdl
        )
    if arch not in SUPPORTED_COMPUTE_CAPABILITIES.values():
        raise ValueError(f"unsupported Kimi-K3 AttnRes architecture {arch!r}")
    num_sms = int(num_sms)
    if num_sms <= 0:
        raise ValueError(f"invalid persistent launch geometry M={M}, num_sms={num_sms}")
    write_idx = int(block_write_idx)
    if write_idx >= 0:
        # Round r4 (direction 4 / 4b): the block-boundary snapshot write (block K written)
        # runs the write variant of the dense cell's program: the small-M write programs
        # inside the dense family's M table, the persistent write variant above it (K >= 1,
        # also where the dense cell runs a native port), one small-M CTA per token for K = 0
        # only. The written index is the
        # program's K; any other index keeps the bootstrap (``common_path_eligible``).
        if write_idx != K or not 0 <= K < MAX_BLOCKS:
            raise ValueError(
                "dense snapshot-write route requires block_write_idx == num_blocks < "
                f"{MAX_BLOCKS}, observed {write_idx} at K={K}"
            )
        max_m = _SMALL_M_DIRECT_MAX_M[arch].get(K)
        if max_m is not None and max_m >= M:
            dense = None
        elif K == 0:
            dense = _plan_route_exact(arch, num_sms, M, K, use_pdl)
        else:
            # Above the table the write runs the persistent write variant even where the dense
            # cell runs a native port (Cake select_route persistent_dense_only).
            dense = _persistent_plan_exact(arch, num_sms, M, K, use_pdl, write=True)
        if dense is None or dense.kind != "persistent":
            return _small_m_plan(
                arch,
                M,
                K,
                use_pdl,
                _small_m_cluster(arch, M, K, num_sms=num_sms, write=True),
                None,
                sources_per_chunk=_small_m_sources_per_chunk(arch, M, K),
                write_block=True,
                num_sms=num_sms,
            )
        schedule_id = f"{dense.schedule_id}_write"
        return RoutePlan(
            "persistent",
            f"{dense.kernel_key}_write",
            dense.grid_x,
            dense.threads,
            schedule_id,
            dense.route_id.replace(dense.schedule_id, schedule_id, 1).replace(
                ".write0.", ".write1.", 1
            ),
            arch,
            use_pdl,
        )
    max_m = _SMALL_M_DIRECT_MAX_M[arch].get(K)
    if (M, K) in _R5_SMALL_M_INCLUDED_CELLS.get(arch, ()) or (
        max_m is not None
        and max_m >= M
        and (M, K) not in _R5_SMALL_M_EXCLUDED_CELLS[arch]
    ):
        return _small_m_plan(
            arch,
            M,
            K,
            use_pdl,
            _small_m_cluster(arch, M, K, num_sms=num_sms),
            None,
            sources_per_chunk=_small_m_sources_per_chunk(arch, M, K),
            num_sms=num_sms,
        )
    pair = _r2(_R2_PAIR_CELLS, arch, num_sms).get((M, K))
    if pair is not None:
        return _pair_plan(arch, num_sms, M, K, use_pdl, pair)
    native = _native_route(arch, num_sms, M, K)
    if native is not None:
        schedule_id = f"native_{native}_nc3_d2_ws288_grid{num_sms}"
        route_id = f"{schedule_id}.{arch}.native_no_hint.k{K}.delta1.write0.norm1.pdl{int(use_pdl)}.native_full_sm"
        return RoutePlan(
            "native",
            f"native_{native}",
            num_sms,
            PERSISTENT_THREADS,
            schedule_id,
            route_id,
            arch,
            use_pdl,
        )
    if K == 0:
        multiplier = _K0_TMA_GRID_MULTIPLIER_M[arch].get(M)
        if multiplier is None:
            grid_x, grid_policy = min(M, num_sms), "token_or_full_sm"
        else:
            grid_x, grid_policy = multiplier * num_sms, f"full_sm_x{multiplier}"
        schedule_id = "k0_tma_persistent_ws288_vec128_fp32x2"
        route_id = f"{schedule_id}.{arch}.none_k0.k0.delta1.write0.norm1.pdl{int(use_pdl)}.{grid_policy}"
        return RoutePlan(
            "k0_tma",
            "k0_tma",
            grid_x,
            PERSISTENT_THREADS,
            schedule_id,
            route_id,
            arch,
            use_pdl,
        )
    return _persistent_plan_exact(arch, num_sms, M, K, use_pdl)


def _pair_plan(
    arch: str, num_sms: int, M: int, K: int, use_pdl: bool, key: str
) -> RoutePlan:
    """The plan of a round-2 pair cell: one CTA per SM, 320 threads (384 for a ``_cr232`` or a
    ``_cr200_pk1`` key), the key as the table names it."""
    match = _PAIR_KEY.match(key)
    if match is None or int(match["K"]) != K or match["write"]:
        raise ValueError(
            f"round-2 cell ({M}, {K}) names {key!r}, not a dense pair key of K={K}"
        )
    nc = int(match["nc"])
    threads = PAIR_CONSUMER_REGS_THREADS if match["cr"] else PAIR_THREADS
    schedule_id = f"pair_ws{threads}_nc{nc}"
    if match["pbf16a"]:
        schedule_id += "_prefix_bf16_add"
    if match["fm"]:
        schedule_id += "_fastmath"
    if match["park"]:
        schedule_id += f"_consumer_regs{PAIR_PARK_ACC_CONSUMER_REGS}_park_acc1"
    elif match["cr"]:
        schedule_id += f"_consumer_regs{PAIR_CONSUMER_REGS}"
    schedule_id += f"_grid{num_sms}"
    route_id = f"{schedule_id}.{arch}.pair_no_hint.k{K}.delta1.write0.norm1.pdl{int(use_pdl)}.pair_full_sm"
    return RoutePlan(
        "pair", key, num_sms, threads, schedule_id, route_id, arch, use_pdl
    )


def _persistent_plan_for_key(
    arch: str, num_sms: int, M: int, K: int, use_pdl: bool, key: str
) -> RoutePlan:
    """The plan of a round-2 / round-3 persistent cell: the key as the table names it, the grid of
    its ``nc`` (``min(M, 2 x SMs)`` for a ``_cta2`` key) and the labels of its fields (what
    ``_persistent_plan_exact`` derives for a legacy cell; a ``_crfnone`` key labels its schedule
    ``_authored_fences``, a ``_crheld`` key ``_held_release``, a ``_rcp`` key
    ``_packed_register_carry``, a ``_cta2`` key ``_two_ctas_per_sm``)."""
    match = _PERSISTENT_KEY.match(key)
    if match is None or int(match["K"]) != K or match["write"]:
        raise ValueError(
            f"round-2 cell ({M}, {K}) names {key!r}, not a dense persistent key of K={K}"
        )
    nc, depth, bits = int(match["nc"]), int(match["depth"]), match["bits"]
    if match["cta2"]:
        grid_x, grid_policy = min(M, 2 * num_sms), "two_ctas_per_sm"
    else:
        grid_x, grid_policy = _grid(M, num_sms, nc)
    schedule_id = _persistent_schedule_id(
        nc,
        depth,
        bits,
        bool(match["crfnone"]),
        held_release=bool(match["crheld"]),
        packed_register_carry=bool(match["rcp"]),
    )
    if match["fm"]:
        schedule_id += "_fastmath"
    if match["cta2"]:
        schedule_id += "_two_ctas_per_sm"
    wait_policy = "deferred_wait_st" if _wait_policy(arch) else "control_wait_st"
    route_id = f"{schedule_id}.{arch}.{wait_policy}.k{K}.delta1.write0.norm1.pdl{int(use_pdl)}.{grid_policy}"
    return RoutePlan(
        "persistent",
        key,
        grid_x,
        PERSISTENT_THREADS,
        schedule_id,
        route_id,
        arch,
        use_pdl,
    )


def _persistent_plan_exact(
    arch: str, num_sms: int, M: int, K: int, use_pdl: bool, write: bool = False
) -> RoutePlan:
    """The persistent program of a cell (the dense tail of ``_plan_route_exact``), mirrored from the
    Cake dispatcher's persistent policies (``write``: the program the snapshot-write variant is
    derived from; it holds the release from ``_WRITE_HELD_CONSUMED_RELEASE_MIN_M``).  A dense
    round-2 cell or a round-5 write cell takes the key its table names (per PDL mode when the
    cell names two)."""
    if not write:
        cell = _r2(_R2_PERSISTENT_KEY_CELLS, arch, num_sms).get((M, K))
        if cell is not None:
            key = cell[int(use_pdl)] if isinstance(cell, tuple) else cell
            return _persistent_plan_for_key(arch, num_sms, M, K, use_pdl, key)
    else:
        # A round-5 write cell: the dense key its snapshot-write variant derives from.
        cell = _r2(_WRITE_PERSISTENT_KEY_CELLS, arch, num_sms).get((M, K))
        if cell is not None:
            key = cell[int(use_pdl)] if isinstance(cell, tuple) else cell
            return _persistent_plan_for_key(arch, num_sms, M, K, use_pdl, key)
    defer = _wait_policy(arch)
    nc, depth = _schedule(arch, M, K, write)
    grid_x, grid_policy = _grid(M, num_sms, nc)
    ecr = _early_consumed_release(arch, M, K, grid_x, write)
    pwa = _producer_wait_acquire(arch, M, K)
    prefix_bf16_add = (
        (not write and (_cell_m(arch, M, K), K) in _R5_PREFIX_BF16_ADD_CELLS[arch])
        or (
            arch == "sm_100a"
            and (
                K == 2
                or (K in {1, 3, 5, 6, 7} and nc == 4)
                or (K == 4 and nc == 5)
                or (K == 8 and nc == 3)
            )
            and not ecr
        )
        or (arch == "sm_103a" and K in {1, 2, 3} and nc == 4 and not ecr)
        or (K == 4 and nc == 3 and not ecr)
        or (arch == "sm_103a" and K == 8 and nc == 3 and not ecr)
    )
    prefix_round_once = K == 8 and nc == 3 and not ecr
    # Bit 4 is register_carry_only (D3): set by the round-2 cells above only.  Its legacy reading,
    # one token per CTA (K 1-3, nc 4, held release, grid_x == M), was never reached: every such
    # cell is small-M at 148 / 152 SMs.
    register_carry_only = False
    prefetch_fifth_stats = arch == "sm_103a" and K == 4 and nc == 5 and not ecr
    k4_early_nc5_input_first = (
        arch == "sm_100a" and M == 1024 and K == 4 and nc == 5 and ecr
    )
    k1_on_output_norm_cache = (
        arch == "sm_103a" and use_pdl and K == 1 and M in {1024, 2048, 4096, 8192}
    )
    k1_output_norm_packed_cache = (
        arch == "sm_103a"
        and not use_pdl
        and K == 1
        and nc == 4
        and not prefix_bf16_add
        and not prefix_round_once
        and ecr
        and pwa
        and not defer
    )
    k8_sm103_nonallocator_qk_prelude = (
        arch == "sm_103a"
        and not use_pdl
        and K == 8
        and nc == 3
        and not ecr
        and prefix_bf16_add
        and prefix_round_once
    )
    # Positions 7, 8, 10, 11 and 12 (the sm_103a K4 held input-first and source-0 carry
    # programs; the sm_103a M = 1 K1 / K5 and the sm_100a M = 64 K4 programs) are '0' in
    # every registered key and False for every input that reaches this function through
    # plan_route: their cells resolve inside the small-M table or, for (1024, 4) on sm_103a,
    # to a three-source schedule.  They stay in the key as constant 0 bits so its
    # 15-character flag field matches the registered keys.
    flags = (
        ecr,
        pwa,
        prefix_bf16_add,
        prefix_round_once,
        register_carry_only,
        prefetch_fifth_stats,
        k4_early_nc5_input_first,
        False,
        False,
        k1_on_output_norm_cache,
        False,
        False,
        False,
        k1_output_norm_packed_cache,
        k8_sm103_nonallocator_qk_prelude,
    )
    bits = "".join("1" if flag else "0" for flag in flags)
    fast_math = _fast_math(arch, M, K, "persistent_write" if write else "persistent")
    key = f"persistent:k{K}_nc{nc}_d{depth}_f{bits}" + ("_fm" if fast_math else "")
    schedule_id = _persistent_schedule_id(nc, depth, bits) + (
        "_fastmath" if fast_math else ""
    )
    wait_policy = "deferred_wait_st" if defer else "control_wait_st"
    route_id = f"{schedule_id}.{arch}.{wait_policy}.k{K}.delta1.write0.norm1.pdl{int(use_pdl)}.{grid_policy}"
    return RoutePlan(
        "persistent",
        key,
        grid_x,
        PERSISTENT_THREADS,
        schedule_id,
        route_id,
        arch,
        use_pdl,
    )


# ---------------------------------------------------------------------------
# Input validation and common-path eligibility (Cake ``_can_use_persistent_common_path``)
# ---------------------------------------------------------------------------


def _dense_vector(tensor: torch.Tensor, row_strides=()) -> bool:
    if tensor.stride(-1) != 1 or tensor.data_ptr() % 16 != 0:
        return False
    elem = tensor.element_size()
    return all(tensor.stride(dim) * elem % 16 == 0 for dim in row_strides)


def validate_kimi_k3_attn_res_inputs(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> None:
    tensors = {
        "prefix": prefix,
        "blocks": blocks,
        "norm_weight": norm_weight,
        "qk_weight": qk_weight,
        "out": out,
    }
    if delta is not None:
        tensors["delta"] = delta
    if output_norm_weight is not None:
        tensors["output_norm_weight"] = output_norm_weight
    device = prefix.device
    for name, tensor in tensors.items():
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be a CUDA tensor on {device}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have a dense last dimension")
    M = int(prefix.shape[0])
    if prefix.dim() != 2 or prefix.shape[1] != HIDDEN_SIZE or M <= 0:
        raise ValueError(f"prefix must be [M > 0, {HIDDEN_SIZE}]")
    if delta is not None and tuple(delta.shape) != (M, HIDDEN_SIZE):
        raise ValueError(f"delta must be [{M}, {HIDDEN_SIZE}]")
    if tuple(out.shape) != (M, HIDDEN_SIZE) or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous [{M}, {HIDDEN_SIZE}] tensor")
    if blocks.dim() != 3 or blocks.shape[0] != M or blocks.shape[2] != HIDDEN_SIZE:
        raise ValueError(f"blocks must be [{M}, <= {MAX_BLOCKS}, {HIDDEN_SIZE}]")
    if not 0 <= int(num_blocks) <= MAX_BLOCKS or int(num_blocks) > blocks.shape[1]:
        raise ValueError(
            f"num_blocks must be in [0, min({MAX_BLOCKS}, blocks.shape[1])]"
        )
    if int(block_write_idx) != -1 and not 0 <= int(block_write_idx) < blocks.shape[1]:
        raise ValueError("block_write_idx must be -1 or a valid snapshot index")
    if blocks.stride(1) < HIDDEN_SIZE or blocks.stride(0) < blocks.shape[
        1
    ] * blocks.stride(1):
        raise ValueError("blocks rows must not overlap")
    for name in ("norm_weight", "qk_weight", "output_norm_weight"):
        tensor = tensors.get(name)
        if tensor is not None and tuple(tensor.shape) != (HIDDEN_SIZE,):
            raise ValueError(f"{name} must be [{HIDDEN_SIZE}]")
    if float(eps) <= 0.0 or (
        output_norm_weight is not None and float(output_norm_eps) <= 0.0
    ):
        raise ValueError("eps values in use must be positive")


def common_path_eligible(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int,
) -> bool:
    """True when the call takes the dense path: residual add + fused output norm, no snapshot
    write or (round r4) the snapshot write of block ``num_blocks`` (the write variants of the
    dense programs; an index below K names a block the call also reads, any other index keeps
    the bootstrap), token-major dense layouts with 16-byte aligned rows and a full
    ``[M, 8, H]`` snapshot bank."""
    if delta is None or output_norm_weight is None:
        return False
    write_idx = int(block_write_idx)
    if (write_idx >= 0 and write_idx != int(num_blocks)) or write_idx >= int(
        blocks.shape[1]
    ):
        return False
    if (
        prefix.stride(0) != HIDDEN_SIZE
        or delta.stride(0) != HIDDEN_SIZE
        or out.stride(0) != HIDDEN_SIZE
    ):
        return False
    if blocks.shape[1] < MAX_BLOCKS:
        return False
    return all(
        (
            _dense_vector(prefix, (0,)),
            _dense_vector(delta, (0,)),
            _dense_vector(out, (0,)),
            _dense_vector(blocks, (0, 1)),
            _dense_vector(norm_weight),
            _dense_vector(qk_weight),
            _dense_vector(output_norm_weight),
        )
    )


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


def _record_name(arch: str, kernel_key: str) -> str:
    """The registry record of ``kernel_key`` on ``arch``; a key this checkout does not register
    (a bootstrap variant without a program) is ``NotImplementedError``."""
    try:
        return record_of(arch, kernel_key)
    except RuntimeError as exc:
        raise NotImplementedError(str(exc)) from None


def _bind(
    name: str, arch: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the argument plan of record ``name`` (its kernel parameters through their
    logical names, the ``launch`` switch ``enable_pdl``, then the grid) and load its entry."""
    record = MODULES[name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    logical = record["logical"]
    arguments = []
    for kind, parameter in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[parameter])
            continue
        if kind == "launch":
            arguments.append(int(kwargs["enable_pdl"]))
            continue
        key = logical.get(parameter)
        if key is None or key not in kwargs:
            raise KeyError(
                f"generated program {name!r} expects argument {parameter!r} ({kind}, "
                f"logical {key!r}); host binding provides {sorted(kwargs)}"
            )
        arguments.append(kwargs[key])
    module = load(name, arch)
    return module.run, tuple(arguments)


@dataclass(frozen=True)
class KimiK3AttnResRunner:
    """One prepared AttnRes call.  ``launch()`` runs the single generated program on the
    current torch stream, mutates ``prefix`` (and the written snapshot) in place, writes
    ``out`` and returns it; no CUDA allocation, no host synchronisation, CUDA-graph
    capturable (capture belongs to the caller).  The program reads its inputs on device at
    every launch, so the same runner (or a graph capturing it) stays valid when the caller
    writes new values into the bound tensors.  Prepare a new runner when a shape, stride,
    flag, tensor binding or the PDL policy changes."""

    plan: RoutePlan
    module_name: str
    #: SM count of the device the plan was resolved for (the grids of the persistent, K = 0, native
    #: and pair programs derive from it).
    sm_count: int
    prefix: torch.Tensor
    blocks: torch.Tensor
    out: torch.Tensor
    launches: tuple[tuple[Callable[..., Any], tuple], ...] = field(repr=False)

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            for entry, arguments in self.launches:
                entry(*arguments)
        return self.out

    __call__ = launch

    @property
    def launch_count(self) -> int:
        return len(self.launches)


def generated_program_available(
    device: torch.device,
    M: Optional[int] = None,
    num_blocks: Optional[int] = None,
    *,
    enable_pdl: bool = False,
    common_path: bool = True,
    has_delta: bool = True,
    block_write_idx: int = -1,
    apply_output_norm: bool = True,
) -> bool:
    """True when this checkout registers the program of that call on ``device`` (or any
    program for the device's architecture when ``M`` / ``num_blocks`` are omitted)."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(_device_capability(_device_index(device)))
    if arch is None:
        return False
    if M is None and num_blocks is None:
        return bool(MODULES) and bool(KERNELS.get(arch))
    if M is None or num_blocks is None:
        raise ValueError("pass both M and num_blocks or neither")
    plan = plan_route(
        arch,
        sm_count_for(device),
        M,
        num_blocks,
        enable_pdl,
        has_delta=has_delta,
        block_write_idx=block_write_idx,
        apply_output_norm=apply_output_norm,
        common_path=common_path,
    )
    return plan.kernel_key in KERNELS.get(arch, {})


def prepare_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
) -> KimiK3AttnResRunner:
    """Validate one call, select its generated program and bind the launch."""
    validate_kimi_k3_attn_res_inputs(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
    )
    device = prefix.device
    arch = arch_for(device)
    M = int(prefix.shape[0])
    common = common_path_eligible(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
    )
    sm_count = sm_count_for(device)
    plan = plan_route(
        arch,
        sm_count,
        M,
        num_blocks,
        enable_pdl,
        has_delta=delta is not None,
        block_write_idx=int(block_write_idx),
        apply_output_norm=output_norm_weight is not None,
        common_path=common,
    )
    module_name = _record_name(arch, plan.kernel_key)
    # The bootstrap program receives valid dummy pointers for absent optional operands
    # and never dereferences them (Cake production substitutes the same operands).
    bound_delta = delta if delta is not None else prefix
    bound_output_norm_weight = (
        output_norm_weight if output_norm_weight is not None else norm_weight
    )
    kwargs: dict[str, Any] = dict(
        grid=(int(plan.grid_x), 1, 1),
        out=out,
        prefix=prefix,
        delta=bound_delta,
        blocks=blocks,
        norm_weight=norm_weight,
        qk_weight=qk_weight,
        output_norm_weight=bound_output_norm_weight,
        out_stride=int(out.stride(0)),
        prefix_stride=int(prefix.stride(0)),
        delta_stride=int(bound_delta.stride(0)),
        blocks_m_stride=int(blocks.stride(0)),
        blocks_k_stride=int(blocks.stride(1)),
        eps=float(eps),
        output_norm_eps=float(output_norm_eps),
        block_write_idx=int(block_write_idx),
        enable_pdl=int(enable_pdl),
        M=M,
    )
    return KimiK3AttnResRunner(
        plan,
        module_name,
        sm_count,
        prefix,
        blocks,
        out,
        (_bind(module_name, arch, kwargs),),
    )


def kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Prepare and launch one call (see :func:`prepare_kimi_k3_attn_res`)."""
    return prepare_kimi_k3_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
        enable_pdl=enable_pdl,
    ).launch()


def reference_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Independent FP32 PyTorch reference with the exact BF16 state mutations (tests / benchmark)."""
    if delta is not None:
        prefix.add_(delta)
    if int(block_write_idx) >= 0:
        blocks[:, int(block_write_idx), :].copy_(prefix)
    K = int(num_blocks)
    if K:
        candidates = torch.cat(
            (blocks[:, :K, :].float(), prefix[:, None, :].float()), dim=1
        )
    else:
        candidates = prefix[:, None, :].float()
    inv_rms = torch.rsqrt(candidates.square().mean(dim=-1) + float(eps))
    scores = (
        candidates * inv_rms[..., None] * (norm_weight.float() * qk_weight.float())
    ).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=1)
    mixed = (probabilities[..., None] * candidates).sum(dim=1)
    if output_norm_weight is not None:
        mixed = (
            mixed
            * torch.rsqrt(
                mixed.square().mean(dim=-1, keepdim=True) + float(output_norm_eps)
            )
            * output_norm_weight.float()
        )
    out.copy_(mixed.to(torch.bfloat16))
    return out
