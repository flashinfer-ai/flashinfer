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

Cake backend: MiniMax-M3 ICP decode indexer chain (SM100 / SM103).

Two generated programs serve one decode step of the index-context-parallel
(ICP) indexer on every rank of a 2- or 4-rank group:

* the **fused scorer** -- one CTA per (request, split) streams this rank's
  FP8 index-key fragments of the request's pages with TMA, scores the
  request's query rows against them with tensor-core MMAs and keeps, in its
  epilogue, each (token, head) row's top-16 (canonical key: score descending,
  global block id ascending) as compact candidate lists in a per-row pool;
* the **exchange merge** -- one CTA per (token, head) row merges the row's
  lists into the rank-local top-16, pushes the 16 words into the owner rank's
  symmetric pool and, on the rows whose head this rank owns, polls the peers'
  words and writes the merged, forced-block-injected global ids ascending
  into ``out``.

Preparation validates the tensors on the host from their metadata only (no
device reads), decides the split count from the device's SM count and the
pool / merge geometry from the block table's column count, binds both
programs and returns a runner that launches with no allocation and no host
synchronization, so it can be captured into a CUDA Graph.  The exchange is a
collective: every rank of the group launches the same chain for the same
step, and the status word of each rank's workspace must be read (outside
capture) after a step; a non-zero value is fatal for the step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from ...utils import get_compute_capability, get_device_sm_count
from .cake_jit import (
    ARG_PLANS,
    POLICY,
    PROGRAMS,
    load_program,
    merge_programs,
    programs_for,
    scorer_programs,
    select_merge_program,
    select_scorer_program,
)
from .cake_workspace import NUM_HEADS, OUT_POISON, TOPK, MSAIcpIndexerWorkspace

HEAD_DIM = 128
WORLD_SIZES = (2, 4)
PAGE_SIZES = (128, 256)
MAX_QUERY_LEN = 8
TMA_ALIGN = 16
TRANSPORT_ID_LIMIT = (1 << 20) - 1  # 20-bit block ids in the transport word
SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
# Launch phases of the exchange merge: the real transport, or the serialised
# push / poll halves the single-process emulation of several ranks uses.
PHASE_PUSH_POLL = 0
PHASE_PUSH_ONLY = 1
PHASE_POLL_ONLY = 2
DEFAULT_TIMEOUT_US = 50_000.0


def _policy(key: str):
    if key not in POLICY:
        raise NotImplementedError(
            f"the generated ICP indexer registry of this checkout carries no planning constant {key!r}"
        )
    return POLICY[key]


# ---------------------------------------------------------------------------
# Device geometry
# ---------------------------------------------------------------------------


def arch_for(device: torch.device) -> Optional[str]:
    return SUPPORTED_COMPUTE_CAPABILITIES.get(get_compute_capability(device))


def generated_program_available(device: torch.device) -> bool:
    """True when this checkout registers both chain programs for ``device``."""
    arch = arch_for(device)
    return (
        arch is not None and bool(scorer_programs(arch)) and bool(merge_programs(arch))
    )


# ---------------------------------------------------------------------------
# Host planning (mirrors of the source's host policy; the registry POLICY
# literal carries the constants so both sides plan identically)
# ---------------------------------------------------------------------------


def pool_layout(nreq: int, split_count: int, bpt: int, slots: int) -> dict[str, int]:
    """Entry layout of one pool row for a request of ``nreq`` blocks on this rank.

    The scorer's (request, split) CTAs hold ``bpt``-block tiles in ``slots``
    epilogue warps; the layout gives the list count and the list lengths of
    the ``r`` splits with one extra tile (A) and the remaining splits (B) and
    the row extent ``E`` the merge reads.
    """
    nreq, split_count, bpt, slots = int(nreq), int(split_count), int(bpt), int(slots)
    ns = min(nreq, split_count)
    q, r = divmod(nreq, split_count)
    ta = (q + bpt) // bpt
    tb = (q + bpt - 1) // bpt
    va, vb = min(ta, slots), min(tb, slots)
    ca = min(TOPK, ((ta + slots - 1) // slots) * bpt)
    cb = min(TOPK, ((tb + slots - 1) // slots) * bpt)
    la, lb = r * va, (ns - r) * vb
    return dict(r=r, vA=va, vB=vb, cA=ca, cB=cb, LA=la, LB=lb, E=ca * la + cb * lb)


def merge_geometry(e_row_max: int, *, min_w: int = 1) -> dict[str, int]:
    """Merge CTA geometry for rows of up to ``e_row_max`` entries: warps per CTA
    (``POLICY['merge_w_policy']``, at least ``min_w``), even entries per thread
    and chunk (at most ``POLICY['merge_epl_max']``), the chunk size and the
    chunks a maximal row takes; ``pool_stride = chunk * chunks``."""
    e_row_max = max(1, int(e_row_max))
    w_policy = [(int(bound), int(wp)) for bound, wp in _policy("merge_w_policy")]
    w_default = int(_policy("merge_w_default"))
    w_max = int(_policy("merge_w_max"))
    epl_max = int(_policy("merge_epl_max"))
    w = next((wp for bound, wp in w_policy if e_row_max <= bound), w_default)
    w = max(int(w), int(min_w))
    if w not in (1, 2, 4, 8, w_max):
        raise ValueError(f"merge warps per CTA must be 1, 2, 4, 8 or {w_max}, got {w}")
    epl = min(epl_max, max(2, -(-e_row_max // (32 * w))))
    epl += epl & 1
    if epl < 2 or epl % 2 or epl > epl_max:
        raise ValueError(
            f"merge entries per thread must be even and in 2..{epl_max}, got {epl}"
        )
    chunk = 32 * w * epl
    chunks = -(-e_row_max // chunk)
    return dict(
        w=int(w), epl=int(epl), chunk=chunk, chunks=chunks, pool_stride=chunk * chunks
    )


def pool_geometry(
    nreq_bound: int, split_count: int, bpt: int, slots: int, *, world_size: int
) -> dict[str, int]:
    """Merge geometry and pool stride covering every request of up to ``nreq_bound``
    blocks on this rank (the block table's column count: the structural bound).
    The exchange tail keeps ``16 * world_size`` entries one per thread, so the
    CTA has at least ``16 * world_size / 32`` warps."""
    nreq_bound = int(nreq_bound)
    e_max = max(
        pool_layout(n, split_count, bpt, slots)["E"] for n in range(nreq_bound + 1)
    )
    min_w = max(1, (TOPK * int(world_size)) // 32)
    g = merge_geometry(e_max, min_w=min_w)
    return dict(
        E_row_max=e_max,
        w=g["w"],
        epl=g["epl"],
        chunk=g["chunk"],
        chunks=g["chunks"],
        pool_stride=g["pool_stride"],
        nreq_bound=nreq_bound,
    )


def default_split_count(batch: int, sm_count: int, ctas_per_sm: int) -> int:
    """Largest split that keeps every scorer CTA resident at once (one wave, no tail)."""
    return max(1, (int(sm_count) * int(ctas_per_sm)) // max(1, int(batch)))


@dataclass(frozen=True)
class ChainPlan:
    """The frozen launch plan of one chain geometry on one device."""

    arch: str
    world_size: int
    query_len: int
    rows_per_rank: int
    page_size: int
    batch: int
    block_table_columns: int
    sm_count: int
    scorer_program: str
    merge_program: str
    bpt: int
    slots: int
    stages: int
    ctas_per_sm: int
    split_count: int
    merge_warps: int
    merge_entries_per_thread: int
    pool_stride: int
    e_row_max: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def plan_chain(
    arch: str,
    *,
    world_size: int,
    query_len: int,
    rows_per_rank: int,
    page_size: int,
    batch: int,
    block_table_columns: int,
    sm_count: int,
) -> ChainPlan:
    """Programs, split count and pool / merge geometry for one chain geometry.

    Fails closed (``NotImplementedError``) when the checkout registers no
    scorer program for the index geometry or no exchange merge for the
    resulting merge geometry.
    """
    scorer = select_scorer_program(
        arch, query_len=query_len, rows_per_rank=rows_per_rank, page_size=page_size
    )
    record = PROGRAMS[scorer]
    split = default_split_count(batch, sm_count, int(record["ctas_per_sm"]))
    geometry = pool_geometry(
        block_table_columns,
        split,
        int(record["bpt"]),
        int(record["slots"]),
        world_size=world_size,
    )
    merge = select_merge_program(
        arch,
        world_size=world_size,
        warps=geometry["w"],
        entries_per_thread=geometry["epl"],
    )
    merge_record = PROGRAMS[merge]
    production = _policy("merge_production")
    if bool(merge_record["late_wait"]) != bool(production["late_wait"]) or bool(
        merge_record["small_fast"]
    ) != bool(production["small_fast"]):
        raise NotImplementedError(
            f"the registered exchange merge {merge!r} is not the production variant {production}"
        )
    if str(record["pdl_trigger"]) != str(production["scorer_pdl_trigger"]):
        raise NotImplementedError(
            f"the registered scorer {scorer!r} triggers PDL at {record['pdl_trigger']!r}; the production chain "
            f"expects {production['scorer_pdl_trigger']!r}"
        )
    return ChainPlan(
        arch=arch,
        world_size=int(world_size),
        query_len=int(query_len),
        rows_per_rank=int(rows_per_rank),
        page_size=int(page_size),
        batch=int(batch),
        block_table_columns=int(block_table_columns),
        sm_count=int(sm_count),
        scorer_program=scorer,
        merge_program=merge,
        bpt=int(record["bpt"]),
        slots=int(record["slots"]),
        stages=int(record["stages"]),
        ctas_per_sm=int(record["ctas_per_sm"]),
        split_count=int(split),
        merge_warps=int(geometry["w"]),
        merge_entries_per_thread=int(geometry["epl"]),
        pool_stride=int(geometry["pool_stride"]),
        e_row_max=int(geometry["E_row_max"]),
    )


# ---------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------


def allocate_pool(
    total_tokens: int, pool_stride: int, device: torch.device
) -> dict[str, Any]:
    """Candidate pool between the two programs: int32 ``[T * 4, pool_stride, 2]``
    (sortable key bits, global block id) pairs padded with the pad pair, plus
    ``row_meta`` int32 ``[T * 4]`` (each row's entry extent, written by the
    scorer).  Entries at or beyond a row's extent are never read, so the pool
    is never cleared between steps."""
    pool = torch.empty(
        (int(total_tokens) * NUM_HEADS, int(pool_stride), 2),
        dtype=torch.int32,
        device=device,
    )
    pool[..., 0] = int(_policy("pad_key"))
    pool[..., 1] = int(_policy("pad_id"))
    return dict(
        pool=pool,
        row_meta=torch.zeros(
            (int(total_tokens) * NUM_HEADS,), dtype=torch.int32, device=device
        ),
        pool_stride=int(pool_stride),
    )


def allocate_candidates(total_tokens: int, device: torch.device) -> torch.Tensor:
    """Rank-local merged lists: fp32 ``[T, 4, 16, 2]`` = (score, int32 id bits), best first, (-inf, -1) padded."""
    return torch.empty(
        (int(total_tokens), NUM_HEADS, TOPK, 2), dtype=torch.float32, device=device
    )


def allocate_out(
    total_tokens: int, world_size: int, device: torch.device
) -> torch.Tensor:
    """Owner output: int32 ``[T, 4 // world_size, 16]`` merged global block ids ascending, -1 tail."""
    return torch.full(
        (int(total_tokens), NUM_HEADS // int(world_size), TOPK),
        OUT_POISON,
        dtype=torch.int32,
        device=device,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _bind(
    program: str,
    arch: str,
    role: str,
    values: dict[str, Any],
    grid: tuple[int, int, int],
):
    """Order ``values`` by the role's generated argument plan and load the program."""
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    first_tensor = next(
        (v for v in values.values() if isinstance(v, torch.Tensor)), None
    )
    for kind, name in ARG_PLANS[role]:
        if kind == "grid":
            arguments.append(int(grid_values[name]))
        elif kind == "device":
            if first_tensor is None:
                raise ValueError(
                    f"the {role} program needs a device index but binds no tensor"
                )
            arguments.append(int(first_tensor.device.index))
        elif kind == "launch":
            if name != "enable_pdl":
                raise ValueError(
                    f"the {role} program's argument plan names an unknown launch flag {name!r}"
                )
            arguments.append(1)
        elif name in values:
            arguments.append(values[name])
        else:
            raise ValueError(
                f"the {role} program's argument plan names {name!r}, which this backend does not bind"
            )
    module = load_program(program, arch)
    return module.run, tuple(arguments)


@dataclass
class MSAIcpIndexerRunner:
    """Launch one prepared decode step of the ICP indexer chain on this rank.

    ``launch()`` submits the fused scorer and the exchange merge on the
    current stream with no allocation and no host synchronization and returns
    ``out``.  Every tensor is read on device at each launch, so the same
    runner (or a CUDA Graph capturing it) stays valid when the caller writes
    new queries, keys, page ids or lengths into the bound buffers; the
    workspace generation words advance per launch and keep the peers in step.
    Prepare a new runner when shapes, dtypes, bindings or the workspace change.

    After a step completes, read ``workspace.status_value()`` (outside any
    capture); a non-zero status is fatal for the step: 1 = NaN among the
    ordinary candidate scores, 2 = a row's list layout exceeded the pool
    stride, 4 = the merge's refine loop made no progress (a pool this scorer
    did not write), 8 = a poll passed ``timeout_us`` (the missing peer words
    merged as absent), 16 = ``n_ordinary == 16`` together with a forced block.
    Reset the workspace before reusing it after a non-zero status.
    """

    plan: ChainPlan
    workspace: MSAIcpIndexerWorkspace
    out: torch.Tensor
    candidates: torch.Tensor
    pool: dict[str, Any]
    status: torch.Tensor
    scorer_entry: Callable[..., Any]
    scorer_arguments: tuple
    merge_entry: Callable[..., Any]
    merge_arguments: dict[int, tuple]
    retained: tuple

    def launch_scorer(self) -> None:
        with tvm_ffi.use_torch_stream():
            self.scorer_entry(*self.scorer_arguments)

    def launch_merge(self, phase: int = PHASE_PUSH_POLL) -> None:
        """The exchange merge; ``phase`` 1 / 2 are the serialised push / poll halves of the emulation."""
        with tvm_ffi.use_torch_stream():
            self.merge_entry(*self.merge_arguments[int(phase)])

    def launch(self) -> torch.Tensor:
        self.launch_scorer()
        self.launch_merge(PHASE_PUSH_POLL)
        return self.out

    __call__ = launch

    @property
    def scorer_grid(self) -> tuple[int, int, int]:
        return (self.plan.batch, self.plan.split_count, 1)

    @property
    def merge_grid(self) -> tuple[int, int, int]:
        return (int(self.out.shape[0]) * NUM_HEADS, 1, 1)


# ---------------------------------------------------------------------------
# Validation and preparation
# ---------------------------------------------------------------------------


def _check(
    tensor: torch.Tensor,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    *,
    contiguous: bool = True,
):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def validate_msa_icp_indexer_inputs(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    active_rows: torch.Tensor,
    forced_column: torch.Tensor,
    forced_block: torch.Tensor,
    n_ordinary: torch.Tensor,
    *,
    workspace: MSAIcpIndexerWorkspace,
    query_len: int,
    out: Optional[torch.Tensor],
    candidates: Optional[torch.Tensor],
) -> dict[str, int]:
    """Check dtypes, shapes and strides against the kernels' contract from metadata alone.

    Returns ``total_tokens``, ``request_count``, ``rows_per_rank``,
    ``page_size``, ``num_pages`` and ``block_table_columns``.  Device
    placement is checked by :func:`prepare_msa_icp_indexer`.  The device
    contents the kernels rely on -- ``query_start_loc[0] == 0``,
    ``query_start_loc[-1] == total_tokens``, every request of at most
    ``query_len`` tokens, page ids inside ``index_k``, ``n_ordinary <= 15``
    whenever ``forced_block >= 0`` -- are the caller's contract and are not
    read here.

    One launch covers the whole batch: tokens ``0 .. total_tokens - 1`` of
    ``index_q`` are the launch window, every per-token tensor is indexed by
    that token, and the window occupies rows ``0 .. total_tokens - 1`` of the
    workspace (``total_tokens <= tokens_cap``).  The kernels' ``token_begin``
    argument is bound as 0.
    """
    query_len = int(query_len)
    if not 1 <= query_len <= MAX_QUERY_LEN:
        raise ValueError(f"query_len must be in [1, {MAX_QUERY_LEN}], got {query_len}")
    if not isinstance(workspace, MSAIcpIndexerWorkspace):
        raise ValueError("workspace must be an MSAIcpIndexerWorkspace")
    if getattr(workspace, "destroyed", False):
        raise ValueError("the workspace has been destroyed")
    world_size = int(workspace.world_size)
    if world_size not in WORLD_SIZES:
        raise ValueError(
            f"the ICP indexer chain runs on a world of {WORLD_SIZES} ranks, got {world_size}"
        )
    if (
        not isinstance(index_q, torch.Tensor)
        or index_q.dtype != torch.float8_e4m3fn
        or index_q.ndim != 3
    ):
        raise ValueError(
            f"index_q must be float8_e4m3fn [total_tokens, {NUM_HEADS}, {HEAD_DIM}]"
        )
    total_tokens = int(index_q.shape[0])
    if tuple(index_q.shape[1:]) != (NUM_HEADS, HEAD_DIM) or not index_q.is_contiguous():
        raise ValueError(
            f"index_q must be contiguous float8_e4m3fn [total_tokens, {NUM_HEADS}, {HEAD_DIM}]"
        )
    if total_tokens < 1 or total_tokens % query_len:
        raise ValueError(
            "index_q rows must be request_count * query_len with a positive request_count"
        )
    request_count = total_tokens // query_len
    if (
        not isinstance(index_k, torch.Tensor)
        or index_k.dtype != torch.float8_e4m3fn
        or index_k.ndim != 3
    ):
        raise ValueError(
            f"index_k must be a float8_e4m3fn [num_pages, rows_per_rank, {HEAD_DIM}] view"
        )
    num_pages, rows_per_rank = int(index_k.shape[0]), int(index_k.shape[1])
    if int(index_k.shape[2]) != HEAD_DIM:
        raise ValueError(f"index_k rows must hold {HEAD_DIM} FP8 values")
    strides = tuple(int(s) for s in index_k.stride())
    if strides[2] != 1 or strides[1] != HEAD_DIM:
        raise ValueError(
            f"index_k must hold each page's {rows_per_rank} x {HEAD_DIM} bytes contiguously (strides {strides})"
        )
    if strides[0] < rows_per_rank * HEAD_DIM or strides[0] % TMA_ALIGN:
        raise ValueError(
            f"index_k page pitch must be a multiple of {TMA_ALIGN} bytes covering the fragment, got {strides[0]}"
        )
    if index_k.data_ptr() % TMA_ALIGN:
        raise ValueError(f"index_k must start at a {TMA_ALIGN}-byte aligned address")
    page_size = rows_per_rank * world_size
    if page_size not in PAGE_SIZES:
        raise ValueError(
            f"rows_per_rank {rows_per_rank} x world_size {world_size} is not an admitted page size {PAGE_SIZES}"
        )
    if page_size == 256 and world_size != 2:
        raise ValueError("256-token pages are admitted for a world of 2 ranks only")
    if (
        not isinstance(block_table, torch.Tensor)
        or block_table.dtype != torch.int32
        or block_table.ndim != 2
    ):
        raise ValueError("block_table must be int32 [request_count, columns]")
    if int(block_table.shape[0]) != request_count or not block_table.is_contiguous():
        raise ValueError(
            f"block_table must be contiguous int32 [{request_count}, columns]"
        )
    columns = int(block_table.shape[1])
    if columns < 1:
        raise ValueError("block_table must hold at least one column")
    if columns * 2 >= TRANSPORT_ID_LIMIT:
        raise ValueError(
            "the transport word holds 20-bit block ids: the block table admits ids beyond that range"
        )
    _check(query_start_loc, "query_start_loc", torch.int32, (request_count + 1,))
    _check(seq_lens, "seq_lens", torch.int32, (request_count,))
    _check(positions, "positions", torch.int64, (total_tokens,))
    _check(active_rows, "active_rows", torch.uint8, (total_tokens,))
    _check(forced_column, "forced_column", torch.int32, (total_tokens,))
    _check(forced_block, "forced_block", torch.int32, (total_tokens,))
    _check(n_ordinary, "n_ordinary", torch.int32, (total_tokens,))
    if total_tokens > int(workspace.tokens_cap):
        raise ValueError(
            f"{total_tokens} tokens exceed the workspace capacity of {workspace.tokens_cap} tokens"
        )
    if int(workspace.stamp_rows) < total_tokens * NUM_HEADS:
        raise ValueError("the workspace stamps fewer rows than the merge launch writes")
    heads_local = NUM_HEADS // world_size
    if out is not None:
        _check(out, "out", torch.int32, (total_tokens, heads_local, TOPK))
    if candidates is not None:
        _check(
            candidates, "candidates", torch.float32, (total_tokens, NUM_HEADS, TOPK, 2)
        )
    return dict(
        total_tokens=total_tokens,
        request_count=request_count,
        rows_per_rank=rows_per_rank,
        page_size=page_size,
        num_pages=num_pages,
        block_table_columns=columns,
        world_size=world_size,
    )


def prepare_msa_icp_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    active_rows: torch.Tensor,
    forced_column: torch.Tensor,
    forced_block: torch.Tensor,
    n_ordinary: torch.Tensor,
    *,
    workspace: MSAIcpIndexerWorkspace,
    query_len: int,
    out: Optional[torch.Tensor] = None,
    candidates: Optional[torch.Tensor] = None,
    timeout_us: float = DEFAULT_TIMEOUT_US,
    backend: str = "cake",
) -> MSAIcpIndexerRunner:
    """Validate and bind one decode step of the ICP indexer chain on this rank.

    The only allocations are the candidate pool between the two programs,
    the two small scratch words and the optional ``out`` / ``candidates``
    outputs when the caller does not pass them; the returned runner launches
    with none.  The split count follows the device's SM count and the
    scorer's resident CTAs per SM; the merge geometry follows the block
    table's column count (every request the buffers admit).  The generated
    programs for the index geometry and the resulting merge geometry must be
    registered in this checkout (``NotImplementedError`` otherwise).  One
    launch covers all ``total_tokens`` tokens of ``index_q`` (the kernels'
    launch window starts at token 0).
    """
    if backend != "cake":
        raise ValueError("the ICP indexer chain supports backend='cake'")
    meta = validate_msa_icp_indexer_inputs(
        index_q,
        index_k,
        block_table,
        query_start_loc,
        seq_lens,
        positions,
        active_rows,
        forced_column,
        forced_block,
        n_ordinary,
        workspace=workspace,
        query_len=query_len,
        out=out,
        candidates=candidates,
    )
    device = index_q.device
    tensors = [
        index_q,
        index_k,
        block_table,
        query_start_loc,
        seq_lens,
        positions,
        active_rows,
        forced_column,
        forced_block,
        n_ordinary,
    ]
    tensors += [t for t in (out, candidates) if t is not None]
    if not all(t.is_cuda and t.device == device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    if workspace.device != device:
        raise ValueError(
            f"the workspace lives on {workspace.device}, the tensors on {device}"
        )
    arch = arch_for(device)
    if arch is None:
        capability = get_compute_capability(device)
        raise ValueError(
            "the ICP indexer chain requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    if float(timeout_us) <= 0.0:
        raise ValueError("timeout_us must be positive")
    total_tokens, request_count = meta["total_tokens"], meta["request_count"]
    world_size = meta["world_size"]
    plan = plan_chain(
        arch,
        world_size=world_size,
        query_len=int(query_len),
        rows_per_rank=meta["rows_per_rank"],
        page_size=meta["page_size"],
        batch=request_count,
        block_table_columns=meta["block_table_columns"],
        sm_count=get_device_sm_count(device),
    )
    if out is None:
        out = allocate_out(total_tokens, world_size, device)
    if candidates is None:
        candidates = allocate_candidates(total_tokens, device)
    pool = allocate_pool(total_tokens, plan.pool_stride, device)
    status = workspace.status
    # The fused scorer writes lists only; its score / validity plane pointers are
    # never dereferenced, so one-element carriers with a zero stride are bound.
    plane_score = torch.zeros((1,), dtype=torch.float32, device=device)
    plane_valid = torch.zeros((1,), dtype=torch.uint8, device=device)
    scorer_gen = torch.zeros((1,), dtype=torch.int32, device=device)
    q2d = index_q.view(torch.uint8).reshape(total_tokens * NUM_HEADS, HEAD_DIM)
    k_u8 = index_k.view(torch.uint8)
    scorer_values = dict(
        Q=q2d,
        K=k_u8,
        block_table=block_table,
        score=plane_score,
        valid=plane_valid,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        positions=positions,
        active_rows=active_rows,
        total_tokens=total_tokens,
        token_begin=0,
        token_count=total_tokens,
        request_begin=0,
        score_stride=0,
        block_table_stride=meta["block_table_columns"],
        split_count=plan.split_count,
        rank=int(workspace.rank),
        forced_column=forced_column,
        pool=pool["pool"],
        pool_stride=pool["pool_stride"],
        row_meta=pool["row_meta"],
        status=status,
        gen_ctr=scorer_gen,
        x_region_stride=0,
        x_par_stride=0,
    )
    scorer_grid = (request_count, plan.split_count, 1)
    scorer_entry, scorer_arguments = _bind(
        plan.scorer_program, arch, "scorer", scorer_values, scorer_grid
    )
    merge_common = dict(
        pool=pool["pool"],
        row_meta=pool["row_meta"],
        query_start_loc=query_start_loc,
        candidates=candidates.view(torch.int32),
        status=status,
        pool_stride=pool["pool_stride"],
        request_count=request_count,
        forced_block=forced_block,
        n_ordinary=n_ordinary,
        out=out,
        rank=int(workspace.rank),
        token_begin=0,
        timeout_ns=int(float(timeout_us) * 1000.0),
        **workspace.merge_bindings(),
    )
    merge_grid = (total_tokens * NUM_HEADS, 1, 1)
    merge_arguments: dict[int, tuple] = {}
    merge_entry = None
    for phase in (PHASE_PUSH_POLL, PHASE_PUSH_ONLY, PHASE_POLL_ONLY):
        merge_entry, merge_arguments[phase] = _bind(
            plan.merge_program,
            arch,
            "merge",
            dict(merge_common, x_phase=phase),
            merge_grid,
        )
    assert merge_entry is not None
    return MSAIcpIndexerRunner(
        plan=plan,
        workspace=workspace,
        out=out,
        candidates=candidates,
        pool=pool,
        status=status,
        scorer_entry=scorer_entry,
        scorer_arguments=scorer_arguments,
        merge_entry=merge_entry,
        merge_arguments=merge_arguments,
        retained=(q2d, k_u8, plane_score, plane_valid, scorer_gen),
    )


__all__ = [
    "ChainPlan",
    "DEFAULT_TIMEOUT_US",
    "HEAD_DIM",
    "MSAIcpIndexerRunner",
    "NUM_HEADS",
    "PHASE_POLL_ONLY",
    "PHASE_PUSH_ONLY",
    "PHASE_PUSH_POLL",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "TOPK",
    "allocate_candidates",
    "allocate_out",
    "allocate_pool",
    "arch_for",
    "default_split_count",
    "generated_program_available",
    "merge_geometry",
    "merge_programs",
    "plan_chain",
    "pool_geometry",
    "pool_layout",
    "prepare_msa_icp_indexer",
    "programs_for",
    "scorer_programs",
    "validate_msa_icp_indexer_inputs",
]
