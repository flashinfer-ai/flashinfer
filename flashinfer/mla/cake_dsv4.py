"""Source-level CAKE backend for DeepSeek V4 sparse MLA on SM100 and SM103.

Host contract (flashinfer#4671 hardening)
-----------------------------------------

* **Metadata ABI.** Every generated variant consumes its per-token metadata
  through the same eight kernel parameters (:data:`KERNEL_METADATA_PARAMS`):
  ``swa_indices``, ``compressed_indices``, ``sparse_topk_lens``,
  ``swa_index_stride``, ``compressed_index_stride``, ``sparse_topk_lens_offset``,
  ``sparse_topk`` and ``num_query_tokens``. Combined-space column ``c`` of row
  ``t`` lives at ``swa_indices[t * swa_index_stride + c]`` when ``c < 128`` and
  at ``compressed_indices[t * compressed_index_stride + (c - 128)]`` otherwise.
  The active length of a row is
  ``clamp(sparse_topk_lens[t] + sparse_topk_lens_offset, 0, sparse_topk)``.
  The host resolves the parameters from either the combined FlashInfer table
  ``sparse_indices [T, sparse_topk]`` (the compressed view is a column offset of
  the same storage) or from separate tables (``sparse_indices [T, 128]`` plus
  ``extra_sparse_indices [T, topk_c]``; ``extra_sparse_topk_lens`` counts
  compressed slots only and implies ``sparse_topk_lens_offset += 128``).
  Resolution never copies: a non-unit column stride or a non-int32 table is an
  error.
* **Padded rows.** ``num_query_tokens`` is the metadata row count ``T``.
  ``query`` and ``out`` may carry more rows; rows ``>= T`` are neither read nor
  written. Grids and workspace views derive from ``T``, never from query rows.
* **Argument binding.** Kernel arguments are bound *by name* through the
  registration ``arg_plan`` (:func:`_prepare_variant`), so regenerated bindings
  only need names from the host vocabulary (:func:`is_bindable_arg`). Every
  route launches its variant kernels directly; two-stage routes bind the
  producer and the reducer first and issue both through one FFI call
  (``run_sequence`` in ``cake_dsv4_launch_sequence.cc``), so no host work
  sits between the two kernels.
* **Query layout.** Every variant receives the five query-layout parameters
  (:data:`QUERY_LAYOUT_PARAMS`): ``seq_lens`` (int32, the cache length of
  every request), ``cum_seq_lens_q``, ``ragged_query``, ``max_q_len`` and
  ``batch_size``. The kernels derive the trtllm-gen SWA validity window from
  them: with ``b`` the request owning metadata row ``t``, ``q_len_b`` its
  query length and ``q_off`` the row's position in the request, combined
  column ``c < 128`` is attended only when
  ``c < clamp(seq_lens[b] - (q_len_b - 1 - q_off), 0, 128)`` (CAKE-957). A
  dense call (``cum_seq_lens_q=None``) binds ``ragged_query = 0``,
  ``max_q_len`` = the caller's per-request query length, ``batch_size =
  seq_lens.numel()`` and ``seq_lens`` in place of the never-read
  ``cum_seq_lens_q`` pointer; a ragged call binds the caller's offsets with
  ``ragged_query = 1``. The producers in :data:`_RAGGED_ONLY_ROUTES` read the
  request boundaries from ``cum_seq_lens_q`` unconditionally and receive the
  cached dense offsets instead. No host value is ever ``None``.
* **Preconditions on the metadata.** Every metadata row must keep at least
  one column inside the three validity predicates' domain: the active length
  ``sparse_topk_lens[t] + sparse_topk_lens_offset`` must be ``>= 1`` and the
  owning request must satisfy ``seq_lens[b] >= q_len_b`` (so the SWA window
  holds at least the row's own token). sglang satisfies both by construction
  (constant 128 on SWA-only layers, the window always contains the current
  token). A row whose columns are all ``-1`` or all beyond the window is a
  defined input and returns the all-invalid zero result. An active length of
  0 is outside the trtllm-gen contract (the stock FP8 kernels read past the
  active length or return NaN there); the Cake programs regenerated for
  CAKE-957 all return the zero result for it (tile-count guards in the FP8
  persistent, BF16 H64 guard/prefill and FP8 H64 source-exact bodies). The
  host does not synchronise to check these values.
* **Workspace.** One caller-owned ``workspace_buffer`` is carved
  deterministically (:func:`cake_dsv4_workspace_layout`)::

      [0,      1024)          reserved (formerly the TMA descriptor slab; see below)
      [1024,   1024 + 256 KiB) split-merge counters, uint32[65536]
      [P,      P + O_bytes)    partial_O  bf16 [T * H * S * 512]  (P = 1024 + 256 KiB;
                               O_bytes = 0 when the route runs one split)
      [P + O_bytes, + L_bytes) partial_lse f32 [T * H * S]
      [..., + Q_bytes)         shifted request offsets int32 [B + 1]
                               (row-tiled launches after the first chunk only)

  with every region 128-byte aligned. The requirement is exact per route
  (:func:`cake_dsv4_workspace_requirement`, from the same
  :func:`_route_plan` table the dispatcher launches with): routes without
  partial buffers need no workspace bytes, one-split routes need the LSE
  region only, split routes the partial O and LSE regions. When
  ``workspace_buffer`` cannot hold one launch over all ``T`` metadata rows,
  the host tiles the rows into consecutive launches that fit
  (:func:`_dispatch_row_tiles`): ``Q``, ``O`` and the metadata tables are
  sliced by row (views, no copies) and every chunk after the first receives
  its request boundaries as ``cum_seq_lens_q - first_row`` (``B + 1`` int32
  written into the workspace's offsets region by one ``torch.sub`` in stream
  order, capturable) with ``ragged_query = 1``, so the window resolves for
  any chunk start. The smallest accepted workspace is the one-row
  requirement; :func:`get_cake_dsv4_workspace_bytes` is the single-launch
  upper bound. sglang's fixed 128 MiB buffer therefore admits every row
  count on every route; the only route whose single launch can outgrow it is
  ``bf16_h32_topk128x_early_v47`` (``ceil(sparse_topk / 128)`` splits of
  ``32 * 1028`` bytes per row, no token bound), which tiles.
* **Split-merge counters.** ``bf16_h32_topk128x_early_v47`` (the only route
  with an in-kernel last-arriver merge) needs its ``T * ceil(H / 8)``
  counters zero at launch and leaves them zero afterwards. Zeroing is part
  of the launch contract and never raises: the first eager launch through a
  workspace zeroes the whole counter region once and registers the
  workspace in :data:`_primed_workspaces` (keyed by device and base address,
  guarded by a weak reference to the tensor owning the storage, so views of
  one buffer share the state and a freed and reused address is re-primed);
  a launch under CUDA Graph capture through an unregistered workspace
  records a zero fill of exactly the counters it uses into the graph
  (replays stay self-contained, nothing is registered). The kernels do not
  depend on any other host state. :func:`cake_dsv4_workspace_reset` zeroes
  and registers explicitly.
* **Descriptor storage.** The SM103 bindings that read their TMA descriptors
  from device memory (``tma_workspace_bytes`` in their registration; the SM100
  twins pass descriptors by value and use none of this) take a private,
  host-retained 1 KiB tensor per launch and write the descriptors of the call
  into it when they differ from what it holds, in stream order and never
  inside CUDA Graph capture. The host keeps a pool of such tensors per
  (variant, device) (:func:`_descriptor_storage`) keyed by the *descriptor
  set* = the TMA source geometry (pointer, shape, strides, dtype of ``Q`` and
  the KV caches). Rules:

  1. A set launched eagerly becomes a *live* entry; at most
     :data:`_DESCRIPTOR_POOL_CAPACITY` live entries exist per pool, and a new
     set beyond that takes the least recently used live entry's storage (the
     binding rewrites it before the launch). A live hit does no device write.
  2. A set launched under graph capture must be live at that moment (prepare
     it with one eager call on the same tensors); it then becomes a *captured*
     entry, which is never evicted or reassigned for the process lifetime
     because its graph may replay at any time. Eviction applies to live
     entries only.
  3. A set that is not live when a capture reaches it raises before any
     allocation or binding call.
  4. One lock covers the pool bookkeeping, the bindings' descriptor checks
     and the launch enqueues of a call; a launch on another stream than the
     storage's last reader waits for that stream first.

  Successive calls through one workspace (the layers of a model) therefore do
  not collide, and a stream of fresh query tensors does not grow memory
  without bound. Nothing else allocates device memory.
"""

from __future__ import annotations

import collections
import functools
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import torch

from ..utils import get_compute_capability


_HEAD_DIM = 512
_TILE_KV = 128
_SWA_WIDTH = 128
_ALIGN = 128

# Deterministic workspace layout (byte offsets inside workspace_buffer).
_DESCRIPTOR_SLAB_OFFSET = 0
_DESCRIPTOR_SLAB_BYTES = 1024
_COUNTER_OFFSET = _DESCRIPTOR_SLAB_OFFSET + _DESCRIPTOR_SLAB_BYTES
_COUNTER_REGION_BYTES = 256 * 1024
_MAX_MERGE_GROUPS = _COUNTER_REGION_BYTES // 4
_PARTIAL_OFFSET = _COUNTER_OFFSET + _COUNTER_REGION_BYTES
# Largest fixed split count used by any route (bf16_h128_topk4x_v52 / fp8_h128).
_MAX_FIXED_SPLITS = 5
# BF16/H128 SWA-only and topk4x rows with this many metadata tokens or more use
# the persistent KV-reuse prefill body (mirrors the Cake dispatcher's
# BF16_H128_PREFILL_MIN_TOKENS).
_BF16_H128_PREFILL_MIN_TOKENS = 64
# The two-stage split4 program (4 owners x 2 CTAs per token) runs one wave
# only up to this many query tokens; wider grids lose to trtllm-gen.
# Mirrors the Cake seed's BF16_TOPK128X_SPLIT_MAX_TOKENS.
_BF16_TOPK128X_SPLIT_MAX_TOKENS = 16
# Mirrors the Cake seed's BF16_ROW_FIRST_V_HALF_SPLIT_MAX_TOKENS: row-first rows
# with at most this many tokens run the V-half split program (two 2-CTA
# clusters per token = 4 CTAs per token, one wave on 148+ SMs).
_BF16_ROW_FIRST_V_HALF_SPLIT_MAX_TOKENS = 37
# Widths the BF16/H128 four-owner split and row-first producers cover (two or
# three live KV tiles); other widths below the prefill token bound have no
# exported kernel.
_BF16_TOPK128X_MIN_WIDTH = 256
_BF16_TOPK128X_MAX_WIDTH = 388
# One FP8 low-head producer partition owns up to three sparse tiles; mirrors
# the producer's FP8_ONE_PARTITION_MAX_TILES = 3.
_FP8_ONE_PARTITION_MAX_WIDTH = 384
_BF16_H64_COMPRESSED_PREFILL_TOKENS = 24
_BF16_H64_PREFILL_MAX_SPARSE_WIDTH = 640
# Workspaces whose split-merge counter region is known to be zero:
# (device index, address of the workspace view) -> weak reference to the
# tensor owning the storage (the view's ``_base`` or the tensor itself). A dead
# reference or another owner at the same address means the memory was freed
# and possibly reused: the workspace is primed again.
_primed_workspaces: dict[tuple[Optional[int], int], "weakref.ref[torch.Tensor]"] = {}
_primed_lock = threading.Lock()


# Work feed of the BF16/H128 persistent prefill body (mirrors the Cake seed's
# bf16_h128_prefill_uses_snake_feed).  With C = min(T, SMs // 2)
# clusters the striped feed gives base = T // C strided PREFIX items to the
# C - T % C regular clusters and base + 1 contiguous SUFFIX items to the T % C
# tail clusters; when the tail clusters are the majority the few regular
# clusters own a short heavy prefix (hardening-000037: 23-30 -> 17 critical
# tiles), so the boustrophedon program is launched instead.  One-tile items
# (SWA-only rows) deal identically under both feeds and keep the striped
# program.
def _bf16_h128_prefill_uses_snake_feed(
    num_query_tokens: int, sparse_topk: int, num_clusters: int
) -> bool:
    tokens = int(num_query_tokens)
    clusters = min(tokens, int(num_clusters))
    if clusters <= 0 or tokens <= clusters:
        return False
    if (int(sparse_topk) + _TILE_KV - 1) // _TILE_KV < 2:
        return False
    return 2 * (tokens % clusters) > clusters


@functools.lru_cache(maxsize=None)
def _bf16_h128_prefill_num_clusters(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    return max(int(props.multi_processor_count) // 2, 1)


KERNEL_METADATA_PARAMS = (
    "swa_indices",
    "compressed_indices",
    "sparse_topk_lens",
    "swa_index_stride",
    "compressed_index_stride",
    "sparse_topk_lens_offset",
    "sparse_topk",
    "num_query_tokens",
)

# Query-layout parameters of every generated variant (the SWA validity window;
# see the module docstring). Mirrors the Cake seeds' SWA_WINDOW_PARAMS.
QUERY_LAYOUT_PARAMS = (
    "seq_lens",
    "cum_seq_lens_q",
    "ragged_query",
    "max_q_len",
    "batch_size",
)

_scale_cache: dict[tuple[str, Optional[int], float], torch.Tensor] = {}
_scale_cache_lock = threading.Lock()
_dense_offsets_cache: dict[tuple[str, Optional[int], int, int], torch.Tensor] = {}


def _target_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"CAKE DSv4 requires CUDA tensors, got {device}")
    major, minor = get_compute_capability(device)
    return f"sm_{major}{minor}a"


def _variant_module(variant: str, *, arch: str):
    from ..jit.cake_dsv4 import get_cake_dsv4_module

    return get_cake_dsv4_module(variant, arch=arch)


def _is_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return bool(torch.cuda.is_current_stream_capturing())


# --------------------------------------------------------------------------- #
# Sparse metadata resolution                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SparseMetadata:
    """Resolved kernel-side view of the DSv4 sparse metadata (no copies)."""

    swa_indices: torch.Tensor
    compressed_indices: torch.Tensor
    sparse_topk_lens: torch.Tensor
    swa_index_stride: int
    compressed_index_stride: int
    sparse_topk_lens_offset: int
    sparse_topk: int
    num_query_tokens: int
    separate_tables: bool

    def kernel_kwargs(self) -> dict[str, Any]:
        """Keyword arguments in the shared kernel parameter vocabulary.

        The two index tables are handed over as contiguous spans (see
        ``_flat_index_span``): the generated bindings check every pointer
        buffer for contiguity, and the column-sliced compressed view of a
        combined table is not contiguous even though the kernel only reads
        its base pointer plus the row stride.
        """
        values = {name: getattr(self, name) for name in KERNEL_METADATA_PARAMS}
        values["swa_indices"] = _flat_index_span(self.swa_indices)
        values["compressed_indices"] = _flat_index_span(self.compressed_indices)
        return values

    @property
    def compressed_width(self) -> int:
        return self.sparse_topk - _SWA_WIDTH

    @property
    def legacy_combined_table(self) -> Optional[torch.Tensor]:
        """Combined ``[T, sparse_topk]`` table for bindings that predate the split ABI.

        Only a combined table without an explicit length offset can be handed
        to a binding that still declares the pre-hardening ``sparse_indices``
        argument; separate tables and offsets need regenerated bindings.
        """
        if self.separate_tables or self.sparse_topk_lens_offset != 0:
            return None
        return self.swa_indices


def _flat_index_span(table: torch.Tensor) -> torch.Tensor:
    """Expose a row-strided int32 table as one contiguous span without a copy.

    A contiguous table is returned as is. A column-sliced view (row stride
    larger than its width) becomes the 1-D span from its first to its last
    element, which keeps ``data_ptr`` and the row stride the kernel indexes
    with while satisfying the binding's contiguity check.
    """
    if table.is_contiguous() or table.ndim != 2:
        return table
    rows, cols = int(table.shape[0]), int(table.shape[1])
    if rows == 0 or cols == 0:
        return table.reshape(-1)
    return table.as_strided(((rows - 1) * int(table.stride(0)) + cols,), (1,))


def _int32_table(tensor: torch.Tensor, name: str, *, rows: Optional[int] = None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must be a [rows, columns] table, got shape {tuple(tensor.shape)}"
        )
    if tensor.ndim > 2:
        try:
            tensor = tensor.view(-1, tensor.shape[-1])
        except RuntimeError as exc:
            raise ValueError(
                f"{name} leading dimensions must be densely packed so they fold "
                "into rows without a copy"
            ) from exc
    if rows is not None and tensor.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {tensor.shape[0]}")
    if tensor.shape[1] and tensor.stride(1) != 1:
        raise ValueError(
            f"{name} must have a unit column stride; pass a row-strided view "
            "instead of a copy"
        )
    return tensor


def _int32_lens(tensor: torch.Tensor, name: str, *, rows: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.ndim != 1:
        try:
            tensor = tensor.view(-1)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} must flatten to one row per token without a copy"
            ) from exc
    if tensor.numel() != rows:
        raise ValueError(f"{name} must have {rows} entries, got {tensor.numel()}")
    if tensor.numel() and tensor.stride(0) != 1:
        raise ValueError(f"{name} must have unit stride")
    return tensor


def resolve_cake_dsv4_sparse_metadata(
    sparse_indices: torch.Tensor,
    sparse_topk_lens: Optional[torch.Tensor] = None,
    *,
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    sparse_topk_lens_offset: int = 0,
    query_rows: int,
) -> SparseMetadata:
    """Resolve the shared metadata ABI from combined or separate host tables.

    Combined form: ``sparse_indices [T, sparse_topk]`` whose first 128 columns
    are SWA slots, with ``sparse_topk_lens`` counting those 128 slots.

    Separate form: ``sparse_indices [T, 128]`` is the SWA table and
    ``extra_sparse_indices [T, topk_c]`` the compressed table. Lengths come from
    ``sparse_topk_lens`` (combined convention) or ``extra_sparse_topk_lens``
    (compressed slots only; the host adds 128 to ``sparse_topk_lens_offset``).

    ``query_rows`` is the number of rows the query tensor provides; the metadata
    may describe fewer tokens (padded batch) but never more.
    """
    if isinstance(sparse_topk_lens_offset, bool) or not isinstance(
        sparse_topk_lens_offset, int
    ):
        raise TypeError("sparse_topk_lens_offset must be an int")
    offset = int(sparse_topk_lens_offset)
    if extra_sparse_indices is not None:
        swa = _int32_table(sparse_indices, "sparse_indices")
        if swa.shape[1] != _SWA_WIDTH:
            raise ValueError(
                "with extra_sparse_indices, sparse_indices is the SWA table and "
                f"must have {_SWA_WIDTH} columns, got {swa.shape[1]}"
            )
        rows = int(swa.shape[0])
        compressed = _int32_table(
            extra_sparse_indices, "extra_sparse_indices", rows=rows
        )
        compressed_width = int(compressed.shape[1])
        if compressed_width == 0:
            compressed = swa
        if extra_sparse_topk_lens is not None:
            if sparse_topk_lens is not None:
                raise ValueError(
                    "pass either sparse_topk_lens (combined, counting the 128 SWA "
                    "slots) or extra_sparse_topk_lens (compressed slots only), not both"
                )
            lens, lens_name = extra_sparse_topk_lens, "extra_sparse_topk_lens"
            offset += _SWA_WIDTH
        else:
            if sparse_topk_lens is None:
                raise ValueError(
                    "sparse_topk_lens or extra_sparse_topk_lens is required"
                )
            lens, lens_name = sparse_topk_lens, "sparse_topk_lens"
        separate = True
    else:
        if extra_sparse_topk_lens is not None:
            raise ValueError("extra_sparse_topk_lens requires extra_sparse_indices")
        if sparse_topk_lens is None:
            raise ValueError(
                "sparse_topk_lens is required with a combined sparse_indices table"
            )
        table = _int32_table(sparse_indices, "sparse_indices")
        rows = int(table.shape[0])
        if table.shape[1] < _SWA_WIDTH:
            raise ValueError(
                f"sparse_indices must have at least {_SWA_WIDTH} columns, got {table.shape[1]}"
            )
        swa = table
        compressed_width = int(table.shape[1]) - _SWA_WIDTH
        compressed = table[:, _SWA_WIDTH:] if compressed_width else table
        lens, lens_name = sparse_topk_lens, "sparse_topk_lens"
        separate = False

    if rows < 1:
        raise ValueError("sparse metadata must describe at least one query token")
    if rows > query_rows:
        raise ValueError(
            f"metadata has {rows} rows but the query only has {query_rows}"
        )
    sparse_topk = _SWA_WIDTH + compressed_width
    if sparse_topk % 4:
        raise ValueError(
            f"sparse_topk (128 + compressed columns) must be a multiple of 4, got {sparse_topk}"
        )
    lens = _int32_lens(lens, lens_name, rows=rows)
    return SparseMetadata(
        swa_indices=swa,
        compressed_indices=compressed,
        sparse_topk_lens=lens,
        swa_index_stride=int(swa.stride(0)),
        compressed_index_stride=int(compressed.stride(0)),
        sparse_topk_lens_offset=offset,
        sparse_topk=sparse_topk,
        num_query_tokens=rows,
        separate_tables=separate,
    )


# --------------------------------------------------------------------------- #
# Workspace                                                                   #
# --------------------------------------------------------------------------- #


def _align_up(num_bytes: int) -> int:
    return -(-num_bytes // _ALIGN) * _ALIGN


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _upper_bound_num_splits(sparse_topk: int) -> int:
    return max(-(-sparse_topk // _TILE_KV), _MAX_FIXED_SPLITS)


@dataclass(frozen=True)
class WorkspaceLayout:
    """Byte ``(offset, size)`` of every region carved from ``workspace_buffer``."""

    descriptor_slab: tuple[int, int]
    counters: tuple[int, int]
    partial_o: tuple[int, int]
    partial_lse: tuple[int, int]
    # Shifted request offsets of a row-tiled chunk (size 0 for whole launches).
    query_offsets: tuple[int, int]
    total_bytes: int


def cake_dsv4_workspace_layout(
    num_query_tokens: int,
    num_heads: int,
    num_splits: int,
    *,
    num_query_offsets: int = 0,
) -> WorkspaceLayout:
    """Deterministic carve of ``workspace_buffer`` for one launch shape.

    One-split launches (``num_splits == 1``) write the final output directly
    and carry no ``partial_O`` region (size 0); the LSE region follows the
    counters immediately. ``num_query_offsets`` (``batch_size + 1`` for a
    row-tiled chunk after the first, else 0) sizes the trailing int32 region
    that holds the chunk's shifted ``cum_seq_lens_q``.
    """
    tokens = _positive_int(num_query_tokens, "num_query_tokens")
    heads = _positive_int(num_heads, "num_heads")
    splits = _positive_int(num_splits, "num_splits")
    if (
        isinstance(num_query_offsets, bool)
        or not isinstance(num_query_offsets, int)
        or num_query_offsets < 0
    ):
        raise ValueError(
            f"num_query_offsets must be a non-negative int, got {num_query_offsets!r}"
        )
    partial_elems = tokens * heads * splits
    o_bytes = (
        _align_up(partial_elems * _HEAD_DIM * torch.bfloat16.itemsize)
        if splits > 1
        else 0
    )
    lse_bytes = _align_up(partial_elems * torch.float32.itemsize)
    offsets_bytes = _align_up(num_query_offsets * torch.int32.itemsize)
    o_offset = _PARTIAL_OFFSET
    lse_offset = o_offset + o_bytes
    offsets_offset = lse_offset + lse_bytes
    return WorkspaceLayout(
        descriptor_slab=(_DESCRIPTOR_SLAB_OFFSET, _DESCRIPTOR_SLAB_BYTES),
        counters=(_COUNTER_OFFSET, _COUNTER_REGION_BYTES),
        partial_o=(o_offset, o_bytes),
        partial_lse=(lse_offset, lse_bytes),
        query_offsets=(offsets_offset, offsets_bytes),
        total_bytes=offsets_offset + offsets_bytes,
    )


def get_cake_dsv4_workspace_bytes(
    num_query_tokens: int,
    num_heads: int,
    sparse_topk: int,
    dtype: torch.dtype,
    *,
    num_splits: Optional[int] = None,
) -> int:
    """Single-launch upper bound of ``workspace_buffer`` for ``backend="cake"``.

    ``num_query_tokens`` is the metadata row count (padded query rows do not
    count). The result bounds every route's one-launch carve::

        S      = num_splits if given else max(ceil(sparse_topk / 128), 5)
        bytes  = 1024                                   # reserved slab
               + 262144                                 # split-merge counters (uint32[65536])
               + align128(num_query_tokens * num_heads * S * 512 * 2)   # partial_O (BF16), S > 1 only
               + align128(num_query_tokens * num_heads * S * 4)         # partial_lse (FP32)

    Partial buffers are BF16/FP32 for both BF16 and FP8 inputs; ``dtype`` is
    validated only. Pass ``num_splits`` to size for a known route (routes use
    ``ceil(sparse_topk / 128)`` or a fixed 1..5 splits). A smaller buffer is
    not an error: the host tiles the metadata rows into launches that fit,
    down to the one-row requirement; :func:`cake_dsv4_workspace_requirement`
    reports the exact per-route numbers for a call.
    """
    if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(f"unsupported CAKE DSv4 dtype: {dtype}")
    topk = _positive_int(sparse_topk, "sparse_topk")
    if topk < _SWA_WIDTH or topk % 4:
        raise ValueError(
            f"sparse_topk must be a multiple of 4 and at least {_SWA_WIDTH}, got {topk}"
        )
    splits = (
        _upper_bound_num_splits(topk)
        if num_splits is None
        else _positive_int(num_splits, "num_splits")
    )
    return cake_dsv4_workspace_layout(num_query_tokens, num_heads, splits).total_bytes


def _workspace_bytes(workspace: torch.Tensor) -> torch.Tensor:
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace_buffer must be a torch.Tensor")
    if not workspace.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous")
    raw = workspace.view(torch.uint8).reshape(-1)
    if raw.data_ptr() % _ALIGN:
        raise ValueError(f"workspace_buffer must be {_ALIGN}-byte aligned")
    return raw


def _require_workspace_bytes(raw: torch.Tensor, needed: int) -> None:
    if raw.numel() < needed:
        raise ValueError(
            f"workspace_buffer requires at least {needed} bytes for this CAKE DSv4 "
            f"launch, got {raw.numel()}; size it with get_cake_dsv4_workspace_bytes()"
        )


def _partial_views(
    raw: torch.Tensor,
    out_rows: torch.Tensor,
    num_query_tokens: int,
    num_heads: int,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = cake_dsv4_workspace_layout(num_query_tokens, num_heads, num_splits)
    _require_workspace_bytes(raw, layout.total_bytes)
    elems = num_query_tokens * num_heads * num_splits
    lse_offset = layout.partial_lse[0]
    partial_lse = raw[lse_offset : lse_offset + elems * torch.float32.itemsize].view(
        torch.float32
    )
    if num_splits == 1:
        # Single-partition routes write the final output directly.
        return out_rows.reshape(-1), partial_lse
    o_offset = layout.partial_o[0]
    partial_o = raw[
        o_offset : o_offset + elems * _HEAD_DIM * torch.bfloat16.itemsize
    ].view(torch.bfloat16)
    return partial_o, partial_lse


def _counters(raw: torch.Tensor, merge_groups: int) -> torch.Tensor:
    if merge_groups > _MAX_MERGE_GROUPS:
        raise ValueError(
            f"CAKE DSv4 split-merge route needs {merge_groups} counters; the workspace "
            f"counter region holds {_MAX_MERGE_GROUPS}"
        )
    _require_workspace_bytes(raw, _PARTIAL_OFFSET)
    return raw[_COUNTER_OFFSET : _COUNTER_OFFSET + merge_groups * 4].view(torch.uint32)


# Descriptor storage pools (see the module docstring, "Descriptor storage").
# Bound on the storages a pool hands to eager launches; descriptor sets that
# were launched under CUDA Graph capture are retained separately for the
# process lifetime because their graphs keep reading them.
_DESCRIPTOR_POOL_CAPACITY = 4096
_descriptor_lock = threading.Lock()
_descriptor_pools: dict[tuple[str, str, torch.device], "_DescriptorPool"] = {}


@dataclass
class _DescriptorStorage:
    tensor: torch.Tensor
    # Stream of the last launch that read this storage; a launch on another
    # stream waits for it before the binding may rewrite the descriptors.
    stream: Optional[torch.cuda.Stream]


@dataclass
class _DescriptorPool:
    """Descriptor storages of one variant module on one device."""

    # descriptor set -> storage, least recently used first; reassignable
    live: "collections.OrderedDict[tuple, _DescriptorStorage]"
    # descriptor sets launched under graph capture: never reassigned
    captured: dict[tuple, _DescriptorStorage]
    # storages released by a capacity change, reused before allocating
    spare: list[_DescriptorStorage]


def _new_descriptor_storage(device: torch.device) -> _DescriptorStorage:
    backing = torch.empty(
        _DESCRIPTOR_SLAB_BYTES + _ALIGN, dtype=torch.uint8, device=device
    )
    offset = (-backing.data_ptr()) % _ALIGN
    return _DescriptorStorage(backing[offset : offset + _DESCRIPTOR_SLAB_BYTES], None)


def _descriptor_storage(
    variant: str,
    arch: str,
    num_bytes: int,
    sources: Sequence[tuple[str, torch.Tensor]],
    *,
    capturing: bool,
) -> torch.Tensor:
    """Private descriptor storage for ``variant`` over these TMA source tensors.

    A descriptor set is a pure function of each source tensor's pointer, shape,
    strides and dtype, so that geometry is the key. The pool of one variant on
    one device hands out at most :data:`_DESCRIPTOR_POOL_CAPACITY` storages to
    eager launches: a hit reuses the storage whose bytes the binding already
    holds, a miss takes a fresh storage until the pool is full and the least
    recently used one afterwards (the binding rewrites its descriptors in
    stream order before the launch). A set launched under CUDA Graph capture
    moves to the pool's retained part and is never reassigned, so replays keep
    reading the descriptors they captured; a set that is not resident when a
    capture reaches it is an error, because the binding cannot initialize
    descriptors inside a capture. The caller holds :data:`_descriptor_lock`
    from this lookup through the launch, so a storage is never reassigned
    between the binding's descriptor check and the launch that reads it.
    """
    if num_bytes > _DESCRIPTOR_SLAB_BYTES:
        raise ValueError(
            f"CAKE DSv4 variant needs {num_bytes} TMA descriptor bytes; the "
            f"descriptor storage holds {_DESCRIPTOR_SLAB_BYTES}"
        )
    device = sources[0][1].device if sources else torch.device("cpu")
    key = tuple(
        (name, t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
        for name, t in sources
    )
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    pool = _descriptor_pools.get((variant, arch, device))
    if pool is None:
        pool = _DescriptorPool(collections.OrderedDict(), {}, [])
        _descriptor_pools[variant, arch, device] = pool
    storage = pool.captured.get(key)
    if storage is not None:
        return storage.tensor
    storage = pool.live.get(key)
    if capturing:
        if storage is None:
            raise RuntimeError(
                f"CAKE DSv4 {variant}: the TMA descriptors for these query / "
                "KV-cache tensors have not been initialised and the current "
                "stream is capturing a CUDA graph; run one eager call with the "
                "same tensors (pointers, shapes and strides) before capture"
            )
        del pool.live[key]
        pool.captured[key] = storage
        return storage.tensor
    if storage is not None:
        pool.live.move_to_end(key)
    else:
        capacity = max(1, int(_DESCRIPTOR_POOL_CAPACITY))
        while len(pool.live) >= capacity:
            pool.spare.append(pool.live.popitem(last=False)[1])
        storage = pool.spare.pop() if pool.spare else _new_descriptor_storage(device)
        pool.live[key] = storage
    if stream is not None and storage.stream is not None and storage.stream != stream:
        # Order this launch (and a descriptor rewrite the binding may issue
        # on this stream) after the last launch that read the storage.
        stream.wait_stream(storage.stream)
    storage.stream = stream
    return storage.tensor


def _workspace_owner(workspace: torch.Tensor) -> torch.Tensor:
    """The tensor whose lifetime owns ``workspace``'s storage (its ``_base`` or itself)."""
    base = workspace._base
    return workspace if base is None else base


def _primed_key(raw: torch.Tensor) -> tuple[Optional[int], int]:
    return (raw.device.index, raw.data_ptr())


def _counters_primed(workspace: torch.Tensor, raw: torch.Tensor) -> bool:
    """Whether the counter region behind ``raw`` is registered as zero."""
    with _primed_lock:
        ref = _primed_workspaces.get(_primed_key(raw))
    return ref is not None and ref() is _workspace_owner(workspace)


def _register_primed(workspace: torch.Tensor, raw: torch.Tensor) -> None:
    with _primed_lock:
        for key in [k for k, ref in _primed_workspaces.items() if ref() is None]:
            del _primed_workspaces[key]
        _primed_workspaces[_primed_key(raw)] = weakref.ref(_workspace_owner(workspace))


def cake_dsv4_workspace_reset(workspace_buffer: torch.Tensor) -> None:
    """Zero the split-merge counter region of ``workspace_buffer`` and register it.

    Optional: the launch contract zeroes the counters itself (first eager use
    zeroes and registers the workspace; a capture through an unregistered
    workspace records the zero fill into the graph). Calling this after
    allocating a workspace only moves that first fill out of the hot path.
    Zeroing is an in-place fill; nothing is allocated.
    """
    raw = _workspace_bytes(workspace_buffer)
    _require_workspace_bytes(raw, _PARTIAL_OFFSET)
    raw[_COUNTER_OFFSET:_PARTIAL_OFFSET].zero_()
    _register_primed(workspace_buffer, raw)


def _ensure_counters_zeroed(
    workspace: torch.Tensor, raw: torch.Tensor, merge_groups: int
) -> None:
    """Make the ``merge_groups`` counters of this launch zero; never raises for capture.

    Registered workspace: nothing (the kernels leave the counters zero).
    Unregistered, eager: zero the whole region once and register it.
    Unregistered, under CUDA Graph capture: record a zero fill of exactly the
    counters this launch uses into the graph; the registry is left alone
    because the fill has not executed and only replays carry it.
    """
    if _counters_primed(workspace, raw):
        return
    if _is_capturing(workspace.device):
        raw[_COUNTER_OFFSET : _COUNTER_OFFSET + merge_groups * 4].zero_()
        return
    raw[_COUNTER_OFFSET:_PARTIAL_OFFSET].zero_()
    _register_primed(workspace, raw)


# --------------------------------------------------------------------------- #
# Name-based argument binding                                                 #
# --------------------------------------------------------------------------- #

_TMA_SOURCE_ALIASES: Mapping[str, str] = {
    "tmap_q": "Q",
    "tmap_swa_k": "SWA_cache",
    "tmap_swa_v": "SWA_cache",
    "tmap_swa_kv": "SWA_cache",
    "tmap_compressed_k": "compressed_KV_cache",
    "tmap_compressed_v": "compressed_KV_cache",
    "tmap_compressed_kv": "compressed_KV_cache",
    # The FP8 persistent bodies (round 5) store O through a TMA descriptor over
    # the same [tokens, heads, 512] rows the plain ``O`` pointer argument sees.
    "tmap_o": "O",
    # The bf16 H64 guard program binds its ``O`` parameter itself as a 3-D
    # tensor map (box 64 x 16 x 1 over [tokens, heads, 512]; head rows >=
    # num_heads are clipped by the map), so the registration carries
    # ("tma_buffer", "O"): the same output rows, encoded by the grid_constant
    # binding -- no descriptor workspace and no host copy.
    "O": "O",
}
_SCALAR_ALIASES: Mapping[str, str] = {
    "num_q_heads": "num_heads",
    "num_split": "num_splits",
}
_TENSOR_VALUE_NAMES = frozenset(
    {
        "Q",
        "SWA_cache",
        "compressed_KV_cache",
        "O",
        "partial_O",
        "partial_lse",
        "partition_arrivals",
        "seq_lens",
        "cum_seq_lens_q",
        "sinks",
        "bmm1_scale",
        "bmm2_scale",
        "swa_indices",
        "compressed_indices",
        "sparse_topk_lens",
        # Pre-hardening combined table; bound only for combined metadata.
        "sparse_indices",
    }
)
_SCALAR_VALUE_NAMES = frozenset(
    {
        "swa_index_stride",
        "compressed_index_stride",
        "sparse_topk_lens_offset",
        "sparse_topk",
        "num_query_tokens",
        "num_heads",
        "num_head_tiles",
        "has_sinks",
        "num_splits",
        "total_work_items",
        "batch_size",
        "max_q_len",
        "ragged_query",
    }
)
_GRID_NAMES = ("grid_x", "grid_y", "grid_z")
_DESCRIPTOR_WORKSPACE_NAME = "tma_descriptor_workspace"
_RETIRED_ARG_REASONS: Mapping[str, str] = {
    "completion_base": (
        "host-side split-merge generation state was removed; the kernel resets "
        "partition_arrivals itself"
    ),
}
# Producers whose generated ABI reads request boundaries from cum_seq_lens_q
# only; run_cake_dsv4 synthesizes the dense offsets for them.
_RAGGED_ONLY_ROUTES = frozenset(
    {
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
    }
)
# Mirrors the Cake seed's LANE_GATHER_MIN_TOKENS. Below it the
# persistent FP8 body runs the program with elected-lane uniform K/V gathers;
# from 128 tokens on, the lane-issued gathers (several waves per cluster) win.
_FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS = 128


def _fp8_persistent_program(num_query_tokens: int) -> str:
    if num_query_tokens < _FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS:
        return "fp8_h128_prefill_source_persistent_uniform"
    return "fp8_h128_prefill_source_persistent"


# Mirrors the Cake seed's H64_M64_MIN_TOKENS
# (portfolio_v34.persistent_program): FP8/H64 rows admitted to the persistent
# body with at least this many tokens run the H64-specific single-CTA M64 body.
_FP8_H64_M64_MIN_TOKENS = 128


def _fp8_h64_uses_persistent_body(sparse_topk: int, num_query_tokens: int) -> bool:
    """FP8/H64 persistent-body rule (Cake seed ``portfolio_v34.uses_persistent_body``)."""
    full_tiles = sparse_topk // 128
    if num_query_tokens <= 12:
        return full_tiles >= 3
    return full_tiles >= 2 or num_query_tokens >= 128


# Mirrors the Cake M64 seed's box_k_gather_for_width (TILE_KV = 128).  The M64
# body is exported twice: single-tile items (the SWA tile is the whole item)
# run the program whose load warp gathers contiguous 16-key SWA chunks through
# one box TMA (-0.35..-0.42 us on the 128/256-token SWA rows); every wider item
# runs the program without that block, which measured at +0.03..+0.11 us of
# load-warp code layout on the multi-tile rows even when the in-kernel gate
# kept it off.  Same bits from both programs.
def _fp8_h64_m64_program(sparse_topk: int) -> str:
    if sparse_topk == _TILE_KV:
        return "fp8_h64_prefill_source_persistent_m64"
    return "fp8_h64_prefill_source_persistent_m64_multi_tile"


_UNAVAILABLE_HINTS: Mapping[str, str] = {
    "sparse_indices": (
        "this binding predates the split-table metadata ABI and only accepts a "
        "combined sparse_indices table with sparse_topk_lens_offset == 0; "
        "regenerate the bindings for separate tables or length offsets"
    ),
}


def canonical_arg_name(kind: str, name: str) -> str:
    """Map a registration ``arg_plan`` entry onto the host value vocabulary."""
    if kind == "tma_buffer":
        return _TMA_SOURCE_ALIASES.get(name, name)
    if kind == "parameter":
        return _SCALAR_ALIASES.get(name, name)
    return name


def is_bindable_arg(kind: str, name: str) -> bool:
    """Whether the host can supply this ``arg_plan`` entry at all."""
    canonical = canonical_arg_name(kind, name)
    if kind in ("buffer", "tma_buffer"):
        return canonical in _TENSOR_VALUE_NAMES
    if kind == "parameter":
        return canonical in _SCALAR_VALUE_NAMES
    if kind == "workspace":
        return name == _DESCRIPTOR_WORKSPACE_NAME
    if kind == "grid":
        return name in _GRID_NAMES
    return False


def _bind_argument(
    values: Mapping[str, Any],
    kind: str,
    name: str,
    *,
    variant: str,
    grid: Mapping[str, int],
    descriptor_slab: Optional[torch.Tensor],
) -> Any:
    if kind == "grid":
        if name not in grid:
            raise ValueError(
                f"CAKE DSv4 {variant} has an unknown grid argument: {name}"
            )
        return grid[name]
    if kind == "workspace":
        if name != _DESCRIPTOR_WORKSPACE_NAME or descriptor_slab is None:
            raise ValueError(
                f"CAKE DSv4 {variant} has an unresolved workspace argument: {name}"
            )
        return descriptor_slab
    if kind not in ("buffer", "tma_buffer", "parameter"):
        raise ValueError(
            f"CAKE DSv4 {variant} has an unknown argument kind {kind!r} for {name}"
        )
    canonical = canonical_arg_name(kind, name)
    if canonical in _RETIRED_ARG_REASONS:
        raise ValueError(
            f"CAKE DSv4 {variant} binds the retired argument {name!r}: "
            f"{_RETIRED_ARG_REASONS[canonical]}; regenerate the bindings"
        )
    if canonical not in values:
        raise ValueError(
            f"CAKE DSv4 {variant} argument {name!r} ({kind}) has no host value; "
            f"known names: {sorted(values)}"
        )
    value = values[canonical]
    if value is None:
        hint = _UNAVAILABLE_HINTS.get(canonical, "it is not available for this call")
        raise ValueError(f"CAKE DSv4 {variant} argument {name!r}: {hint}")
    if kind == "parameter":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"CAKE DSv4 {variant} parameter {name!r} must be an int, got {type(value).__name__}"
            )
        return int(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"CAKE DSv4 {variant} buffer {name!r} must be a tensor, got {type(value).__name__}"
        )
    return value


def _grid_values(grid: tuple[int, int, int]) -> dict[str, int]:
    if len(grid) != 3 or any(
        isinstance(g, bool) or not isinstance(g, int) or g < 1 for g in grid
    ):
        raise ValueError(f"launch grid must be three positive ints, got {grid!r}")
    return dict(zip(_GRID_NAMES, grid, strict=True))


class _PreparedLaunch:
    """A bound variant launch: the FFI call is the only work left to do."""

    __slots__ = ("module", "bound")

    def __init__(self, module, bound: list):
        self.module = module
        self.bound = bound

    def __call__(self):
        # Direct-source bindings use the target FFI current stream.
        return self.module.run(*self.bound)


def _sequence_module():
    from ..jit.cake_dsv4 import get_cake_dsv4_launch_sequence_module

    return get_cake_dsv4_launch_sequence_module()


def _prepare_variant(
    variant: str,
    *,
    arch: str,
    grid: tuple[int, int, int],
    values: Mapping[str, Any],
) -> _PreparedLaunch:
    """Bind the generated ABI by name through the registration ``arg_plan``.

    Two-stage routes prepare the producer and the reducer before launching
    either, so no Python-side binding sits between the two kernels. The caller
    holds :data:`_descriptor_lock` from the first preparation through the last
    launch of the call.
    """
    from ..jit.cake_dsv4 import get_cake_dsv4_spec

    contract = get_cake_dsv4_spec(variant, arch=arch)
    plan = contract["arg_plan"]
    grid_values = _grid_values(grid)

    def bind(kind, name, storage=None):
        return _bind_argument(
            values,
            kind,
            name,
            variant=variant,
            grid=grid_values,
            descriptor_slab=storage,
        )

    tma_bytes = int(contract.get("tma_workspace_bytes", 0) or 0)
    storage = None
    if tma_bytes:
        sources = [
            (name, bind(kind, name)) for kind, name in plan if kind == "tma_buffer"
        ]
        device = sources[0][1].device if sources else torch.device("cpu")
        storage = _descriptor_storage(
            variant, arch, tma_bytes, sources, capturing=_is_capturing(device)
        )
    bound = [bind(kind, name, storage) for kind, name in plan]
    return _PreparedLaunch(_variant_module(variant, arch=arch), bound)


def _launch_variant(
    variant: str,
    *,
    arch: str,
    grid: tuple[int, int, int],
    values: Mapping[str, Any],
):
    """Prepare and launch one variant (single-launch convenience)."""
    with _descriptor_lock:
        return _prepare_variant(variant, arch=arch, grid=grid, values=values)()


# --------------------------------------------------------------------------- #
# Routing                                                                     #
# --------------------------------------------------------------------------- #


def _route(
    *,
    dtype: torch.dtype,
    num_heads: int,
    max_q_len: int,
    ragged: bool,
    sparse_topk: int,
    batch_size: int,
    compressed_page_size: int,
    num_query_tokens: int,
) -> str:
    if max_q_len <= 0:
        raise ValueError("max_q_len must be positive")
    is_swa = sparse_topk == 128
    is_topk4x = not is_swa and compressed_page_size == 64
    is_topk128x = not is_swa and compressed_page_size == 2
    if dtype == torch.float8_e4m3fn:
        if num_heads == 128:
            # One producer for every FP8/H128 row: the persistent body (one
            # work feed over T x 2 CTAs, runtime q_lens) matched or beat the
            # former per-(token, split) cluster producer on the 12-token
            # decode rows and beat trtllm-gen 1.12-1.75x on the MTP rows.
            return _fp8_persistent_program(num_query_tokens)
        if num_heads not in (8, 16, 32, 64):
            raise ValueError(f"unsupported CAKE FP8 DSv4 head count: {num_heads}")
        if (
            num_heads == 64
            and batch_size == 2
            and max_q_len == 257
            and ragged
            and is_topk4x
            and sparse_topk == 640
        ):
            return "fp8_h64_source_exact"
        if max_q_len >= 257:
            return "fp8_lowhead_prefill"
        if (
            num_heads in (8, 16)
            and batch_size == 3
            and max_q_len == 5
            and ragged
            and (
                (is_topk128x and sparse_topk == 260)
                or (is_topk4x and sparse_topk == (192 if num_heads == 8 else 256))
            )
        ):
            # FP8 port of the 1-CTA SwapsAb body trtllm-gen runs
            # on these rows (heads on the MMA N side, kind::f8f6f4, P e4m3
            # x448); same shape lock as the BF16 source-exact route.  Paired
            # vs trtllm-gen: rows 55/58/67/70 GB300 1.27-1.30x / B200
            # 1.24-1.25x, width-260 rows 56/59/68/71 GB300 1.38-1.49x /
            # B200 1.33-1.43x (one-partition producer 0.99-1.27x).
            return "fp8_h8_h16_source_exact"
        if num_heads == 64 and _fp8_h64_uses_persistent_body(
            sparse_topk, num_query_tokens
        ):
            # Mirrors the Cake seed's H64_PERSISTENT_* rule:
            # 12-token rows with >= 3 complete sparse tiles; every many-token
            # compressed row (>= 2 complete tiles); SWA-only rows from 128
            # tokens.  Same persistent FP8 body as FP8/H128 (heads >= num_heads
            # predicated): GB300 1.23-1.26x / B200 1.15x on the 12-token rows,
            # 1.02-1.92x on the 16-128-token hardening rows where the
            # per-token cluster body and the SWA producer sat at 0.6-0.87x.
            # Evaluated before the SWA-only test on purpose.
            if num_query_tokens >= _FP8_H64_M64_MIN_TOKENS:
                # H64-specific single-CTA M64 persistent body (one
                # CTA per token, unified 128-row KV stage, no V gathers): GB300
                # 1.25-1.55x / B200 1.22-1.44x on the 128-512 token rows where
                # the FP8/H128 body sat at 0.81-1.13x.  Two exported programs,
                # selected by the item width.
                return _fp8_h64_m64_program(sparse_topk)
            return _fp8_persistent_program(num_query_tokens)
        if is_swa:
            return "fp8_lowhead_prefill"
        if num_heads == 64:
            return "fp8_lowhead_h64"
        # One producer partition owns up to three sparse tiles (widths up to
        # 384) and writes final O directly (one partition measured 1.18-1.26x
        # on the width-260 rows).  No two-partition producer is exported for
        # wider low-head rows.
        if sparse_topk > _FP8_ONE_PARTITION_MAX_WIDTH:
            raise ValueError(
                f"backend='cake' has no FP8 kernel for {num_heads} heads with "
                f"sparse_topk {sparse_topk} > {_FP8_ONE_PARTITION_MAX_WIDTH} "
                f"below max_q_len 257"
            )
        return "fp8_lowhead_one_partition"
    if dtype != torch.bfloat16:
        raise ValueError(f"unsupported CAKE DSv4 dtype: {dtype}")
    if (
        num_heads in (8, 16)
        and batch_size == 3
        and max_q_len == 5
        and ragged
        and (
            (is_topk128x and sparse_topk == 260)
            or (is_topk4x and sparse_topk == (192 if num_heads == 8 else 256))
        )
    ):
        return "bf16_h8_h16_source_exact"
    if num_heads in (8, 16):
        if is_swa:
            return "bf16_h8_swa128_v43" if num_heads == 8 else "bf16_h16_h32_swa128_v44"
        raise ValueError(
            f"backend='cake' has no BF16 kernel for {num_heads} heads with a "
            "compressed cache outside the batch-3, max_q_len-5 ragged rows"
        )
    if num_heads == 32:
        if is_swa:
            return "bf16_h16_h32_swa128_v44"
        if is_topk4x or is_topk128x:
            # The retained-KV body with the last-arriver merge
            # beats the topk4x body on the H32 topk4x rows on both targets.
            return "bf16_h32_topk128x_early_v47"
        raise ValueError("BF16 H32 compressed cache requires page size 64 or 2")
    if num_heads == 64:
        # Dense Q with equal q_lens is the same row memory as ragged Q: the
        # not-ragged rows follow the ragged rules (the portfolio producer beats
        # the former fixed-Q guard program on the sparse decode rows, GB300
        # 1.03x -> 1.21x; the fixed-Q programs are no longer exported).
        if max_q_len >= 257:
            return "bf16_h64_prefill"
        if is_swa:
            return (
                "bf16_h64_guard_q_tma_batch_r25"
                if max_q_len > 5
                else "bf16_swa128_single_cta"
            )
        if (
            num_query_tokens >= _BF16_H64_COMPRESSED_PREFILL_TOKENS
            and sparse_topk <= _BF16_H64_PREFILL_MAX_SPARSE_WIDTH
        ):
            # The KV-reuse prefill body (one Q64 CTA per token, full V) beats
            # the one-tile-per-split portfolio producer from 24 tokens on.
            return "bf16_h64_prefill"
        return "bf16_h64_compressed_q8_v38"
    if num_heads == 128:
        if max_q_len >= 257:
            return "bf16_h128_prefill_v42"
        if (is_swa or is_topk4x) and num_query_tokens >= _BF16_H128_PREFILL_MIN_TOKENS:
            # Persistent KV-reuse prefill body for the many-token MTP rows
            # (Cake rule BF16_H128_PREFILL_MIN_TOKENS): the dedicated SWA
            # producer repeats QK per V chunk and the per-token owners lose
            # from 64 tokens on.
            return "bf16_h128_prefill_v42"
        if is_swa:
            return "bf16_h128_swa128"
        if is_topk4x and sparse_topk == 1152:
            return "bf16_h128_topk4x_v52"
        if _BF16_TOPK128X_MIN_WIDTH < sparse_topk <= _BF16_TOPK128X_MAX_WIDTH:
            return "bf16_h128_topk128x"
        raise ValueError(
            f"backend='cake' has no BF16 H128 kernel for sparse_topk {sparse_topk} "
            f"below {_BF16_H128_PREFILL_MIN_TOKENS} tokens (supported: 128, "
            f"{_BF16_TOPK128X_MIN_WIDTH + 4}-{_BF16_TOPK128X_MAX_WIDTH} and 1152 "
            "with page size 64)"
        )
    raise ValueError(f"unsupported CAKE BF16 DSv4 head count: {num_heads}")


# --------------------------------------------------------------------------- #
# Route workspace plan                                                        #
# --------------------------------------------------------------------------- #

# Routes whose producers write the final output directly and keep no LSE:
# they never touch the workspace.
_NO_WORKSPACE_ROUTES = frozenset(
    {
        "bf16_swa128_single_cta",
        "bf16_h128_swa128",
        "bf16_h8_swa128_v43",
        "bf16_h16_h32_swa128_v44",
        "bf16_h8_h16_source_exact",
        "fp8_h8_h16_source_exact",
        "bf16_h64_guard_q_tma_batch_r25",
        "bf16_h64_prefill",
        "fp8_h64_source_exact",
    }
)
# Routes that launch one producer with ``partials(1)``: final O written
# directly, the LSE region of the workspace only.
_ONE_SPLIT_ROUTES = frozenset(
    {
        "bf16_h128_prefill_v42",
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "fp8_lowhead_one_partition",
        "fp8_lowhead_h64",
        "fp8_lowhead_prefill",
    }
)


@dataclass(frozen=True)
class _RoutePlan:
    """Workspace use of one route at one shape: the single source for the
    dispatcher's ``partials`` / ``counters`` calls and for the sizing API."""

    # Partial buffers per (token, head); 1 = LSE region only, no partial_O.
    num_splits: int
    # False: the route binds no partial buffers and needs no workspace bytes.
    uses_partials: bool
    # Split-merge counters per metadata row (0: no in-kernel merge).
    merge_groups_per_row: int


def _route_plan(
    route: str, *, num_query_tokens: int, num_heads: int, sparse_topk: int
) -> _RoutePlan:
    if route == "bf16_h64_compressed_q8_v38":
        # One KV tile per split plus the H64 reducer (single split: direct O).
        return _RoutePlan(_ceil_div(sparse_topk, _TILE_KV), True, 0)
    if route == "bf16_h32_topk128x_early_v47":
        # One KV tile per split; the last-arriving split of each (token, head
        # tile) merges in-kernel through one counter per group.
        return _RoutePlan(
            _ceil_div(sparse_topk, _TILE_KV), True, _ceil_div(num_heads, 8)
        )
    if route == "bf16_h128_topk4x_v52":
        # Five fixed full-V KV owners per token plus the split-5 reducer.
        return _RoutePlan(5, True, 0)
    if route == "bf16_h128_topk128x":
        # Four disjoint full-V owners plus the LSE reducer up to the token
        # bound; one row-first owner per token above it.
        return _RoutePlan(
            4 if num_query_tokens <= _BF16_TOPK128X_SPLIT_MAX_TOKENS else 1, True, 0
        )
    if route in _ONE_SPLIT_ROUTES:
        return _RoutePlan(1, True, 0)
    if route in _NO_WORKSPACE_ROUTES:
        return _RoutePlan(1, False, 0)
    raise RuntimeError(f"unhandled CAKE DSv4 route: {route}")


def _launch_workspace_bytes(
    plan: _RoutePlan, rows: int, num_heads: int, num_query_offsets: int = 0
) -> int:
    """Workspace bytes one launch of ``plan`` over ``rows`` metadata rows carves."""
    if not plan.uses_partials:
        return 0
    return cake_dsv4_workspace_layout(
        rows, num_heads, plan.num_splits, num_query_offsets=num_query_offsets
    ).total_bytes


def _launch_fits(
    plan: _RoutePlan,
    rows: int,
    num_heads: int,
    num_query_offsets: int,
    workspace_bytes: int,
) -> bool:
    if (
        plan.merge_groups_per_row
        and rows * plan.merge_groups_per_row > _MAX_MERGE_GROUPS
    ):
        return False
    return (
        _launch_workspace_bytes(plan, rows, num_heads, num_query_offsets)
        <= workspace_bytes
    )


def _rows_per_launch(
    plan: _RoutePlan,
    *,
    num_query_tokens: int,
    num_heads: int,
    batch_size: int,
    workspace_bytes: int,
) -> int:
    """Metadata rows per launch for a workspace of ``workspace_bytes``.

    All rows when one launch fits (no row tiling, no offsets region);
    otherwise the largest chunk whose carve, including the shifted request
    offsets (``batch_size + 1`` int32) of the chunks after the first, fits;
    0 when not even one row fits.
    """
    total = num_query_tokens
    if _launch_fits(plan, total, num_heads, 0, workspace_bytes):
        return total
    offsets = batch_size + 1
    per_row = (
        num_heads
        * plan.num_splits
        * (
            (_HEAD_DIM * torch.bfloat16.itemsize if plan.num_splits > 1 else 0)
            + torch.float32.itemsize
        )
    )
    rows = (
        workspace_bytes
        - _PARTIAL_OFFSET
        - _align_up(offsets * torch.int32.itemsize)
        - 2 * _ALIGN
    ) // per_row
    if plan.merge_groups_per_row:
        rows = min(rows, _MAX_MERGE_GROUPS // plan.merge_groups_per_row)
    rows = max(0, min(rows, total - 1))
    while rows > 0 and not _launch_fits(
        plan, rows, num_heads, offsets, workspace_bytes
    ):
        rows -= 1
    while rows + 1 < total and _launch_fits(
        plan, rows + 1, num_heads, offsets, workspace_bytes
    ):
        rows += 1
    return rows


@dataclass(frozen=True)
class WorkspaceRequirement:
    """Exact workspace numbers of one ``backend="cake"`` call.

    ``single_launch_bytes`` is what one launch over all metadata rows carves
    (0 for routes without partial buffers); ``minimum_bytes`` is the smallest
    workspace the call accepts (one row per launch, row tiling); any size in
    between works with :meth:`rows_per_launch` rows per launch.
    """

    route: str
    num_splits: int
    uses_workspace: bool
    single_launch_bytes: int
    minimum_bytes: int
    num_query_tokens: int
    num_heads: int
    batch_size: int
    plan: _RoutePlan = field(repr=False)

    def rows_per_launch(self, workspace_bytes: int) -> int:
        """Metadata rows per launch with ``workspace_bytes`` (0: too small)."""
        return _rows_per_launch(
            self.plan,
            num_query_tokens=self.num_query_tokens,
            num_heads=self.num_heads,
            batch_size=self.batch_size,
            workspace_bytes=int(workspace_bytes),
        )


def cake_dsv4_workspace_requirement(
    *,
    dtype: torch.dtype,
    num_heads: int,
    num_query_tokens: int,
    sparse_topk: int,
    compressed_page_size: int,
    max_q_len: int,
    batch_size: int,
    ragged: bool,
) -> WorkspaceRequirement:
    """Route and exact workspace requirement of one call, without device access.

    The arguments are the route inputs of :func:`run_cake_dsv4`:
    ``num_query_tokens`` = metadata rows, ``sparse_topk`` = combined table
    width, ``compressed_page_size`` = ``compressed_kv_cache.shape[-2]``
    (irrelevant for SWA-only calls), ``max_q_len`` = dense per-request query
    length or the ragged maximum, ``batch_size`` = ``seq_lens.numel()``,
    ``ragged`` = ``cum_seq_lens_q is not None``. Raises ``ValueError`` for a
    shape ``backend="cake"`` has no kernel for.
    """
    tokens = _positive_int(num_query_tokens, "num_query_tokens")
    heads = _positive_int(num_heads, "num_heads")
    batch = _positive_int(batch_size, "batch_size")
    topk = _positive_int(sparse_topk, "sparse_topk")
    route = _route(
        dtype=dtype,
        num_heads=heads,
        max_q_len=_positive_int(max_q_len, "max_q_len"),
        ragged=bool(ragged),
        sparse_topk=topk,
        batch_size=batch,
        compressed_page_size=int(compressed_page_size),
        num_query_tokens=tokens,
    )
    plan = _route_plan(
        route, num_query_tokens=tokens, num_heads=heads, sparse_topk=topk
    )
    single = _launch_workspace_bytes(plan, tokens, heads, 0)
    minimum = min(single, _launch_workspace_bytes(plan, 1, heads, batch + 1))
    return WorkspaceRequirement(
        route=route,
        num_splits=plan.num_splits,
        uses_workspace=plan.uses_partials,
        single_launch_bytes=single,
        minimum_bytes=minimum,
        num_query_tokens=tokens,
        num_heads=heads,
        batch_size=batch,
        plan=plan,
    )


# --------------------------------------------------------------------------- #
# Output placement                                                            #
# --------------------------------------------------------------------------- #
_OUT_PHASE_PERIOD = 4096

# Output placement: on the routes below the kernel time is a 4 KiB-periodic
# function of the output buffer's base address (bits 0..11 only; bits >= 12 do
# nothing).  ``base % 4096 == 0`` -- what a fresh >= 2 MiB caching-allocator
# block gives, i.e. the default ``torch.empty`` for these outputs -- is the SLOW
# phase; ``0x800`` is the fast one.  Pinned-phase paired measurements (cold-L2
# CUPTI, both arm orders) on the exported programs, us at phase 0 ->
# phase 0x800, B200 / GB300:
#   bf16_h128_prefill_v42 (every bf16 H128 persistent row, striped and snake):
#     hardening-000035 40.13 -> 36.96 / 35.74 -> 33.25, hardening-000023 17.98 ->
#     16.70 / 16.58 -> 15.42, hardening-000029 26.78 -> 24.77 / 24.16 -> 22.94,
#     prefill-style-000088 90.66 -> 88.10 / 74.88 -> 72.48, prefill-style-000092
#     94.56 -> 92.26 / 78.21 -> 76.10 (-2.3 .. -8 %).
#   fp8_lowhead_prefill, 64 heads, SWA-128 table (the single-CTA K1 body):
#     decode-000006 7.07 -> 6.72 / 6.75 -> 6.43, decode-000009 7.07 -> 6.72 /
#     6.66 -> 6.30, hardening-000020 7.84 -> 7.49 / 7.55 -> 7.20 (-4.5 .. -5.3 %);
#     32 heads move +-0.1 us with a process-dependent sign and 8 / 16 heads do
#     not move, so they keep the allocator default.
# The kernels are untouched: the same program writes the same bits to the same
# (token, head, column) positions; only the buffer's base address is chosen.
# Applies to ``out=None`` only -- a caller-provided ``out`` is used as is, and a
# 2 MiB-aligned caller buffer sits in the slow phase.
_OUT_PHASE_BY_ROUTE: dict[str, int] = {
    "bf16_h128_prefill_v42": 0x800,
    "fp8_lowhead_prefill": 0x800,
    # 134-row public-API pass, allocator default -> 0x800, both arm orders:
    #   bf16_h64_prefill (086/087/090/091) -0.70..-1.50 us GB300,
    #   fp8_h64_source_exact -1.09..-1.18 us GB300, fp8_lowhead_h64 (008/011) -0.34 us GB300.
    "bf16_h64_prefill": 0x800,
    "fp8_h64_source_exact": 0x800,
    "fp8_lowhead_h64": 0x800,
    #   bf16_h128_topk128x: hardening-000025 / -000031 -0.74 / -0.70 us GB300, the
    #   other rows of the route flat (|d| <= 0.03 us).
    "bf16_h128_topk128x": 0x800,
}


def cake_dsv4_out_phase(
    route: str, *, num_heads: int, sparse_topk: int
) -> Optional[int]:
    """Preferred ``out.data_ptr() % 4096`` for ``route`` (None = allocator default)."""
    phase = _OUT_PHASE_BY_ROUTE.get(route)
    if phase is None:
        return None
    if route == "fp8_lowhead_prefill" and not (num_heads == 64 and sparse_topk == 128):
        return None
    return phase


def allocate_cake_dsv4_output(
    shape: tuple[int, ...], device: torch.device, *, phase: Optional[int]
) -> torch.Tensor:
    """bf16 output of ``shape``; with ``phase`` the view's ``data_ptr() % 4096 == phase``
    (one extra 4 KiB page is allocated), with None a plain ``torch.empty``."""
    if phase is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)
    if phase % 2 or not 0 <= phase < _OUT_PHASE_PERIOD:
        raise ValueError(
            f"output phase must be an even byte offset below {_OUT_PHASE_PERIOD}, got {phase}"
        )
    numel = 1
    for dim in shape:
        numel *= int(dim)
    raw = torch.empty(
        numel + _OUT_PHASE_PERIOD // 2, dtype=torch.bfloat16, device=device
    )
    start = ((phase - raw.data_ptr()) % _OUT_PHASE_PERIOD) // 2
    return raw[start : start + numel].view(shape)


# --------------------------------------------------------------------------- #
# Launch orchestration                                                        #
# --------------------------------------------------------------------------- #


def _device_scale(
    value: Union[float, torch.Tensor], *, device: torch.device, name: str
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise ValueError(f"{name} must be a one-element FP32 tensor")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}, got {value.device}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return value

    device_index = device.index
    if device_index is None and device.type == "cuda":
        device_index = torch.cuda.current_device()
    key = (device.type, device_index, float(value))
    with _scale_cache_lock:
        result = _scale_cache.get(key)
        if result is None:
            # Process-lifetime constant; not a per-call allocation.
            result = torch.tensor([key[2]], dtype=torch.float32, device=device)
            _scale_cache[key] = result
    return result


def _dense_query_offsets(
    batch_size: int, q_len: int, *, device: torch.device
) -> torch.Tensor:
    """``cum_seq_lens_q`` for a dense ``[batch, q_len]`` query (process-lifetime constant).

    Variants that only accept ragged queries read the request boundaries
    from ``cum_seq_lens_q``; a dense query is the ragged query with every
    request ``q_len`` long. The offsets are cached per (device, batch,
    q_len) so no call path allocates and CUDA-graph replay sees a stable
    pointer.
    """
    device_index = device.index
    if device_index is None and device.type == "cuda":
        device_index = torch.cuda.current_device()
    key = (device.type, device_index, int(batch_size), int(q_len))
    with _scale_cache_lock:
        result = _dense_offsets_cache.get(key)
        if result is None:
            result = torch.arange(
                0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device
            )
            _dense_offsets_cache[key] = result
    return result


def _dense_rows(cache: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
    if cache.dtype != dtype:
        raise ValueError(
            f"{name} dtype must match the query dtype {dtype}, got {cache.dtype}"
        )
    if cache.shape[-1] != _HEAD_DIM:
        raise ValueError(
            f"{name} must have head dim {_HEAD_DIM}, got {cache.shape[-1]}"
        )
    flat = cache.reshape(-1, _HEAD_DIM)
    if flat.data_ptr() != cache.data_ptr() or not flat.is_contiguous():
        raise ValueError(
            f"{name} must be a densely packed [..., {_HEAD_DIM}] pool; backend='cake' "
            "makes no host copy, so strided page layouts are not supported"
        )
    return flat


def _int32_vector(
    tensor: torch.Tensor, name: str, device: torch.device
) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    # A 1-D vector is bound as the caller's own object (reshape would wrap it
    # in a new view object even when nothing changes); higher-rank inputs
    # flatten without a copy or are rejected.
    flat = tensor if tensor.dim() == 1 else tensor.reshape(-1)
    if flat.data_ptr() != tensor.data_ptr() or not flat.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; backend='cake' makes no host copy"
        )
    return flat


class _Launcher:
    """Per-call launch context: shared values plus per-variant overrides."""

    def __init__(
        self,
        *,
        arch: str,
        workspace: torch.Tensor,
        raw: torch.Tensor,
        values: dict[str, Any],
    ):
        self.arch = arch
        self.workspace = workspace
        self.raw = raw
        self.values = values

    def variant(
        self, name: str, *, grid: tuple[int, int, int], **overrides: Any
    ) -> _PreparedLaunch:
        return _prepare_variant(
            name, arch=self.arch, grid=grid, values={**self.values, **overrides}
        )

    @staticmethod
    def run(*launches: _PreparedLaunch):
        """Issue the launches of one route.

        A single launch is one FFI call. Several launches go through the
        ``run_sequence`` host helper in one FFI call, so the host work of the
        second launch (argument conversion, Python-to-C transition) does not
        sit between the two kernels.
        """
        if len(launches) == 1:
            return launches[0]()
        flat: list[Any] = []
        for launch in launches:
            flat.append(launch.module.run)
            flat.append(len(launch.bound))
            flat.extend(launch.bound)
        return _sequence_module().run_sequence(*flat)

    def partials(self, num_splits: int) -> dict[str, Any]:
        partial_o, partial_lse = _partial_views(
            self.raw,
            self.values["O"],
            self.values["num_query_tokens"],
            self.values["num_heads"],
            num_splits,
        )
        return {
            "partial_O": partial_o,
            "partial_lse": partial_lse,
            "num_splits": num_splits,
        }

    def counters(self, merge_groups: int) -> torch.Tensor:
        counters = _counters(self.raw, merge_groups)
        _ensure_counters_zeroed(self.workspace, self.raw, merge_groups)
        return counters

    def reduce(self, reducer: str, **overrides: Any) -> _PreparedLaunch:
        tokens = self.values["num_query_tokens"]
        heads = self.values["num_heads"]
        return self.variant(reducer, grid=(tokens, heads, 1), **overrides)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def run_cake_dsv4(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[torch.Tensor],
    max_q_len: int,
    cum_seq_lens_q: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    backend: Literal["cake"],
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    sparse_topk_lens_offset: int = 0,
    out_shape: Optional[tuple[int, ...]] = None,
) -> torch.Tensor:
    """Launch the CAKE DSv4 route for flattened ``query [rows, num_heads, 512]``.

    ``out=None`` allocates the output here, after the route is known, so the
    routes in ``_OUT_PHASE_BY_ROUTE`` get their measured-fast base phase;
    ``out_shape`` (default ``[rows, num_heads, 512]``) is the shape to allocate.

    ``query`` / ``out`` may have more rows than the metadata; only the first
    ``num_query_tokens`` (metadata rows) are read and written. ``max_q_len`` is
    the dense per-request query length (``query.shape[1]`` of the caller's
    ``[B, Q, H, 512]``) or the ragged maximum; ``seq_lens`` has one entry per
    request and ``cum_seq_lens_q`` (ragged only) ``batch_size + 1`` offsets.
    Apart from the ``out=None`` output no device memory is allocated here
    (the dense request offsets of the ragged-only producers and of row-tiled
    dense launches are process-lifetime constants, created on first use:
    warm such shapes up eagerly before a capture); see the module docstring
    for the workspace, counter and row-tiling contract.
    """
    if backend != "cake":
        raise ValueError(f"expected backend='cake', got {backend!r}")
    if query.ndim != 3:
        raise ValueError(
            f"query must be [num_tokens, num_heads, {_HEAD_DIM}], got shape {tuple(query.shape)}"
        )
    query_capacity, num_heads, head_dim = query.shape
    if head_dim != _HEAD_DIM:
        raise ValueError(f"CAKE DSv4 requires head dim {_HEAD_DIM}, got {head_dim}")
    if query.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(f"unsupported CAKE DSv4 dtype: {query.dtype}")
    if not query.is_contiguous():
        raise ValueError("query must be contiguous; backend='cake' makes no host copy")
    device = query.device
    arch = _target_arch(device)

    meta = resolve_cake_dsv4_sparse_metadata(
        sparse_indices,
        sparse_topk_lens,
        extra_sparse_indices=extra_sparse_indices,
        extra_sparse_topk_lens=extra_sparse_topk_lens,
        sparse_topk_lens_offset=sparse_topk_lens_offset,
        query_rows=query_capacity,
    )
    for tensor, name in (
        (meta.swa_indices, "sparse_indices"),
        (meta.compressed_indices, "extra_sparse_indices"),
        (meta.sparse_topk_lens, "sparse_topk_lens"),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    num_query_tokens = meta.num_query_tokens
    sparse_topk = meta.sparse_topk

    if out is not None:
        if out.dtype != torch.bfloat16:
            raise ValueError(f"out must be bfloat16, got {out.dtype}")
        if out.device != device:
            raise ValueError(f"out must be on {device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError(
                "out must be contiguous; backend='cake' makes no host copy"
            )
        row_elems = num_heads * _HEAD_DIM
        if out.numel() % row_elems or out.numel() < num_query_tokens * row_elems:
            raise ValueError(
                f"out must hold at least {num_query_tokens} rows of [{num_heads}, {_HEAD_DIM}], "
                f"got {tuple(out.shape)}"
            )
    query_rows = query[:num_query_tokens]

    swa = _dense_rows(swa_kv_cache, "swa_kv_cache", query.dtype)
    compressed = _dense_rows(compressed_kv_cache, "compressed_kv_cache", query.dtype)
    seq_lens = _int32_vector(seq_lens, "seq_lens", device)
    batch_size = seq_lens.numel()
    ragged = cum_seq_lens_q is not None
    if ragged:
        cum_seq_lens_q = _int32_vector(cum_seq_lens_q, "cum_seq_lens_q", device)
    scale1 = _device_scale(bmm1_scale, device=device, name="bmm1_scale")
    scale2 = _device_scale(bmm2_scale, device=device, name="bmm2_scale")
    if sinks is not None:
        if (
            sinks.dtype != torch.float32
            or sinks.device != device
            or not sinks.is_contiguous()
        ):
            raise ValueError(f"sinks must be a contiguous FP32 tensor on {device}")
    has_sinks = int(sinks is not None)
    sink_tensor = sinks if sinks is not None else scale1

    if workspace_buffer.device != device:
        raise ValueError(
            f"workspace_buffer must be on {device}, got {workspace_buffer.device}"
        )
    raw = _workspace_bytes(workspace_buffer)
    route = _route(
        dtype=query.dtype,
        num_heads=num_heads,
        max_q_len=max_q_len,
        ragged=ragged,
        sparse_topk=sparse_topk,
        batch_size=batch_size,
        compressed_page_size=compressed_kv_cache.shape[-2],
        num_query_tokens=meta.num_query_tokens,
    )

    if out is None:
        out = allocate_cake_dsv4_output(
            tuple(out_shape)
            if out_shape is not None
            else (query_capacity, num_heads, _HEAD_DIM),
            device,
            phase=cake_dsv4_out_phase(
                route, num_heads=num_heads, sparse_topk=sparse_topk
            ),
        )
        if out.numel() < num_query_tokens * num_heads * _HEAD_DIM:
            raise ValueError(
                f"out_shape {tuple(out.shape)} holds fewer than {num_query_tokens} rows"
            )
    out_rows = out.view(-1, num_heads, _HEAD_DIM)[:num_query_tokens]

    if not ragged and batch_size * max_q_len < num_query_tokens:
        raise ValueError(
            f"dense query layout [batch_size={batch_size}, max_q_len={max_q_len}] "
            f"holds {batch_size * max_q_len} rows but the metadata describes "
            f"{num_query_tokens}"
        )
    if ragged:
        request_offsets: Optional[torch.Tensor] = cum_seq_lens_q
        cum_seq_lens_q_value = cum_seq_lens_q
    else:
        # Dense rows: the kernels map t -> (t // max_q_len, t % max_q_len) and
        # never read cum_seq_lens_q, which is bound to the seq_lens pointer.
        # The ragged-only producers read request boundaries unconditionally
        # and receive the cached dense offsets (every request max_q_len long).
        request_offsets = None
        cum_seq_lens_q_value = (
            _dense_query_offsets(batch_size, max_q_len, device=device)
            if route in _RAGGED_ONLY_ROUTES
            else seq_lens
        )

    is_fp8 = query.dtype == torch.float8_e4m3fn
    values: dict[str, Any] = {
        "Q": query_rows.view(torch.uint8) if is_fp8 else query_rows,
        "SWA_cache": swa.view(torch.uint8) if is_fp8 else swa,
        "compressed_KV_cache": compressed.view(torch.uint8) if is_fp8 else compressed,
        "O": out_rows,
        "seq_lens": seq_lens,
        "cum_seq_lens_q": cum_seq_lens_q_value,
        "sinks": sink_tensor,
        "bmm1_scale": scale1,
        "bmm2_scale": scale2,
        **meta.kernel_kwargs(),
        "sparse_indices": meta.legacy_combined_table,
        "num_heads": num_heads,
        "num_head_tiles": _ceil_div(num_heads, 64),
        "has_sinks": has_sinks,
        "batch_size": batch_size,
        "max_q_len": max_q_len,
        "ragged_query": int(ragged),
        "num_splits": 1,
        "total_work_items": num_query_tokens,
    }
    plan = _route_plan(
        route,
        num_query_tokens=num_query_tokens,
        num_heads=num_heads,
        sparse_topk=sparse_topk,
    )
    rows_per_launch = _rows_per_launch(
        plan,
        num_query_tokens=num_query_tokens,
        num_heads=num_heads,
        batch_size=batch_size,
        workspace_bytes=raw.numel(),
    )
    if rows_per_launch < 1:
        minimum = _launch_workspace_bytes(plan, 1, num_heads, batch_size + 1)
        raise ValueError(
            f"workspace_buffer holds {raw.numel()} bytes; the CAKE DSv4 route "
            f"{route} needs at least {minimum} bytes for one metadata row "
            f"({num_heads} heads, {plan.num_splits} splits); size it with "
            "get_cake_dsv4_workspace_bytes() or cake_dsv4_workspace_requirement()"
        )
    # One host-side critical section per call: descriptor-pool bookkeeping,
    # the bindings' descriptor checks and the launch enqueues (see
    # _descriptor_storage).
    with _descriptor_lock:
        if rows_per_launch >= num_query_tokens:
            _dispatch_route(
                route,
                _Launcher(
                    arch=arch, workspace=workspace_buffer, raw=raw, values=values
                ),
            )
        else:
            if request_offsets is None:
                request_offsets = _dense_query_offsets(
                    batch_size, max_q_len, device=device
                )
            _dispatch_row_tiles(
                route,
                arch=arch,
                workspace=workspace_buffer,
                raw=raw,
                values=values,
                meta=meta,
                plan=plan,
                rows_per_launch=rows_per_launch,
                request_offsets=request_offsets,
            )
    return out


def _dispatch_row_tiles(
    route: str,
    *,
    arch: str,
    workspace: torch.Tensor,
    raw: torch.Tensor,
    values: Mapping[str, Any],
    meta: SparseMetadata,
    plan: _RoutePlan,
    rows_per_launch: int,
    request_offsets: torch.Tensor,
) -> None:
    """Launch ``route`` over consecutive metadata-row chunks that fit the workspace.

    Every chunk sees row-sliced views of ``Q``, ``O`` and the metadata tables
    (no copies; the index tables keep their row strides) and its own row
    count. Chunks after the first cannot use the caller's layout (the kernels
    index rows from 0), so they run as ragged rows whose request boundaries
    are ``request_offsets - first_row`` (all ``batch_size + 1`` entries; the
    kernels' scan tolerates negative entries of the requests before the
    chunk), written into the workspace's offsets region in stream order --
    one captured ``torch.sub`` per chunk, nothing allocated.
    """
    total = values["num_query_tokens"]
    num_heads = values["num_heads"]
    num_offsets = int(request_offsets.numel())
    legacy = values["sparse_indices"]
    for first in range(0, total, rows_per_launch):
        rows = min(rows_per_launch, total - first)
        stop = first + rows
        chunk: dict[str, Any] = dict(values)
        chunk.update(
            Q=values["Q"][first:stop],
            O=values["O"][first:stop],
            swa_indices=_flat_index_span(meta.swa_indices[first:stop]),
            compressed_indices=_flat_index_span(meta.compressed_indices[first:stop]),
            sparse_topk_lens=meta.sparse_topk_lens[first:stop],
            sparse_indices=None if legacy is None else legacy[first:stop],
            num_query_tokens=rows,
            total_work_items=rows,
        )
        if first:
            layout = cake_dsv4_workspace_layout(
                rows, num_heads, plan.num_splits, num_query_offsets=num_offsets
            )
            begin = layout.query_offsets[0]
            shifted = raw[begin : begin + num_offsets * torch.int32.itemsize].view(
                torch.int32
            )
            torch.sub(request_offsets, first, out=shifted)
            chunk.update(cum_seq_lens_q=shifted, ragged_query=1)
        _dispatch_route(
            route, _Launcher(arch=arch, workspace=workspace, raw=raw, values=chunk)
        )


def _dispatch_route(route: str, L: _Launcher) -> None:
    v = L.values
    T = v["num_query_tokens"]
    H = v["num_heads"]
    topk = v["sparse_topk"]
    plan = _route_plan(route, num_query_tokens=T, num_heads=H, sparse_topk=topk)

    if route in (
        "bf16_swa128_single_cta",
        "bf16_h128_swa128",
        "bf16_h8_swa128_v43",
        "bf16_h16_h32_swa128_v44",
    ):
        head_tiles = _ceil_div(H, 64) if route == "bf16_h128_swa128" else 1
        L.run(
            L.variant(route, grid=(T * head_tiles * 4, 1, 1), num_head_tiles=head_tiles)
        )
        return

    if route == "bf16_h8_h16_source_exact":
        L.run(L.variant(route, grid=(v["max_q_len"], (H // 8) * 4, v["batch_size"])))
        return

    if route == "fp8_h8_h16_source_exact":
        # Same launch shape as the BF16 source-exact body (one CTA per
        # (query-within-sequence, value quarter, batch)); FP8 Q/KV pools.
        L.run(L.variant(route, grid=(v["max_q_len"], (H // 8) * 4, v["batch_size"])))
        return

    if route == "bf16_h64_guard_q_tma_batch_r25":
        L.run(L.variant(route, grid=(T, 2, 1)))
        return

    if route == "bf16_h64_compressed_q8_v38":
        num_splits = plan.num_splits
        parts = L.partials(num_splits)
        compressed = L.variant(route, grid=(T * num_splits * 2, 1, 1), **parts)
        if num_splits > 1:
            L.run(compressed, L.reduce("bf16_h64_compressed_reduce", **parts))
        else:
            L.run(compressed)
        return

    if route == "bf16_h32_topk128x_early_v47":
        num_splits = plan.num_splits
        head_tiles = plan.merge_groups_per_row
        parts = L.partials(num_splits)
        arrivals = L.counters(T * head_tiles)
        L.run(
            L.variant(
                route,
                grid=(T * num_splits * head_tiles, 1, 1),
                partition_arrivals=arrivals,
                num_head_tiles=head_tiles,
                **parts,
            )
        )
        return

    if route == "bf16_h64_prefill":
        L.run(L.variant(route, grid=(T, 1, 1), total_work_items=T))
        return

    if route in ("bf16_h128_topk128x", "bf16_h128_topk4x_v52", "bf16_h128_prefill_v42"):
        # The BF16/H128 producers launch directly with the grids their former
        # single-route family libraries computed: two CTAs per work item.
        producer = route
        num_splits = plan.num_splits
        grid_x = 2 * T
        if route == "bf16_h128_prefill_v42":
            # Persistent KV-reuse body: one two-CTA cluster per item, at most
            # half the SMs in clusters; the boustrophedon feed of the same body
            # serves the tail-majority rows (see the predicate above:
            # hardening-000037 0.77-0.90x -> 0.98-1.01x, hardening-000027
            # +8-10 % vs the striped feed).
            clusters = _bf16_h128_prefill_num_clusters(v["Q"].device)
            if _bf16_h128_prefill_uses_snake_feed(T, topk, clusters):
                producer = "bf16_h128_prefill_v42_snake"
            grid_x = min(2 * T, 2 * clusters)
        elif route == "bf16_h128_topk4x_v52":
            # Five fixed full-V KV owners per token plus the split-5 reducer
            # (plan.num_splits == 5).
            pass
        elif T > _BF16_TOPK128X_SPLIT_MAX_TOKENS:
            # Above the token bound one full-V owner per token whose invalid
            # (-1) sparse rows gather the tile's first index
            # (hardening-000025/31 0.45-0.95x -> 1.13-1.65x).  While four
            # CTAs per token still fit one wave, each token runs two 2-CTA
            # clusters that gather the full K tiles and only their 256-column
            # V half (same MMA operands and order per element -> identical
            # bits; -1.1 us / -8 % on the 32-token row on both targets): grid
            # = 4 * tokens, total_work_items keeps meaning query tokens.
            # Mirrors the Cake seed rule bf16_topk128x_uses_row_first_vsplit.
            if T <= _BF16_ROW_FIRST_V_HALF_SPLIT_MAX_TOKENS:
                producer = "bf16_h128_topk128x_row_first_vsplit"
                grid_x = 4 * T
            else:
                producer = "bf16_h128_topk128x_row_first"
        else:
            # Two-stage split program on both Blackwell targets: four disjoint
            # full-V KV owners (a three-tile row runs with the fourth tile
            # masked: 12.3-13.0 us vs 14.8 us for a three-owner pair) plus one
            # LSE reducer; GB300 width 260/388 rows measured 1.18-1.26x vs
            # trtllm-gen against 0.83-1.05x for the single-owner kernel
            # (plan.num_splits == 4).
            producer = "bf16_h128_topk128x_split4_sm100"
        parts = L.partials(num_splits)
        if num_splits == 1:
            L.run(L.variant(producer, grid=(grid_x, 1, 1), total_work_items=T, **parts))
            return
        # The split producers write their per-owner outputs through ``O``.
        work_items = T * num_splits
        split = L.variant(
            producer,
            grid=(2 * work_items, 1, 1),
            total_work_items=work_items,
            **{**parts, "O": parts["partial_O"]},
        )
        if num_splits == 5:
            # The two split-5 reducers are different kernels with different
            # launch contracts (copied from their former family libraries):
            # the sm_103a body runs grid (tokens, heads / 4), the sm_100a
            # body grid (tokens, heads).
            reducer_heads = H // 4 if L.arch == "sm_103a" else H
            reduce = L.variant(
                "bf16_h128_split5_reduce", grid=(T, reducer_heads, 1), **parts
            )
        else:
            reduce = L.reduce("split_reduce", **parts)
        L.run(split, reduce)
        return

    if route == "fp8_h64_source_exact":
        head_tiles = _ceil_div(H, 64)
        L.run(
            L.variant(
                route,
                grid=(v["max_q_len"], head_tiles, v["batch_size"]),
                num_head_tiles=head_tiles,
                total_work_items=v["max_q_len"] * head_tiles * v["batch_size"],
            )
        )
        return

    if route in (
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
    ):
        parts = L.partials(1)
        L.run(L.variant(route, grid=(T * 2, 1, 1), total_work_items=T, **parts))
        return

    if route in (
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
    ):
        # One CTA per token (cta_group::1, M = 64); same kernel kwargs as the
        # FP8/H128 persistent body (num_heads runtime, -1 masking, padded rows,
        # caller-owned workspace), grid = tokens.  Both M64 programs share the
        # kernel ABI; _fp8_h64_m64_program picks one by the item width.
        parts = L.partials(1)
        L.run(L.variant(route, grid=(T, 1, 1), total_work_items=T, **parts))
        return

    if route in (
        "fp8_lowhead_one_partition",
        "fp8_lowhead_h64",
        "fp8_lowhead_prefill",
    ):
        # One producer partition writes the final output directly.  The
        # prefill producer runs one CTA per work item over two items per
        # token; the decode producers run one two-CTA cluster per token.
        work_items = 2 * T if route == "fp8_lowhead_prefill" else T
        L.run(
            L.variant(
                route, grid=(2 * T, 1, 1), total_work_items=work_items, **L.partials(1)
            )
        )
        return

    raise RuntimeError(f"unhandled CAKE DSv4 route: {route}")


__all__ = [
    "KERNEL_METADATA_PARAMS",
    "QUERY_LAYOUT_PARAMS",
    "SparseMetadata",
    "WorkspaceLayout",
    "WorkspaceRequirement",
    "cake_dsv4_workspace_layout",
    "cake_dsv4_workspace_requirement",
    "cake_dsv4_workspace_reset",
    "canonical_arg_name",
    "get_cake_dsv4_workspace_bytes",
    "is_bindable_arg",
    "resolve_cake_dsv4_sparse_metadata",
    "run_cake_dsv4",
]
