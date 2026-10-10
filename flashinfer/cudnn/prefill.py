import contextlib
import functools
import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.attention import cudnn_batch_prefill_trace
from ..utils import check_lse_base, get_compute_capability, ln2, log2e
from .utils import (
    get_cudnn_fmha_gen_module,
    get_cudnn_attention_handle,
    supports_ordered_cudnn_execution,
    supports_native_cudnn_log2,
    build_cudnn_graph_with_log2,
    require_native_cudnn_log2,
)

try:
    import cudnn

    CUDNN_AVAILABLE = True
except Exception:
    cudnn = None
    CUDNN_AVAILABLE = False


# Older FROST bindings specialize HN head stride on the declared token count.
# Keep their established NH staging path: native HN must also preserve graph
# reuse when serving changes packed token totals inside an override bucket.
_CUDNN_NATIVE_HN_SUPPORTED = bool(
    getattr(
        getattr(getattr(cudnn, "_pybind_module", None), "_SdpaThdBinder", None),
        "supports_stats_stride_override",
        False,
    )
)


@functools.cache
def _cudnn_supports_direct_seqlens(dtype: torch.dtype, *, mixed: bool = False) -> bool:
    """True if cuDNN can consume token-unit indptr buffers directly for `dtype`.

    Requires cu_seq_len_q/kv SDPA inputs and per-tensor ragged-offset
    multipliers on the unified SDPA engine (forward only):
    - fp16/bf16: cuDNN backend 9.24+ with cudnn-frontend 1.25+
    - fp8 (e4m3/e5m2): cuDNN backend 9.25+ with cudnn-frontend 1.27+ (the first
      release whose sdpa_fp8 python binding exposes cu_seq_len_q/kv)

    `mixed=True` additionally requires *mixed-form* sequence lengths (cu_seq_len
    on one side, per-batch on the other). This is the paged path, which pairs a
    token-unit cu_seq_len_q with an actual-length seq_len_kv (KV addressed via
    the page table). It needs cuDNN backend 9.25+ and a cudnn-frontend carrying
    the per-side relaxation (frontend PR #430; released in 1.27+).

    Note: this is a pure version compare against the runtime backend and the FE
    package version (the practical proxy for the FE's compiled-against cuDNN).
    The FE exposes no compiled/effective-version query, so a feature-probe with
    NOT_SUPPORTED fallback is left as a follow-up.
    """
    if not CUDNN_AVAILABLE:
        return False
    if mixed or (dtype in (torch.float8_e4m3fn, torch.float8_e5m2)):
        min_backend, min_frontend = 92500, (1, 27)
    elif dtype in (torch.float16, torch.bfloat16):
        min_backend, min_frontend = 92400, (1, 25)
    else:
        return False
    try:
        if cudnn.backend_version() < min_backend:
            return False
        major, minor = map(int, cudnn.__version__.split(".")[:2])
        return (major, minor) >= min_frontend
    except Exception:
        return False


# Execute-time shape override for the ragged (token-indptr) path. cuDNN
# finalizes one execution plan per declared (batch, max_seq_len) -- 55-70 ms on
# SM100, ~1 s when a kernel variant compiles -- and serving changes both every
# step. With override the graph is built once at a "cache shape" and every
# execute passes the real (b, s_q, s_kv) through override_uids/shapes/strides:
# no per-shape plan, no padding rows. q and kv lengths are classed as 1, <=128
# or 65536+: s_q == 1 is cuDNN's decode kernel (an override may not cross that
# boundary), and the heuristic picks the short-row engine for a declared
# max_len <= 128 and another engine above (flip measured between 128 and 256 on
# SM100 and SM107, independent of batch, LSE and head dims). Within a class
# override matches a natively built plan. Bounded SM100/SM107 MLA and D128 use
# smaller power-of-two classes so plan-time occupancy and partial workspace
# bounds remain useful without specializing every live length.
_PREFILL_SHAPE_OVERRIDE_ENV = "FLASHINFER_CUDNN_PREFILL_SHAPE_OVERRIDE"
_OVERRIDE_SHORT_SEQ = 128
_OVERRIDE_CACHE_SEQ_LONG = 65536
_OVERRIDE_CACHE_BATCH = 4096


@functools.cache
def _cudnn_single_token_gqa_ragged_stats_broken() -> bool:
    """True if cuDNN's single-token (s_q == 1) decode-class kernel mis-stores a
    ragged Stats tensor when h_qo != h_kv (NVBug 6783545): it used the sequence
    stride for the per-head tile rows, so most heads of a packed LSE were never
    written. The output is unaffected. Fixed in cuDNN 9.27.0."""
    if not CUDNN_AVAILABLE:
        return False
    return cudnn.backend_version() < 92700


def _cudnn_version_supports_shape_override() -> bool:
    """SDPA shape override needs the unified engine's support (cuDNN 9.22+) and a
    cudnn-frontend whose pygraph takes is_override_shape_enabled (1.29+)."""
    if not CUDNN_AVAILABLE:
        return False
    try:
        if cudnn.backend_version() < 92200:
            return False
        major, minor = map(int, cudnn.__version__.split(".")[:2])
        return (major, minor) >= (1, 29)
    except Exception:
        return False


@functools.cache
def _cudnn_supports_bounded_ragged(*, d128: bool = False) -> bool:
    """Packed capacity declarations require a matching FE Python/native stack."""
    if not CUDNN_AVAILABLE:
        return False
    try:
        version = tuple(map(int, cudnn.__version__.split(".")[:2]))
        binder = getattr(getattr(cudnn, "_pybind_module", None), "_SdpaThdBinder", None)
        return version >= (1, 31) and bool(
            getattr(
                binder,
                "supports_nonpaged_d128_packed_split"
                if d128
                else "supports_nonpaged_packed_split",
                False,
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _cudnn_supports_shape_override() -> bool:
    if os.environ.get(_PREFILL_SHAPE_OVERRIDE_ENV, "1") == "0":
        return False
    return _cudnn_version_supports_shape_override()


def _override_seq_class(max_seq: int, *, is_q: bool) -> int:
    s = max(int(max_seq), 1)
    if is_q and s == 1:
        # cuDNN builds a distinct decode kernel for s_q == 1 and rejects an
        # override that crosses the s_q == 1 boundary in either direction.
        return 1
    if s <= _OVERRIDE_SHORT_SEQ:
        return _OVERRIDE_SHORT_SEQ
    return max(_OVERRIDE_CACHE_SEQ_LONG, 1 << (s - 1).bit_length())


def _override_cache_shape(
    batch_size: int, max_seq_q: int, max_seq_kv: int, *, bounded_ragged: bool = False
) -> tuple[int, int, int]:
    """Declared (batch, s_q, s_kv) of the override graph for a real (b, s_q,
    s_kv). q and kv are classed separately (a short-q / long-kv step must not
    be declared as short kv). Grows by powers of two when a caller exceeds the
    defaults; that changes the cache key and builds one more plan."""
    # A bounded declaration lets either FE provider reserve packed partials and
    # select an underfilled packed launch. Powers of two retain graph reuse across
    # steps; the packed-Q capacity follows the declaration, not the live total.
    # Q=1 stays in the distinct decode class; shorter prefill shares Q<=128.
    if (
        bounded_ragged
        and 1 <= batch_size <= 4
        and 2 <= max_seq_q <= 1024
        and 2048 <= max_seq_kv <= 32768
        and 4 * max_seq_q <= max_seq_kv
    ):
        cache_b, cache_q, cache_kv = (
            1 << (int(n) - 1).bit_length()
            for n in (batch_size, max(128, max_seq_q), max_seq_kv)
        )
        return cache_b, cache_q, cache_kv
    cache_b = max(
        _OVERRIDE_CACHE_BATCH, 1 << (max(int(batch_size), 1) - 1).bit_length()
    )
    return (
        cache_b,
        _override_seq_class(max_seq_q, is_q=True),
        _override_seq_class(max_seq_kv, is_q=False),
    )


# Workspace bytes a built graph needs, memoized by the graph-cache key (the
# same tuple `_sdpa_prefill_key_fn` builds): the size is a function of the
# declared shape, so the memo stays valid when the FE cache evicts and rebuilds
# a graph, whereas an id(graph) key could be reused by a later object. The
# override graphs reserve per-declared-batch TMA descriptors (~1 MiB at batch
# 4096), so a caller's small workspace is checked before use.
_graph_workspace_bytes: dict[tuple, int] = {}


def _graph_workspace_size(graph, key: tuple) -> int:
    size = _graph_workspace_bytes.get(key)
    if size is None:
        size = int(graph.get_workspace_size())
        _graph_workspace_bytes[key] = size
    return size


# Graph builds (cache misses) since import; read by tests.
_prefill_graph_builds = 0


_dummy_scale_tensors: dict[torch.device, torch.Tensor] = {}


def _get_dummy_scale_tensor(device: torch.device):
    t = _dummy_scale_tensors.get(device)
    if t is None:
        t = torch.tensor([1.0], device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
        _dummy_scale_tensors[device] = t
    return t


def _create_cudnn_handle(stream: torch.cuda.Stream):
    return get_cudnn_attention_handle(cudnn, stream)


# Tensor ids
class UIDs(Enum):
    RESERVED_INVALID_UID = 0

    Q_UID = 1  # Query tensor
    K_UID = 2  # Key cache tensor
    V_UID = 3  # Value cache tensor

    ACTUAL_SEQ_LENS_Q_UID = 100  # Actual sequence lengths for query tensor
    ACTUAL_SEQ_LENS_KV_UID = 101  # Actual sequence lengths for key/value tensor

    BLOCK_TABLES_UID = 200  # Block tables tensor
    BLOCK_TABLES_K_UID = 201  # Block tables tensor for key
    BLOCK_TABLES_V_UID = 202  # Block tables tensor for value

    RAGGED_Q_UID = 50  # Ragged query tensor
    RAGGED_O_UID = 51  # Ragged output tensor
    RAGGED_STATS_UID = 52  # Ragged stats tensor
    RAGGED_K_UID = 53  # Ragged key tensor
    RAGGED_V_UID = 54  # Ragged value tensor

    O_UID = 1000  # Output tensor
    STATS_UID = 1001  # Stats tensor

    Q_SCALE_UID = 150  # Query scale tensor
    K_SCALE_UID = 151  # Key scale tensor
    V_SCALE_UID = 152  # Value scale tensor
    S_SCALE_UID = 153  # Scale tensor
    S_DESCALE_UID = 154  # Descale tensor
    O_SCALE_UID = 155  # Output scale tensor

    S_AMAX_UID = 160  # Scale amax tensor
    O_AMAX_UID = 161  # Output amax tensor


def _prefill_runtime_key(q, k_cache, v_cache, scale, o_data_type):
    # Only execute-time tensors belong here. Plan metadata is keyed once by
    # _prefill_descriptor_key; pointer values never select a graph.
    return (
        q.device,
        q.dtype,
        q.dtype if o_data_type is None else o_data_type,
        q.dim(),
        q.shape[1:],
        k_cache.dtype,
        v_cache.dtype,
        k_cache.shape if k_cache.dim() == 4 else k_cache.shape[1:],
        v_cache.shape if v_cache.dim() == 4 else v_cache.shape[1:],
        q.stride(),
        k_cache.stride(),
        v_cache.stride(),
        scale,
    )


def _prefill_descriptor_key(
    *,
    max_token_seq_q: Optional[int] = None,
    max_sequence_kv: Optional[int] = None,
    actual_seq_lens_q: Optional[torch.Tensor] = None,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    cu_seq_lens_q: Optional[torch.Tensor] = None,
    cu_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    bottom_right_causal_mask: Optional[bool] = None,
    return_lse: Optional[bool] = False,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    batch_offsets_k: Optional[torch.Tensor] = None,
    batch_offsets_v: Optional[torch.Tensor] = None,
    batch_offsets_stats: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    o_data_type: Optional[torch.dtype] = None,
    override_cache: Optional[tuple[int, int, int]] = None,
    max_total_num_rows: Optional[int] = None,
):
    if actual_seq_lens_q is not None:
        graph_b = actual_seq_lens_q.shape[0]
    elif cu_seq_lens_q is not None:
        graph_b = cu_seq_lens_q.shape[0] - 1
    else:
        raise ValueError("Either actual_seq_lens_q or cu_seq_lens_q must be provided")
    layouts = {}

    def layout(t):
        if t is None:
            return None
        if id(t) not in layouts:
            layouts[id(t)] = (tuple(t.shape), tuple(t.stride()), t.dtype)
        return layouts[id(t)]

    core = (
        (graph_b, max_token_seq_q, max_sequence_kv, max_total_num_rows),
        bottom_right_causal_mask,
        False,
        layout(block_tables),
        layout(actual_seq_lens_q) if cu_seq_lens_q is None else None,
        layout(actual_seq_lens_kv) if cu_seq_lens_kv is None else None,
        tuple(
            layout(t)
            for t in (
                cu_seq_lens_q,
                cu_seq_lens_kv,
                batch_offsets_q,
                batch_offsets_o,
                batch_offsets_k,
                batch_offsets_v,
            )
        ),
    )
    key = (core, (return_lse, layout(batch_offsets_stats) if return_lse else None))
    return (
        _prefill_override_descriptor_key(key, override_cache)
        if override_cache is not None
        else key
    )


def _prefill_graph_total_q(override_cache, max_total_num_rows):
    if override_cache is None:
        return max_total_num_rows
    if override_cache[0] >= _OVERRIDE_CACHE_BATCH:
        # Preserve the broad fallback's existing declaration and cache domain.
        return None
    capacity = override_cache[0] * override_cache[1]
    return (
        min(capacity, max_total_num_rows)
        if max_total_num_rows is not None
        else capacity
    )


def _prefill_override_descriptor_key(key, override_cache):
    core, (return_lse, stats) = key
    shape, causal, _, table, seq_q, seq_kv, offsets = core
    total_q = _prefill_graph_total_q(override_cache, shape[3])

    def indptr_layout(layout):
        if layout is None:
            return None
        # Match indptr_tensor's declaration at the cache batch size.
        return ((override_cache[0] + 1, 1, 1, 1), (1, 1, 1, 1), layout[2])

    return (
        (
            (*override_cache, total_q),
            causal,
            True,
            table,
            seq_q,
            seq_kv,
            tuple(indptr_layout(t) for t in offsets),
        ),
        (return_lse, indptr_layout(stats)),
    )


def _sdpa_prefill_key_fn(
    q,
    k_cache,
    v_cache,
    scale,
    *,
    o_data_type=None,
    stats_head_stride=0,
    stats_use_log2=False,
    workspace_limit=None,
    **metadata,
):
    return (
        _prefill_runtime_key(q, k_cache, v_cache, scale, o_data_type),
        _prefill_descriptor_key(**metadata),
        bool(stats_head_stride)
        if metadata.get("override_cache") is not None and _CUDNN_NATIVE_HN_SUPPORTED
        else stats_head_stride,
        stats_use_log2,
        workspace_limit,
    )


if CUDNN_AVAILABLE:

    @cudnn.graph_cache(key_fn=_sdpa_prefill_key_fn)
    def _build_prefill_graph(*args, stats_use_log2=False, **kwargs):
        return build_cudnn_graph_with_log2(
            cudnn, _make_prefill_graph, args, kwargs, stats_use_log2
        )

    @cudnn.jit(heur_modes=[cudnn.heur_mode.A])
    def _make_prefill_graph(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        scale: float,
        *,
        max_token_seq_q: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
        actual_seq_lens_q: Optional[torch.Tensor] = None,
        actual_seq_lens_kv: Optional[torch.Tensor] = None,
        cu_seq_lens_q: Optional[torch.Tensor] = None,
        cu_seq_lens_kv: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        bottom_right_causal_mask: Optional[bool] = True,
        return_lse: Optional[bool] = False,
        batch_offsets_q: Optional[torch.Tensor] = None,
        batch_offsets_o: Optional[torch.Tensor] = None,
        batch_offsets_k: Optional[torch.Tensor] = None,
        batch_offsets_v: Optional[torch.Tensor] = None,
        batch_offsets_stats: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        o_data_type: Optional[torch.dtype] = None,
        override_cache: Optional[tuple[int, int, int]] = None,
        max_total_num_rows: Optional[int] = None,
        stats_head_stride: int = 0,
        stats_use_log2: bool = False,
        workspace_limit: Optional[int] = None,
    ):
        global _prefill_graph_builds
        _prefill_graph_builds += 1
        handle = _create_cudnn_handle(torch.cuda.current_stream(q.device))

        # cu_seq_lens_q/kv select the direct path: cuDNN consumes (b+1)-shaped
        # token-unit prefix sums for the padding mask, and the batch_offsets_*
        # ragged offsets are token-unit indptrs scaled by per-tensor
        # multipliers. Mutually exclusive with actual_seq_lens_q/kv for the
        # mask role; dtype/version support is the caller's responsibility
        # (_cudnn_supports_direct_seqlens).
        # cu_seq_lens_q selects the direct (token-unit) path for Q: the buffer
        # feeds the padding mask and, via batch_offsets_q, the Q/O ragged
        # offsets. The KV side chooses its form independently -- cumulative
        # (cu_seq_lens_kv, non-paged) or per-batch (actual_seq_lens_kv, paged).
        # The paged case is the *mixed* form (cu_seq_len_q + seq_len_kv); KV is
        # addressed through block_tables and carries no ragged offset. Mixed
        # forms need cuDNN 9.25+ (_cudnn_supports_direct_seqlens(mixed=True)).
        use_cu_seq_lens = cu_seq_lens_q is not None
        if use_cu_seq_lens:
            assert batch_offsets_q is not None, (
                "cu_seq_lens_q requires token-unit batch_offsets_q"
            )
            if block_tables is None:
                assert cu_seq_lens_kv is not None and batch_offsets_k is not None, (
                    "non-paged cu_seq_lens requires cu_seq_lens_kv and token-unit "
                    "batch_offsets_k"
                )
            else:
                assert (
                    actual_seq_lens_kv is not None
                    and cu_seq_lens_kv is None
                    and batch_offsets_k is None
                ), (
                    "paged cu_seq_lens requires actual_seq_lens_kv and no "
                    "cu_seq_lens_kv / batch_offsets_k (KV is paged)"
                )

        if actual_seq_lens_q is not None:
            graph_b = actual_seq_lens_q.shape[0]
        elif cu_seq_lens_q is not None:
            graph_b = cu_seq_lens_q.shape[0] - 1
        else:
            raise ValueError(
                "Either actual_seq_lens_q or cu_seq_lens_q must be provided"
            )
        graph_s_qo = max_token_seq_q
        graph_s_kv = max_sequence_kv
        graph_total_q = _prefill_graph_total_q(override_cache, max_total_num_rows)
        if override_cache is not None:
            graph_b, graph_s_qo, graph_s_kv = override_cache

        def indptr_tensor(like: torch.Tensor):
            # (b+1)-row int32 buffers (ragged offsets, cu_seq_lens). Declared at
            # the cache batch under override, else shaped like the caller's.
            if override_cache is None:
                return g.tensor_like(like)
            return g.tensor(
                dim=(graph_b + 1, 1, 1, 1),
                stride=(1, 1, 1, 1),
                data_type=cudnn.datatypes._torch_to_cudnn_data_type(like.dtype),
            )

        if not cudnn.datatypes.is_torch_available():
            raise RuntimeError("torch is not available")

        cudnn_q_data_type = cudnn.datatypes._torch_to_cudnn_data_type(q.dtype)
        cudnn_k_data_type = cudnn.datatypes._torch_to_cudnn_data_type(k_cache.dtype)
        cudnn_v_data_type = cudnn.datatypes._torch_to_cudnn_data_type(v_cache.dtype)

        if o_data_type is None:
            o_data_type = q.dtype

        cudnn_o_data_type = cudnn.datatypes._torch_to_cudnn_data_type(o_data_type)

        if (
            cudnn_q_data_type == cudnn.data_type.FP8_E4M3
            or cudnn_q_data_type == cudnn.data_type.FP8_E5M2
        ) and cudnn.backend_version() < 91701:
            raise RuntimeError(
                f"FP8 is not supported in cuDNN backend version < 9.17.1, current version is {cudnn.backend_version()}"
            )

        graph_ctx: contextlib.AbstractContextManager
        if override_cache is not None:
            graph_ctx = contextlib.nullcontext(
                (
                    cudnn.pygraph(
                        name="cudnn_graph",
                        io_data_type=cudnn.data_type.HALF,
                        intermediate_data_type=cudnn.data_type.FLOAT,
                        compute_data_type=cudnn.data_type.FLOAT,
                        handle=handle,
                        is_override_shape_enabled=True,
                    ),
                    [],
                )
            )
        else:
            graph_ctx = cudnn.graph(handle)
        with graph_ctx as (g, _):
            # Create tensors from the input tensors
            if q.dim() == 3:
                h_qo, d_qk = q.shape[1], q.shape[2]
                s_stride, h_stride, d_stride = q.stride()
            elif q.dim() == 4:
                h_qo, d_qk = q.shape[2], q.shape[3]
                s_stride, h_stride, d_stride = q.stride()
            else:
                raise ValueError(f"Invalid query tensor shape: {q.shape}")

            q_token_stride = s_stride
            cudnn_q = g.tensor(
                name="q",
                dim=(graph_b, h_qo, graph_s_qo, d_qk),
                stride=(h_qo * d_qk, h_stride, s_stride, d_stride),
                data_type=cudnn_q_data_type,
            )

            if (
                cudnn_q_data_type == cudnn.data_type.FP8_E4M3
                or cudnn_q_data_type == cudnn.data_type.FP8_E5M2
            ):
                cudnn_q_scale = g.tensor(
                    name="q_scale",
                    dim=(1, 1, 1, 1),
                    stride=(1, 1, 1, 1),
                    data_type=cudnn.data_type.FLOAT,
                )

                cudnn_k_scale = g.tensor(
                    name="k_scale",
                    dim=(1, 1, 1, 1),
                    stride=(1, 1, 1, 1),
                    data_type=cudnn.data_type.FLOAT,
                )

                cudnn_v_scale = g.tensor(
                    name="v_scale",
                    dim=(1, 1, 1, 1),
                    stride=(1, 1, 1, 1),
                    data_type=cudnn.data_type.FLOAT,
                )

                cudnn_s_scale = g.tensor(
                    name="s_scale",
                    dim=(1, 1, 1, 1),
                    stride=(1, 1, 1, 1),
                    data_type=cudnn.data_type.FLOAT,
                )

                cudnn_s_descale = g.tensor(
                    name="s_descale",
                    dim=(1, 1, 1, 1),
                    stride=(1, 1, 1, 1),
                    data_type=cudnn.data_type.FLOAT,
                )

                cudnn_o_scale = g.tensor(
                    name="o_scale",
                    dim=(1, 1, 1, 1),
                    stride=(1, 1, 1, 1),
                    data_type=cudnn.data_type.FLOAT,
                )

                cudnn_q_scale.set_uid(UIDs.Q_SCALE_UID.value)
                cudnn_k_scale.set_uid(UIDs.K_SCALE_UID.value)
                cudnn_v_scale.set_uid(UIDs.V_SCALE_UID.value)
                cudnn_s_scale.set_uid(UIDs.S_SCALE_UID.value)
                cudnn_s_descale.set_uid(UIDs.S_DESCALE_UID.value)
                cudnn_o_scale.set_uid(UIDs.O_SCALE_UID.value)

            if batch_offsets_q is not None:
                ragged_q = indptr_tensor(batch_offsets_q)
                ragged_q.set_uid(UIDs.RAGGED_Q_UID.value)
                cudnn_q.set_ragged_offset(ragged_q)
                if use_cu_seq_lens:
                    # Offsets are token-unit indptrs; the engine scales them
                    # back to elements with the tensor's own token stride, so a
                    # non-contiguous q (T3HD: 3*h*d per token) addresses correctly.
                    cudnn_q.set_ragged_offset_multiplier(q_token_stride)

            if v_cache.dim() == 3:
                assert block_tables is None, (
                    "block_tables needs 4 dimensions of kv cache"
                )
                h_kv, d_vo = v_cache.shape[1], v_cache.shape[2]
            elif v_cache.dim() == 4:
                h_kv, d_vo = (
                    v_cache.shape[1],
                    v_cache.shape[3],
                )
            else:
                raise ValueError(f"Invalid kv cache tensor shape: {k_cache.shape}")

            if k_cache.dim() == 3:
                s_stride, h_stride, d_stride = k_cache.stride()
                cudnn_k_cache = g.tensor(
                    name="k_cache",
                    dim=(graph_b, h_kv, graph_s_kv, d_qk),
                    stride=(h_kv * d_qk * graph_s_kv, h_stride, s_stride, d_stride),
                    data_type=cudnn_k_data_type,
                )

                if batch_offsets_k is not None:
                    ragged_k = indptr_tensor(batch_offsets_k)
                    ragged_k.set_uid(UIDs.RAGGED_K_UID.value)
                    cudnn_k_cache.set_ragged_offset(ragged_k)
                    if use_cu_seq_lens:
                        cudnn_k_cache.set_ragged_offset_multiplier(s_stride)

                assert v_cache.dim() == 3, (
                    "v_cache must have 3 dimensions since k_cache has 3 dimensions"
                )
                s_stride, h_stride, d_stride = v_cache.stride()
                cudnn_v_cache = g.tensor(
                    name="v_cache",
                    dim=(graph_b, h_kv, graph_s_kv, d_vo),
                    stride=(h_kv * d_vo * graph_s_kv, h_stride, s_stride, d_stride),
                    data_type=cudnn_v_data_type,
                )

                if batch_offsets_v is not None:
                    ragged_v = indptr_tensor(batch_offsets_v)
                    ragged_v.set_uid(UIDs.RAGGED_V_UID.value)
                    cudnn_v_cache.set_ragged_offset(ragged_v)
                    if use_cu_seq_lens:
                        cudnn_v_cache.set_ragged_offset_multiplier(s_stride)

            elif k_cache.dim() == 4:
                cudnn_k_cache = g.tensor(
                    name="k_cache",
                    dim=k_cache.shape,
                    stride=k_cache.stride(),
                    data_type=cudnn_k_data_type,
                )

                cudnn_v_cache = g.tensor(
                    name="v_cache",
                    dim=v_cache.shape,
                    stride=v_cache.stride(),
                    data_type=cudnn_v_data_type,
                )

            cudnn_q.set_uid(UIDs.Q_UID.value)
            cudnn_k_cache.set_uid(UIDs.K_UID.value)
            cudnn_v_cache.set_uid(UIDs.V_UID.value)

            if block_tables is not None:
                nd_block_tables = block_tables.reshape(
                    block_tables.shape[0], 1, block_tables.shape[1], 1
                )
                cudnn_k_block_tables = g.tensor_like(nd_block_tables)
                cudnn_k_block_tables.set_uid(UIDs.BLOCK_TABLES_K_UID.value)

                cudnn_v_block_tables = g.tensor_like(nd_block_tables)
                cudnn_v_block_tables.set_uid(UIDs.BLOCK_TABLES_V_UID.value)

            if use_cu_seq_lens:
                # cu_seq_len_q occupies the Q seq-len UID slot (mutually
                # exclusive with a per-batch seq_len_q, same role). On the
                # ragged path it is the same buffer as the Q ragged offset.
                cudnn_cu_seq_lens_q = indptr_tensor(cu_seq_lens_q)
                cudnn_cu_seq_lens_q.set_name("cu_seq_lens_q")
                cudnn_cu_seq_lens_q.set_uid(UIDs.ACTUAL_SEQ_LENS_Q_UID.value)

                padding_mask = True
                # These kwargs postdate cudnn-frontend 1.13 (cu_seq_len_*:
                # 1.25+; implementation / attention_implementation: 1.14+). The
                # declared >=1.30 floor covers both, but the runtime cuDNN
                # backend version still gates this path, so they stay confined
                # to it, which _cudnn_supports_direct_seqlens guards.
                seq_len_kwargs = {
                    "cu_seq_len_q": cudnn_cu_seq_lens_q,
                    # cu_seq_lens are unified-engine-only; pin the
                    # implementation so an unsupported config fails with the
                    # unified engine's specific error instead of
                    # auto-selection's generic failure.
                    "implementation": cudnn.attention_implementation.UNIFIED,
                }
                # KV side, independent form. Both tensors take the shared
                # ACTUAL_SEQ_LENS_KV UID; the execute var_map binds cu_seq_lens_kv
                # or actual_seq_lens_kv to it accordingly.
                if cu_seq_lens_kv is not None:
                    # Non-paged: both-cumulative (KV also token-unit ragged).
                    cudnn_cu_seq_lens_kv = indptr_tensor(cu_seq_lens_kv)
                    cudnn_cu_seq_lens_kv.set_name("cu_seq_lens_kv")
                    cudnn_cu_seq_lens_kv.set_uid(UIDs.ACTUAL_SEQ_LENS_KV_UID.value)
                    seq_len_kwargs["cu_seq_len_kv"] = cudnn_cu_seq_lens_kv
                else:
                    # Mixed (paged): per-batch KV lengths mask; KV addressed via
                    # block_tables. This form requires cuDNN 9.25+.
                    cudnn_seq_len_kv = g.tensor_like(actual_seq_lens_kv)
                    cudnn_seq_len_kv.set_name("seq_len_kv")
                    cudnn_seq_len_kv.set_uid(UIDs.ACTUAL_SEQ_LENS_KV_UID.value)
                    seq_len_kwargs["seq_len_kv"] = cudnn_seq_len_kv
            else:
                if actual_seq_lens_q is not None:
                    cudnn_actual_seq_lens_q = g.tensor_like(actual_seq_lens_q)
                    cudnn_actual_seq_lens_q.set_name("actual_seq_lens_q")
                    cudnn_actual_seq_lens_q.set_uid(UIDs.ACTUAL_SEQ_LENS_Q_UID.value)

                if actual_seq_lens_kv is not None:
                    cudnn_actual_seq_lens_kv = g.tensor_like(actual_seq_lens_kv)
                    cudnn_actual_seq_lens_kv.set_name("actual_seq_lens_kv")
                    cudnn_actual_seq_lens_kv.set_uid(UIDs.ACTUAL_SEQ_LENS_KV_UID.value)

                padding_mask = (
                    actual_seq_lens_q is not None and actual_seq_lens_kv is not None
                )
                seq_len_kwargs = {
                    "seq_len_q": (
                        cudnn_actual_seq_lens_q
                        if actual_seq_lens_q is not None
                        else None
                    ),
                    "seq_len_kv": (
                        cudnn_actual_seq_lens_kv
                        if actual_seq_lens_kv is not None
                        else None
                    ),
                }

            if (
                cudnn_q_data_type == cudnn.data_type.BFLOAT16
                or cudnn_q_data_type == cudnn.data_type.HALF
            ):
                O, Stats = g.sdpa(
                    name="sdpa",
                    q=cudnn_q,
                    k=cudnn_k_cache,
                    v=cudnn_v_cache,
                    **seq_len_kwargs,
                    use_padding_mask=padding_mask,
                    attn_scale=scale,
                    generate_stats=return_lse,
                    **({"stats_use_log2": True} if stats_use_log2 else {}),
                    use_causal_mask_bottom_right=bottom_right_causal_mask,
                    paged_attention_k_table=(
                        cudnn_k_block_tables if block_tables is not None else None
                    ),
                    paged_attention_v_table=(
                        cudnn_v_block_tables if block_tables is not None else None
                    ),
                    paged_attention_max_seq_len_kv=(
                        graph_s_kv if block_tables is not None else None
                    ),
                    # A fixed upper bound, shared by backend and FROST. The
                    # generic large envelope retains its previous contract.
                    **(
                        {"max_total_seq_len_q": graph_total_q}
                        if graph_total_q is not None
                        else {}
                    ),
                    compute_data_type=cudnn.data_type.FLOAT,
                )

            elif (
                cudnn_q_data_type == cudnn.data_type.FP8_E4M3
                or cudnn_q_data_type == cudnn.data_type.FP8_E5M2
            ):
                O, Stats, amax_s, amax_o = g.sdpa_fp8(
                    q=cudnn_q,
                    k=cudnn_k_cache,
                    v=cudnn_v_cache,
                    descale_q=cudnn_q_scale,
                    descale_k=cudnn_k_scale,
                    descale_v=cudnn_v_scale,
                    scale_s=cudnn_s_scale,
                    descale_s=cudnn_s_descale,
                    scale_o=cudnn_o_scale,
                    generate_stats=True,
                    attn_scale=scale,
                    use_causal_mask_bottom_right=bottom_right_causal_mask,
                    use_padding_mask=padding_mask,
                    # cu_seq_len kwargs (direct path) exist on sdpa_fp8 only in
                    # cudnn-frontend 1.27+; the version gate guarantees that.
                    **seq_len_kwargs,
                    paged_attention_k_table=(
                        cudnn_k_block_tables if block_tables is not None else None
                    ),
                    paged_attention_v_table=(
                        cudnn_v_block_tables if block_tables is not None else None
                    ),
                    paged_attention_max_seq_len_kv=(
                        graph_s_kv if block_tables is not None else None
                    ),
                )

                amax_s.set_uid(UIDs.S_AMAX_UID.value).set_output(False).set_dim(
                    (1, 1, 1, 1)
                ).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
                amax_o.set_uid(UIDs.O_AMAX_UID.value).set_output(False).set_dim(
                    (1, 1, 1, 1)
                ).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

            if batch_offsets_o is not None:
                ragged_o = indptr_tensor(batch_offsets_o)
                ragged_o.set_uid(UIDs.RAGGED_O_UID.value)
                O.set_ragged_offset(ragged_o)
                if use_cu_seq_lens:
                    O.set_ragged_offset_multiplier(h_qo * d_vo)

            if batch_offsets_stats is not None:
                ragged_stats = indptr_tensor(batch_offsets_stats)
                ragged_stats.set_uid(UIDs.RAGGED_STATS_UID.value)
                Stats.set_ragged_offset(ragged_stats)
                if use_cu_seq_lens:
                    Stats.set_ragged_offset_multiplier(1 if stats_head_stride else h_qo)

            O.set_uid(UIDs.O_UID.value).set_output(True).set_dim(
                [graph_b, h_qo, graph_s_qo, d_vo]
            ).set_stride(
                [graph_s_qo * d_vo * h_qo, d_vo, d_vo * h_qo, 1]
            ).set_data_type(cudnn_o_data_type)

            if return_lse:
                Stats.set_uid(UIDs.STATS_UID.value).set_output(
                    return_lse
                ).set_data_type(cudnn.data_type.FLOAT).set_dim(
                    [graph_b, h_qo, graph_s_qo, 1]
                ).set_stride(
                    [h_qo * stats_head_stride, stats_head_stride, 1, 1]
                    if stats_head_stride
                    else [graph_s_qo * h_qo, 1, h_qo, 1]
                )

            tensors_to_return = [cudnn_q, cudnn_k_cache, cudnn_v_cache, O]
            if return_lse:
                tensors_to_return.append(Stats)

            if use_cu_seq_lens:
                tensors_to_return.append(cudnn_cu_seq_lens_q)
                # KV tensor is cu_seq_len_kv (both-cumulative) or seq_len_kv
                # (mixed/paged), whichever the KV branch above created.
                tensors_to_return.append(
                    cudnn_cu_seq_lens_kv
                    if cu_seq_lens_kv is not None
                    else cudnn_seq_len_kv
                )
            else:
                if actual_seq_lens_q is not None:
                    tensors_to_return.append(cudnn_actual_seq_lens_q)
                if actual_seq_lens_kv is not None:
                    tensors_to_return.append(cudnn_actual_seq_lens_kv)

            if workspace_limit is not None:
                # The same constraint applies to backend and Python engines.
                # This graph has its own cache key; never filter a shared graph.
                g.deselect_workspace_greater_than(workspace_limit)
            if stats_use_log2:
                require_native_cudnn_log2(g, cudnn)
            return g, tensors_to_return


def _override_execute_kwargs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    batch_size: int,
    s_qo: int,
    s_kv: int,
    with_stats: bool,
    stats_head_stride: int = 0,
) -> dict:
    """Real shapes for an override-graph execute: the dims/strides
    _build_prefill_graph would have declared for this call, in the same form."""
    h_qo, d_qk = q.shape[1], q.shape[2]
    h_kv, d_vo = v_cache.shape[1], v_cache.shape[2]
    q_s, q_h, q_d = q.stride()
    k_s, k_h, k_d = k_cache.stride()
    v_s, v_h, v_d = v_cache.stride()
    rows, unit = [batch_size + 1, 1, 1, 1], [1, 1, 1, 1]
    uids = [
        UIDs.Q_UID.value,
        UIDs.K_UID.value,
        UIDs.V_UID.value,
        UIDs.O_UID.value,
        UIDs.RAGGED_Q_UID.value,
        UIDs.RAGGED_K_UID.value,
        UIDs.RAGGED_V_UID.value,
        UIDs.RAGGED_O_UID.value,
        UIDs.ACTUAL_SEQ_LENS_Q_UID.value,
        UIDs.ACTUAL_SEQ_LENS_KV_UID.value,
    ]
    shapes = [
        [batch_size, h_qo, s_qo, d_qk],
        [batch_size, h_kv, s_kv, d_qk],
        [batch_size, h_kv, s_kv, d_vo],
        [batch_size, h_qo, s_qo, d_vo],
    ] + [rows] * 6
    strides = [
        [h_qo * d_qk, q_h, q_s, q_d],
        [h_kv * d_qk * s_kv, k_h, k_s, k_d],
        [h_kv * d_vo * s_kv, v_h, v_s, v_d],
        [s_qo * d_vo * h_qo, d_vo, d_vo * h_qo, 1],
    ] + [unit] * 6
    if with_stats:
        uids += [UIDs.STATS_UID.value, UIDs.RAGGED_STATS_UID.value]
        shapes += [[batch_size, h_qo, s_qo, 1], rows]
        strides += [
            [h_qo * stats_head_stride, stats_head_stride, 1, 1]
            if stats_head_stride
            else [s_qo * h_qo, 1, h_qo, 1],
            unit,
        ]
    return dict(override_uids=uids, override_shapes=shapes, override_strides=strides)


@dataclass(slots=True)
class _PrefillMetadata:
    """Resolved cuDNN metadata; wrappers resolve this once per plan.

    Cumulative lengths and ragged offsets already have the units expected by
    the graph. One explicit mapping supplies both build and match, including
    every descriptor argument even if today's cache key ignores some of them.
    FP8 scales are runtime bindings, and outputs stay call-local.
    """

    max_token_per_sequence: int
    max_sequence_kv: int
    causal: bool
    return_lse: bool
    o_data_type: Optional[torch.dtype] = None
    actual_seq_lens_q: Optional[torch.Tensor] = None
    actual_seq_lens_kv: Optional[torch.Tensor] = None
    cu_seq_lens_q: Optional[torch.Tensor] = None
    cu_seq_lens_kv: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    batch_offsets_q: Optional[torch.Tensor] = None
    batch_offsets_o: Optional[torch.Tensor] = None
    batch_offsets_k: Optional[torch.Tensor] = None
    batch_offsets_v: Optional[torch.Tensor] = None
    batch_offsets_stats: Optional[torch.Tensor] = None
    q_scale: Optional[torch.Tensor] = None
    k_scale: Optional[torch.Tensor] = None
    v_scale: Optional[torch.Tensor] = None

    # Only a wrapper-enforced lifetime capacity is safe here. A live total is
    # insufficient: captured graphs can outlive later plan() calls.
    max_total_num_rows: Optional[int] = None
    _bounded_ragged: bool = False

    def resolve(self, q, k_cache, v_cache, *, batch_offsets_units="elements"):
        # Element offsets count elements of each tensor's own storage, so the
        # conversion path scales by the real token strides (a non-contiguous q
        # such as a T3HD view has stride(0) > num_heads * head_dim), matching
        # the multipliers the direct path declares on the graph.
        token_strides = (
            q.stride(0),
            k_cache.stride(0) if k_cache.dim() == 3 else None,
            v_cache.stride(0) if v_cache.dim() == 3 else None,
        )
        return self.resolve_from_plan(
            q.dtype,
            q.shape[1],
            k_cache.shape[1],
            q.shape[-1],
            v_cache.shape[-1],
            batch_offsets_units=batch_offsets_units,
            token_strides=token_strides,
        )

    def resolve_from_plan(
        self,
        q_dtype,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        *,
        batch_offsets_units="tokens",
        token_strides=None,
    ):
        self._bounded_ragged = (
            q_dtype in (torch.float16, torch.bfloat16)
            # The bounded-override contract is part of FE 1.31. Keep older
            # stacks on the established broad cache rather than multiplying
            # their graphs without a usable packed split implementation.
            and head_dim_vo == 128
            and 4 <= num_qo_heads <= 64
            and (
                (
                    head_dim_qk == 192
                    and q_dtype == torch.bfloat16
                    and num_qo_heads == num_kv_heads
                )
                or (
                    head_dim_qk == 128
                    and num_kv_heads > 0
                    and num_qo_heads % num_kv_heads == 0
                    and num_qo_heads // num_kv_heads in (1, 2, 4, 8, 16)
                )
            )
            and _cudnn_supports_bounded_ragged(d128=head_dim_qk == 128)
            and self.batch_offsets_q is not None
            and self.batch_offsets_q.device.type == "cuda"
            and get_compute_capability(self.batch_offsets_q.device)
            in ((10, 0), (10, 7))
        )
        if batch_offsets_units != "tokens":
            self.max_total_num_rows = None
            return self
        if self.batch_offsets_o is None:
            self.batch_offsets_o = self.batch_offsets_q
        if self.batch_offsets_v is None:
            self.batch_offsets_v = self.batch_offsets_k
        paged = self.block_tables is not None
        direct = (
            _cudnn_supports_direct_seqlens(q_dtype, mixed=paged)
            and self.batch_offsets_q is not None
            and (
                self.actual_seq_lens_kv is not None
                if paged
                else self.batch_offsets_k is not None
            )
        )
        # The wrapper's initialization capacity is already fixed across
        # graph-mode replans. Preserve it without padding the packed total.
        if self.max_total_num_rows is not None and not (
            self.max_total_num_rows > 0
            and direct
            and q_dtype in (torch.float16, torch.bfloat16)
            and _cudnn_supports_bounded_ragged(d128=head_dim_qk == 128)
        ):
            self.max_total_num_rows = None
        if direct:
            self.cu_seq_lens_q = self.batch_offsets_q
            self.cu_seq_lens_kv = None if paged else self.batch_offsets_k
        else:
            if self.actual_seq_lens_q is None:
                self.actual_seq_lens_q = (
                    self.batch_offsets_q[1:] - self.batch_offsets_q[:-1]
                ).view(-1, 1, 1, 1)
            if self.actual_seq_lens_kv is None:
                self.actual_seq_lens_kv = (
                    self.batch_offsets_k[1:] - self.batch_offsets_k[:-1]
                ).view(-1, 1, 1, 1)
            q_stride, k_stride, v_stride = token_strides or (None, None, None)
            for name, multiplier in (
                ("batch_offsets_q", q_stride or num_qo_heads * head_dim_qk),
                ("batch_offsets_o", num_qo_heads * head_dim_vo),
                ("batch_offsets_k", k_stride or num_kv_heads * head_dim_qk),
                ("batch_offsets_v", v_stride or num_kv_heads * head_dim_vo),
                ("batch_offsets_stats", num_qo_heads),
            ):
                offsets = getattr(self, name)
                if offsets is not None:
                    setattr(self, name, offsets * multiplier)
        return self

    def bind_single_token_stats_unragged(
        self, num_tokens: int, num_qo_heads: int, num_kv_heads: int
    ) -> bool:
        """Work around NVBug 6783545 on cuDNN < 9.27 for a packed LSE.

        The single-token GQA kernel mis-stores a ragged Stats tensor. When every
        request has exactly one query token, the unragged Stats declaration
        ``(b, h_qo, 1, 1)`` with token-major strides addresses the same bytes as
        the packed ``(tokens, h_qo)`` buffer, so drop the Stats ragged offset
        and let the fixed-stride store write it. Returns ``False`` when the
        packed LSE cannot be served: a batch that mixes zero-length and
        one-token requests, where the two layouts differ.
        """
        if not (
            self.return_lse
            and self.batch_offsets_stats is not None
            and self.max_token_per_sequence == 1
            and num_qo_heads != num_kv_heads
            and _cudnn_single_token_gqa_ragged_stats_broken()
        ):
            return True
        if self.cu_seq_lens_q is not None:
            batch = self.cu_seq_lens_q.shape[0] - 1
        elif self.actual_seq_lens_q is not None:
            batch = self.actual_seq_lens_q.shape[0]
        else:
            batch = self.batch_offsets_q.shape[0] - 1
        if num_tokens != batch:
            return False
        self.batch_offsets_stats = None
        return True

    def override_shape(self, q, k_cache):
        return self.override_shape_from_plan(
            q.dtype, q.dim() == 3 and k_cache.dim() == 3
        )

    def override_shape_from_plan(self, q_dtype, ragged):
        # Override descriptors declare contiguous indptrs at the cache batch.
        if (
            self.cu_seq_lens_q is not None
            and self.cu_seq_lens_kv is not None
            and self.block_tables is None
            and ragged
            and q_dtype in (torch.float16, torch.bfloat16)
            and self.q_scale is None
            and self.k_scale is None
            and self.v_scale is None
            and self.batch_offsets_q is not None
            and self.batch_offsets_k is not None
            and self.batch_offsets_v is not None
            and self.batch_offsets_o is not None
            and (self.batch_offsets_stats is not None or not self.return_lse)
            and all(
                t.is_contiguous()
                for t in (
                    self.cu_seq_lens_q,
                    self.cu_seq_lens_kv,
                    self.batch_offsets_q,
                    self.batch_offsets_k,
                    self.batch_offsets_v,
                    self.batch_offsets_o,
                )
            )
            and (
                self.batch_offsets_stats is None
                or self.batch_offsets_stats.is_contiguous()
            )
            and _cudnn_supports_shape_override()
        ):
            return _override_cache_shape(
                self.cu_seq_lens_q.shape[0] - 1,
                self.max_token_per_sequence,
                self.max_sequence_kv,
                bounded_ragged=self._bounded_ragged,
            )
        return None

    def graph_kwargs(self, override_cache):
        return dict(
            max_token_seq_q=self.max_token_per_sequence,
            max_total_num_rows=self.max_total_num_rows,
            max_sequence_kv=self.max_sequence_kv,
            override_cache=override_cache,
            actual_seq_lens_q=self.actual_seq_lens_q,
            actual_seq_lens_kv=self.actual_seq_lens_kv,
            cu_seq_lens_q=self.cu_seq_lens_q,
            cu_seq_lens_kv=self.cu_seq_lens_kv,
            block_tables=self.block_tables,
            bottom_right_causal_mask=self.causal,
            return_lse=self.return_lse,
            batch_offsets_q=self.batch_offsets_q,
            batch_offsets_o=self.batch_offsets_o,
            batch_offsets_k=self.batch_offsets_k,
            batch_offsets_v=self.batch_offsets_v,
            batch_offsets_stats=self.batch_offsets_stats,
            o_data_type=self.o_data_type,
        )


def _prefill_plan_bindings(metadata, dtype, device):
    """Bindings whose layout and ownership are fixed by a plan."""
    var_map = {}
    # The cumulative seq lens feed the padding mask via the seq-lens UID
    # slots (on the ragged path these are the same buffers as the
    # token-unit ragged offsets below).
    if metadata.cu_seq_lens_q is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_Q_UID.value] = metadata.cu_seq_lens_q
    elif metadata.actual_seq_lens_q is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_Q_UID.value] = metadata.actual_seq_lens_q
    if metadata.cu_seq_lens_kv is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_KV_UID.value] = metadata.cu_seq_lens_kv
    elif metadata.actual_seq_lens_kv is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_KV_UID.value] = metadata.actual_seq_lens_kv
    if metadata.batch_offsets_q is not None:
        var_map[UIDs.RAGGED_Q_UID.value] = metadata.batch_offsets_q
    if metadata.batch_offsets_o is not None:
        var_map[UIDs.RAGGED_O_UID.value] = metadata.batch_offsets_o
    if metadata.batch_offsets_k is not None:
        var_map[UIDs.RAGGED_K_UID.value] = metadata.batch_offsets_k
    if metadata.batch_offsets_v is not None:
        var_map[UIDs.RAGGED_V_UID.value] = metadata.batch_offsets_v
    if metadata.block_tables is not None:
        var_map[UIDs.BLOCK_TABLES_K_UID.value] = metadata.block_tables
        var_map[UIDs.BLOCK_TABLES_V_UID.value] = metadata.block_tables
    if metadata.return_lse:
        if metadata.batch_offsets_stats is not None:
            var_map[UIDs.RAGGED_STATS_UID.value] = metadata.batch_offsets_stats
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        dummy_scale_tensor = _get_dummy_scale_tensor(device)
        for uid, scale in (
            (UIDs.Q_SCALE_UID, metadata.q_scale),
            (UIDs.K_SCALE_UID, metadata.k_scale),
            (UIDs.V_SCALE_UID, metadata.v_scale),
        ):
            if scale is None:
                scale = dummy_scale_tensor
            elif not isinstance(scale, torch.Tensor):
                # Fill on the GPU: torch.tensor would copy unpinned host data
                # during capture. Each call owns its scalar storage.
                scale = torch.full((1,), scale, device=device, dtype=torch.float32)
            if (
                scale.dtype != torch.float32
                or scale.numel() != 1
                or not scale.is_contiguous()
            ):
                raise ValueError("cuDNN FP8 scales must be scalar float32 tensors")
            var_map[uid.value] = scale
        var_map[UIDs.S_SCALE_UID.value] = dummy_scale_tensor
        var_map[UIDs.S_DESCALE_UID.value] = dummy_scale_tensor
        var_map[UIDs.O_SCALE_UID.value] = dummy_scale_tensor

    for tensor in var_map.values():
        if tensor.device != device:
            raise ValueError(
                "all cuDNN prefill buffers must be on the same device as q"
            )
    return var_map


class _CudnnPrefillPlan:
    """Plan-owned metadata snapshots, keys and static bindings.

    Tensor values can change in place. Layout changes require another plan;
    detached views keep a caller's later Tensor metadata mutation from changing
    these descriptors. Each execute creates call-local bindings before rebinding.
    """

    _tensor_fields = (
        "actual_seq_lens_q",
        "actual_seq_lens_kv",
        "cu_seq_lens_q",
        "cu_seq_lens_kv",
        "block_tables",
        "batch_offsets_q",
        "batch_offsets_o",
        "batch_offsets_k",
        "batch_offsets_v",
        "batch_offsets_stats",
    )

    @classmethod
    def prepare(cls, metadata, dtype, device, previous=None):
        inputs = tuple(getattr(metadata, name) for name in cls._tensor_fields)
        exact = _prefill_descriptor_key(**metadata.graph_kwargs(None))
        enabled = _cudnn_supports_shape_override()
        if (
            previous is not None
            and previous.device == device
            and previous.dtype == dtype
            and previous.metadata.o_data_type == metadata.o_data_type
            and previous.override_enabled == enabled
            and previous.metadata._bounded_ragged == metadata._bounded_ragged
            and previous.exact_keys[True] == exact
            and all(
                a is b
                or (
                    a is not None
                    and b is not None
                    and a.dtype == b.dtype
                    and a.is_set_to(b)
                )
                for a, b in zip(previous.inputs, inputs, strict=True)
            )
        ):
            # Fresh views over the same buffers can reuse the static bindings.
            # Compare owned snapshots: set_() can rebind a caller's tensor
            # without changing either its Python identity or descriptor.
            return previous
        return cls(
            metadata,
            dtype,
            device,
            previous,
            exact=exact,
            inputs=inputs,
            override_enabled=enabled,
        )

    def __init__(
        self,
        metadata,
        dtype,
        device,
        previous=None,
        *,
        exact,
        inputs,
        override_enabled,
    ):
        views = {}
        for name in self._tensor_fields:
            tensor = getattr(metadata, name)
            if tensor is not None:
                if id(tensor) not in views:
                    views[id(tensor)] = tensor.detach()
                setattr(metadata, name, views[id(tensor)])
        self.inputs = tuple(None if t is None else views[id(t)] for t in inputs)
        self.metadata = metadata
        self.device = device
        self.override_enabled = override_enabled
        self.dtype = dtype
        if (
            previous is not None
            and exact == previous.exact_keys[True]
            and dtype == previous.dtype
            and self.override_enabled == previous.override_enabled
            and metadata._bounded_ragged == previous.metadata._bounded_ragged
        ):
            self.override = previous.override
            self.exact_keys = previous.exact_keys
            self.override_keys = previous.override_keys
        else:
            self.override = metadata.override_shape_from_plan(
                dtype, metadata.block_tables is None
            )
            self.exact_keys = ((exact[0], (False, None)), exact)
            if self.override is not None:
                override = _prefill_override_descriptor_key(exact, self.override)
                self.override_keys = ((override[0], (False, None)), override)
            else:
                self.override_keys = self.exact_keys
        bindings = _prefill_plan_bindings(metadata, dtype, device)
        without_stats = bindings.copy()
        without_stats.pop(UIDs.RAGGED_STATS_UID.value, None)
        self.bindings = (without_stats, bindings)
        dynamic_uids = (
            UIDs.Q_UID.value,
            UIDs.K_UID.value,
            UIDs.V_UID.value,
            UIDs.O_UID.value,
        )
        self.ordered_bindings = (
            (
                dynamic_uids + tuple(without_stats),
                (None,) * 4 + tuple(without_stats.values()),
            ),
            (
                dynamic_uids + (UIDs.STATS_UID.value,) + tuple(bindings),
                (None,) * 5 + tuple(bindings.values()),
            ),
        )
        self.execution_shape = exact[0][0][:3]
        self.bound_graph = None
        self.bound_stats_head_stride = 0
        self.execute_kwargs = {}
        # Real shape overrides depend on bounds and on the prepared graph's
        # runtime tensor layout, not on the new indptr pointers or values.
        if previous is not None and previous.execution_shape == self.execution_shape:
            self.bound_graph = previous.bound_graph
            self.bound_stats_head_stride = previous.bound_stats_head_stride
            self.execute_kwargs = previous.execute_kwargs

    def build_metadata(self, return_lse):
        # The output-only variant is needed only when preparing a new graph.
        return (
            self.metadata
            if return_lse
            else replace(self.metadata, return_lse=False, batch_offsets_stats=None)
        )


class CudnnPrefillGraph:
    """A built prefill graph prepared once for one graph signature, so a
    wrapper's per-step ``run()`` only rebinds the tensors and executes.

    The signature is the graph-cache key of :func:`_build_prefill_graph`
    (declared shape or override class, dtypes, strides, scale, mask, flags), so
    a prepared graph is reused exactly where the graph cache would replay the
    same graph and re-prepared otherwise. The wrapper owns a metadata plan
    that caches real-shape execute kwargs, checks matches_plan before running,
    and invalidates the prepared graph on workspace replacement. Each run
    binds fresh pointers. Planning and workspace replacement must not race
    with a run on the same wrapper.

    ``out`` and ``lse`` are bound as given: callers validate their shapes
    (the wrappers do, and :func:`cudnn_batch_prefill_with_kv_cache` does
    before preparing).
    """

    __slots__ = (
        "key",
        "graph",
        "override_cache",
        "return_lse",
        "ordered_execution",
        "stats_head_stride",
        "requested_stats_head_stride",
        "lse_base",
        "stats_use_log2",
    )

    def __init__(
        self,
        key,
        graph,
        *,
        override_cache,
        return_lse: bool,
        stats_head_stride=0,
        requested_stats_head_stride=0,
        lse_base="log2",
    ):
        self.key = key
        self.graph = graph
        self.override_cache = override_cache
        self.return_lse = return_lse
        self.ordered_execution = supports_ordered_cudnn_execution(type(graph))
        self.stats_head_stride = stats_head_stride
        self.requested_stats_head_stride = requested_stats_head_stride
        self.lse_base = lse_base
        self.stats_use_log2 = return_lse and getattr(
            graph, "_flashinfer_stats_use_log2", False
        )

    def matches_plan(
        self,
        q,
        k_cache,
        v_cache,
        scale,
        plan,
        return_lse,
        stats_head_stride=0,
        lse_base="log2",
    ):
        if return_lse and lse_base != self.lse_base:
            return False
        if self.override_cache is not None and _CUDNN_NATIVE_HN_SUPPORTED:
            if bool(stats_head_stride) != bool(self.requested_stats_head_stride):
                return False
        elif stats_head_stride != self.requested_stats_head_stride:
            return False
        metadata = plan.metadata
        if self.override_cache is not None:
            if plan.override != self.override_cache:
                return False
            key = plan.override_keys[return_lse]
        else:
            key = plan.exact_keys[return_lse]
        return self.key[1] == key and self.key[0] == _prefill_runtime_key(
            q, k_cache, v_cache, scale, metadata.o_data_type
        )

    def run_planned(
        self, q, k_cache, v_cache, out, lse, workspace_buffer, *, plan, lse_base="log2"
    ):
        device = plan.device
        if any(
            t.device != device for t in (q, k_cache, v_cache, out, workspace_buffer)
        ):
            raise ValueError(
                "all cuDNN prefill buffers must be on the same device as q"
            )
        if self.return_lse and (lse is None or lse.device != device):
            raise ValueError(
                "all cuDNN prefill buffers must be on the same device as q"
            )
        stats_head_stride = q.size(0) if self.stats_head_stride else 0
        if (
            plan.bound_graph is not self
            or plan.bound_stats_head_stride != stats_head_stride
        ):
            metadata = plan.metadata
            plan.execute_kwargs = (
                _override_execute_kwargs(
                    q,
                    k_cache,
                    v_cache,
                    batch_size=metadata.cu_seq_lens_q.shape[0] - 1,
                    s_qo=metadata.max_token_per_sequence,
                    s_kv=metadata.max_sequence_kv,
                    with_stats=self.return_lse,
                    stats_head_stride=stats_head_stride,
                )
                if self.override_cache is not None
                else {}
            )
            plan.bound_graph = self
            plan.bound_stats_head_stride = stats_head_stride
        if self.ordered_execution:
            uids, template = plan.ordered_bindings[self.return_lse]
            buffers = list(template)
            buffers[:4] = q, k_cache, v_cache, out
            if self.return_lse:
                buffers[4] = lse
            return self._execute(
                q,
                out,
                lse,
                workspace_buffer,
                buffers,
                plan.execute_kwargs,
                lse_base,
                tensor_uids=uids,
            )
        var_map = plan.bindings[self.return_lse].copy()
        var_map.update(
            {
                UIDs.Q_UID.value: q,
                UIDs.K_UID.value: k_cache,
                UIDs.V_UID.value: v_cache,
                UIDs.O_UID.value: out,
            }
        )
        if self.return_lse:
            var_map[UIDs.STATS_UID.value] = lse
        return self._execute(
            q,
            out,
            lse,
            workspace_buffer,
            var_map,
            plan.execute_kwargs,
            lse_base,
        )

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        lse: Optional[torch.Tensor],
        workspace_buffer: torch.Tensor,
        *,
        metadata: _PrefillMetadata,
        lse_base: str = "log2",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = q.device
        var_map = _prefill_plan_bindings(metadata, q.dtype, device)
        var_map.update(
            {
                UIDs.Q_UID.value: q,
                UIDs.K_UID.value: k_cache,
                UIDs.V_UID.value: v_cache,
                UIDs.O_UID.value: out,
            }
        )
        if self.return_lse:
            var_map[UIDs.STATS_UID.value] = lse
        for tensor in (k_cache, v_cache, out, workspace_buffer):
            if tensor.device != device:
                raise ValueError(
                    "all cuDNN prefill buffers must be on the same device as q"
                )
        if self.return_lse and (lse is None or lse.device != device):
            raise ValueError(
                "all cuDNN prefill buffers must be on the same device as q"
            )

        execute_kwargs: dict = {}
        if self.override_cache is not None:
            # The public entry point creates this wrapper per call. Repeated
            # wrapper execution uses run_planned and its plan-owned kwargs.
            execute_kwargs = _override_execute_kwargs(
                q,
                k_cache,
                v_cache,
                batch_size=metadata.cu_seq_lens_q.shape[0] - 1,
                s_qo=metadata.max_token_per_sequence,
                s_kv=metadata.max_sequence_kv,
                with_stats=self.return_lse,
                stats_head_stride=q.size(0) if self.stats_head_stride else 0,
            )

        return self._execute(
            q, out, lse, workspace_buffer, var_map, execute_kwargs, lse_base
        )

    def _execute(
        self,
        q,
        out,
        lse,
        workspace_buffer,
        var_map,
        execute_kwargs,
        lse_base,
        tensor_uids=None,
    ):
        device = q.device
        handle = _create_cudnn_handle(
            torch.cuda.current_stream(device.index if device.type == "cuda" else device)
        )
        if tensor_uids is None:
            self.graph.execute(
                var_map, workspace=workspace_buffer, handle=handle, **execute_kwargs
            )
        else:
            self.graph.execute(
                var_map,
                workspace=workspace_buffer,
                handle=handle,
                tensor_uids=tensor_uids,
                **execute_kwargs,
            )

        if self.return_lse:
            # The built graph's Stats base is immutable, including after a
            # capability fallback. Convert only when the caller needs the other base.
            if self.stats_use_log2 and lse_base == "ln":
                lse.mul_(ln2)
            elif not self.stats_use_log2 and lse_base == "log2":
                lse.mul_(log2e)
            return out, lse
        return out, None


def prepare_cudnn_batch_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    metadata: _PrefillMetadata,
    stats_head_stride: int = 0,
    lse_base: str = "log2",
) -> CudnnPrefillGraph:
    """Fetch (or build into the graph cache) the prefill graph for this call's
    signature and wrap it for execution.

    Takes the resolved form :func:`cudnn_batch_prefill_with_kv_cache` hands to
    the low level: token-unit ``cu_seq_lens_*`` on the direct path (with the
    same buffers as ``batch_offsets_*``), or per-batch ``actual_seq_lens_*``
    with element-unit offsets. The wrappers call this directly and keep the
    result across steps, using :meth:`CudnnPrefillGraph.matches_plan` and
    :meth:`CudnnPrefillGraph.run_planned` with their plan-owned metadata.
    The public low-level entry point instead calls :meth:`CudnnPrefillGraph.run`
    with metadata resolved for that call.
    """
    override_cache = metadata.override_shape(q, k_cache)
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()

    stats_use_log2 = (
        metadata.return_lse
        and lse_base == "log2"
        and q.dtype in (torch.float16, torch.bfloat16)
        and supports_native_cudnn_log2(cudnn, q.device)
    )
    requested_stats_head_stride = stats_head_stride
    try:
        graph, _ = _build_prefill_graph(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            scale=scale,
            stats_use_log2=stats_use_log2,
            stats_head_stride=stats_head_stride,
            **metadata.graph_kwargs(override_cache),
        )
        if stats_head_stride and override_cache is not None:
            # Released native cuDNN can retain the declared HN stride after an
            # override. Only the prepared half FROST binders currently prove
            # native HN across changing token totals; preserve NH elsewhere.
            engine = getattr(graph, "selected_engine", None)
            if getattr(engine, "name", None) not in (
                "sdpa_fwd_prefill_sm100",
                "sdpa_fwd_prefill_sm107",
                "sdpa_fwd_prefill_sm120",
            ):
                raise cudnn.cudnnGraphNotSupportedError(
                    "selected engine cannot override packed HN Stats stride"
                )
    except cudnn.cudnnGraphNotSupportedError:
        if not stats_head_stride:
            raise
        # Older engines can decline HN. Retain the established NH graph and
        # remember the requested layout so warm runs do not retry this build.
        stats_head_stride = 0
        graph, _ = _build_prefill_graph(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            scale=scale,
            stats_use_log2=stats_use_log2,
            **metadata.graph_kwargs(override_cache),
        )
    key = _sdpa_prefill_key_fn(
        q,
        k_cache,
        v_cache,
        scale,
        stats_head_stride=stats_head_stride,
        stats_use_log2=stats_use_log2,
        **metadata.graph_kwargs(override_cache),
    )
    if override_cache is not None:
        if _graph_workspace_size(graph, key) > workspace_bytes:
            # The override graph reserves TMA descriptors for its declared
            # batch (~1 MiB at 4096). A caller whose workspace cannot hold them
            # gets the exact-shape graph instead of an out-of-bounds execute.
            override_cache = None
            graph, _ = _build_prefill_graph(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                scale=scale,
                stats_head_stride=stats_head_stride,
                stats_use_log2=stats_use_log2,
                **metadata.graph_kwargs(None),
            )
            key = _sdpa_prefill_key_fn(
                q,
                k_cache,
                v_cache,
                scale,
                stats_head_stride=stats_head_stride,
                stats_use_log2=stats_use_log2,
                **metadata.graph_kwargs(None),
            )
    if _graph_workspace_size(graph, key) > workspace_bytes:
        # An exact graph may still prefer split-KV with large FP32 partials.
        # Let FE walk its ordinary plan list under the caller's workspace limit.
        # Cache this separately so earlier captures and larger-workspace callers
        # retain their selected plan. Prepared warm runs do not revisit this.
        constrained = dict(
            **metadata.graph_kwargs(override_cache), workspace_limit=workspace_bytes
        )
        graph, _ = _build_prefill_graph(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            scale=scale,
            stats_head_stride=stats_head_stride,
            stats_use_log2=stats_use_log2,
            **constrained,
        )
        key = _sdpa_prefill_key_fn(
            q,
            k_cache,
            v_cache,
            scale,
            stats_head_stride=stats_head_stride,
            stats_use_log2=stats_use_log2,
            **constrained,
        )
    return CudnnPrefillGraph(
        key,
        graph,
        override_cache=override_cache,
        return_lse=metadata.return_lse,
        stats_head_stride=stats_head_stride,
        requested_stats_head_stride=requested_stats_head_stride,
        lse_base=lse_base,
    )


@flashinfer_api(trace=cudnn_batch_prefill_trace)
def cudnn_batch_prefill_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    max_token_per_sequence: int,
    max_sequence_kv: int,
    actual_seq_lens_q: Optional[torch.Tensor] = None,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    causal: bool,
    return_lse: bool,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    batch_offsets_k: Optional[torch.Tensor] = None,
    batch_offsets_v: Optional[torch.Tensor] = None,
    batch_offsets_stats: Optional[torch.Tensor] = None,
    batch_offsets_units: str = "elements",
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_cuda_graph_compatible: bool = False,
    backend: Optional[str] = None,
    o_data_type: Optional[torch.dtype] = None,
    lse_base: str = "log2",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Batched prefill attention with paged KV cache, backed by cuDNN SDPA.

    Parameters
    ----------
    q : torch.Tensor
        Packed query tensor with shape ``(total_qo_tokens, num_heads_qo, head_dim_qk)``.
    k_cache : torch.Tensor
        Key cache.  If paged:
        ``(total_num_pages, num_heads_kv, page_size, head_dim_qk)``; otherwise
        ``(total_kv_tokens, num_heads_kv, head_dim_qk)``.
    v_cache : torch.Tensor
        Value cache.  If paged:
        ``(total_num_pages, num_heads_kv, page_size, head_dim_vo)``; otherwise
        ``(total_kv_tokens, num_heads_kv, head_dim_vo)``.
    scale : float
        Softmax scaling factor, typically ``1 / sqrt(head_dim_qk)``.
    workspace_buffer : torch.Tensor
        Workspace buffer for cuDNN.  Scales with batch size; 128 MB is sufficient
        for typical prefill workloads.
    max_token_per_sequence : int
        Maximum number of tokens per query sequence (``s_qo_max``).
    max_sequence_kv : int
        Maximum number of tokens per KV sequence (``s_kv_max``).
    actual_seq_lens_q : Optional[torch.Tensor]
        Per-request query lengths, shape ``(batch_size, 1, 1, 1)``.  When cuDNN is
        available (the default backend) this tensor must reside on the same
        CUDA device as ``q``.  Only the fallback non-cuDNN path accepts (and
        internally copies) a CPU tensor; that fallback is also the only path
        that requires a CPU tensor when ``is_cuda_graph_compatible`` is
        ``False``.  May be omitted with ``batch_offsets_units="tokens"`` and
        ``batch_offsets_q`` set (the lengths are then implied by the indptr);
        the cubin backend always requires it.
    actual_seq_lens_kv : Optional[torch.Tensor]
        Per-request KV lengths, shape ``(batch_size, 1, 1, 1)``.  Same device
        rules as ``actual_seq_lens_q``.  May be omitted with
        ``batch_offsets_units="tokens"`` and ``batch_offsets_k`` set (non-paged
        KV); the cubin backend always requires it.
    block_tables : Optional[torch.Tensor]
        Paged KV block table, shape ``(batch_size, num_pages_per_seq)`` on GPU.
        Pass ``None`` for non-paged KV layouts.
    causal : bool
        Whether to apply a causal mask.
    return_lse : bool
        Whether to return the log-sum-exp tensor (currently must be ``True`` in the
        cubin backend).
    q_scale : Optional[torch.Tensor]
        FP8 dequantization scale for the query, shape ``(1, 1, 1, 1)`` on GPU.
    k_scale : Optional[torch.Tensor]
        FP8 dequantization scale for the key, shape ``(1, 1, 1, 1)`` on GPU.
    v_scale : Optional[torch.Tensor]
        FP8 dequantization scale for the value, shape ``(1, 1, 1, 1)`` on GPU.
    batch_offsets_q : Optional[torch.Tensor]
        Cumulative per-request start offsets into the packed query tensor,
        shape ``(batch_size + 1,)``, int32, on GPU, in the units given by
        ``batch_offsets_units`` (element offsets count elements of ``q``'s own
        storage: ``token_offset * q.stride(0)``, which is
        ``token_offset * num_heads_qo * head_dim_qk`` for a contiguous ``q``).
        Required when ``batch_size > 1`` on the cuDNN graph path; may be
        omitted only for ``batch_size == 1``.
    batch_offsets_o : Optional[torch.Tensor]
        Cumulative per-request start offsets into the packed output tensor,
        shape ``(batch_size + 1,)``, int32, on GPU, in the units given by
        ``batch_offsets_units`` (element offsets are
        ``token_offset * num_heads_qo * head_dim_vo``; ``out`` is contiguous).
        Required when ``batch_size > 1`` on the cuDNN graph path; with
        ``batch_offsets_units="tokens"`` it defaults to ``batch_offsets_q``.
    batch_offsets_k : Optional[torch.Tensor]
        Cumulative per-request start offsets into the key tensor, shape
        ``(batch_size + 1,)`` on GPU, in the units given by
        ``batch_offsets_units``.  Only used for non-paged (3-D) KV.
    batch_offsets_v : Optional[torch.Tensor]
        Cumulative per-request start offsets into the value tensor, shape
        ``(batch_size + 1,)`` on GPU, in the units given by
        ``batch_offsets_units``.  Only used for non-paged (3-D) KV; with
        ``batch_offsets_units="tokens"`` it defaults to ``batch_offsets_k``.
    batch_offsets_stats : Optional[torch.Tensor]
        Cumulative per-request start offsets into a packed LSE tensor, shape
        ``(batch_size + 1,)``, in the units given by ``batch_offsets_units``
        (element offsets are ``token_offset * num_heads_qo``).  Derived from
        ``batch_offsets_q`` when omitted and the LSE is packed; rejected with a
        padded LSE.
    batch_offsets_units : str
        Units of the ``batch_offsets_*`` tensors. ``"elements"`` (default, the
        historical behavior): offsets are pre-scaled tensor-element offsets,
        e.g. ``cumsum(seq_lens) * num_heads * head_dim`` for the query.
        ``"tokens"``: offsets are plain token-unit prefix sums
        (``qo_indptr``/``kv_indptr`` style, the FlashInfer convention). For
        non-paged KV, token-unit indptrs are consumed directly by the kernel
        with no conversion pre-pass on cuDNN backend 9.24+ with cudnn-frontend
        1.25+ (fp16/bf16) or backend 9.25+ with cudnn-frontend 1.27+ (fp8);
        otherwise FlashInfer scales them to element units internally.
    out : Optional[torch.Tensor]
        Pre-allocated contiguous output tensor on ``q``'s device, shape
        ``(total_qo_tokens, num_heads_qo, head_dim_vo)``.  Allocated internally
        when ``None``.
    lse : Optional[torch.Tensor]
        Pre-allocated contiguous float32 LSE tensor on ``q``'s device, either
        packed ``(total_qo_tokens, num_heads_qo)`` (one row per query token, the
        FlashInfer convention) or padded
        ``(batch_size, max_token_per_sequence, num_heads_qo)``.  When ``None``
        and ``return_lse`` is ``True`` it is allocated packed on the cuDNN graph
        path and padded on the ``"cubin"`` backend, which writes only that
        form.
    is_cuda_graph_compatible : bool
        Whether to plan the operation in a CUDA-graph-capture-safe mode.
    backend : Optional[str]
        Optional cuDNN backend selector (e.g. ``"cubin"``).  When ``None``,
        autodetects based on cuDNN availability.
    o_data_type : Optional[torch.dtype]
        Optional output dtype; defaults to ``q.dtype``.
    lse_base : str
        ``"log2"`` (default): return the LSE in base 2 like every other FlashInfer
        backend. ``"ln"``: return cuDNN's native natural-log stats, skipping the
        conversion kernel.

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        ``(output, lse)`` where ``output`` has shape
        ``(total_qo_tokens, num_heads_qo, head_dim_vo)``; ``lse`` is the
        packed ``(total_qo_tokens, num_heads_qo)`` tensor (or the caller's
        padded buffer, or the cubin backend's padded form) when
        ``return_lse=True``, else ``None``.

    Note
    ----
    Query and KV heads may differ (``num_heads_qo >= num_heads_kv``, MQA / GQA).
    When using CUDA graph capture, ``actual_seq_lens_q`` and ``actual_seq_lens_kv``
    must reside on the same device as ``q``.  ``head_dim_qk`` must be 128 or 192,
    and ``head_dim_vo`` must be 128.
    """
    check_lse_base(lse_base)

    num_tokens = q.shape[0]

    if batch_offsets_units not in ("elements", "tokens"):
        raise ValueError(
            f"batch_offsets_units must be 'elements' or 'tokens', got {batch_offsets_units!r}"
        )

    # actual_seq_lens_q/kv may be omitted only when they are derivable from
    # token-unit indptrs (the direct path consumes the indptrs as-is; the
    # conversion path derives per-request lengths from them below).
    if actual_seq_lens_q is not None:
        num_sequences = actual_seq_lens_q.shape[0]
    elif batch_offsets_units == "tokens" and batch_offsets_q is not None:
        num_sequences = batch_offsets_q.shape[0] - 1
    else:
        raise ValueError(
            "actual_seq_lens_q may be omitted only with "
            'batch_offsets_units="tokens" and batch_offsets_q set'
        )
    if actual_seq_lens_kv is None and not (
        batch_offsets_units == "tokens" and batch_offsets_k is not None
    ):
        raise ValueError(
            "actual_seq_lens_kv may be omitted only with "
            'batch_offsets_units="tokens" and batch_offsets_k set (non-paged KV)'
        )

    if q.dim() == 3:
        h_qo, d_qk = q.shape[1], q.shape[2]
    elif q.dim() == 4:
        h_qo, d_qk = q.shape[1], q.shape[3]

    if v_cache.dim() == 3:
        d_vo = v_cache.shape[2]
    elif v_cache.dim() == 4:
        d_vo = v_cache.shape[3]

    use_cudnn_graph = CUDNN_AVAILABLE and backend != "cubin"

    # The LSE is packed (total_qo_tokens, h_qo) -- FlashInfer's convention, the
    # shape every other backend returns and the wrappers allocate -- unless the
    # caller hands in the historical padded (batch, max_token_per_sequence,
    # h_qo) buffer. The cubin backend writes the padded form only. A packed LSE
    # is declared to cuDNN as a ragged Stats tensor over the query token
    # indptr, exactly like the packed q it accompanies.
    packed_shape = (num_tokens, h_qo)
    padded_shape = (num_sequences, max_token_per_sequence, h_qo)
    if return_lse and lse is None:
        lse = torch.empty(
            packed_shape if use_cudnn_graph else padded_shape,
            device=q.device,
            dtype=torch.float32,
        )
    if lse is not None:
        # The graph declares Stats as contiguous float32 on q's device and binds
        # this buffer to it directly, so check here rather than letting cuDNN
        # execute against storage the declared strides do not describe.
        if lse.dtype != torch.float32:
            raise ValueError(f"lse must have dtype torch.float32, got {lse.dtype}")
        if lse.device != q.device:
            raise ValueError(f"lse must be on {q.device}, got {lse.device}")
        if not lse.is_contiguous():
            raise ValueError("lse must be contiguous")
    lse_packed = lse is not None and tuple(lse.shape) == packed_shape
    if lse is not None and not lse_packed and tuple(lse.shape) != padded_shape:
        raise ValueError(
            f"lse must have shape {packed_shape} (packed, one row per query token) "
            f"or {padded_shape} (padded, one block per request); got {tuple(lse.shape)}"
        )
    if lse_packed and not use_cudnn_graph:
        raise ValueError(
            f"the cubin backend writes a padded LSE of shape {padded_shape}; got {tuple(lse.shape)}"
        )
    if lse is not None and not lse_packed and batch_offsets_stats is not None:
        # Stats offsets address packed rows; declaring the padded buffer as a
        # ragged Stats tensor would scatter every request after the first to
        # the wrong rows.
        raise ValueError(
            f"batch_offsets_stats addresses a packed LSE of shape {packed_shape}; "
            f"drop it for the padded form {padded_shape}"
        )

    if o_data_type is None:
        o_data_type = q.dtype

    if out is None:
        out_shape = (num_tokens, h_qo, d_vo)
        out = torch.empty(out_shape, device=q.device, dtype=o_data_type)
    else:
        # The graph declares O with contiguous (tokens, h_qo, d_vo) strides and
        # binds this buffer to it directly.
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.device != q.device:
            raise ValueError(f"out must be on {q.device}, got {out.device}")

    if batch_offsets_units == "tokens":
        # Convenience feature to allow user to set just batch_offsets_{q,k} if desired.
        if batch_offsets_o is None:
            batch_offsets_o = batch_offsets_q
        if batch_offsets_v is None:
            batch_offsets_v = batch_offsets_k

    if use_cudnn_graph:
        # The cuDNN graph declares packed q/out with THD nominal strides
        # (batch stride == one token), which is only addressable through ragged
        # offsets. Without them the graph is well-formed but reads/writes batch
        # b at token offset b, silently corrupting every batch except the
        # first, so reject instead (batch_size == 1 needs no offsets: the only
        # batch starts at 0).
        if num_sequences > 1 and (batch_offsets_q is None or batch_offsets_o is None):
            raise ValueError(
                "batch_offsets_q and batch_offsets_o are required when batch_size > 1: "
                "packed q/out cannot be addressed without ragged offsets. Pass "
                "cumulative element offsets of shape (batch_size + 1,), e.g. "
                "cumsum([0, *actual_seq_lens_q]) * q.stride(0) for q (and "
                "* num_qo_heads * head_dim_vo for out), or token-unit indptrs "
                'with batch_offsets_units="tokens".'
            )
        if return_lse and lse_packed and batch_offsets_stats is None:
            # Each request's rows of the packed LSE start where its query
            # tokens start. Token-unit offsets are the q indptr itself (the
            # graph applies the per-tensor multiplier h_qo); element-unit q
            # offsets are token_offset * q.stride(0), so the token offset is
            # recovered with q's real token stride and rescaled by h_qo. A
            # single request without offsets starts at row 0 and spans the
            # buffer; built on the device so the call stays capturable.
            if batch_offsets_q is None:
                batch_offsets_stats = torch.arange(
                    2, dtype=torch.int32, device=q.device
                ) * (
                    num_tokens if batch_offsets_units == "tokens" else num_tokens * h_qo
                )
            elif batch_offsets_units == "tokens":
                batch_offsets_stats = batch_offsets_q
            else:
                batch_offsets_stats = (
                    torch.div(batch_offsets_q, q.stride(0), rounding_mode="floor")
                    * h_qo
                )
        metadata = _PrefillMetadata(
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            causal=causal,
            return_lse=return_lse,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            o_data_type=o_data_type,
            batch_offsets_q=batch_offsets_q,
            batch_offsets_o=batch_offsets_o,
            batch_offsets_k=batch_offsets_k,
            batch_offsets_v=batch_offsets_v,
            batch_offsets_stats=batch_offsets_stats,
        ).resolve(q, k_cache, v_cache, batch_offsets_units=batch_offsets_units)
        if not metadata.bind_single_token_stats_unragged(
            num_tokens, h_qo, k_cache.shape[1]
        ):
            raise ValueError(
                "cuDNN < 9.27 mis-stores a ragged Stats tensor for single-token "
                "GQA (NVBug 6783545); with zero-length requests in the batch the "
                f"packed LSE cannot be served, pass a padded lse of shape "
                f"{padded_shape} instead"
            )
        prepared = prepare_cudnn_batch_prefill(
            q,
            k_cache,
            v_cache,
            scale,
            workspace_buffer,
            metadata=metadata,
            lse_base=lse_base,
        )
        return prepared.run(
            q,
            k_cache,
            v_cache,
            out,
            lse,
            workspace_buffer,
            metadata=metadata,
            lse_base=lse_base,
        )
    else:
        if actual_seq_lens_q is None or actual_seq_lens_kv is None:
            raise ValueError(
                "the cubin backend requires actual_seq_lens_q and actual_seq_lens_kv"
            )

        assert return_lse, "Currently only supports return_lse = True"

        assert (d_qk == 192 and block_tables is None) or (
            d_qk == 128 and block_tables is not None
        ), (
            "Currently only supports if d_qk = 192 and block_tables is None or d_qk = 128 and block_tables is not None"
        )

        if max_sequence_kv is None:
            max_sequence_kv = max_token_per_sequence

        actual_seq_lens_q_gpu = actual_seq_lens_q.to(q.device, non_blocking=True)

        actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.device, non_blocking=True)

        if lse_base != "log2":
            raise NotImplementedError(
                "the cubin cuDNN prefill path only returns base-2 LSE"
            )
        run_func = get_cudnn_fmha_gen_module().prefill
        run_func(
            num_sequences,
            max_token_per_sequence,  # max_s_qo
            max_sequence_kv,  # max_s_kv
            q,
            k_cache,
            v_cache,
            scale,
            workspace_buffer,
            actual_seq_lens_q,  # actual_seq_lens_q
            actual_seq_lens_kv,  # actual_seq_lens_kv
            actual_seq_lens_q_gpu,
            actual_seq_lens_kv_gpu,
            block_tables,
            causal,
            return_lse,
            out,
            lse,
            None,
            None,
            None,
            None,
            is_cuda_graph_compatible,
        )

    return out, lse
