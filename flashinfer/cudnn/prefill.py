import contextlib
import functools
import os
from enum import Enum
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.attention import cudnn_batch_prefill_trace
from ..utils import check_lse_base, log2e
from .utils import get_cudnn_fmha_gen_module

try:
    import cudnn

    CUDNN_AVAILABLE = True
except Exception:
    cudnn = None
    CUDNN_AVAILABLE = False


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
# override matches a natively built plan.
_PREFILL_SHAPE_OVERRIDE_ENV = "FLASHINFER_CUDNN_PREFILL_SHAPE_OVERRIDE"
_OVERRIDE_SHORT_SEQ = 128
_OVERRIDE_CACHE_SEQ_LONG = 65536
_OVERRIDE_CACHE_BATCH = 4096


@functools.cache
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
    batch_size: int, max_seq_q: int, max_seq_kv: int
) -> tuple[int, int, int]:
    """Declared (batch, s_q, s_kv) of the override graph for a real (b, s_q,
    s_kv). q and kv are classed separately (a short-q / long-kv step must not
    be declared as short kv). Grows by powers of two when a caller exceeds the
    defaults; that changes the cache key and builds one more plan."""
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


# Global cudnn handle. need to make it per device in future
_cudnn_handle = None

_dummy_scale_tensors: dict[torch.device, torch.Tensor] = {}


def _get_dummy_scale_tensor(device: torch.device):
    t = _dummy_scale_tensors.get(device)
    if t is None:
        t = torch.tensor([1.0], device=device, dtype=torch.float32).reshape(1, 1, 1, 1)
        _dummy_scale_tensors[device] = t
    return t


def _create_cudnn_handle(stream: torch.cuda.Stream):
    global _cudnn_handle

    if _cudnn_handle is None:
        _cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(_cudnn_handle, stream.cuda_stream)
    return _cudnn_handle


def _tensor_descriptor_signature(
    tensor: Optional[torch.Tensor], *, ignore_leading_dim: bool = False
):
    """Return hashable metadata used to build a cuDNN tensor descriptor."""
    if tensor is None:
        return None
    shape = tuple(tensor.shape)
    if ignore_leading_dim and tensor.dim() == 3:
        shape = shape[1:]
    return tensor.dim(), tensor.dtype, tensor.device, shape, tuple(tensor.stride())


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


def _sdpa_prefill_key_fn(
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
):
    if actual_seq_lens_q is not None:
        graph_b = actual_seq_lens_q.shape[0]
    elif cu_seq_lens_q is not None:
        graph_b = cu_seq_lens_q.shape[0] - 1
    else:
        raise ValueError("Either actual_seq_lens_q or cu_seq_lens_q must be provided")
    if override_cache is not None:
        # The graph is declared at the cache shape; the real (b, s) arrive at
        # execute time, so they must not split the cache.
        graph_b, max_token_seq_q, max_sequence_kv = override_cache

    if q.dim() == 3:
        h_qo, d_qk = q.shape[1], q.shape[2]
    elif q.dim() == 4:
        h_qo, d_qk = q.shape[1], q.shape[3]

    if v_cache.dim() == 3:
        h_kv, d_vo = v_cache.shape[1], v_cache.shape[2]
    elif v_cache.dim() == 4:
        h_kv, d_vo = v_cache.shape[1], v_cache.shape[3]

    block_tables_signature = None
    if block_tables is not None:
        page_size = k_cache.shape[2]
        nd_block_tables = block_tables.reshape(
            block_tables.shape[0], 1, block_tables.shape[1], 1
        )
        block_tables_signature = _tensor_descriptor_signature(nd_block_tables)

    if o_data_type is None:
        o_data_type = q.dtype

    key = (
        graph_b,
        q.dim(),
        q.dtype,
        q.dtype if o_data_type is None else o_data_type,
        k_cache.dim(),
        max_token_seq_q,
        max_sequence_kv,
        h_qo,
        d_qk,
        h_kv,
        d_vo,
        block_tables is not None,
        return_lse,
        bottom_right_causal_mask,
        page_size,
        cu_seq_lens_q is not None,
        override_cache is not None,
        # The graph tensors carry the callers' strides (a packed T3HD q/k/v has a
        # token stride of 3*h*d, not h*d), so two same-shape calls with different
        # strides need different graphs.
        tuple(q.stride()),
        tuple(k_cache.stride()),
        tuple(v_cache.stride()),
        # attn_scale is baked into the built graph as a compile-time constant
        # (see _build_prefill_graph); omitting it here silently replays a
        # stale-scale graph for any same-shape call with a different scale.
        scale,
        o_data_type,
        _tensor_descriptor_signature(q, ignore_leading_dim=True),
        _tensor_descriptor_signature(k_cache, ignore_leading_dim=True),
        _tensor_descriptor_signature(v_cache, ignore_leading_dim=True),
        _tensor_descriptor_signature(actual_seq_lens_q),
        _tensor_descriptor_signature(actual_seq_lens_kv),
        _tensor_descriptor_signature(cu_seq_lens_q),
        _tensor_descriptor_signature(cu_seq_lens_kv),
        block_tables_signature,
        _tensor_descriptor_signature(batch_offsets_q),
        _tensor_descriptor_signature(batch_offsets_o),
        _tensor_descriptor_signature(batch_offsets_k),
        _tensor_descriptor_signature(batch_offsets_v),
        _tensor_descriptor_signature(batch_offsets_stats),
    )
    return key


if CUDNN_AVAILABLE:

    @cudnn.jit(heur_modes=[cudnn.heur_mode.A])
    @cudnn.graph_cache(key_fn=_sdpa_prefill_key_fn)
    def _build_prefill_graph(
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
                # These kwargs need a newer cudnn-frontend than the declared
                # >=1.13 floor (cu_seq_len_*: 1.25+; implementation /
                # attention_implementation: 1.14+), so they are only mentioned
                # on this path, which _cudnn_supports_direct_seqlens guards.
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
                    Stats.set_ragged_offset_multiplier(h_qo)

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
                ).set_stride([graph_s_qo * h_qo, 1, h_qo, 1])

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
        strides += [[s_qo * h_qo, 1, h_qo, 1], unit]
    return dict(override_uids=uids, override_shapes=shapes, override_strides=strides)


def _batch_prefill_with_kv_cache(
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
    cu_seq_lens_q: Optional[torch.Tensor] = None,
    cu_seq_lens_kv: Optional[torch.Tensor] = None,
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
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    o_data_type: Optional[torch.dtype] = None,
    lse_base: str = "log2",
) -> tuple[torch.Tensor, torch.Tensor]:
    # Shape override covers the fully ragged token-indptr path (3-D q/k/v with
    # cu_seq_lens on both sides, 16-bit inputs); paged, per-batch-length and fp8
    # graphs keep the exact declaration.
    override_cache = None
    if (
        cu_seq_lens_q is not None
        and cu_seq_lens_kv is not None
        and block_tables is None
        and q.dim() == 3
        and k_cache.dim() == 3
        and q_scale is None
        and k_scale is None
        and v_scale is None
        and batch_offsets_q is not None
        and batch_offsets_k is not None
        and batch_offsets_v is not None
        and batch_offsets_o is not None
        and (batch_offsets_stats is not None or not return_lse)
        and _cudnn_supports_shape_override()
    ):
        override_cache = _override_cache_shape(
            cu_seq_lens_q.shape[0] - 1, max_token_per_sequence, max_sequence_kv
        )

    def graph_kwargs(cache):
        return dict(
            max_token_seq_q=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            override_cache=cache,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_kv=cu_seq_lens_kv,
            block_tables=block_tables,
            bottom_right_causal_mask=causal,
            return_lse=return_lse,
            batch_offsets_q=batch_offsets_q,
            batch_offsets_o=batch_offsets_o,
            batch_offsets_k=batch_offsets_k,
            batch_offsets_v=batch_offsets_v,
            batch_offsets_stats=batch_offsets_stats,
            out=out,
            lse=lse,
            o_data_type=o_data_type,
        )

    graph, tensors = _build_prefill_graph(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        scale=scale,
        **graph_kwargs(override_cache),
    )
    if override_cache is not None:
        workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
        key = _sdpa_prefill_key_fn(
            q, k_cache, v_cache, scale, **graph_kwargs(override_cache)
        )
        if _graph_workspace_size(graph, key) > workspace_bytes:
            # The override graph reserves TMA descriptors for its declared
            # batch (~1 MiB at 4096). A caller whose workspace cannot hold them
            # gets the exact-shape graph instead of an out-of-bounds execute.
            override_cache = None
            graph, tensors = _build_prefill_graph(
                q=q, k_cache=k_cache, v_cache=v_cache, scale=scale, **graph_kwargs(None)
            )

    var_map = {
        UIDs.Q_UID.value: q,
        UIDs.K_UID.value: k_cache,
        UIDs.V_UID.value: v_cache,
        UIDs.O_UID.value: out,
    }

    # The cumulative seq lens feed the padding mask via the seq-lens UID
    # slots (on the ragged path these are the same buffers as the
    # token-unit ragged offsets below).
    if cu_seq_lens_q is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_Q_UID.value] = cu_seq_lens_q
    elif actual_seq_lens_q is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_Q_UID.value] = actual_seq_lens_q
    if cu_seq_lens_kv is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_KV_UID.value] = cu_seq_lens_kv
    elif actual_seq_lens_kv is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_KV_UID.value] = actual_seq_lens_kv

    if batch_offsets_q is not None:
        var_map[UIDs.RAGGED_Q_UID.value] = batch_offsets_q
    if batch_offsets_o is not None:
        var_map[UIDs.RAGGED_O_UID.value] = batch_offsets_o

    if batch_offsets_k is not None:
        var_map[UIDs.RAGGED_K_UID.value] = batch_offsets_k
    if batch_offsets_v is not None:
        var_map[UIDs.RAGGED_V_UID.value] = batch_offsets_v

    if block_tables is not None:
        var_map[UIDs.BLOCK_TABLES_K_UID.value] = block_tables
        var_map[UIDs.BLOCK_TABLES_V_UID.value] = block_tables

    if return_lse:
        var_map[UIDs.STATS_UID.value] = lse
        if batch_offsets_stats is not None:
            var_map[UIDs.RAGGED_STATS_UID.value] = batch_offsets_stats

    if q_scale is not None:
        dummy_scale_tensor = _get_dummy_scale_tensor(q.device)
        var_map[UIDs.Q_SCALE_UID.value] = q_scale
        var_map[UIDs.S_SCALE_UID.value] = dummy_scale_tensor
        var_map[UIDs.S_DESCALE_UID.value] = dummy_scale_tensor
        var_map[UIDs.O_SCALE_UID.value] = dummy_scale_tensor
    if k_scale is not None:
        var_map[UIDs.K_SCALE_UID.value] = k_scale
    if v_scale is not None:
        var_map[UIDs.V_SCALE_UID.value] = v_scale

    handle = _create_cudnn_handle(torch.cuda.current_stream(q.device))
    execute_kwargs = {}
    if override_cache is not None:
        execute_kwargs = _override_execute_kwargs(
            q,
            k_cache,
            v_cache,
            batch_size=cu_seq_lens_q.shape[0] - 1,
            s_qo=max_token_per_sequence,
            s_kv=max_sequence_kv,
            with_stats=return_lse,
        )
    graph.execute(var_map, workspace=workspace_buffer, handle=handle, **execute_kwargs)

    if return_lse:
        # cuDNN emits softmax stats as natural-log LSE; every other FlashInfer
        # backend returns base-2 LSE (they fold log2e into the softmax scale, so
        # their kernels emit base-2 directly). Convert here so the cuDNN backend
        # matches that contract. log2(sum exp(x)) = ln(sum exp(x)) * log2(e).
        # A caller that wants natural log gets the stats as written.
        if lse_base == "log2":
            lse.mul_(log2e)
        return out, lse
    else:
        return out, None


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
        ``batch_offsets_units`` (element offsets are
        ``token_offset * num_heads_qo * head_dim_qk``).  Required when
        ``batch_size > 1`` on the cuDNN graph path; may be omitted only for
        ``batch_size == 1``.
    batch_offsets_o : Optional[torch.Tensor]
        Cumulative per-request start offsets into the packed output tensor,
        shape ``(batch_size + 1,)``, int32, on GPU, in the units given by
        ``batch_offsets_units`` (element offsets are
        ``token_offset * num_heads_qo * head_dim_vo``).  Required when
        ``batch_size > 1`` on the cuDNN graph path; with
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
        Cumulative per-request start offsets into the LSE / stats tensor,
        shape ``(batch_size + 1,)``, in the units given by
        ``batch_offsets_units``.
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
        Pre-allocated output tensor, shape
        ``(total_qo_tokens, num_heads_qo, head_dim_vo)``.  Allocated internally
        when ``None``.
    lse : Optional[torch.Tensor]
        Pre-allocated LSE tensor, shape
        ``(batch_size, max_token_per_sequence, num_heads_qo)``.  Allocated
        internally when ``None`` and ``return_lse`` is ``True``.
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
        ``(total_qo_tokens, num_heads_qo, head_dim_vo)``; ``lse`` has shape
        ``(batch_size, max_token_per_sequence, num_heads_qo)`` when
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

    if return_lse:
        if lse is None:
            lse = torch.empty(
                num_sequences,
                max_token_per_sequence,
                h_qo,
                device=q.device,
                dtype=torch.float32,
            )

    if lse is not None:
        padded = (num_sequences, max_token_per_sequence, h_qo)
        # With a stats ragged offset cuDNN writes each request at its token
        # offset, so a packed [num_tokens, h_qo] buffer (the wrapper contract)
        # is a valid target too.
        packed_ok = batch_offsets_stats is not None and lse.shape == (num_tokens, h_qo)
        if lse.shape != padded and not packed_ok:
            raise ValueError(
                "lse must have shape (num_sequences, max_token_per_sequence, h_qo)"
                + (
                    " or, with batch_offsets_stats, (num_tokens, h_qo)"
                    if batch_offsets_stats is not None
                    else ""
                )
            )

    if o_data_type is None:
        o_data_type = q.dtype

    if out is None:
        out_shape = (num_tokens, h_qo, d_vo)
        out = torch.empty(out_shape, device=q.device, dtype=o_data_type)

    if batch_offsets_units == "tokens":
        # Convenience feature to allow user to set just batch_offsets_{q,k} if desired.
        if batch_offsets_o is None:
            batch_offsets_o = batch_offsets_q
        if batch_offsets_v is None:
            batch_offsets_v = batch_offsets_k

    if CUDNN_AVAILABLE and backend != "cubin":
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
                "cumsum([0, *actual_seq_lens_q]) * num_qo_heads * head_dim, or "
                'token-unit indptrs with batch_offsets_units="tokens".'
            )
        run_kwargs = dict(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            scale=scale,
            workspace_buffer=workspace_buffer,
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
            out=out,
            lse=lse,
            o_data_type=o_data_type,
            lse_base=lse_base,
        )

        if batch_offsets_units == "tokens":
            h_kv = k_cache.shape[1]
            if block_tables is None:
                # Non-paged: both-cumulative direct path (KV is ragged). The
                # token-unit q/k indptrs double as cu_seq_lens.
                use_direct = (
                    _cudnn_supports_direct_seqlens(q.dtype)
                    and batch_offsets_q is not None
                    and batch_offsets_k is not None
                )
            else:
                # Paged: mixed direct path -- cu_seq_len_q (token-unit q indptr)
                # + per-batch actual_seq_lens_kv for the mask; KV addressed via
                # block_tables (no k/v ragged offsets). Requires mixed-form
                # support (cuDNN 9.25+).
                use_direct = (
                    _cudnn_supports_direct_seqlens(q.dtype, mixed=True)
                    and batch_offsets_q is not None
                    and actual_seq_lens_kv is not None
                )
            if use_direct:
                # The token-unit q indptr is both cu_seq_len_q and the Q/O
                # ragged offset (per-tensor multipliers applied in the builder).
                run_kwargs["cu_seq_lens_q"] = batch_offsets_q
                if block_tables is None:
                    run_kwargs["cu_seq_lens_kv"] = batch_offsets_k
                # Paged: KV masked by actual_seq_lens_kv (already in run_kwargs);
                # batch_offsets_k/v stay None.
            else:
                # Old cuDNN/frontend or paged: convert the token-unit indptrs
                # to the element units the legacy graph expects. Names are
                # rebound, not mutated, so the aliasing defaults above (o from
                # q, v from k) still read the original token-unit buffers.

                # The legacy graph's padding mask needs per-request lengths;
                # derive them from the token-unit indptrs when omitted.
                if actual_seq_lens_q is None:
                    run_kwargs["actual_seq_lens_q"] = (
                        batch_offsets_q[1:] - batch_offsets_q[:-1]
                    ).view(-1, 1, 1, 1)
                if actual_seq_lens_kv is None:
                    run_kwargs["actual_seq_lens_kv"] = (
                        batch_offsets_k[1:] - batch_offsets_k[:-1]
                    ).view(-1, 1, 1, 1)

                def apply_multiplier(offsets, multiplier):
                    return offsets * multiplier if offsets is not None else None

                batch_offsets_q = apply_multiplier(batch_offsets_q, h_qo * d_qk)
                batch_offsets_o = apply_multiplier(batch_offsets_o, h_qo * d_vo)
                batch_offsets_k = apply_multiplier(batch_offsets_k, h_kv * d_qk)
                batch_offsets_v = apply_multiplier(batch_offsets_v, h_kv * d_vo)
                batch_offsets_stats = apply_multiplier(batch_offsets_stats, h_qo)

        return _batch_prefill_with_kv_cache(
            **run_kwargs,
            batch_offsets_q=batch_offsets_q,
            batch_offsets_o=batch_offsets_o,
            batch_offsets_k=batch_offsets_k,
            batch_offsets_v=batch_offsets_v,
            batch_offsets_stats=batch_offsets_stats,
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
