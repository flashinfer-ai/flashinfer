from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.attention import cudnn_batch_decode_trace
from ..utils import log2e
from .utils import (
    get_cudnn_fmha_gen_module,
    get_cudnn_attention_handle,
    supports_ordered_cudnn_execution,
)

try:
    import cudnn

    CUDNN_AVAILABLE = True
except ImportError:
    cudnn = None
    CUDNN_AVAILABLE = False


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

    SINK_UID = 300  # Attention sink logits, (1, num_heads_qo, 1, 1) fp32

    O_UID = 1000  # Output tensor
    STATS_UID = 1001  # Stats tensor


def _tensor_layout_key(t: Optional[torch.Tensor]):
    if t is None:
        return None
    return (tuple(t.shape), tuple(t.stride()), t.dtype)


def _sdpa_decode_key_fn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    *,
    max_sequence_kv: int,
    actual_seq_lens_q: Optional[torch.Tensor] = None,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    q_len_per_req: int = 1,
    window_left: int = -1,
    sinks: Optional[torch.Tensor] = None,
):
    return (
        "decode",
        q.device,
        max_sequence_kv,
        tuple(q.shape),
        # Q strides are baked into the graph descriptor: a query sliced out of
        # a packed QKV buffer (batch stride > h*d) must not replay a graph built
        # for a contiguous query.
        tuple(q.stride()),
        # K/V shapes and strides are baked into the built graph via
        # tensor_like (v_cache also supplies d_vo for the O dims), so both
        # caches key on their full layout, not just k_cache's shape.
        tuple(k_cache.shape),
        tuple(v_cache.shape),
        tuple(k_cache.stride()),
        tuple(v_cache.stride()),
        # I/O data types are baked into the built graph; same-shape calls that
        # differ only in dtype must not share a graph (a replayed graph would
        # silently reinterpret the buffers as the first caller's dtype).
        q.dtype,
        k_cache.dtype,
        v_cache.dtype,
        # attn_scale is baked into the built graph as a compile-time constant;
        # omitting it silently replays a stale-scale graph on same-shape calls.
        scale,
        # These presence flags change the built graph's structure (padding
        # mask, paged tables, Stats output) the same way: same-shape calls
        # that differ only in them must not share a graph.
        actual_seq_lens_q is not None,
        actual_seq_lens_kv is not None,
        # The block table's dims/strides/dtype are baked via tensor_like: a
        # same-batch table with a different pages-per-seq width must not share
        # a graph (the replay would walk rows with the stale row stride).
        tuple(block_tables.shape) if block_tables is not None else None,
        tuple(block_tables.stride()) if block_tables is not None else None,
        block_tables.dtype if block_tables is not None else None,
        # The seq-len tensors are bound via tensor_like, which bakes their
        # dtypes, dims and strides (an int64 buffer bound to a graph built for
        # int32 would be silently read as int32; a (bs,) buffer and a
        # (bs, 1, 1, 1) buffer need different descriptors).
        _tensor_layout_key(actual_seq_lens_q),
        _tensor_layout_key(actual_seq_lens_kv),
        return_lse,
        # The mask is baked into the graph: q_len_per_req > 1 adds the
        # bottom-right causal diagonal, window_left its left band bound; the
        # sink tensor's layout is baked via tensor_like. q's shape already
        # carries q_len_per_req, the explicit entries keep the key readable.
        q_len_per_req,
        window_left,
        _tensor_layout_key(sinks),
    )


if CUDNN_AVAILABLE:

    @cudnn.jit(heur_modes=[cudnn.heur_mode.A])
    @cudnn.graph_cache(key_fn=_sdpa_decode_key_fn)
    def _build_decode_graph(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        scale: float,
        *,
        max_sequence_kv: int,
        actual_seq_lens_q: Optional[torch.Tensor] = None,
        actual_seq_lens_kv: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        q_len_per_req: int = 1,
        window_left: int = -1,
        sinks: Optional[torch.Tensor] = None,
    ):
        handle = _create_cudnn_handle(torch.cuda.current_stream(q.device))

        # Decode represents fixed-length queries with dense strided Q/O
        # descriptors; public batch offsets are only used by the cubin path.

        # Q and O carry explicit data types (K/V inherit theirs from the torch
        # tensors via tensor_like below); derive them from q.dtype instead of
        # hardcoding, so fp16 callers are not silently reinterpreted as bf16.
        cudnn_q_data_type = cudnn.datatypes._torch_to_cudnn_data_type(q.dtype)
        cudnn_o_data_type = cudnn_q_data_type

        with cudnn.graph(handle) as (g, _):
            if q.dim() == 3:
                s_qo = 1
                b, h_qo, d_qk = q.shape[0], q.shape[1], q.shape[2]
            elif q.dim() == 4:
                b, h_qo, s_qo, d_qk = (
                    q.shape[0],
                    q.shape[1],
                    q.shape[2],
                    q.shape[3],
                )
            else:
                raise ValueError(f"q must have 3 or 4 dimensions, got {q.dim()}")

            assert s_qo == q_len_per_req, (
                f"q's sequence dim ({s_qo}) must equal q_len_per_req ({q_len_per_req})"
            )
            assert k_cache.dim() == 4, "k_cache must have 4 dimensions"

            d_vo = v_cache.shape[3]

            # Use the caller's strides: a query sliced from a packed QKV buffer
            # has a batch stride larger than h_qo * d_qk. For a 3-D q, s_qo == 1
            # and the sequence stride is immaterial; reuse the batch stride. A
            # 4-D q is the (batch, heads, q_len_per_req, d) view the public
            # entry point builds for multi-token decode.
            if q.dim() == 3:
                q_stride = (q.stride(0), q.stride(1), q.stride(0), q.stride(2))
            else:
                q_stride = tuple(q.stride())
            cudnn_q = g.tensor(
                name="q",
                dim=(b, h_qo, s_qo, d_qk),
                stride=q_stride,
                data_type=cudnn_q_data_type,
            )

            cudnn_k_cache = g.tensor_like(k_cache)
            cudnn_v_cache = g.tensor_like(v_cache)

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

            if actual_seq_lens_q is not None:
                cudnn_actual_seq_lens_q = g.tensor_like(actual_seq_lens_q)
                cudnn_actual_seq_lens_q.set_uid(UIDs.ACTUAL_SEQ_LENS_Q_UID.value)

            if actual_seq_lens_kv is not None:
                cudnn_actual_seq_lens_kv = g.tensor_like(actual_seq_lens_kv)
                cudnn_actual_seq_lens_kv.set_uid(UIDs.ACTUAL_SEQ_LENS_KV_UID.value)
                cudnn_actual_seq_lens_kv.set_is_pass_by_value(False)

            padding_mask = actual_seq_lens_kv is not None

            cudnn_sinks = None
            if sinks is not None:
                # (1, H, 1, 1) fp32 sink logits: one extra softmax column per
                # head with a zero value row (Streaming-LLM / gpt-oss sinks),
                # the same contract as FlashInfer's fa2 / trtllm-gen ``sinks``.
                cudnn_sinks = g.tensor_like(sinks)
                cudnn_sinks.set_uid(UIDs.SINK_UID.value)

            # Multi-token decode (speculative / MTP verification) is the
            # bottom-right causal diagonal: row i of a request's s_qo rows sees
            # keys 0 .. kv_len - s_qo + i. A sliding window needs the same
            # alignment (the diagonal is its right bound). FlashInfer's
            # window_left counts the keys strictly before the diagonal; cuDNN's
            # left bound counts the diagonal too, hence the + 1.
            mask_kwargs: dict[str, Any] = {}
            if s_qo > 1 or window_left >= 0:
                mask_kwargs["use_causal_mask_bottom_right"] = True
            if window_left >= 0:
                mask_kwargs["diagonal_band_left_bound"] = window_left + 1
            if cudnn_sinks is not None:
                mask_kwargs["sink_token"] = cudnn_sinks

            O, Stats = g.sdpa(
                name="sdpa",
                q=cudnn_q,
                k=cudnn_k_cache,
                v=cudnn_v_cache,
                seq_len_q=(
                    cudnn_actual_seq_lens_q if actual_seq_lens_q is not None else None
                ),
                seq_len_kv=(
                    cudnn_actual_seq_lens_kv if actual_seq_lens_kv is not None else None
                ),
                use_padding_mask=padding_mask,
                generate_stats=return_lse,
                attn_scale=scale,
                paged_attention_k_table=cudnn_k_block_tables,
                paged_attention_v_table=cudnn_v_block_tables,
                paged_attention_max_seq_len_kv=max_sequence_kv,
                compute_data_type=cudnn.data_type.FLOAT,
                **mask_kwargs,
            )

            # O is bound to a contiguous (batch * s_qo, h_qo, d_vo) buffer:
            # token-major rows, heads inside a row.
            O.set_uid(UIDs.O_UID.value).set_output(True).set_dim(
                [b, h_qo, s_qo, d_vo]
            ).set_stride([s_qo * h_qo * d_vo, d_vo, h_qo * d_vo, 1]).set_data_type(
                cudnn_o_data_type
            )

            if return_lse:
                # Same layout as prefill's Stats: fp32, (b, h_qo, s_qo, 1) with
                # token-major strides, which is exactly a contiguous
                # (batch * s_qo, num_heads_qo) fp32 buffer.
                Stats.set_uid(UIDs.STATS_UID.value).set_output(True).set_data_type(
                    cudnn.data_type.FLOAT
                ).set_dim([b, h_qo, s_qo, 1]).set_stride([s_qo * h_qo, 1, h_qo, 1])

        tensors_to_return = [cudnn_q, cudnn_k_cache, cudnn_v_cache, O]
        if return_lse:
            tensors_to_return.append(Stats)
        if cudnn_sinks is not None:
            tensors_to_return.append(cudnn_sinks)

        if actual_seq_lens_q is not None:
            tensors_to_return.append(cudnn_actual_seq_lens_q)
        if actual_seq_lens_kv is not None:
            tensors_to_return.append(cudnn_actual_seq_lens_kv)

        return g, tensors_to_return


@dataclass(frozen=True)
class _DecodeInputs:
    """Validated call buffers; only the prepared path retains the sink view."""

    q: torch.Tensor
    out: torch.Tensor
    lse: Optional[torch.Tensor]
    sinks: Optional[torch.Tensor]
    batch_size: int


def _decode_q_view(q: torch.Tensor, batch_size: int, q_len_per_req: int):
    if q_len_per_req == 1:
        return q
    return q.view(batch_size, q_len_per_req, q.shape[1], q.shape[2]).transpose(1, 2)


def _validate_decode_devices(device, tensors):
    # Keys describe tensor geometry, not placement. Rebound pointers must be
    # checked even when a graph signature matches.
    for name, tensor in tensors:
        if tensor is not None and tensor.device != device:
            raise ValueError(
                f"{name} must be on the same device as q ({device}), got {tensor.device}"
            )


def _decode_sinks(q, sinks, *, normalize=True):
    if sinks is None:
        return None
    if (
        sinks.dim() != 1
        or sinks.shape[0] != q.shape[1]
        or sinks.dtype != torch.float32
        or sinks.device != q.device
    ):
        raise ValueError(
            f"sinks must be a float32 tensor of shape (num_heads_qo,) = "
            f"({q.shape[1]},) on {q.device}, got shape {tuple(sinks.shape)} "
            f"with dtype {sinks.dtype} on {sinks.device}"
        )
    return sinks.contiguous().view(1, q.shape[1], 1, 1) if normalize else sinks


def _check_dense_decode_offsets(
    q, out, batch_size, q_len_per_req, offsets_q, offsets_o
):
    """Keep legacy dense offsets without silently accepting ragged addressing.

    This is only used by the public API when explicit offsets are supplied.
    CUDA value checks stay on device, so capture also checks them on replay.
    """
    for name, offsets, tensor in (
        ("batch_offsets_q", offsets_q, q),
        ("batch_offsets_o", offsets_o, out),
    ):
        if offsets is None:
            continue
        if (
            offsets.device != q.device
            or offsets.dtype not in (torch.int32, torch.int64)
            or offsets.dim() != 1
            or offsets.numel() not in (batch_size, batch_size + 1)
        ):
            raise ValueError(
                f"{name} must be an int32 or int64 tensor of shape "
                f"({batch_size},) or ({batch_size + 1},) on {q.device}"
            )
        expected = torch.arange(
            offsets.numel(), device=offsets.device, dtype=torch.int64
        ) * (q_len_per_req * tensor.stride(0))
        torch._assert_async(
            (offsets == expected).all(),
            f"{name}: the cuDNN decode path only supports dense offsets "
            "matching the tensor's batch stride; omit offsets for dense decode",
        )


def _normalize_decode_inputs(
    q,
    k_cache,
    v_cache,
    workspace_buffer,
    *,
    actual_seq_lens_kv,
    block_tables,
    out,
    return_lse,
    lse,
    q_len_per_req,
    sinks,
) -> _DecodeInputs:
    """Shared cold validation, output allocation and query/sink normalization.

    The public entry can allocate outputs and copy a strided query. The
    prepared entry checks its narrower, supplied-buffer contract first.
    """
    if q_len_per_req < 1:
        raise ValueError(f"q_len_per_req must be >= 1, got {q_len_per_req}")
    if q.dim() != 3 or q.shape[0] % q_len_per_req != 0:
        raise ValueError(
            "q must have shape (batch_size * q_len_per_req, num_heads_qo, head_dim); "
            f"got {tuple(q.shape)} with q_len_per_req={q_len_per_req}"
        )
    rows = q.shape[0]
    bs = rows // q_len_per_req
    h_qo = q.shape[1]
    d_vo = v_cache.shape[3]

    supported_dtypes = (torch.float16, torch.bfloat16)
    for name, t in (("q", q), ("k_cache", k_cache), ("v_cache", v_cache)):
        if t.dtype not in supported_dtypes:
            raise ValueError(
                f"cudnn_batch_decode_with_kv_cache only supports torch.float16 "
                f"and torch.bfloat16, got {name}.dtype={t.dtype}"
            )
    if out is not None and out.dtype != q.dtype:
        raise ValueError(
            f"out.dtype ({out.dtype}) must match q.dtype ({q.dtype}); the "
            "output is produced in the query's data type"
        )

    if return_lse:
        if not CUDNN_AVAILABLE:
            raise NotImplementedError(
                "return_lse=True requires the cuDNN graph backend; it is not "
                "supported by the fallback cubin decode path"
            )
        if lse is None:
            lse = torch.empty(rows, h_qo, device=q.device, dtype=torch.float32)
        elif (
            lse.shape != (rows, h_qo)
            or lse.dtype != torch.float32
            or not lse.is_contiguous()
            or lse.device != q.device
        ):
            raise ValueError(
                "lse must be a contiguous float32 tensor of shape "
                f"(batch_size * q_len_per_req, num_heads_qo) = ({rows}, {h_qo}), "
                f"got shape {tuple(lse.shape)} with dtype {lse.dtype}"
            )

    if out is None:
        out = torch.empty(rows, h_qo, d_vo, device=q.device, dtype=q.dtype)
    elif (
        out.shape != (rows, h_qo, d_vo)
        or out.device != q.device
        or not out.is_contiguous()
    ):
        # O is bound with contiguous (rows, heads, d_vo) strides.
        raise ValueError(
            f"out must be a contiguous tensor of shape ({rows}, {h_qo}, {d_vo}) on "
            f"{q.device}, got shape {tuple(out.shape)} on {out.device}"
        )
    sinks_view = _decode_sinks(q, sinks)
    # Every tensor bound to the graph (and the workspace) must live where q
    # does; the graph executes on q's device with raw pointers.
    _validate_decode_devices(
        q.device,
        (
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("workspace_buffer", workspace_buffer),
            ("lse", lse),
            ("block_tables", block_tables),
            ("actual_seq_lens_kv", actual_seq_lens_kv if CUDNN_AVAILABLE else None),
        ),
    )
    if q.stride(-1) != 1:
        # The graph honors arbitrary batch/head strides but needs a unit
        # innermost stride (TMA/vector loads); such inputs are rare, copy them.
        q = q.contiguous()

    return _DecodeInputs(q, out, lse, sinks_view, bs)


def _execute_decode(
    graph,
    q,
    k_cache,
    v_cache,
    out,
    lse,
    workspace_buffer,
    *,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    block_tables,
    return_lse,
    sinks,
    execution_bindings=None,
):
    """Bind call-local pointers and apply the public base-2 LSE contract."""
    handle_ = _create_cudnn_handle(torch.cuda.current_stream(q.device))

    if execution_bindings is not None:
        # UIDs are fixed by preparation; tensor observation remains FE's job.
        # Each call owns its sequence, including during capture and rebind.
        buffers = [q, k_cache, v_cache, out]
        if return_lse:
            buffers.append(lse)
        if sinks is not None:
            buffers.append(sinks)
        buffers.extend(
            (actual_seq_lens_q, actual_seq_lens_kv, block_tables, block_tables)
        )
        graph.execute(
            buffers,
            workspace=workspace_buffer,
            handle=handle_,
            tensor_uids=execution_bindings,
        )
    else:
        var_map = {
            UIDs.Q_UID.value: q,
            UIDs.K_UID.value: k_cache,
            UIDs.V_UID.value: v_cache,
            UIDs.O_UID.value: out,
        }
        if return_lse:
            var_map[UIDs.STATS_UID.value] = lse
        if sinks is not None:
            var_map[UIDs.SINK_UID.value] = sinks
        if actual_seq_lens_q is not None:
            var_map[UIDs.ACTUAL_SEQ_LENS_Q_UID.value] = actual_seq_lens_q
        if actual_seq_lens_kv is not None:
            var_map[UIDs.ACTUAL_SEQ_LENS_KV_UID.value] = actual_seq_lens_kv

        if block_tables is not None:
            var_map[UIDs.BLOCK_TABLES_K_UID.value] = block_tables
            var_map[UIDs.BLOCK_TABLES_V_UID.value] = block_tables

        graph.execute(var_map, workspace=workspace_buffer, handle=handle_)
    if return_lse:
        # cuDNN emits natural-log softmax stats; FlashInfer's LSE contract is
        # base-2 (the cascade-merge kernels consume it), as in the prefill path.
        lse.mul_(log2e)

    return out


class CudnnDecodeGraph:
    """Reusable paged-decode graph with retained constant query-length storage.

    Each run binds a fresh variant pack; the plan never retains changing
    query, cache, output or metadata pointers. Callers check ``matches`` before
    running. Compatible replans retain this object; workspace replacement
    invalidates it. Sink values are rebound or copied from the current input.

    The signature is the graph-cache key of :func:`_build_decode_graph`
    (shapes, strides, dtypes, scale, presence flags, mask parameters), so a
    prepared graph is reused exactly where the graph cache would replay the
    same graph; anything else re-prepares. Input validation happens in
    :func:`prepare_cudnn_batch_decode`; runtime devices are checked on every
    rebind. Wrappers own the constant per-batch query-length storage across
    graph replacement; standalone preparation allocates it when needed.
    """

    __slots__ = (
        "key",
        "graph",
        "return_lse",
        "batch_size",
        "q_len_per_req",
        "out_shape",
        "lse_shape",
        "sinks_view",
        "seq_lens_q",
        "execution_bindings",
    )

    def __init__(
        self,
        key,
        graph,
        *,
        return_lse: bool,
        batch_size: int,
        q_len_per_req: int,
        out_shape: tuple,
        lse_shape: Optional[tuple],
        sinks_view: Optional[torch.Tensor],
        seq_lens_q: torch.Tensor,
    ):
        self.key = key
        self.graph = graph
        self.return_lse = return_lse
        self.batch_size = batch_size
        self.q_len_per_req = q_len_per_req
        self.out_shape = out_shape
        self.lse_shape = lse_shape
        self.sinks_view = sinks_view
        self.seq_lens_q = seq_lens_q
        self.execution_bindings = None
        if supports_ordered_cudnn_execution(type(graph)):
            uids = [
                UIDs.Q_UID.value,
                UIDs.K_UID.value,
                UIDs.V_UID.value,
                UIDs.O_UID.value,
            ]
            if return_lse:
                uids.append(UIDs.STATS_UID.value)
            if sinks_view is not None:
                uids.append(UIDs.SINK_UID.value)
            uids.extend(
                (
                    UIDs.ACTUAL_SEQ_LENS_Q_UID.value,
                    UIDs.ACTUAL_SEQ_LENS_KV_UID.value,
                    UIDs.BLOCK_TABLES_K_UID.value,
                    UIDs.BLOCK_TABLES_V_UID.value,
                )
            )
            self.execution_bindings = tuple(uids)

    def _q_graph(self, q: torch.Tensor) -> torch.Tensor:
        return _decode_q_view(q, self.batch_size, self.q_len_per_req)

    def matches(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        scale: float,
        *,
        max_sequence_kv: int,
        actual_seq_lens_kv: torch.Tensor,
        block_tables: torch.Tensor,
        return_lse: bool,
        q_len_per_req: int,
        window_left: int,
        out: torch.Tensor,
        lse: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor],
    ) -> bool:
        """Whether this prepared graph serves the call: same graph-cache key
        (shapes, strides, dtypes, scale, flags, mask), out / lse keep their
        layout, and sink presence is unchanged."""
        if (
            q_len_per_req != self.q_len_per_req
            or q.shape[0] != self.batch_size * q_len_per_req
        ):
            return False
        if (sinks is None) != (self.sinks_view is None):
            return False
        _decode_sinks(q, sinks, normalize=False)
        if (
            out.shape != self.out_shape
            or not out.is_contiguous()
            or out.dtype != q.dtype
            or out.device != q.device
        ):
            return False
        if self.return_lse != return_lse or (
            return_lse
            and (
                lse is None
                or lse.shape != self.lse_shape
                or not lse.is_contiguous()
                or lse.dtype != torch.float32
                or lse.device != q.device
            )
        ):
            return False
        key = _sdpa_decode_key_fn(
            self._q_graph(q),
            k_cache,
            v_cache,
            scale,
            max_sequence_kv=max_sequence_kv,
            actual_seq_lens_q=self.seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            return_lse=return_lse,
            q_len_per_req=q_len_per_req,
            window_left=window_left,
            sinks=self.sinks_view,
        )
        return key == self.key

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        lse: Optional[torch.Tensor],
        workspace_buffer: torch.Tensor,
        *,
        actual_seq_lens_kv: torch.Tensor,
        block_tables: torch.Tensor,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Matching graph metadata does not enforce the runtime device contract.
        # Revalidate every rebound buffer before handing raw pointers to cuDNN,
        # including the plan-owned constant and a replaced workspace.
        _validate_decode_devices(
            q.device,
            (
                ("seq_lens_q", self.seq_lens_q),
                ("k_cache", k_cache),
                ("v_cache", v_cache),
                ("workspace_buffer", workspace_buffer),
                ("actual_seq_lens_kv", actual_seq_lens_kv),
                ("block_tables", block_tables),
            ),
        )
        return _execute_decode(
            self.graph,
            self._q_graph(q),
            k_cache,
            v_cache,
            out,
            lse,
            workspace_buffer,
            actual_seq_lens_q=self.seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            return_lse=self.return_lse,
            # Normalize current values on this call's stream. In capture, a
            # strided sink's copy is captured too, rather than cached stale.
            sinks=self.sinks_view if sinks is None else _decode_sinks(q, sinks),
            execution_bindings=self.execution_bindings,
        )


def prepare_cudnn_batch_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    *,
    max_sequence_kv: int,
    actual_seq_lens_kv: torch.Tensor,
    block_tables: torch.Tensor,
    out: torch.Tensor,
    return_lse: bool,
    lse: Optional[torch.Tensor],
    q_len_per_req: int,
    window_left: int,
    sinks: Optional[torch.Tensor],
    actual_seq_lens_q: Optional[torch.Tensor] = None,
) -> CudnnDecodeGraph:
    """Validate once and build (or fetch from the graph cache) the paged-decode
    graph for this signature; steps then execute through
    :meth:`CudnnDecodeGraph.run`. Same contract as
    :func:`cudnn_batch_decode_with_kv_cache` in the paged, per-batch-length,
    ``out``-provided form the wrappers use."""
    if not CUDNN_AVAILABLE:
        raise NotImplementedError(
            "prepare_cudnn_batch_decode requires the cuDNN graph backend"
        )
    if out is None or (return_lse and lse is None):
        raise ValueError("prepared decode requires supplied out and active lse buffers")
    if q.dim() > 0 and q.stride(-1) != 1:
        raise ValueError("q must have a unit innermost stride")
    if actual_seq_lens_kv is None or block_tables is None:
        raise ValueError("prepared decode requires KV lengths and block tables")
    inputs = _normalize_decode_inputs(
        q,
        k_cache,
        v_cache,
        None,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        out=out,
        return_lse=return_lse,
        lse=lse,
        q_len_per_req=q_len_per_req,
        sinks=sinks,
    )
    bs, sinks_view = inputs.batch_size, inputs.sinks
    seq_lens_q = actual_seq_lens_q
    if seq_lens_q is None:
        seq_lens_q = torch.full(
            (bs, 1, 1, 1), q_len_per_req, device=q.device, dtype=torch.int32
        )
    elif (
        seq_lens_q.shape != (bs, 1, 1, 1)
        or seq_lens_q.dtype != torch.int32
        or seq_lens_q.device != q.device
        or not seq_lens_q.is_contiguous()
    ):
        raise ValueError(
            "actual_seq_lens_q must be contiguous int32 (batch_size, 1, 1, 1) on q.device"
        )
    q_graph = _decode_q_view(q, bs, q_len_per_req)
    kwargs = dict(
        max_sequence_kv=max_sequence_kv,
        actual_seq_lens_q=seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        return_lse=return_lse,
        q_len_per_req=q_len_per_req,
        window_left=window_left,
        sinks=sinks_view,
    )
    graph, _ = _build_decode_graph(q_graph, k_cache, v_cache, scale, **kwargs)
    key = _sdpa_decode_key_fn(q_graph, k_cache, v_cache, scale, **kwargs)
    return CudnnDecodeGraph(
        key,
        graph,
        return_lse=return_lse,
        batch_size=bs,
        q_len_per_req=q_len_per_req,
        out_shape=tuple(out.shape),
        lse_shape=tuple(lse.shape) if return_lse else None,
        sinks_view=sinks_view,
        seq_lens_q=seq_lens_q,
    )


@flashinfer_api(trace=cudnn_batch_decode_trace)
def cudnn_batch_decode_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    max_sequence_kv: int,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    is_cuda_graph_compatible: bool = False,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    batch_offsets_k: Optional[torch.Tensor] = None,
    batch_offsets_v: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse: Optional[torch.Tensor] = None,
    q_len_per_req: int = 1,
    window_left: int = -1,
    sinks: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Batched decode attention with paged KV cache, backed by cuDNN SDPA.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape ``(batch_size * q_len_per_req, num_heads_qo, head_dim)``
        (``(batch_size, num_heads_qo, head_dim)`` for plain one-token decode),
        ``torch.float16`` or ``torch.bfloat16`` (the output uses ``q.dtype``).
        With ``q_len_per_req > 1`` the rows of one request are consecutive.
        ``torch.float16`` requires the cuDNN graph backend; the fallback
        (cubin) path is bf16-only and raises ``NotImplementedError``.
    k_cache : torch.Tensor
        Key cache, shape ``(total_num_pages, num_heads_kv, page_size, head_dim)``.
    v_cache : torch.Tensor
        Value cache, shape ``(total_num_pages, num_heads_kv, page_size, head_dim)``.
    scale : float
        Softmax scaling factor, typically ``1 / sqrt(head_dim)``.
    workspace_buffer : torch.Tensor
        Workspace buffer for cuDNN.  Scales with batch size; 128 MB is sufficient
        for typical decode workloads.
    max_sequence_kv : int
        Maximum number of tokens per KV sequence in the batch (``s_kv_max``).
    actual_seq_lens_kv : Optional[torch.Tensor]
        Per-request KV lengths, shape ``(batch_size,)``.  When cuDNN is
        available (the default backend) this tensor must reside on the
        same CUDA device as ``q``.  Only the fallback non-cuDNN path
        accepts (and internally copies) a CPU tensor.
    block_tables : Optional[torch.Tensor]
        Page-table mapping for the paged KV cache, shape
        ``(batch_size, num_pages_per_seq)`` on GPU.
    is_cuda_graph_compatible : bool
        Whether to plan the operation in a CUDA-graph-capture-safe mode.
    batch_offsets_q : Optional[torch.Tensor]
        Per-request element offsets into the query tensor, int32 or int64, shape
        ``(batch_size,)`` or ``(batch_size + 1,)`` on GPU (optional end offset).
        The cuDNN graph path accepts only dense offsets matching the query's
        batch stride. Prefer ``None`` to avoid redundant device-side checks.
    batch_offsets_o : Optional[torch.Tensor]
        Like ``batch_offsets_q``, but for the contiguous output tensor.
        On the cuDNN graph path, non-dense Q/O offsets trigger an asynchronous
        device assertion, including if changed before CUDA graph replay.
    batch_offsets_k : Optional[torch.Tensor]
        Per-request offsets into the key tensor, shape ``(batch_size,)`` on GPU.
    batch_offsets_v : Optional[torch.Tensor]
        Per-request offsets into the value tensor, shape ``(batch_size,)`` on GPU.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor, shape ``(batch_size, num_heads_qo, head_dim)``
        with dtype ``q.dtype``; allocated internally when ``None``.
    return_lse : bool
        Whether to also return the log-sum-exp of the attention scores
        (cuDNN's SDPA ``Stats`` output).  Requires the cuDNN graph backend;
        raises ``NotImplementedError`` on the fallback (cubin) path.
    lse : Optional[torch.Tensor]
        Pre-allocated LSE tensor, shape ``(batch_size * q_len_per_req, num_heads_qo)``,
        ``torch.float32``, contiguous, on the same device as ``q``; allocated
        internally when ``None`` and ``return_lse`` is ``True``.
    q_len_per_req : int
        Query rows per request (speculative / multi-token-prediction
        verification). Rows of one request attend under the bottom-right causal
        diagonal: row ``i`` of a request with ``kv_len`` keys sees keys
        ``0 .. kv_len - q_len_per_req + i``. Every request needs
        ``kv_len >= q_len_per_req``. Defaults to ``1`` (no mask beyond padding).
    window_left : int
        Left sliding-window bound in FlashInfer's convention: a row attends to
        the ``window_left`` keys before its diagonal position plus that position
        itself; ``-1`` (default) disables the window.
    sinks : Optional[torch.Tensor]
        Per-head attention sink logits, shape ``(num_heads_qo,)``,
        ``torch.float32``, on ``q``'s device. ``sinks[h]`` joins each row's
        softmax denominator as one extra logit with a zero value row, as in
        FlashInfer's other backends (gpt-oss / Streaming-LLM sinks). Whether the
        cuDNN stack serves a sink at ``q_len_per_req == 1`` is decided by its
        SDPA engines (cudnn-frontend 1.30+ with the FROST engines enabled does;
        the backend engine raises a not-supported error at graph build).

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Output tensor of shape ``(batch_size * q_len_per_req, num_heads_qo, head_dim)``
        when ``return_lse=False``; otherwise ``(output, lse)`` where ``lse`` has
        shape ``(batch_size * q_len_per_req, num_heads_qo)`` and dtype
        ``torch.float32``.

    Note
    ----
    All tensors must be on the same CUDA device. ``q`` may carry arbitrary batch/head strides (e.g. a slice of a
    packed QKV projection) as long as ``head_dim`` is innermost and dense;
    ``out``/``lse`` must be contiguous.  Query and KV heads may differ
    (``num_heads_qo >= num_heads_kv``, multi-query / grouped-query attention).

    LSE convention: ``lse[b, h]`` is the **base-2** log-sum-exp of the
    pre-softmax attention row with ``scale`` folded in, i.e.
    ``log2(sum_j(exp(scale * q[b, h] . k[b, h // (num_heads_qo // num_heads_kv), j])))``
    summed over the valid KV positions ``j < actual_seq_lens_kv[b]`` — the same
    contract as every other FlashInfer backend (``torch.logsumexp(...) * log2(e)``),
    so it can be fed to the cascade-merge kernels. cuDNN emits natural-log stats;
    they are folded to base-2 here.
    """

    inputs = _normalize_decode_inputs(
        q,
        k_cache,
        v_cache,
        workspace_buffer,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        out=out,
        return_lse=return_lse,
        lse=lse,
        q_len_per_req=q_len_per_req,
        sinks=sinks,
    )
    if CUDNN_AVAILABLE and (batch_offsets_q is not None or batch_offsets_o is not None):
        _check_dense_decode_offsets(
            q,
            inputs.out,
            inputs.batch_size,
            q_len_per_req,
            batch_offsets_q,
            batch_offsets_o,
        )
    q, out, lse, sinks_view, bs = (
        inputs.q,
        inputs.out,
        inputs.lse,
        inputs.sinks,
        inputs.batch_size,
    )

    if not CUDNN_AVAILABLE:
        if q_len_per_req > 1 or window_left >= 0 or sinks is not None:
            raise NotImplementedError(
                "q_len_per_req > 1, window_left and sinks require the cuDNN graph "
                "backend (the cudnn-frontend python package); the fallback cubin "
                "decode path serves plain one-token decode only"
            )
        for name, t in (("q", q), ("k_cache", k_cache), ("v_cache", v_cache)):
            if t.dtype != torch.bfloat16:
                # The fallback cubins are compiled for bf16 only; passing fp16
                # buffers through would silently reinterpret them as bf16.
                raise NotImplementedError(
                    f"{name}.dtype={t.dtype} requires the cuDNN graph backend; the "
                    "fallback cubin decode path only supports torch.bfloat16"
                )
        actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.device, non_blocking=True)

        run_func = get_cudnn_fmha_gen_module().decode
        run_func(
            max_sequence_kv,
            q.contiguous(),  # the cubin path assumes a dense (b, h, d) query
            k_cache,
            v_cache,
            scale,
            workspace_buffer,
            actual_seq_lens_kv,
            actual_seq_lens_kv_gpu,
            block_tables,
            out,
            batch_offsets_q,
            batch_offsets_o,
            is_cuda_graph_compatible,
        )
    else:
        actual_seq_lens_q = torch.full(
            (bs, 1, 1, 1), q_len_per_req, device=q.device, dtype=torch.int32
        )
        # Multi-token rows are presented to the graph as (batch, heads,
        # q_len_per_req, d): a strided view, no copy (the rows of one request
        # are consecutive in q, whatever its batch stride).
        q_graph = _decode_q_view(q, bs, q_len_per_req)

        graph, _ = _build_decode_graph(
            q=q_graph,
            k_cache=k_cache,
            v_cache=v_cache,
            scale=scale,
            max_sequence_kv=max_sequence_kv,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            return_lse=return_lse,
            q_len_per_req=q_len_per_req,
            window_left=window_left,
            sinks=sinks_view,
        )
        _execute_decode(
            graph,
            q_graph,
            k_cache,
            v_cache,
            out,
            lse,
            workspace_buffer,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            return_lse=return_lse,
            sinks=sinks_view,
        )

    if return_lse:
        return out, lse
    return out
