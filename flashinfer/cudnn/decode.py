from enum import Enum
from typing import Optional, Union

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.attention import cudnn_batch_decode_trace
from .utils import get_cudnn_fmha_gen_module

try:
    import cudnn

    CUDNN_AVAILABLE = True
except ImportError:
    cudnn = None
    CUDNN_AVAILABLE = False

# Global cudnn handle. need to make it per device in future
_cudnn_handle = None


def _create_cudnn_handle(stream: torch.cuda.Stream):
    global _cudnn_handle
    if _cudnn_handle is None:
        _cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(_cudnn_handle, stream.cuda_stream)
    return _cudnn_handle


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

    O_UID = 1000  # Output tensor
    STATS_UID = 1001  # Stats tensor


def _sdpa_decode_key_fn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    *,
    max_sequence_kv: int,
    block_size: Optional[int] = 1,
    actual_seq_lens_q: Optional[torch.Tensor] = None,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    return_lse: bool = False,
):
    return (
        "decode",
        max_sequence_kv,
        tuple(q.shape),
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
        block_tables.dtype if block_tables is not None else None,
        # The seq-len tensors are bound via tensor_like, which bakes their
        # dtypes (an int64 buffer bound to a graph built for int32 would be
        # silently read as int32).
        actual_seq_lens_q.dtype if actual_seq_lens_q is not None else None,
        actual_seq_lens_kv.dtype if actual_seq_lens_kv is not None else None,
        return_lse,
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
        block_size: Optional[int] = 1,
        actual_seq_lens_q: Optional[torch.Tensor] = None,
        actual_seq_lens_kv: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        batch_offsets_q: Optional[torch.Tensor] = None,
        batch_offsets_o: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ):
        handle = _create_cudnn_handle(torch.cuda.current_stream())

        # WAR: override batch offsets for now, as it leads to a poor performance
        batch_offsets_q = None
        batch_offsets_o = None

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

            assert s_qo == 1, "q must have a sequence length of 1"
            assert k_cache.dim() == 4, "k_cache must have 4 dimensions"

            d_vo = v_cache.shape[3]

            cudnn_q = g.tensor(
                name="q",
                dim=(b, h_qo, s_qo, d_qk),
                stride=(h_qo * d_qk, d_qk, d_qk * h_qo, 1),
                data_type=cudnn_q_data_type,
            )
            if batch_offsets_q is not None:
                ragged_q = g.tensor_like(batch_offsets_q)
                ragged_q.set_uid(UIDs.RAGGED_Q_UID.value)
                cudnn_q.set_ragged_offset(ragged_q)

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
            )

            if batch_offsets_o is not None:
                ragged_o = g.tensor_like(batch_offsets_o)
                ragged_o.set_uid(UIDs.RAGGED_O_UID.value)
                O.set_ragged_offset(ragged_o)

            O.set_uid(UIDs.O_UID.value).set_output(True).set_dim(
                [b, h_qo, s_qo, d_vo]
            ).set_stride([d_vo * h_qo, d_vo, d_vo * h_qo, 1]).set_data_type(
                cudnn_o_data_type
            )

            if return_lse:
                # Same layout as prefill's Stats with s_qo == 1: fp32,
                # (b, h_qo, 1, 1) with token-major strides, which is exactly a
                # contiguous (batch, num_heads_qo) fp32 buffer.
                Stats.set_uid(UIDs.STATS_UID.value).set_output(True).set_data_type(
                    cudnn.data_type.FLOAT
                ).set_dim([b, h_qo, s_qo, 1]).set_stride([s_qo * h_qo, 1, h_qo, 1])

        tensors_to_return = [cudnn_q, cudnn_k_cache, cudnn_v_cache, O]
        if return_lse:
            tensors_to_return.append(Stats)

        if actual_seq_lens_q is not None:
            tensors_to_return.append(cudnn_actual_seq_lens_q)
        if actual_seq_lens_kv is not None:
            tensors_to_return.append(cudnn_actual_seq_lens_kv)

        return g, tensors_to_return


def _batch_decode_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    max_sequence_kv: int,
    actual_seq_lens_q: Optional[torch.Tensor] = None,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    block_size: Optional[int] = 1,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    batch_offsets_k: Optional[torch.Tensor] = None,
    batch_offsets_v: Optional[torch.Tensor] = None,
    out: torch.Tensor,
    return_lse: bool = False,
    lse: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    graph, tensors = _build_decode_graph(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        scale=scale,
        max_sequence_kv=max_sequence_kv,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        block_size=block_size,
        batch_offsets_q=batch_offsets_q if batch_offsets_q is not None else None,
        batch_offsets_o=batch_offsets_q if batch_offsets_q is not None else None,
        return_lse=return_lse,
    )

    handle_ = _create_cudnn_handle(torch.cuda.current_stream())

    var_map = {
        UIDs.Q_UID.value: q,
        UIDs.K_UID.value: k_cache,
        UIDs.V_UID.value: v_cache,
        UIDs.O_UID.value: out,
    }
    if return_lse:
        var_map[UIDs.STATS_UID.value] = lse
    if actual_seq_lens_q is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_Q_UID.value] = actual_seq_lens_q
    if actual_seq_lens_kv is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_KV_UID.value] = actual_seq_lens_kv

    if batch_offsets_q is not None:
        var_map[UIDs.RAGGED_Q_UID.value] = batch_offsets_q
    if batch_offsets_o is not None:
        var_map[UIDs.RAGGED_O_UID.value] = batch_offsets_o

    if block_tables is not None:
        var_map[UIDs.BLOCK_TABLES_K_UID.value] = block_tables
        var_map[UIDs.BLOCK_TABLES_V_UID.value] = block_tables

    graph.execute(var_map, workspace=workspace_buffer, handle=handle_)

    return out


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
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Batched decode attention with paged KV cache, backed by cuDNN SDPA.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape ``(batch_size, num_heads_qo, head_dim)``,
        ``torch.float16`` or ``torch.bfloat16`` (the output uses ``q.dtype``).
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
        Per-request offsets into the query tensor, shape ``(batch_size,)`` on GPU.
    batch_offsets_o : Optional[torch.Tensor]
        Per-request offsets into the output tensor, shape ``(batch_size,)`` on GPU.
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
        Pre-allocated LSE tensor, shape ``(batch_size, num_heads_qo)``,
        ``torch.float32``, contiguous, on the same device as ``q``; allocated
        internally when ``None`` and ``return_lse`` is ``True``.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Output tensor of shape ``(batch_size, num_heads_qo, head_dim)`` when
        ``return_lse=False``; otherwise ``(output, lse)`` where ``lse`` has
        shape ``(batch_size, num_heads_qo)`` and dtype ``torch.float32``.

    Note
    ----
    Currently only supports causal attention; all tensors must be contiguous and
    on the same CUDA device.  Query and KV heads may differ
    (``num_heads_qo >= num_heads_kv``, multi-query / grouped-query attention).

    LSE convention: ``lse[b, h]`` is the natural-log log-sum-exp of the
    pre-softmax attention row with ``scale`` folded in, i.e.
    ``log(sum_j(exp(scale * q[b, h] . k[b, h // (num_heads_qo // num_heads_kv), j])))``
    summed over the valid KV positions ``j < actual_seq_lens_kv[b]`` (matching
    ``torch.logsumexp`` on the masked, scaled scores).
    """

    bs = q.shape[0]
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
            lse = torch.empty(bs, h_qo, device=q.device, dtype=torch.float32)
        elif (
            lse.shape != (bs, h_qo)
            or lse.dtype != torch.float32
            or not lse.is_contiguous()
        ):
            raise ValueError(
                "lse must be a contiguous float32 tensor of shape "
                f"(batch_size, num_heads_qo) = ({bs}, {h_qo}), got shape "
                f"{tuple(lse.shape)} with dtype {lse.dtype}"
            )

    if out is None:
        out = torch.empty(bs, h_qo, d_vo, device=q.device, dtype=q.dtype)

    if not CUDNN_AVAILABLE:
        if q.dtype != torch.bfloat16:
            # The fallback cubins are compiled for bf16 only; passing fp16
            # buffers through would silently reinterpret them as bf16.
            raise NotImplementedError(
                f"q.dtype={q.dtype} requires the cuDNN graph backend; the "
                "fallback cubin decode path only supports torch.bfloat16"
            )
        actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.device, non_blocking=True)

        run_func = get_cudnn_fmha_gen_module().decode
        run_func(
            max_sequence_kv,
            q,
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
        actual_seq_lens_q = torch.ones(
            (bs, 1, 1, 1), device=q.device, dtype=torch.int32
        )
        block_size = k_cache.shape[2]

        _batch_decode_with_kv_cache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            scale=scale,
            workspace_buffer=workspace_buffer,
            max_sequence_kv=max_sequence_kv,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            batch_offsets_q=batch_offsets_q,
            batch_offsets_o=batch_offsets_o,
            block_size=block_size,
            out=out,
            return_lse=return_lse,
            lse=lse,
        )

    if return_lse:
        return out, lse
    return out
