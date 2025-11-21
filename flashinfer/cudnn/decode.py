from enum import Enum
from typing import Optional

import torch

from ..api_logging import flashinfer_api
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
):
    return (
        "decode",
        max_sequence_kv,
        tuple(q.shape),
        tuple(k_cache.shape),
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
    ):
        handle = _create_cudnn_handle(torch.cuda.current_stream())

        # WAR: override batch offsets for now, as it leads to a poor performance
        batch_offsets_q = None
        batch_offsets_o = None

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
                data_type=cudnn.data_type.BFLOAT16,
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

            O, _ = g.sdpa(
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
                is_inference=True,
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
                cudnn.data_type.BFLOAT16
            )

        tensors_to_return = [cudnn_q, cudnn_k_cache, cudnn_v_cache, O]

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
    )

    handle_ = _create_cudnn_handle(torch.cuda.current_stream())

    var_map = {
        UIDs.Q_UID.value: q,
        UIDs.K_UID.value: k_cache,
        UIDs.V_UID.value: v_cache,
        UIDs.O_UID.value: out,
    }
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


@flashinfer_api
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
) -> torch.Tensor:
    """Performs batched decode attention with paged KV cache using cuDNN.

    Args:
        q: Query tensor of shape (batch_size, num_heads_qo, head_dim), seq_len_q is the maximum sequence length of queries in the batch
        k_cache: Key cache tensor of shape   (total_num_pages, num_heads_kv, page_size, head_dim)
        v_cache: Value cache tensor of shape (total_num_pages, num_heads_kv, page_size, head_dim)
        scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        workspace_buffer: Workspace buffer for cuDNN operations. Scales with batch size. 128 MB should be sufficient for most cases
        max_sequence_kv: Maximum number of tokens per key/value sequence (s_kv_max)
        actual_seq_lens_kv: Actual sequence lengths for key/values per batch, shape (batch_size,) on CPU
        block_tables: Page table mapping for KV cache, shape (batch_size, num_pages_per_seq) on GPU
        is_cuda_graph_compatible: Whether the decode operation is compatible with CUDA graph
        batch_offsets: Optional batch offsets tensor of shape (batch_size,) on GPU
        out: Optional pre-allocated output tensor
        batch_offsets_q: Optional batch offsets for query tensor of shape (batch_size,) on GPU
        batch_offsets_o: Optional batch offsets for output tensor of shape (batch_size,) on GPU
        batch_offsets_k: Optional batch offsets for key tensor of shape (batch_size,) on GPU
        batch_offsets_v: Optional batch offsets for value tensor of shape (batch_size,) on GPU

    Returns:
        Output tensor of shape (batch_size, num_heads_qo, head_dim)

    Note:
        Currently only supports causal attention (causal must be True)
        All tensors must be contiguous and on the same CUDA device
        Query and KV heads can have different sizes (num_heads_qo >= num_heads_kv)
    """

    bs = q.shape[0]
    h_qo = q.shape[1]
    d_vo = v_cache.shape[3]

    if out is None:
        out = torch.empty(bs, h_qo, d_vo, device=q.device, dtype=q.dtype)

    if not CUDNN_AVAILABLE:
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
        )

    return out
