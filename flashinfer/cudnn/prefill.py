from enum import Enum
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from .utils import get_cudnn_fmha_gen_module

try:
    import cudnn

    CUDNN_AVAILABLE = True
except Exception:
    cudnn = None
    CUDNN_AVAILABLE = False

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
    actual_seq_lens_kv: torch.Tensor,
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
):
    graph_b = actual_seq_lens_q.shape[0]

    if q.dim() == 3:
        h_qo, d_qk = q.shape[1], q.shape[2]
    elif q.dim() == 4:
        h_qo, d_qk = q.shape[1], q.shape[3]

    if v_cache.dim() == 3:
        h_kv, d_vo = k_cache.shape[1], k_cache.shape[2]
    elif k_cache.dim() == 4:
        h_kv, d_vo = k_cache.shape[1], k_cache.shape[3]

    if block_tables is not None:
        page_size = k_cache.shape[2]

    key = (
        graph_b,
        q.dim(),
        q.dtype,
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
    ):
        handle = _create_cudnn_handle(torch.cuda.current_stream(q.device))

        graph_b = actual_seq_lens_q.shape[0]
        graph_s_qo = max_token_seq_q
        graph_s_kv = max_sequence_kv

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
        ) and cudnn.backend_version() < 91800:
            raise RuntimeError(
                f"FP8 is not supported in cuDNN backend version < 9.18.0, current version is {cudnn.backend_version()}"
            )

        with cudnn.graph(handle) as (g, _):
            # Create tensors from the input tensors
            if q.dim() == 3:
                h_qo, d_qk = q.shape[1], q.shape[2]
            elif q.dim() == 4:
                h_qo, d_qk = q.shape[2], q.shape[3]
            else:
                raise ValueError(f"Invalid query tensor shape: {q.shape}")

            cudnn_q = g.tensor(
                name="q",
                dim=(graph_b, h_qo, graph_s_qo, d_qk),
                stride=(h_qo * d_qk, d_qk, d_qk * h_qo, 1),
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
                ragged_q = g.tensor_like(batch_offsets_q)
                ragged_q.set_uid(UIDs.RAGGED_Q_UID.value)
                cudnn_q.set_ragged_offset(ragged_q)

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
                cudnn_k_cache = g.tensor(
                    name="k_cache",
                    dim=(graph_b, h_kv, graph_s_kv, d_qk),
                    stride=(h_kv * d_qk * graph_s_kv, d_qk, d_qk * h_kv, 1),
                    data_type=cudnn_k_data_type,
                )

                if batch_offsets_k is not None:
                    ragged_k = g.tensor_like(batch_offsets_k)
                    ragged_k.set_uid(UIDs.RAGGED_K_UID.value)
                    cudnn_k_cache.set_ragged_offset(ragged_k)

                cudnn_v_cache = g.tensor(
                    name="v_cache",
                    dim=(graph_b, h_kv, graph_s_kv, d_vo),
                    stride=(h_kv * d_vo * graph_s_kv, d_vo, d_vo * h_kv, 1),
                    data_type=cudnn_v_data_type,
                )

                if batch_offsets_v is not None:
                    ragged_v = g.tensor_like(batch_offsets_v)
                    ragged_v.set_uid(UIDs.RAGGED_V_UID.value)
                    cudnn_v_cache.set_ragged_offset(ragged_v)

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

            if (
                cudnn_q_data_type == cudnn.data_type.BFLOAT16
                or cudnn_q_data_type == cudnn.data_type.HALF
            ):
                O, Stats = g.sdpa(
                    name="sdpa",
                    q=cudnn_q,
                    k=cudnn_k_cache,
                    v=cudnn_v_cache,
                    seq_len_q=(
                        cudnn_actual_seq_lens_q
                        if actual_seq_lens_q is not None
                        else None
                    ),
                    seq_len_kv=(
                        cudnn_actual_seq_lens_kv
                        if actual_seq_lens_kv is not None
                        else None
                    ),
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
                    seq_len_q=(
                        cudnn_actual_seq_lens_q
                        if actual_seq_lens_q is not None
                        else None
                    ),
                    seq_len_kv=(
                        cudnn_actual_seq_lens_kv
                        if actual_seq_lens_kv is not None
                        else None
                    ),
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
                ragged_o = g.tensor_like(batch_offsets_o)
                ragged_o.set_uid(UIDs.RAGGED_O_UID.value)
                O.set_ragged_offset(ragged_o)

            if batch_offsets_stats is not None:
                ragged_stats = g.tensor_like(batch_offsets_stats)
                ragged_stats.set_uid(UIDs.RAGGED_STATS_UID.value)
                Stats.set_ragged_offset(ragged_stats)

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

            if actual_seq_lens_q is not None:
                tensors_to_return.append(cudnn_actual_seq_lens_q)
            if actual_seq_lens_kv is not None:
                tensors_to_return.append(cudnn_actual_seq_lens_kv)

            return g, tensors_to_return


def _batch_prefill_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    max_token_per_sequence: int,
    max_sequence_kv: int,
    actual_seq_lens_q: torch.Tensor,
    actual_seq_lens_kv: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    graph, tensors = _build_prefill_graph(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        scale=scale,
        max_token_seq_q=max_token_per_sequence,
        max_sequence_kv=max_sequence_kv,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
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
    graph.execute(var_map, workspace=workspace_buffer, handle=handle)

    if return_lse:
        return out, lse
    else:
        return out, None


@flashinfer_api
def cudnn_batch_prefill_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    max_token_per_sequence: int,
    max_sequence_kv: int,
    actual_seq_lens_q: torch.Tensor,
    actual_seq_lens_kv: torch.Tensor,
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
    is_cuda_graph_compatible: bool = False,
    backend: Optional[str] = None,
    o_data_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Performs batched prefill attention with paged KV cache using cuDNN.

    Args:
        q: Query tensor of shape (Total number of tokens, num_heads_qo, head_dim)
        k_cache: Key cache tensor of shape   (total_num_pages, num_heads_kv, page_size, head_dim) if paged kv cache is enabled else (Total sequence length of kv, num_heads_kv, d_qk)
        v_cache: Value cache tensor of shape (total_num_pages, num_heads_kv, page_size, head_dim) if paged kv cache is enabled else (Total sequence length of kv, num_heads_kv, d_vo)
        scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        workspace_buffer: Workspace buffer for cuDNN operations. Scales with batch size. 128 MB should be sufficient for most cases
        max_token_per_sequence: Maximum number of tokens per query sequence (s_qo_max)
        max_sequence_kv: Maximum number of tokens per key/value sequence (s_kv_max)
        actual_seq_lens_q:  Actual number of tokens per query sequence shape (batch_size,) on cpu or device (cpu if cuda_graph is False)
        actual_seq_lens_kv: Actual sequence lengths for key/values per batch, shape (batch_size,) on CPU or device (cpu if cuda_graph is False)
        block_tables: Page table mapping for KV cache, shape (batch_size, num_pages_per_seq) on GPU
        causal: Whether to apply causal masking
        return_lse: Whether to return log-sum-exp values (must be True)
        out: Optional pre-allocated output tensor
        lse: Optional pre-allocated tensor for log-sum-exp values if return_lse is True else returns None
        is_cuda_graph_compatible: Whether the prefill operation is compatible with CUDA graph
        q_scale: Optional scale tensor for query tensor of shape (1, 1, 1, 1) on GPU
        k_scale: Optional scale tensor for key tensor of shape (1, 1, 1, 1) on GPU
        v_scale: Optional scale tensor for value tensor of shape (1, 1, 1, 1) on GPU
        batch_offsets_q: Optional batch offsets for query tensor of shape (batch_size,) on GPU
        batch_offsets_o: Optional batch offsets for output tensor of shape (batch_size,) on GPU
        batch_offsets_k: Optional batch offsets for key tensor of shape (batch_size,) on GPU
        batch_offsets_v: Optional batch offsets for value tensor of shape (batch_size,) on GPU
        o_data_type: Optional data type for output tensor
    Returns:
        Output tensor of shape (batch_size * seq_len_q, num_heads_qo, head_dim)
        If return_lse is True, also returns log-sum-exp tensor of shape (batch_size, seq_len_q, num_heads_qo)

    Note:
        Query and KV heads can have different sizes (num_heads_qo >= num_heads_kv)
        When using cuda graph, actual_seq_lens_q and actual_seq_lens_kv must be on the same device as q
        Head dimension of query and key must be 128 or 192
        Head dimension of value and output must be 128
    """

    num_tokens = q.shape[0]

    num_sequences = actual_seq_lens_q.shape[0]

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

    if lse is not None and lse.shape != (num_sequences, max_token_per_sequence, h_qo):
        raise ValueError(
            "lse must have shape (num_sequences, max_token_per_sequence, h_qo)"
        )

    if o_data_type is None:
        o_data_type = q.dtype

    if out is None:
        out_shape = (num_tokens, h_qo, d_vo)
        out = torch.empty(out_shape, device=q.device, dtype=o_data_type)

    if CUDNN_AVAILABLE and backend != "cubin":
        return _batch_prefill_with_kv_cache(
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
            batch_offsets_q=batch_offsets_q,
            batch_offsets_o=batch_offsets_o,
            batch_offsets_k=batch_offsets_k,
            batch_offsets_v=batch_offsets_v,
            batch_offsets_stats=batch_offsets_stats,
            out=out,
            lse=lse,
            o_data_type=o_data_type,
        )
    else:
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
