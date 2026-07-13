import functools
from enum import Enum
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.attention import cudnn_batch_prefill_trace
from .utils import get_cudnn_fmha_gen_module

try:
    import cudnn

    CUDNN_AVAILABLE = True
except Exception:
    cudnn = None
    CUDNN_AVAILABLE = False


@functools.cache
def _cudnn_supports_direct_seqlens(dtype: torch.dtype) -> bool:
    """True if cuDNN can consume token-unit indptr buffers directly for `dtype`.

    Requires cu_seq_len_q/kv SDPA inputs and per-tensor ragged-offset
    multipliers on the unified SDPA engine (forward only):
    - fp16/bf16: cuDNN backend 9.24+ with cudnn-frontend 1.25+
    - fp8 (e4m3/e5m2): cuDNN backend 9.25+ with cudnn-frontend 1.27+ (the first
      release whose sdpa_fp8 python binding exposes cu_seq_len_q/kv)
    """
    if not CUDNN_AVAILABLE:
        return False
    if dtype in (torch.float16, torch.bfloat16):
        min_backend, min_frontend = 92400, (1, 25)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        min_backend, min_frontend = 92500, (1, 27)
    else:
        return False
    try:
        if cudnn.backend_version() < min_backend:
            return False
        major, minor = map(int, cudnn.__version__.split(".")[:2])
        return (major, minor) >= min_frontend
    except Exception:
        return False


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
):
    if actual_seq_lens_q is not None:
        graph_b = actual_seq_lens_q.shape[0]
    elif cu_seq_lens_q is not None:
        graph_b = cu_seq_lens_q.shape[0] - 1
    else:
        raise ValueError("Either actual_seq_lens_q or cu_seq_lens_q must be provided")

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
        cu_seq_lens_q is not None,
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
    ):
        handle = _create_cudnn_handle(torch.cuda.current_stream(q.device))

        # cu_seq_lens_q/kv select the direct path: cuDNN consumes (b+1)-shaped
        # token-unit prefix sums for the padding mask, and the batch_offsets_*
        # ragged offsets are token-unit indptrs scaled by per-tensor
        # multipliers. Mutually exclusive with actual_seq_lens_q/kv for the
        # mask role; dtype/version support is the caller's responsibility
        # (_cudnn_supports_direct_seqlens).
        assert (cu_seq_lens_q is None) == (cu_seq_lens_kv is None), (
            "cu_seq_lens_q and cu_seq_lens_kv must both be set or both unset"
        )
        use_cu_seq_lens = cu_seq_lens_q is not None
        if use_cu_seq_lens:
            # Non-paged only for now: this is a FlashInfer plumbing limitation,
            # not a cuDNN one (the unified engine supports paged attention with
            # cu_seq_lens). The direct path reuses the token-unit indptrs as
            # both cu_seq_lens and ragged offsets, and no paged caller passes a
            # token-unit KV prefix sum today.
            assert (
                batch_offsets_q is not None
                and batch_offsets_k is not None
                and block_tables is None
            ), (
                "the cu_seq_lens path currently requires non-paged KV with "
                "token-unit q/k batch offsets"
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

        with cudnn.graph(handle) as (g, _):
            # Create tensors from the input tensors
            if q.dim() == 3:
                h_qo, d_qk = q.shape[1], q.shape[2]
                s_stride, h_stride, d_stride = q.stride()
            elif q.dim() == 4:
                h_qo, d_qk = q.shape[2], q.shape[3]
                s_stride, h_stride, d_stride = q.stride()
            else:
                raise ValueError(f"Invalid query tensor shape: {q.shape}")

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
                ragged_q = g.tensor_like(batch_offsets_q)
                ragged_q.set_uid(UIDs.RAGGED_Q_UID.value)
                cudnn_q.set_ragged_offset(ragged_q)
                if use_cu_seq_lens:
                    # Offsets are token-unit indptrs; the engine scales them
                    # back to elements.
                    cudnn_q.set_ragged_offset_multiplier(h_qo * d_qk)

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
                    ragged_k = g.tensor_like(batch_offsets_k)
                    ragged_k.set_uid(UIDs.RAGGED_K_UID.value)
                    cudnn_k_cache.set_ragged_offset(ragged_k)
                    if use_cu_seq_lens:
                        cudnn_k_cache.set_ragged_offset_multiplier(h_kv * d_qk)

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
                    ragged_v = g.tensor_like(batch_offsets_v)
                    ragged_v.set_uid(UIDs.RAGGED_V_UID.value)
                    cudnn_v_cache.set_ragged_offset(ragged_v)
                    if use_cu_seq_lens:
                        cudnn_v_cache.set_ragged_offset_multiplier(h_kv * d_vo)

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
                # The cumulative seq lens occupy the ACTUAL_SEQ_LENS UID slots
                # -- mutually exclusive with per-batch seq lens, same role. On
                # the ragged path the caller passes the token-unit indptrs
                # here, i.e. the same buffers as the ragged offsets.
                cudnn_cu_seq_lens_q = g.tensor_like(cu_seq_lens_q)
                cudnn_cu_seq_lens_q.set_name("cu_seq_lens_q")
                cudnn_cu_seq_lens_q.set_uid(UIDs.ACTUAL_SEQ_LENS_Q_UID.value)

                cudnn_cu_seq_lens_kv = g.tensor_like(cu_seq_lens_kv)
                cudnn_cu_seq_lens_kv.set_name("cu_seq_lens_kv")
                cudnn_cu_seq_lens_kv.set_uid(UIDs.ACTUAL_SEQ_LENS_KV_UID.value)

                padding_mask = True
                # These kwargs need a newer cudnn-frontend than the declared
                # >=1.13 floor (cu_seq_len_*: 1.25+; implementation /
                # attention_implementation: 1.14+), so they are only mentioned
                # on this path, which _cudnn_supports_direct_seqlens guards.
                seq_len_kwargs = {
                    "cu_seq_len_q": cudnn_cu_seq_lens_q,
                    "cu_seq_len_kv": cudnn_cu_seq_lens_kv,
                    # cu_seq_lens are unified-engine-only; pin the
                    # implementation so an unsupported config fails with the
                    # unified engine's specific error instead of
                    # auto-selection's generic failure.
                    "implementation": cudnn.attention_implementation.UNIFIED,
                }
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
                ragged_o = g.tensor_like(batch_offsets_o)
                ragged_o.set_uid(UIDs.RAGGED_O_UID.value)
                O.set_ragged_offset(ragged_o)
                if use_cu_seq_lens:
                    O.set_ragged_offset_multiplier(h_qo * d_vo)

            if batch_offsets_stats is not None:
                ragged_stats = g.tensor_like(batch_offsets_stats)
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
                tensors_to_return.append(cudnn_cu_seq_lens_kv)
            else:
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
    graph.execute(var_map, workspace=workspace_buffer, handle=handle)

    if return_lse:
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

    if lse is not None and lse.shape != (num_sequences, max_token_per_sequence, h_qo):
        raise ValueError(
            "lse must have shape (num_sequences, max_token_per_sequence, h_qo)"
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
        )

        if batch_offsets_units == "tokens":
            h_kv = k_cache.shape[1]
            use_direct = (
                _cudnn_supports_direct_seqlens(q.dtype)
                and block_tables is None
                # The direct path consumes the q/k indptrs as cu_seq_lens.
                and batch_offsets_q is not None
                and batch_offsets_k is not None
            )
            if use_direct:
                # On the ragged path the token-unit indptrs are also the
                # cumulative seq lens; cuDNN consumes them and the offsets
                # directly (scaling offsets by per-tensor multipliers).
                run_kwargs["cu_seq_lens_q"] = batch_offsets_q
                run_kwargs["cu_seq_lens_kv"] = batch_offsets_k
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
