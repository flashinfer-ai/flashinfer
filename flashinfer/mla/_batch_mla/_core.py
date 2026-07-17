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

from collections import namedtuple
import logging
import math
import warnings
from typing import List, Literal, Optional, Tuple, Union, cast, overload

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.api_logging import flashinfer_api
from flashinfer.autotuner import AutoTuner, TunableRunner
from flashinfer.trace.templates.attention import (
    mla_paged_decode_trace,
    xqa_batch_decode_mla_trace,
)
from flashinfer.utils import (
    _check_block_tables_shape,
    _resolve_trtllm_gen_multi_ctas_kv_counter_buffer,
    check_shape_dtype_device,
    determine_mla_backend,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    log2e,
)

from ._backends.cute_dsl_backend import (
    CuteDslMlaDecodeRunner,
    _BatchMLAPagedAttentionCuteDslBackend,
    _cute_dsl_incompatibility_reason,
    _cute_dsl_max_supported_batch,
)
# Private imports plus assignments preserve the Batch MLA core's compatibility surface.
from ._backends.cutlass_backend import (
    _BatchMLAPagedAttentionCutlassBackend,
    _validate_cutlass_plan_metadata,
    get_mla_module as _get_mla_module,
)
from ._backends.fa2_backend import _BatchMLAPagedAttentionFa2Backend
from ._backends.fa3_backend import _BatchMLAPagedAttentionFa3Backend
from ._backends._fa_common import (
    _BatchMLAGeneratedFaWorkspace,
    get_batch_mla_module as _get_batch_mla_module,
)
from ._backends.trtllm_gen_backend import (
    _TRTLLM_GEN_MLA_MAX_BATCH,
    _BatchMLAPagedAttentionTrtllmGenBackend,
    TrtllmGenMlaDecodeRunner,
    get_trtllm_gen_fmha_module as _get_trtllm_gen_fmha_module,
)
from ._backends.xqa_backend import (
    _BatchMLAPagedAttentionXqaBackend,
    _XqaMlaDecodeImplementation,
)
from ._planning import (
    _CSRPlanMetadata,
    _DensePlanMetadata,
    _MLAPlanArguments,
    _MLAWrapperPlanResult,
)


logger = logging.getLogger(__name__)


get_mla_module = _get_mla_module
get_batch_mla_module = _get_batch_mla_module
get_trtllm_gen_fmha_module = _get_trtllm_gen_fmha_module


_MLAHeadDimensions = namedtuple(
    "MLAHeadDimensions",
    ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "kv_lora_rank"),
)
deepseek_mla_dimensions = _MLAHeadDimensions(128, 64, 128, 512)
smaller_mla_dimensions = _MLAHeadDimensions(64, 64, 128, 256)
supported_mla_head_dimensions = [
    deepseek_mla_dimensions,
    smaller_mla_dimensions,
]


def _compute_mla_decode_buckets(
    workspace_buffer: torch.Tensor,
    runner_names: List[str],
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    device: torch.device,
    cute_dsl_max_batch: Optional[int] = None,
) -> Tuple[int, ...]:
    """Compute the autotune bucket list from kernel/workspace limits only."""
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    cap = 0
    if "trtllm-gen" in runner_names:
        cap = max(cap, _TRTLLM_GEN_MLA_MAX_BATCH)
    if "cute-dsl" in runner_names:
        if cute_dsl_max_batch is None:
            from flashinfer.cute_dsl.utils import get_num_sm

            cute_dsl_max_batch = _cute_dsl_max_supported_batch(
                workspace_bytes=(
                    workspace_buffer.numel() * workspace_buffer.element_size()
                ),
                q_len=q_len,
                num_heads=num_heads,
                kv_lora_rank=kv_lora_rank,
                max_active_blocks=get_num_sm(device),
                candidate_max=_TRTLLM_GEN_MLA_MAX_BATCH,
            )
        cap = max(cap, cute_dsl_max_batch)

    return get_hybrid_num_tokens_buckets(max(1, cap))


def _build_mla_decode_tuning_config(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    workspace_buffer: torch.Tensor,
    runner_names: List[str],
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_seq_len: int,
    device: torch.device,
    cute_dsl_max_batch: Optional[int] = None,
):
    """Build the per-call tuning config for the batch sweep."""
    from flashinfer.autotuner import DynamicTensorSpec, TuningConfig, make_bucket_mapper

    page_size = kv_cache.shape[-2]
    provisioned_max_seq_len = block_tables.shape[-1] * page_size
    profile_seq_len = min(max_seq_len, provisioned_max_seq_len)
    num_pages = kv_cache.shape[0]

    buckets = _compute_mla_decode_buckets(
        workspace_buffer,
        runner_names,
        q_len,
        num_heads,
        kv_lora_rank,
        device,
        cute_dsl_max_batch,
    )

    def init_block_tables(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.random_(0, num_pages)
        return tensor

    def init_seq_lens(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.fill_(profile_seq_len)
        return tensor

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 2, 3),
                dim_idx=(0, 0, 0, 0),
                gen_tuning_buckets=buckets,
                map_to_tuning_buckets=make_bucket_mapper(buckets, round_map=False),
                tensor_initializers=[None, init_block_tables, init_seq_lens, None],
            ),
        ),
        use_cuda_graph=True,
        use_cold_l2_cache=True,
    )


def _check_trtllm_gen_mla_shape(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
    page_table: torch.Tensor,
    page_size: int,
    uses_shared_paged_kv_idx: bool = True,
    batch_size: Optional[int] = None,
    max_q_len: Optional[int] = None,
    require_aligned_block_table: bool = True,
) -> torch.Tensor:
    is_flattened_query = False
    if query.ndim == 4:
        num_seqs, num_tokens, _, qk_head_dim = query.shape
    elif query.ndim == 3:
        is_flattened_query = True
        if batch_size is None or max_q_len is None:
            raise ValueError(
                "batch_size and max_q_len are required when query.ndim == 3"
            )
        num_seqs = batch_size
        num_tokens = max_q_len
        _, _, qk_head_dim = query.shape
    else:
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")

    # Support both 3D and 4D kv_cache for backward compatibility
    if kv_cache.ndim == 3:
        # [num_pages, page_size, head_dim_ckv + head_dim_kpe] -> [num_pages, 1, page_size, head_dim_ckv + head_dim_kpe]
        kv_cache = kv_cache.unsqueeze(1)
    elif kv_cache.ndim != 4:
        raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")

    is_deepseek_dimensions = (
        kv_lora_rank == deepseek_mla_dimensions.kv_lora_rank
        and qk_rope_head_dim == deepseek_mla_dimensions.qk_rope_head_dim
    )
    is_smaller_mla_dimensions = (
        kv_lora_rank == smaller_mla_dimensions.kv_lora_rank
        and qk_rope_head_dim == smaller_mla_dimensions.qk_rope_head_dim
    )
    if not (is_deepseek_dimensions or is_smaller_mla_dimensions):
        raise ValueError(
            f"Unsupported MLA dimensions, got kv_lora_rank={kv_lora_rank} and qk_rope_head_dim={qk_rope_head_dim}, supported dimensions are: {supported_mla_head_dimensions}"
        )

    ckv_dim = kv_cache.shape[3]
    expected_qk_head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_head_dim != expected_qk_head_dim or ckv_dim != expected_qk_head_dim:
        raise ValueError(
            f"Expected head dim {expected_qk_head_dim} for query and kv_cache, got {qk_head_dim} and {ckv_dim}"
        )

    if sparse_mla_top_k > 0:
        page_table_shape = page_table.shape
        expected_page_table_shape = (
            (query.size(0), sparse_mla_top_k)
            if is_flattened_query
            else (num_seqs, num_tokens, sparse_mla_top_k)
        )
        if page_table_shape != expected_page_table_shape:
            raise ValueError(
                "Expected page_table.shape == "
                f"{expected_page_table_shape}" + f", got {page_table_shape}"
            )
    else:
        _check_block_tables_shape(page_table, uses_shared_paged_kv_idx)
        B_block_table = page_table.shape[0]
        block_num = page_table.shape[-1]
        block_size = page_size
        if num_seqs != B_block_table:
            raise ValueError(
                f"Expected batch size {num_seqs} for query and block_table, got {num_seqs} and {B_block_table}"
            )
        if require_aligned_block_table and block_num % (128 / block_size) != 0:
            raise ValueError(
                f"Expected block_num % (128 / block_size) == 0, got {block_num=} and {block_size=}"
            )

    return kv_cache


def _run_mla_decode_trtllm_gen_or_cute_dsl_impl(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,  # TODO: remove in 1.0?
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: bool | None = None,
    backend: str = "auto",
    is_var_seq: bool = True,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cute_dsl_impl: str = "auto",
    kv_scale_format: str = "auto",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Decode MLA with TRTLLM-GEN, CuteDSL, XQA, or SM120/SM121 sparse kernels.

    With ``backend="auto"``, SM100/SM103 devices use TRTLLM-GEN for sparse MLA
    when ``sparse_mla_top_k > 0``. SM120/SM121 devices use the packed sparse
    backend for ``sparse_mla_top_k > 0`` and XQA for dense decode.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape
        ``[batch_size, q_len_per_request, num_heads, head_dim_qk]`` where
        ``head_dim_qk = kv_lora_rank + qk_rope_head_dim``. For the SM120/SM121
        v32/GLM sparse backend, this must be BF16 with ``head_dim_qk == 576``.
    kv_cache : torch.Tensor
        For TRTLLM-GEN, CuteDSL, and XQA, the paged KV cache is
        ``[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]`` or
        ``[num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim]`` and uses
        the query-compatible dense dtype. For the SM120/SM121 v32/GLM sparse
        backend, this is a packed uint8 cache with 656 bytes per token, shaped
        ``[num_pages, page_size, 656]`` or ``[num_pages, 1, page_size, 656]``.
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer. Must be zero-initialized on first use
        by kernels that use semaphore state.
    qk_nope_head_dim : int
        Non-RoPE query dimension. Dense MLA paths commonly use ``128`` or
        ``64`` depending on model. The SM120/SM121 sparse v32/GLM backend
        ignores this value and validates ``query.shape[-1] == 576`` instead.
    kv_lora_rank : int
        Latent KV rank. TRTLLM-GEN and SM120/SM121 sparse v32/GLM use ``512``.
    qk_rope_head_dim : int
        RoPE head dimension. Sparse MLA paths use ``64``.
    block_tables : torch.Tensor
        Page table for dense MLA backends when ``sparse_mla_top_k == 0``. For
        SM100/SM103 TRTLLM-GEN sparse MLA it is the usual paged block table.
        When ``cum_seq_lens_q`` is provided with sparse MLA, pass compact
        sparse rows in flattened query-token order with shape
        ``[total_q, sparse_mla_top_k]``.
        For SM120/SM121 sparse v32/GLM, it is the sparse index matrix and must
        have shape ``[batch_size, q_len_per_request, sparse_mla_top_k]`` with
        int32 physical token indices.
    seq_lens : Optional[torch.Tensor]
        Per-request KV sequence lengths for dense and TRTLLM-GEN paths. For
        SM120/SM121 sparse v32/GLM, pass ``[batch_size, q_len_per_request]`` or
        flattened ``[batch_size * q_len_per_request]`` active top-k lengths; if
        ``None``, every column in ``block_tables`` is active.
    max_seq_len : int
        Maximum KV sequence length used for dense/TRTLLM-GEN scheduling.
        Ignored by the SM120/SM121 sparse v32/GLM backend.
    sparse_mla_top_k : int
        Enables sparse MLA when greater than zero. On SM100/SM103 this selects
        the TRTLLM-GEN sparse page-table path. On SM120/SM121 with
        ``backend="auto"`` or ``backend="sparse"``, this is the width of the
        packed v32/GLM sparse index matrix. The TRTLLM-GEN backend supports
        dense query input or flattened query input plus ``cum_seq_lens_q``.
    out : Optional[torch.Tensor]
        Output tensor. If not provided, it is allocated internally.
    bmm1_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM1. TRTLLM-GEN accepts a FP32 tensor or float.
        CuteDSL, XQA, and SM120/SM121 sparse v32/GLM require a float.
    bmm2_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM2. TRTLLM-GEN accepts a FP32 tensor or float.
        CuteDSL and XQA require a float. SM120/SM121 sparse v32/GLM requires
        ``1.0``.
    sinks : Optional[List[torch.Tensor]]
        Additional value per head in the denominator of the softmax.
        Supported by ``trtllm-gen``, ``cute-dsl``, and ``sparse``.
        On ``cute-dsl`` this requires the modular implementation;
        ``cute_dsl_impl="auto"`` (the default) promotes to modular
        automatically, and ``cute_dsl_impl="monolithic"`` with sinks set raises
        :class:`ValueError`.
    skip_softmax_threshold_scale_factor: threshold scale factor for skipping softmax operations.
        Providing a value for this parameter enables skip-softmax sparsity as described in: https://arxiv.org/abs/2512.12087
        If no value is provided, then standard attention is used.
        Setting the threshold to a higher value generally increases kernel performance at the cost of accuracy degradation.
        The actual threshold value equals the provided threshold_scale_factor divided by the context length.
    enable_pdl : Optional[bool]
        Programmatic Dependent Launch toggle.  When ``None`` (default), auto-detects
        support from the query device. Honoured by the ``trtllm-gen``, ``cute-dsl``,
        and ``xqa`` functional backends.
    backend : str = "auto"
        Implementation backend. Valid values are ``"auto"``, ``"xqa"``,
        ``"trtllm-gen"``, ``"cute-dsl"``, and ``"sparse"``. ``"auto"``
        chooses ``"trtllm-gen"`` for SM100/SM103 sparse MLA and chooses
        ``"sparse"`` for SM120/SM121 when ``sparse_mla_top_k > 0``; otherwise
        SM120/SM121 dense decode uses ``"xqa"``.
        The ``cute-dsl`` backend has two interchangeable implementations
        (``monolithic`` and ``modular``) on the same shape/dtype envelope;
        which one runs is controlled by the ``cute_dsl_impl`` kwarg below.
    is_var_seq : bool
        Whether the sequence length is variable.
        If True, the sequence length is variable.
        Otherwise,the sequence length is fixed for all the requests in the batch.
    uses_shared_paged_kv_idx : bool = True
        Whether K and V page indices are shared as a unified index.
        True (default) uses vLLM/FlashInfer layout with a 2D page table.
        False uses TRT-LLM layout with a 3D page table ``[batch_size, 2, max_num_pages_per_seq]``.
        False is only supported by TRTLLM-GEN.
    lse : Optional[torch.Tensor] = None
        Optional pre-allocated buffer for Log-Sum-Exp values. Supported by
        ``trtllm-gen``, ``cute-dsl``, and ``sparse`` backends. Must have
        dtype ``torch.float32``. Accepted shapes:

        * ``[batch_size * q_len_per_request, num_qo_heads]`` (TRTLLM-GEN
          native; accepted by sparse), or
        * ``[batch_size, q_len_per_request, num_qo_heads]`` (cute-dsl native;
          also accepted by cute-dsl).

        If ``return_lse`` is True and this is None, a buffer will be
        allocated by the backend.
    return_lse : bool = False
        Whether to return LSE values. Supported by ``trtllm-gen``,
        ``cute-dsl``, and ``sparse`` backends. When True, the function
        returns ``(out, lse)``.
    cute_dsl_impl : str = "auto"
        Which cute-dsl implementation to use. Honored when
        ``backend="cute-dsl"`` and when ``backend="auto"`` considers the
        cute-dsl candidate; ignored for non-cute-dsl backends.

        * ``"auto"`` (default) — picks monolithic by default, automatically
          promoted to modular when the call uses a feature monolithic
          doesn't support (currently ``sinks``).
        * ``"modular"`` — strict.  Always run the modular kernels.
        * ``"monolithic"`` — strict.  Always run the monolithic kernels;
          raise :class:`ValueError` if the call uses any modular-only
          feature (e.g. ``sinks``).
    kv_scale_format : str = "auto"
        Scale semantics for the SM120/SM121 packed v32/GLM sparse backend.
        ``"auto"`` and ``"pow2_fp32"`` select DSv3.2 power-of-2 FP32 inline
        scales; ``"arbitrary_fp32"`` selects GLM-style arbitrary FP32 inline scales.
        Ignored by the ``trtllm-gen``, ``xqa``, and ``cute-dsl`` backends.
    cum_seq_lens_q : Optional[torch.Tensor] = None
        Cumulative query sequence lengths for variable-length query support,
        shape ``[batch_size + 1]``, dtype ``torch.int32``. Must be a 1D tensor
        with at least two entries. When ``max_q_len`` is not provided, this
        function validates that it starts with 0, ends at ``query.size(0)``,
        and is monotonically non-decreasing. Only supported by the
        ``trtllm-gen`` backend. When provided, ``query`` must have shape
        ``[total_q, num_heads, head_dim_qk]``.
        For best performance, provide ``max_q_len`` together with
        ``cum_seq_lens_q`` to avoid host-side metadata validation.
    max_q_len : Optional[int] = None
        Maximum query sequence length across all requests when using
        ``cum_seq_lens_q``. Provide with ``cum_seq_lens_q`` to avoid
        host-side metadata validation. Must be greater than or equal to the
        maximum segment length represented by ``cum_seq_lens_q``. Over-estimation
        is safe but may waste work; under-estimation is invalid and may produce
        incorrect output.
    multi_ctas_kv_counter_buffer : Optional[torch.Tensor] = None
        Optional caller-owned counter buffer for the ``trtllm-gen`` backend.
        It must be contiguous, remain alive for every launch or CUDA graph replay
        that uses it, and be zero-initialized once. Allocate at least the number
        of bytes returned by ``get_trtllm_gen_multi_ctas_kv_counter_bytes`` for
        the current batch size, query-head count, and device SM count. Autotune
        profiling uses runner-owned storage; this buffer is used only for the
        selected TRTLLM-GEN runner's final request.

    Note
    ----
    In MLA, the actual BMM1 and BMM2 scales applied would be fused as:
    bmm1_scale = q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5)
    bmm2_scale = v_scale * o_scale
    or,
    bmm1_scale = torch.Tensor([q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5))
    bmm2_scale = torch.Tensor([v_scale * o_scale])

    The two scale factors should be static constant for cuda graph capture.
    Either (bmm1_scale, bmm2_scale) or (bmm1_scale_log2_tensor, bmm2_scale_tensor) should be provided.

    For static constant scale factors, the scale factors should be provided as float.
        - (bmm1_scale, bmm2_scale)
    For on-device fused scale tensors, which could dynamically change, the scale factors should be provided as torch.Tensor.
        - (bmm1_scale_log2_tensor, bmm2_scale_tensor)
        - Currently, only fp8 tensor core operation supports this mode.
    When both are provided, the dynamic scale factor tensors will be used.

    Autotune
    --------
    On SM100/SM103 dense MLA, calling under ``flashinfer.autotune(True)`` with
    ``backend="auto"`` profiles both ``trtllm-gen`` and ``cute-dsl`` across a
    bucketed batch sweep up to each runner's kernel/workspace cap and caches the
    winning runner per shape signature. Subsequent calls under
    ``autotune(False)`` dispatch to the cached choice; any batch outside the
    tuned range falls back to a default runner with a one-time warning.

    The autotune bucket range and cache key do **not** depend on
    ``kv_cache.shape[0]`` (the number of pages in the pool), so reallocating the
    pool between tuning and inference does not invalidate cached choices. However,
    the **page-aliasing ratio** during profiling does depend on the pool size:
    synthetic ``block_tables`` are filled by uniform random sampling into
    ``[0, kv_cache.shape[0])``, so a small pool produces high aliasing
    (L2-resident reads) and a large pool produces low aliasing (HBM-bound reads).
    For best profile fidelity, autotune with a ``kv_cache`` whose size reflects
    the production page-sharing pattern of your workload (e.g., heavily shared
    prefix → smaller pool; independent contexts → larger pool).
    """
    if backend not in ("auto", "trtllm-gen", "cute-dsl"):
        raise ValueError(f"Backend {backend} not supported by dense MLA decode")

    if seq_lens is None:
        raise ValueError(
            "seq_lens is required for trtllm-gen and cute-dsl MLA backends"
        )

    # log2e fusion is a trtllm-gen-only transform (the kernel expects
    # log2-form scales for the tensor case). Apply after the xqa branch so
    # that calling this function with backend="xqa" and calling
    # xqa_batch_decode_with_kv_cache_mla directly yield the same kernel input.
    if isinstance(bmm1_scale, torch.Tensor):
        bmm1_scale = bmm1_scale * log2e

    # Shared setup for the trtllm-gen / cute-dsl autotune dispatch.
    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl
    sm_count = get_device_sm_count(query.device)

    block_size = kv_cache.size(-2)
    trtllm_gen_not_supported_reason: Optional[str] = None
    if block_size != 32 and block_size != 64:
        trtllm_gen_not_supported_reason = (
            f"trtllm-gen requires block_size in (32, 64), got {block_size}"
        )

    if skip_softmax_threshold_scale_factor is not None and sparse_mla_top_k != 0:
        raise ValueError("skip_softmax is not supported for sparse MLA")

    has_var_q = cum_seq_lens_q is not None
    if has_var_q:
        if backend == "cute-dsl":
            raise ValueError("cute-dsl MLA does not support cum_seq_lens_q")
        if return_lse or lse is not None:
            raise NotImplementedError(
                "trtllm-gen MLA does not support return_lse/lse with cum_seq_lens_q"
            )
        if query.ndim != 3:
            raise ValueError(
                "query must have shape [total_q, num_heads, head_dim_qk] "
                "when cum_seq_lens_q is provided"
            )
        check_shape_dtype_device(
            cum_seq_lens_q,
            None,
            torch.int32,
            query.device,
            "cum_seq_lens_q",
        )
        if cum_seq_lens_q.ndim != 1:
            raise ValueError(
                f"Expected cum_seq_lens_q.ndim == 1, got {cum_seq_lens_q.ndim}"
            )
        if cum_seq_lens_q.size(0) < 2:
            raise ValueError("cum_seq_lens_q must contain at least two entries")
        batch_size = cum_seq_lens_q.size(0) - 1
        if batch_size != seq_lens.size(0):
            raise ValueError(
                "Batch size mismatch: cum_seq_lens_q describes "
                f"{batch_size} sequences, but seq_lens has {seq_lens.size(0)} entries"
            )
        if max_q_len is None:
            cum_seq_lens_q_host = cum_seq_lens_q.cpu()
            if cum_seq_lens_q_host[0].item() != 0:
                raise ValueError("cum_seq_lens_q must start with 0")
            if cum_seq_lens_q_host[-1].item() != query.size(0):
                raise ValueError(
                    "cum_seq_lens_q[-1] must match the flattened query length"
                )
            q_lens = cum_seq_lens_q_host[1:] - cum_seq_lens_q_host[:-1]
            if torch.any(q_lens < 0).item():
                raise ValueError("cum_seq_lens_q must be monotonically non-decreasing")
            max_q_len = q_lens.max().item()
            if max_q_len <= 0:
                raise ValueError(
                    "cum_seq_lens_q must describe at least one query token"
                )
        elif max_q_len <= 0:
            raise ValueError("max_q_len must be greater than 0")
        elif max_q_len > query.size(0):
            raise ValueError("max_q_len cannot exceed the flattened query length")

        kv_cache = _check_trtllm_gen_mla_shape(
            query,
            kv_cache,
            kv_lora_rank,
            qk_rope_head_dim,
            sparse_mla_top_k,
            block_tables,
            block_size,
            uses_shared_paged_kv_idx,
            batch_size=batch_size,
            max_q_len=max_q_len,
            require_aligned_block_table=False,
        )

        if multi_ctas_kv_counter_buffer is not None:
            multi_ctas_kv_counter_buffer = (
                _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
                    multi_ctas_kv_counter_buffer,
                    batch_size,
                    query.size(1),
                    sm_count,
                    query.device,
                )
            )

        expected_out_shape = query.shape[:-1] + (kv_lora_rank,)
        if out is None:
            out = torch.empty(
                expected_out_shape, dtype=torch.bfloat16, device=query.device
            )
        else:
            check_shape_dtype_device(
                out,
                expected_out_shape,
                torch.bfloat16,
                query.device,
                "out",
            )

        return TrtllmGenMlaDecodeRunner(
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            sm_count=sm_count,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            sparse_mla_top_k=sparse_mla_top_k,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=sinks,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            enable_pdl=enable_pdl,
            is_var_seq=is_var_seq,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            return_lse=False,
            lse=None,
        ).forward_ragged(
            query=query,
            block_tables=block_tables,
            seq_lens=seq_lens,
            out=out,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )

    # Normalize kv_cache to 4D and validate MLA dimensions. Despite the name,
    # the shape/dim checks here apply to both backends.
    kv_cache = _check_trtllm_gen_mla_shape(
        query,
        kv_cache,
        kv_lora_rank,
        qk_rope_head_dim,
        sparse_mla_top_k,
        block_tables,
        block_size,
        uses_shared_paged_kv_idx,
        require_aligned_block_table=backend != "trtllm-gen",
    )

    # Pre-allocate `out` so non-swept dims have a template for autotune
    # profiling (the autotuner inherits non-swept dims from caller tensors).
    expected_out_shape = query.shape[:-1] + (kv_lora_rank,)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=query.device)
    else:
        check_shape_dtype_device(
            out,
            expected_out_shape,
            torch.bfloat16,
            query.device,
            "out",
        )

    # Remember the caller-supplied lse so we can return it in its original
    # shape: 2D ``(B*q_len, H)`` stays 2D, 3D ``(B, q_len, H)`` stays 3D, and
    # an allocated default stays 2D.  Internally we normalize to 2D for the
    # backend dispatch (matches trtllm-gen's native layout).
    user_lse = lse
    if return_lse:
        flat_lse_shape = (query.size(0) * query.size(1), query.size(2))
        nested_lse_shape = (query.size(0), query.size(1), query.size(2))
        if lse is None:
            lse = torch.empty(flat_lse_shape, dtype=torch.float32, device=query.device)
            user_lse = lse
        elif tuple(lse.shape) == flat_lse_shape:
            check_shape_dtype_device(
                lse, flat_lse_shape, torch.float32, query.device, "lse"
            )
        elif tuple(lse.shape) == nested_lse_shape:
            check_shape_dtype_device(
                lse, nested_lse_shape, torch.float32, query.device, "lse"
            )
            # Normalize to 2D for the backend; .view shares storage so the
            # kernel writes propagate back to user_lse automatically.
            lse = lse.view(flat_lse_shape)
        else:
            raise ValueError(
                f"lse must have shape {flat_lse_shape} or {nested_lse_shape}; "
                f"got {tuple(lse.shape)}"
            )

    page_size = kv_cache.shape[-2]
    cute_dsl_reason = _cute_dsl_incompatibility_reason(
        query,
        out.dtype,
        bmm1_scale,
        bmm2_scale,
        sinks,
        sparse_mla_top_k,
        skip_softmax_threshold_scale_factor,
        uses_shared_paged_kv_idx,
        qk_rope_head_dim,
        kv_lora_rank,
        page_size,
        is_var_seq,
        return_lse,
        lse,
        cute_dsl_impl,
    )
    if backend == "cute-dsl":
        if cute_dsl_reason is not None:
            raise ValueError(cute_dsl_reason)
        runner_names = ["cute-dsl"]
    elif backend == "trtllm-gen":
        if trtllm_gen_not_supported_reason is not None:
            raise ValueError(trtllm_gen_not_supported_reason)
        runner_names = ["trtllm-gen"]
    else:  # backend == "auto"
        runner_names = []
        if trtllm_gen_not_supported_reason is None:
            runner_names.append("trtllm-gen")
        if cute_dsl_reason is None:
            runner_names.append("cute-dsl")
        if not runner_names:
            raise ValueError(
                f"auto: no backend supports this configuration "
                f"(trtllm-gen: {trtllm_gen_not_supported_reason}; "
                f"cute-dsl: {cute_dsl_reason})"
            )

    if multi_ctas_kv_counter_buffer is not None and "trtllm-gen" not in runner_names:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is only supported when a "
            "trtllm-gen runner is selected"
        )
    if multi_ctas_kv_counter_buffer is not None:
        multi_ctas_kv_counter_buffer = (
            _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
                multi_ctas_kv_counter_buffer,
                query.size(0),
                query.size(2),
                sm_count,
                query.device,
            )
        )

    runners: List[TunableRunner] = []
    if "trtllm-gen" in runner_names:
        runners.append(
            TrtllmGenMlaDecodeRunner(
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                sm_count=sm_count,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                sparse_mla_top_k=sparse_mla_top_k,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=sinks,
                skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                enable_pdl=enable_pdl,
                is_var_seq=is_var_seq,
                uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
                return_lse=return_lse,
                lse=lse,
            )
        )
    if "cute-dsl" in runner_names:
        # Normalize sinks: public contract accepts Optional[List[Tensor]] for
        # legacy reasons, but cute-dsl's modular variant expects a single
        # per-head tensor or None. The list-of-1 case has been guarded by
        # `_cute_dsl_incompatibility_reason` so we can unpack here safely.
        cute_dsl_sinks: Optional[torch.Tensor] = None
        if sinks is not None:
            cute_dsl_sinks = sinks[0] if isinstance(sinks, (list, tuple)) else sinks
        runners.append(
            CuteDslMlaDecodeRunner(
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                softmax_scale=bmm1_scale,
                output_scale=bmm2_scale,
                out_dtype=out.dtype,
                enable_pdl=enable_pdl,
                is_var_seq=is_var_seq,
                uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
                lse=lse,
                return_lse=return_lse,
                sinks=cute_dsl_sinks,
                cute_dsl_impl=cute_dsl_impl,
            )
        )

    _, q_len, num_heads, _ = query.shape
    tuning_config = _build_mla_decode_tuning_config(
        kv_cache=kv_cache,
        block_tables=block_tables,
        workspace_buffer=workspace_buffer,
        runner_names=runner_names,
        q_len=q_len,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        max_seq_len=max_seq_len,
        device=query.device,
    )
    inputs = [query, block_tables, seq_lens, out]
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm_batch_decode_mla",
        runners,
        tuning_config,
        inputs,
    )
    if multi_ctas_kv_counter_buffer is not None and isinstance(
        runner, TrtllmGenMlaDecodeRunner
    ):
        runner(
            inputs=inputs,
            tactic=tactic,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )
    else:
        runner(inputs=inputs, tactic=tactic)
    if return_lse:
        # Return the lse in the same shape the caller supplied (2D or 3D),
        # or 2D ``(B*q_len, H)`` when we allocated the default.
        return out, user_lse
    return out


def _run_mla_decode_xqa(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: bool | None = None,
    backend: str = "auto",
    is_var_seq: bool = True,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cute_dsl_impl: str = "auto",
    kv_scale_format: str = "auto",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if backend != "xqa":
        raise ValueError(f"XQA adapter requires backend='xqa', got {backend!r}")
    if seq_lens is None:
        raise ValueError("seq_lens is required for XQA MLA")
    if sparse_mla_top_k > 0:
        raise ValueError("XQA MLA does not support sparse_mla_top_k")
    if cum_seq_lens_q is not None or max_q_len is not None:
        raise ValueError("XQA MLA does not support cum_seq_lens_q / max_q_len")
    if multi_ctas_kv_counter_buffer is not None:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is only supported by the "
            "trtllm-gen backend"
        )
    if skip_softmax_threshold_scale_factor is not None:
        raise ValueError("skip_softmax is not supported for XQA backend")
    if not uses_shared_paged_kv_idx:
        raise ValueError(
            "XQA MLA does not support separate KV page indices "
            "(uses_shared_paged_kv_idx=False)"
        )
    del (
        qk_nope_head_dim,
        max_seq_len,
        cute_dsl_impl,
        kv_scale_format,
        is_var_seq,
    )
    return _XqaMlaDecodeImplementation().run(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        enable_pdl=enable_pdl,
        lse=lse,
        return_lse=return_lse,
    )


def _run_mla_decode_trtllm_gen(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: bool | None = None,
    backend: str = "auto",
    is_var_seq: bool = True,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cute_dsl_impl: str = "auto",
    kv_scale_format: str = "auto",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if backend not in ("auto", "trtllm-gen"):
        raise ValueError(
            "TRTLLM-GEN adapter requires backend='auto' or 'trtllm-gen', "
            f"got {backend!r}"
        )
    if backend == "trtllm-gen":
        cute_dsl_impl = "auto"
    kv_scale_format = "auto"
    return _run_mla_decode_trtllm_gen_or_cute_dsl_impl(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=sparse_mla_top_k,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        enable_pdl=enable_pdl,
        backend=backend,
        is_var_seq=is_var_seq,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        lse=lse,
        return_lse=return_lse,
        cute_dsl_impl=cute_dsl_impl,
        kv_scale_format=kv_scale_format,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
    )


def _run_mla_decode_cute_dsl(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: bool | None = None,
    backend: str = "auto",
    is_var_seq: bool = True,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cute_dsl_impl: str = "auto",
    kv_scale_format: str = "auto",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if backend != "cute-dsl":
        raise ValueError(
            f"CuTe DSL adapter requires backend='cute-dsl', got {backend!r}"
        )
    kv_scale_format = "auto"
    if cum_seq_lens_q is not None or max_q_len is not None:
        raise ValueError("cute-dsl MLA does not support cum_seq_lens_q / max_q_len")
    if multi_ctas_kv_counter_buffer is not None:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is only supported by the "
            "trtllm-gen backend"
        )
    return _run_mla_decode_trtllm_gen_or_cute_dsl_impl(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=sparse_mla_top_k,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        enable_pdl=enable_pdl,
        backend=backend,
        is_var_seq=is_var_seq,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        lse=lse,
        return_lse=return_lse,
        cute_dsl_impl=cute_dsl_impl,
        kv_scale_format=kv_scale_format,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
    )


@flashinfer_api(trace=xqa_batch_decode_mla_trace)
def xqa_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,  # TODO: remove in 1.0?
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,  # TODO: remove in 1.0?
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    r"""XQA-backend batched MLA decode.

    Single-query (MTP-aware) MLA decode kernel optimized for SM120a / SM121a tensor cores.
    Accepts the concatenated ``(q_nope || q_rope)`` query and ``(ckv || kpe)`` paged KV
    cache layout used by DeepSeek-V3 / R1 inference.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape
        ``[batch_size, q_len_per_request, num_heads, head_dim_qk]`` where
        ``head_dim_qk = kv_lora_rank + qk_rope_head_dim``.  Must be the concatenation
        ``[q_nope, q_rope]``.  ``q_len_per_request`` is the MTP query length and is
        currently required to be ``1``.
    kv_cache : torch.Tensor
        Paged KV cache, either 3-D
        ``[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]`` or 4-D
        ``[num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim]``.  The last
        dimension is the concatenation ``[ckv_cache, kpe_cache]``.  Both shapes are
        accepted for backward compatibility.
    workspace_buffer : torch.Tensor
        Pre-allocated backend scratch workspace buffer.
    qk_nope_head_dim : int
        Non-RoPE head dimension.  Must be ``128``.  Will be removed in 1.0; pass
        ``kv_lora_rank`` instead going forward.
    kv_lora_rank : int
        Rank of the latent KV projection.  Must be ``512``.
    qk_rope_head_dim : int
        RoPE head dimension appended to the latent projection.  Must be ``64``.
    block_tables : torch.Tensor
        Per-request paged KV block table, shape ``[batch_size, num_pages]``.
    seq_lens : torch.Tensor
        Per-request KV sequence length, shape ``[batch_size]``.
    max_seq_len : int
        Maximum KV sequence length used for kernel scheduling.  Will be removed in
        1.0; the kernel reads the per-request lengths from ``seq_lens``.
    out : Optional[torch.Tensor]
        Optional output tensor of shape ``[batch_size, num_heads, kv_lora_rank]``
        and dtype ``torch.bfloat16``.  If ``None``, it is allocated internally.
    bmm1_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM1 (see Note).  ``float`` for static (CUDA-graph
        safe) scales; ``torch.Tensor`` for on-device dynamic scales (FP8 only).
    bmm2_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM2 (see Note).  Same typing rules as ``bmm1_scale``.
    sinks : Optional[List[torch.Tensor]]
        Attention-sink tensors.  Currently unsupported and must be ``None``.
    enable_pdl : Optional[bool]
        Programmatic Dependent Launch toggle.  When ``None``, auto-detects support
        from the device.

    Returns
    -------
    torch.Tensor
        Attention output, shape ``[batch_size, num_heads, kv_lora_rank]``, dtype
        ``torch.bfloat16``.

    Note
    ----
    In MLA, the BMM1 and BMM2 scales are fused as:

    .. code-block:: text

        bmm1_scale = q_scale * k_scale * sm_scale / sqrt(head_dim_qk)
        bmm2_scale = v_scale * o_scale

    The scale factors must be static constants for CUDA graph capture.  Either the
    ``(bmm1_scale, bmm2_scale)`` (float) pair or the on-device
    ``(bmm1_scale_log2_tensor, bmm2_scale_tensor)`` tensor pair may be passed.
    When tensor inputs are supplied, the on-device path is taken (FP8 only).
    """
    del qk_nope_head_dim, max_seq_len
    return _XqaMlaDecodeImplementation().run(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        enable_pdl=enable_pdl,
    )


class BatchMLAPagedAttentionWrapper:
    r"""Wrapper class for MLA (`Multi-head Latent Attention <https://arxiv.org/abs/2405.04434>`_)
    PagedAttention on DeepSeek models. This kernel can be used in decode, and incremental prefill
    and should be used together with `Matrix Absorption trick
    <https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md>`_:
    where :math:`W_{UQ}` is absorbed with :math:`W_{UK}`, and :math:`W_{UV}` is
    absorbed with :math:`W_{O}`.
    For MLA attention without Matrix Absorption (``head_dim_qk=192`` and ``head_dim_vo=128``, which is
    used in prefilling self-attention stage), please use
    :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`.

    More information about The Paged KV-Cache layout in MLA is explained in our tutorial
    :ref:`MLA Page Layout <mla-page-layout>`.

    For more details about the MLA computation, Matrix Absorption and FlashInfer's MLA implementation,
    please refer to our `blog post <http://flashinfer.ai/2025/02/10/flashinfer-deepseek-mla.html>`_.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_local_heads = 128
    >>> batch_size = 114
    >>> head_dim_ckv = 512
    >>> head_dim_kpe = 64
    >>> page_size = 1
    >>> mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    ...     torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
    ...     backend="fa2"
    ... )
    >>> q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
    >>> kv_lens = torch.full((batch_size,), 999, dtype=torch.int32).to(0)
    >>> kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * 999
    >>> kv_indices = torch.arange(0, batch_size * 999).to(0).int()
    >>> q_nope = torch.randn(
    ...     batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> q_pe = torch.zeros(
    ...     batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> ckv = torch.randn(
    ...     batch_size * 999, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> kpe = torch.zeros(
    ...     batch_size * 999, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    >>> mla_wrapper.plan(
    ...     q_indptr,
    ...     kv_indptr,
    ...     kv_indices,
    ...     kv_lens,
    ...     num_local_heads,
    ...     head_dim_ckv,
    ...     head_dim_kpe,
    ...     page_size,
    ...     False,  # causal
    ...     sm_scale,
    ...     q_nope.dtype,
    ...     ckv.dtype,
    ... )
    >>> o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    >>> o.shape
    torch.Size([114, 128, 512])
    """

    _blackwell_auto_fallback_warned: bool = False
    _run_adapter_names = {
        "fa2": "_run_fa2",
        "fa3": "_run_fa3",
        "cutlass": "_run_cutlass",
        "trtllm-gen": "_run_trtllm_gen",
        "cute-dsl": "_run_cute_dsl",
        "xqa": "_run_xqa",
    }

    @classmethod
    def _maybe_warn_blackwell_auto_fallback(
        cls, device: torch.device, selected_backend: str
    ) -> None:
        if cls._blackwell_auto_fallback_warned:
            return
        major, minor = get_compute_capability(device)
        if major < 10:
            return
        cls._blackwell_auto_fallback_warned = True
        warnings.warn(
            f"BatchMLAPagedAttentionWrapper: backend='auto' selected "
            f"'{selected_backend}' on SM{major}{minor}, which is not Blackwell-native "
            f"and gives poor MLA decode performance. "
            f"For decode, use "
            f"flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla "
            f"(Blackwell-native trtllm-gen); backend='cutlass' is the closest "
            f"in-wrapper alternative but may be slower than this fallback for "
            f"decode shapes.",
            UserWarning,
            stacklevel=3,
        )

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        r"""Constructor for BatchMLAPagedAttentionWrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store intermediate attention results in
            split-k algorithm. The recommended size is 128MB, the device of the workspace buffer
            should be the same as the device of the input tensors. The XQA wrapper backend
            requires at least 128 MiB and initializes its live semaphore range during planning.
        use_cuda_graph : bool, optional
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.
            An initial ``cutlass``, ``trtllm-gen``, ``cute-dsl``, or ``xqa`` plan may
            be captured and replayed, but replanning those dense backends is rejected
            because they do not accept caller-reserved metadata buffers for
            pointer-stable replacement.
        qo_indptr : Optional[torch.Tensor]
            User-reserved buffer to back the ``qo_indptr`` array, shape ``[batch_size + 1]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.  The wrapper
            copies into this buffer at :meth:`plan` time so capture-time pointers remain
            stable.
        kv_indptr : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_indptr`` array, shape ``[batch_size + 1]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.
        kv_indices : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_indices`` array, sized to the maximum
            expected number of pages, dtype ``int32``.  Only consulted when
            ``use_cuda_graph=True``.
        kv_len_arr : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_len_arr`` array, shape ``[batch_size]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.
        backend : str
            One of ``"auto"``, ``"fa2"``, ``"fa3"``, ``"cutlass"``,
            ``"trtllm-gen"``, ``"cute-dsl"``, or ``"xqa"``. Default ``"auto"``.

            ``"auto"`` first tries ``"fa3"`` on SM90a, else ``"fa2"``, then
            falls back to ``"cutlass"`` when the preferred backend reports the
            configuration as unsupported. On SM>=100 neither FA backend is
            Blackwell-native; for MLA decode prefer
            :func:`trtllm_batch_decode_with_kv_cache_mla`. The ``"cutlass"`` option
            in this wrapper is the closest in-wrapper alternative but may be
            slower than the fa2 fallback for decode shapes.

            ``"cutlass"`` uses the SM100/SM110 CUTLASS MLA decode kernel. Only
            ``float_workspace_buffer`` is required at construction. Public
            ``q_nope``/``q_pe`` and ``ckv_cache``/``kpe_cache`` inputs stay split
            and are concatenated internally. ``kv_len`` and ``page_table`` may
            be captured by ``plan()`` and omitted from ``run()``; planned
            metadata takes precedence over cheap-verified aliases supplied at
            run time. For backward compatibility, an explicitly requested
            CUTLASS backend may also run without a preceding ``plan()`` when
            both metadata tensors are supplied to ``run()``.

            ``"trtllm-gen"`` uses the dense TRTLLM-GEN MLA decode path. It is
            explicit-only initially and is not considered by ``backend="auto"``.

            ``"cute-dsl"`` uses the Blackwell CuTe DSL MLA decode path. It is
            explicit-only and is never considered by ``backend="auto"``.

            ``"xqa"`` uses the SM120/SM121 XQA MLA decode path. It is
            explicit-only and is never considered by ``backend="auto"``. Its
            contiguous workspace must contain at least 128 MiB.
        """
        if backend not in (
            "auto",
            "fa2",
            "fa3",
            "cutlass",
            "trtllm-gen",
            "cute-dsl",
            "xqa",
        ):
            raise ValueError(
                "backend must be one of 'auto', 'fa2', 'fa3', 'cutlass', "
                f"'trtllm-gen', 'cute-dsl', or 'xqa', got {backend!r}"
            )
        self._backend = backend
        self._selected_backend: Optional[str] = None
        self._backend_impl = None
        self._csr_plan_metadata: Optional[_CSRPlanMetadata] = None
        self._dense_plan_metadata: Optional[_DensePlanMetadata] = None

        self.device = float_workspace_buffer.device
        self._float_workspace_buffer = float_workspace_buffer
        self._generated_fa_workspace = _BatchMLAGeneratedFaWorkspace(self.device)
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr

    def _reject_unsafe_cuda_graph_replan(self) -> None:
        if self._use_cuda_graph and self._selected_backend in (
            "cutlass",
            "trtllm-gen",
            "cute-dsl",
            "xqa",
        ):
            raise ValueError(
                "CUDA graph dense backend replan is not supported for "
                f"{self._selected_backend!r}: "
                "the first plan remains valid for capture and replay, but replacing "
                "its metadata tensors would invalidate captured launch pointers."
            )

    def _plan_fa2(self, plan_args: _MLAPlanArguments) -> _MLAWrapperPlanResult:
        enable_pdl = plan_args.enable_pdl
        is_var_seq = plan_args.is_var_seq
        cute_dsl_impl = plan_args.cute_dsl_impl
        use_sinks = plan_args.use_sinks
        qk_nope_head_dim = plan_args.qk_nope_head_dim
        kv_len = plan_args.kv_len
        page_table = plan_args.page_table
        if enable_pdl:
            raise ValueError("enable_pdl is not supported by the fa2 wrapper backend.")
        if is_var_seq is not None:
            raise ValueError("is_var_seq is not supported by the fa2 wrapper backend.")
        if cute_dsl_impl != "auto":
            raise ValueError(
                "cute_dsl_impl is not supported by the fa2 wrapper backend."
            )
        if use_sinks:
            raise ValueError("use_sinks is not supported by the fa2 wrapper backend.")
        if qk_nope_head_dim is not None:
            raise ValueError(
                "qk_nope_head_dim is only supported with trtllm-gen backend."
            )
        if self._backend != "auto" and (kv_len is not None or page_table is not None):
            raise ValueError(
                "kv_len and page_table plan options are only supported with "
                "backend='cutlass' (or backend='auto' as CUTLASS fallback metadata)."
            )
        csr = plan_args.csr
        backend_impl = _BatchMLAPagedAttentionFa2Backend(
            self._float_workspace_buffer,
            self._generated_fa_workspace,
            self._use_cuda_graph,
            self._qo_indptr_buf,
            self._kv_indptr_buf,
            self._kv_indices_buf,
            self._kv_len_arr_buf,
            (
                lambda: self._maybe_warn_blackwell_auto_fallback(
                    self.device, "fa2"
                )
            )
            if self._backend == "auto"
            else None,
        )
        backend_impl.plan(
            qo_indptr=csr.qo_indptr,
            kv_indptr=csr.kv_indptr,
            kv_indices=csr.kv_indices,
            kv_len_arr=csr.kv_len_arr,
            num_heads=plan_args.num_heads,
            head_dim_ckv=plan_args.head_dim_ckv,
            head_dim_kpe=plan_args.head_dim_kpe,
            page_size=plan_args.page_size,
            causal=plan_args.causal,
            sm_scale=plan_args.sm_scale,
            q_data_type=plan_args.q_data_type,
            kv_data_type=plan_args.kv_data_type,
            use_profiler=plan_args.use_profiler,
        )
        return _MLAWrapperPlanResult(
            backend_impl=backend_impl,
            csr=csr,
            dense=plan_args.resolved_dense,
        )

    def _plan_fa3(self, plan_args: _MLAPlanArguments) -> _MLAWrapperPlanResult:
        enable_pdl = plan_args.enable_pdl
        is_var_seq = plan_args.is_var_seq
        cute_dsl_impl = plan_args.cute_dsl_impl
        use_sinks = plan_args.use_sinks
        qk_nope_head_dim = plan_args.qk_nope_head_dim
        kv_len = plan_args.kv_len
        page_table = plan_args.page_table
        if enable_pdl:
            raise ValueError("enable_pdl is not supported by the fa3 wrapper backend.")
        if is_var_seq is not None:
            raise ValueError("is_var_seq is not supported by the fa3 wrapper backend.")
        if cute_dsl_impl != "auto":
            raise ValueError(
                "cute_dsl_impl is not supported by the fa3 wrapper backend."
            )
        if use_sinks:
            raise ValueError("use_sinks is not supported by the fa3 wrapper backend.")
        if qk_nope_head_dim is not None:
            raise ValueError(
                "qk_nope_head_dim is only supported with trtllm-gen backend."
            )
        if self._backend != "auto" and (kv_len is not None or page_table is not None):
            raise ValueError(
                "kv_len and page_table plan options are only supported with "
                "backend='cutlass' (or backend='auto' as CUTLASS fallback metadata)."
            )
        csr = plan_args.csr
        backend_impl = _BatchMLAPagedAttentionFa3Backend(
            self._float_workspace_buffer,
            self._generated_fa_workspace,
            self._use_cuda_graph,
            self._qo_indptr_buf,
            self._kv_indptr_buf,
            self._kv_indices_buf,
            self._kv_len_arr_buf,
            (
                lambda: self._maybe_warn_blackwell_auto_fallback(
                    self.device, "fa3"
                )
            )
            if self._backend == "auto"
            else None,
        )
        backend_impl.plan(
            qo_indptr=csr.qo_indptr,
            kv_indptr=csr.kv_indptr,
            kv_indices=csr.kv_indices,
            kv_len_arr=csr.kv_len_arr,
            num_heads=plan_args.num_heads,
            head_dim_ckv=plan_args.head_dim_ckv,
            head_dim_kpe=plan_args.head_dim_kpe,
            page_size=plan_args.page_size,
            causal=plan_args.causal,
            sm_scale=plan_args.sm_scale,
            q_data_type=plan_args.q_data_type,
            kv_data_type=plan_args.kv_data_type,
            use_profiler=plan_args.use_profiler,
        )
        return _MLAWrapperPlanResult(
            backend_impl=backend_impl,
            csr=csr,
            dense=plan_args.resolved_dense,
        )

    def _plan_cutlass(self, plan_args: _MLAPlanArguments) -> _MLAWrapperPlanResult:
        enable_pdl = plan_args.enable_pdl
        is_var_seq = plan_args.is_var_seq
        cute_dsl_impl = plan_args.cute_dsl_impl
        use_sinks = plan_args.use_sinks
        qk_nope_head_dim = plan_args.qk_nope_head_dim
        kv_len = plan_args.kv_len
        page_table = plan_args.page_table
        if enable_pdl:
            raise ValueError(
                "enable_pdl is not supported by the cutlass wrapper backend."
            )
        if is_var_seq is not None:
            raise ValueError(
                "is_var_seq is not supported by the cutlass wrapper backend."
            )
        if cute_dsl_impl != "auto":
            raise ValueError(
                "cute_dsl_impl is not supported by the cutlass wrapper backend."
            )
        if use_sinks:
            raise ValueError(
                "use_sinks is not supported by the cutlass wrapper backend."
            )
        if qk_nope_head_dim is not None:
            raise ValueError(
                "qk_nope_head_dim is only supported with trtllm-gen backend."
            )
        canonical_dense_supplied = plan_args.has_canonical_dense_metadata
        legacy_dense_supplied = plan_args.has_legacy_dense_metadata
        csr = None
        dense = None
        if canonical_dense_supplied or legacy_dense_supplied:
            if (
                not isinstance(plan_args.page_size, int)
                or isinstance(plan_args.page_size, bool)
                or plan_args.page_size <= 0
            ):
                raise ValueError(
                    f"page_size must be a positive int, got {plan_args.page_size!r}."
                )
            if plan_args.page_size > 128 or 128 % plan_args.page_size != 0:
                raise ValueError(
                    "cutlass dense metadata requires page_size to divide 128, "
                    f"got {plan_args.page_size}."
                )
            if legacy_dense_supplied:
                csr = plan_args.csr
                assert kv_len is not None
                assert page_table is not None
                dense = _validate_cutlass_plan_metadata(
                    kv_len,
                    page_table,
                    csr=csr,
                    page_size=plan_args.page_size,
                    device=self._float_workspace_buffer.device,
                )
            else:
                dense = plan_args.dense(
                    table_width_alignment=128 // plan_args.page_size,
                )
                csr = plan_args.resolved_csr
            batch_size = dense.cum_seq_lens_q.shape[0] - 1
        else:
            csr = plan_args.csr
            if self._backend == "auto":
                raise _BackendPlanUnsupportedError(
                    "automatic cutlass selection requires both kv_len and page_table "
                    "at plan time when canonical dense metadata was not supplied."
                )
            batch_size = csr.qo_indptr.shape[0] - 1
        backend_impl = _BatchMLAPagedAttentionCutlassBackend(
            self._float_workspace_buffer
        )
        backend_impl.plan(
            num_heads=plan_args.num_heads,
            head_dim_ckv=plan_args.head_dim_ckv,
            head_dim_kpe=plan_args.head_dim_kpe,
            page_size=plan_args.page_size,
            causal=plan_args.causal,
            sm_scale=plan_args.sm_scale,
            q_data_type=plan_args.q_data_type,
            kv_data_type=plan_args.kv_data_type,
            use_profiler=plan_args.use_profiler,
            batch_size=batch_size,
            kv_len=None if dense is None else dense.seq_lens,
            page_table=None if dense is None else dense.block_tables,
        )
        return _MLAWrapperPlanResult(
            backend_impl=backend_impl,
            csr=plan_args.resolved_csr,
            dense=dense,
        )

    def _plan_trtllm_gen(
        self, plan_args: _MLAPlanArguments
    ) -> _MLAWrapperPlanResult:
        cute_dsl_impl = plan_args.cute_dsl_impl
        kv_len = plan_args.kv_len
        page_table = plan_args.page_table
        if cute_dsl_impl != "auto":
            raise ValueError(
                "cute_dsl_impl is not supported by the trtllm-gen backend."
            )
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table plan options are only supported with "
                "backend='cutlass' (or backend='auto' as CUTLASS fallback metadata)."
            )
        if (
            not isinstance(plan_args.page_size, int)
            or isinstance(plan_args.page_size, bool)
            or plan_args.page_size <= 0
        ):
            raise ValueError(
                f"page_size must be a positive int, got {plan_args.page_size!r}."
            )
        if plan_args.page_size > 128 or 128 % plan_args.page_size != 0:
            raise ValueError(
                "trtllm-gen dense metadata requires page_size to divide 128, "
                f"got {plan_args.page_size}."
            )
        dense = plan_args.native_dense
        backend_impl = _BatchMLAPagedAttentionTrtllmGenBackend(
            self._float_workspace_buffer
        )
        backend_impl.plan(
            cum_seq_lens_q=dense.cum_seq_lens_q,
            block_tables=dense.block_tables,
            seq_lens=dense.seq_lens,
            max_q_len=dense.max_q_len,
            num_heads=plan_args.num_heads,
            head_dim_ckv=plan_args.head_dim_ckv,
            head_dim_kpe=plan_args.head_dim_kpe,
            page_size=plan_args.page_size,
            causal=plan_args.causal,
            sm_scale=plan_args.sm_scale,
            q_data_type=plan_args.q_data_type,
            kv_data_type=plan_args.kv_data_type,
            use_profiler=plan_args.use_profiler,
            qk_nope_head_dim=plan_args.qk_nope_head_dim,
            enable_pdl=plan_args.enable_pdl,
            is_var_seq=plan_args.is_var_seq,
            use_sinks=plan_args.use_sinks,
        )
        return _MLAWrapperPlanResult(
            backend_impl=backend_impl,
            csr=plan_args.resolved_csr,
            dense=dense,
        )

    def _plan_cute_dsl(
        self, plan_args: _MLAPlanArguments
    ) -> _MLAWrapperPlanResult:
        enable_pdl = plan_args.enable_pdl
        qk_nope_head_dim = plan_args.qk_nope_head_dim
        kv_len = plan_args.kv_len
        page_table = plan_args.page_table
        if enable_pdl:
            raise ValueError(
                "enable_pdl is not supported by the cute-dsl wrapper backend."
            )
        if plan_args.use_profiler:
            raise ValueError(
                "use_profiler is not supported by the cute-dsl wrapper backend."
            )
        if plan_args.causal:
            raise ValueError(
                "causal=True is not supported by the cute-dsl wrapper backend."
            )
        if qk_nope_head_dim is not None:
            raise ValueError(
                "qk_nope_head_dim is not supported by the cute-dsl wrapper backend."
            )
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported by the cute-dsl wrapper backend."
            )
        if (
            not isinstance(plan_args.page_size, int)
            or isinstance(plan_args.page_size, bool)
            or plan_args.page_size <= 0
        ):
            raise ValueError(
                f"page_size must be a positive int, got {plan_args.page_size!r}."
            )
        if plan_args.page_size > 128 or 128 % plan_args.page_size != 0:
            raise ValueError(
                "cute-dsl dense metadata requires page_size to divide 128, "
                f"got {plan_args.page_size}."
            )
        dense = plan_args.dense(
            table_width_alignment=128 // plan_args.page_size,
        )
        backend_impl = _BatchMLAPagedAttentionCuteDslBackend(
            self._float_workspace_buffer
        )
        backend_impl.plan(
            cum_seq_lens_q=dense.cum_seq_lens_q,
            block_tables=dense.block_tables,
            seq_lens=dense.seq_lens,
            max_q_len=dense.max_q_len,
            num_heads=plan_args.num_heads,
            head_dim_ckv=plan_args.head_dim_ckv,
            head_dim_kpe=plan_args.head_dim_kpe,
            page_size=plan_args.page_size,
            causal=plan_args.causal,
            sm_scale=plan_args.sm_scale,
            q_data_type=plan_args.q_data_type,
            kv_data_type=plan_args.kv_data_type,
            use_profiler=plan_args.use_profiler,
            is_var_seq=plan_args.is_var_seq,
            cute_dsl_impl=plan_args.cute_dsl_impl,
            use_sinks=plan_args.use_sinks,
        )
        return _MLAWrapperPlanResult(
            backend_impl=backend_impl,
            csr=plan_args.resolved_csr,
            dense=dense,
        )

    def _plan_xqa(self, plan_args: _MLAPlanArguments) -> _MLAWrapperPlanResult:
        if plan_args.use_profiler:
            raise ValueError(
                "use_profiler is not supported by the XQA wrapper backend."
            )
        if plan_args.causal:
            raise ValueError("causal=True is not supported by the XQA wrapper backend.")
        if plan_args.qk_nope_head_dim is not None:
            raise ValueError(
                "qk_nope_head_dim is not supported by the XQA wrapper backend."
            )
        if plan_args.kv_len is not None or plan_args.page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported by the XQA wrapper backend."
            )
        if plan_args.is_var_seq is not None:
            raise ValueError("is_var_seq is not supported by the XQA wrapper backend.")
        if plan_args.cute_dsl_impl != "auto":
            raise ValueError(
                "cute_dsl_impl is not supported by the XQA wrapper backend."
            )
        if plan_args.use_sinks:
            raise ValueError("use_sinks is not supported by the XQA wrapper backend.")
        if (
            not isinstance(plan_args.page_size, int)
            or isinstance(plan_args.page_size, bool)
            or plan_args.page_size <= 0
        ):
            raise ValueError(
                f"page_size must be a positive int, got {plan_args.page_size!r}."
            )
        if plan_args.page_size > 128 or 128 % plan_args.page_size != 0:
            raise ValueError(
                "xqa dense metadata requires page_size to divide 128, "
                f"got {plan_args.page_size}."
            )
        dense = plan_args.dense(
            table_width_alignment=128 // plan_args.page_size,
        )
        backend_impl = _BatchMLAPagedAttentionXqaBackend(self._float_workspace_buffer)
        backend_impl.plan(
            cum_seq_lens_q=dense.cum_seq_lens_q,
            block_tables=dense.block_tables,
            seq_lens=dense.seq_lens,
            max_q_len=dense.max_q_len,
            num_heads=plan_args.num_heads,
            head_dim_ckv=plan_args.head_dim_ckv,
            head_dim_kpe=plan_args.head_dim_kpe,
            page_size=plan_args.page_size,
            causal=plan_args.causal,
            sm_scale=plan_args.sm_scale,
            q_data_type=plan_args.q_data_type,
            kv_data_type=plan_args.kv_data_type,
            use_profiler=plan_args.use_profiler,
            enable_pdl=plan_args.enable_pdl,
        )
        return _MLAWrapperPlanResult(
            backend_impl=backend_impl,
            csr=plan_args.resolved_csr,
            dense=dense,
        )

    # CSR metadata form.
    @overload
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
        *,
        qk_nope_head_dim: Optional[int] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        cute_dsl_impl: str = "auto",
        use_sinks: bool = False,
    ) -> None: ...

    # Complete equivalent CSR and dense metadata forms.
    @overload
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
        *,
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        cute_dsl_impl: str = "auto",
        use_sinks: bool = False,
    ) -> None: ...

    # Dense page-table metadata form.
    @overload
    def plan(
        self,
        *,
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        max_q_len: Optional[int] = None,
        use_profiler: bool = False,
        qk_nope_head_dim: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        cute_dsl_impl: str = "auto",
        use_sinks: bool = False,
    ) -> None: ...

    @flashinfer_api
    def plan(
        self,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
        num_heads: Optional[int] = None,
        head_dim_ckv: Optional[int] = None,
        head_dim_kpe: Optional[int] = None,
        page_size: Optional[int] = None,
        causal: Optional[bool] = None,
        sm_scale: Optional[float] = None,
        q_data_type: Optional[torch.dtype] = None,
        kv_data_type: Optional[torch.dtype] = None,
        use_profiler: bool = False,
        *,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        cute_dsl_impl: str = "auto",
        use_sinks: bool = False,
    ) -> None:
        r"""Plan from one or two equivalent canonical metadata forms.

        **CSR form**

        Uses ``qo_indptr``, ``kv_indptr``, ``kv_indices``, and
        ``kv_len_arr``.  This form supports the existing positional call
        convention.

        **Dense page-table form**

        Uses the keyword-only ``cum_seq_lens_q``, ``block_tables``, and
        ``seq_lens`` arguments with optional ``max_q_len``.

        The CSR and dense page-table forms may be supplied together when they
        are logically equivalent. ``kv_len`` and ``page_table`` remain a
        compatibility exception only for CSR calls that may select CUTLASS.

        Metadata and required common arguments explicitly set to ``None`` are
        treated as omitted. In particular, ``max_q_len=None`` derives the value
        from the supplied query metadata.

        For ``backend="cute-dsl"``, ``cute_dsl_impl`` selects ``"auto"``,
        ``"monolithic"``, or ``"modular"``.  ``use_sinks=True`` plans the
        modular sink-capable kernel, and ``is_var_seq`` must match whether the
        planned KV ``seq_lens`` vary.  Query lengths must remain uniform.  CuTe
        DSL is explicit-only and is never an automatic-backend candidate.

        For ``backend="xqa"``, both canonical metadata forms are accepted and
        normalized to a shared dense page table during planning. XQA supports
        single-token decode on SM120/SM121 and is never an automatic-backend
        candidate in this wrapper.
        """
        self._generated_fa_workspace.raise_if_invalid()
        self._reject_unsafe_cuda_graph_replan()
        if hasattr(self, "_plan_in_progress"):
            raise RuntimeError(
                "BatchMLAPagedAttentionWrapper.plan() cannot be called reentrantly "
                "while planning is already in progress."
            )

        self._plan_in_progress = None
        try:
            common_values = {
                "num_heads": num_heads,
                "head_dim_ckv": head_dim_ckv,
                "head_dim_kpe": head_dim_kpe,
                "page_size": page_size,
                "causal": causal,
                "sm_scale": sm_scale,
                "q_data_type": q_data_type,
                "kv_data_type": kv_data_type,
            }
            missing_common = [
                name for name, value in common_values.items() if value is None
            ]
            if missing_common:
                raise TypeError(
                    "plan() missing required arguments: " + ", ".join(missing_common)
                )

            plan_args = _MLAPlanArguments(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_len_arr=kv_len_arr,
                num_heads=cast(int, num_heads),
                head_dim_ckv=cast(int, head_dim_ckv),
                head_dim_kpe=cast(int, head_dim_kpe),
                page_size=cast(int, page_size),
                causal=cast(bool, causal),
                sm_scale=cast(float, sm_scale),
                q_data_type=cast(torch.dtype, q_data_type),
                kv_data_type=cast(torch.dtype, kv_data_type),
                use_profiler=use_profiler,
                cum_seq_lens_q=cum_seq_lens_q,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_q_len=max_q_len,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_len=kv_len,
                page_table=page_table,
                enable_pdl=enable_pdl,
                is_var_seq=is_var_seq,
                cute_dsl_impl=cute_dsl_impl,
                use_sinks=use_sinks,
                device=self.device,
            )

            if self._backend != "auto":
                if self._backend not in (
                    "fa2",
                    "fa3",
                    "cutlass",
                    "trtllm-gen",
                    "cute-dsl",
                    "xqa",
                ):
                    raise ValueError(f"unsupported MLA backend {self._backend!r}")
                plan_adapter = getattr(self, f"_plan_{self._backend.replace('-', '_')}")
                result = plan_adapter(plan_args)
                self._backend_impl = result.backend_impl
                self._selected_backend = self._backend
                self._csr_plan_metadata = result.csr
                self._dense_plan_metadata = result.dense
                logger.info(
                    "BatchMLAPagedAttentionWrapper requested backend '%s' selected backend '%s'",
                    self._backend,
                    self._backend,
                )
                return

            # auto backend selection logic, TODO @mingyangw
            if self._use_cuda_graph and self._selected_backend in ("fa2", "fa3"):
                candidates = [self._selected_backend]
            else:
                preferred_backend = determine_mla_backend(self.device)
                candidates = [preferred_backend]
                if preferred_backend != "cutlass":
                    candidates.append("cutlass")

            rejections = []
            last_rejection = None
            for candidate in candidates:
                try:
                    if candidate not in ("fa2", "fa3", "cutlass"):
                        raise ValueError(
                            f"unsupported automatic MLA backend {candidate!r}"
                        )
                    plan_adapter = getattr(self, f"_plan_{candidate}")
                    result = plan_adapter(plan_args)
                except _BackendPlanUnsupportedError as err:
                    last_rejection = err
                    reason = str(err)
                    rejections.append((candidate, reason))
                    logger.debug(
                        "BatchMLAPagedAttentionWrapper automatically rejected backend '%s': %s",
                        candidate,
                        reason,
                    )
                    continue

                self._backend_impl = result.backend_impl
                self._selected_backend = candidate
                self._csr_plan_metadata = result.csr
                self._dense_plan_metadata = result.dense
                logger.info(
                    "BatchMLAPagedAttentionWrapper requested backend 'auto' selected backend '%s'",
                    candidate,
                )
                return

            candidate_names = ", ".join(candidates)
            rejection_summary = "; ".join(
                f"{candidate}: {reason}" for candidate, reason in rejections
            )
            raise _BackendPlanUnsupportedError(
                f"backend='auto' rejected all candidates [{candidate_names}]: "
                f"{rejection_summary}"
            ) from last_rejection
        finally:
            del self._plan_in_progress

    def _run_fa2(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if sinks is not None:
            raise ValueError("sinks are not supported by the fa2 wrapper backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported by the "
                "fa2 wrapper backend."
            )
        if bmm1_scale is not None:
            raise ValueError("bmm1_scale is not supported by the fa2 wrapper backend.")
        if bmm2_scale is not None:
            raise ValueError("bmm2_scale is not supported by the fa2 wrapper backend.")
        if kv_len is not None:
            raise ValueError("kv_len is only supported with cutlass backend.")
        if page_table is not None:
            raise ValueError("page_table is only supported with cutlass backend.")
        if o_scale is not None:
            raise ValueError(
                "o_scale is only supported with the cutlass backend for now."
            )
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are only supported with the fa3 backend "
                "and FP8 kv_data_type."
            )
        return self._backend_impl.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            profiler_buffer=profiler_buffer,
            return_lse_base_on_e=return_lse_base_on_e,
        )

    def _run_fa3(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if sinks is not None:
            raise ValueError("sinks are not supported by the fa3 wrapper backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported by the "
                "fa3 wrapper backend."
            )
        if bmm1_scale is not None:
            raise ValueError("bmm1_scale is not supported by the fa3 wrapper backend.")
        if bmm2_scale is not None:
            raise ValueError("bmm2_scale is not supported by the fa3 wrapper backend.")
        if kv_len is not None:
            raise ValueError("kv_len is only supported with cutlass backend.")
        if page_table is not None:
            raise ValueError("page_table is only supported with cutlass backend.")
        if o_scale is not None:
            raise ValueError(
                "o_scale is only supported with the cutlass backend for now."
            )
        return self._backend_impl.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            profiler_buffer=profiler_buffer,
            return_lse_base_on_e=return_lse_base_on_e,
            ckv_scale=ckv_scale,
            kpe_scale=kpe_scale,
        )

    def _run_cutlass(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> torch.Tensor:
        if sinks is not None:
            raise ValueError("sinks are not supported by the cutlass wrapper backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported by the "
                "cutlass wrapper backend."
            )
        if bmm1_scale is not None:
            raise ValueError(
                "bmm1_scale is not supported by the cutlass wrapper backend."
            )
        if bmm2_scale is not None:
            raise ValueError(
                "bmm2_scale is not supported by the cutlass wrapper backend."
            )
        if return_lse:
            raise ValueError("return_lse is not supported with cutlass backend.")
        if lse is not None:
            raise ValueError("lse is not supported with cutlass backend.")
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with cutlass backend.")
        if return_lse_base_on_e:
            raise ValueError(
                "return_lse_base_on_e is not supported with cutlass backend."
            )
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are only supported with the fa3 backend "
                "and FP8 kv_data_type."
            )
        backend_impl = self._backend_impl
        if backend_impl is None:
            backend_impl = _BatchMLAPagedAttentionCutlassBackend(
                self._float_workspace_buffer
            )
            backend_impl.plan(
                num_heads=q_nope.shape[1],
                head_dim_ckv=q_nope.shape[2],
                head_dim_kpe=q_pe.shape[2],
                page_size=ckv_cache.shape[1],
                causal=False,
                sm_scale=1.0 / math.sqrt(128 + q_pe.shape[2]),
                q_data_type=q_nope.dtype,
                kv_data_type=ckv_cache.dtype,
                use_profiler=False,
                batch_size=q_nope.shape[0],
                kv_len=None,
                page_table=None,
            )
        return backend_impl.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            kv_len=kv_len,
            page_table=page_table,
            o_scale=o_scale,
        )

    def _run_trtllm_gen(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if profiler_buffer is not None:
            raise ValueError(
                "profiler_buffer is not supported with trtllm-gen backend."
            )
        if kv_len is not None:
            raise ValueError(
                "kv_len is not supported with trtllm-gen backend; KV lengths "
                "are captured from seq_lens at plan time."
            )
        if page_table is not None:
            raise ValueError(
                "page_table is not supported with trtllm-gen backend; "
                "block_tables are captured at plan time."
            )
        if return_lse_base_on_e:
            raise ValueError(
                "return_lse_base_on_e is not supported with trtllm-gen backend."
            )
        if o_scale is not None:
            raise ValueError("o_scale is not supported with trtllm-gen backend.")
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are only supported with the fa3 backend "
                "and FP8 kv_data_type."
            )
        return self._backend_impl.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            sinks=sinks,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )

    def _run_cute_dsl(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with cute-dsl backend.")
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported with cute-dsl backend."
            )
        if return_lse_base_on_e:
            raise ValueError(
                "return_lse_base_on_e is not supported with cute-dsl backend; "
                "CuTe DSL LSE is already returned in natural-log base."
            )
        if o_scale is not None:
            raise ValueError("o_scale is not supported with cute-dsl backend.")
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are not supported with cute-dsl backend."
            )
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported with "
                "cute-dsl backend."
            )
        return self._backend_impl.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            sinks=sinks,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )

    def _run_xqa(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> torch.Tensor:
        if return_lse or lse is not None:
            raise ValueError("XQA MLA wrapper does not support LSE output.")
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with XQA backend.")
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported with XQA backend."
            )
        if return_lse_base_on_e:
            raise ValueError("return_lse_base_on_e is not supported with XQA backend.")
        if o_scale is not None:
            raise ValueError("o_scale is not supported with XQA backend.")
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are not supported with XQA backend."
            )
        if sinks is not None:
            raise ValueError("sinks are not supported with XQA backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported with XQA backend."
            )
        for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
            if isinstance(scale, torch.Tensor):
                raise ValueError(
                    f"XQA MLA wrapper accepts {name} as a float only; "
                    "tensor scales are not supported."
                )
        return self._backend_impl.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=None,
            return_lse=False,
            bmm1_scale=cast(Optional[float], bmm1_scale),
            bmm2_scale=cast(Optional[float], bmm2_scale),
        )

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        kpe_scale: Optional[float] = None,
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        kpe_scale: Optional[float] = None,
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @flashinfer_api(trace=mla_paged_decode_trace)
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        kpe_scale: Optional[float] = None,
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run the MLA attention computation.

        **Output-only form**

        With ``return_lse=False``, returns the output tensor.

        **Output-and-LSE form**

        With ``return_lse=True``, returns a tuple containing the output tensor
        and the log-sum-exp tensor.

        Parameters
        ----------
        q_nope : torch.Tensor
            The query tensor without rope, shape: ``[batch_size, num_heads, head_dim_ckv]``.
        q_pe : torch.Tensor
            The rope part of the query tensor, shape: ``[batch_size, num_heads, head_dim_kpe]``.
        ckv_cache : torch.Tensor
            The compressed kv-cache tensor (without rope), shape: ``[num_pages, page_size, head_dim_ckv]``.
            ``head_dim_ckv`` is 512 in DeepSeek v2/v3 models.
        kpe_cache : torch.Tensor
            The rope part of the kv-cache tensor, shape: ``[num_pages, page_size, head_dim_kpe]``.
            ``head_dim_kpe`` is 64 in DeepSeek v2/v3 models.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
            When ``o_scale`` is provided, this should be an FP8 tensor.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool, optional
            Whether to return the log-sum-exp value, default is False.
        profiler_buffer : Optional[torch.Tensor]
            The buffer to store the profiler data.
        kv_len : Optional[torch.Tensor]
            The KV length of each request, shape: ``[batch_size]``. For CUTLASS,
            this may be omitted when captured by ``plan()``. If supplied after
            planning, it must alias the same planned tensor view.
        page_table : Optional[torch.Tensor]
            The CUTLASS page table, shape: ``[batch_size, num_pages]``. This may
            be omitted when captured by ``plan()``; run-time values must be
            supplied together with ``kv_len`` and alias planned metadata when
            both exist. Both ``kv_len`` and ``page_table`` are required when an
            explicitly requested CUTLASS backend runs without ``plan()``.
        return_lse_base_on_e : bool, optional
            Controls the base of the returned LSE values when ``return_lse=True``.
            If ``False`` (default), the LSE is returned in base-2
            (``log2(sum(exp2(...)))``) to match the kernel's internal log-base.
            If ``True``, the LSE is converted to natural-log base (``log(sum(exp(...)))``)
            for compatibility with cascade-merging APIs that expect base-e LSEs.
        o_scale : Optional[float]
            FP8 output dequantization scale (``real = quantized * o_scale``).
            When provided, ``out`` must be an FP8 tensor. Only supported with
            the ``cutlass`` backend.
        ckv_scale : Optional[float]
            Per-tensor dequantization scale for the compressed-KV cache when
            ``kv_data_type`` is FP8 (``real = quantized * ckv_scale``). Required
            (together with ``kpe_scale``) for the FP8 KV cache path on the
            ``fa3`` backend. Must be a finite positive value. Must not be
            provided when ``kv_data_type`` is BF16/FP16.
        kpe_scale : Optional[float]
            Per-tensor dequantization scale for the rope-K cache when
            ``kv_data_type`` is FP8 (``real = quantized * kpe_scale``). Same
            usage rules as ``ckv_scale``.
        sinks : Optional[torch.Tensor]
            Per-head float32 attention sinks.  For ``backend="cute-dsl"``,
            sinks must be planned with ``use_sinks=True`` and use the modular
            implementation.
        bmm1_scale : Optional[float]
            Finite run-time attention-logit scale override for CuTe DSL or
            XQA. If omitted, the ``sm_scale`` captured by ``plan()`` is used.
            The XQA wrapper accepts Python floats only; its functional API also
            supports a paired FP8 device-tensor scale mode.
        bmm2_scale : Optional[float]
            Finite run-time output scale override for CuTe DSL or XQA.
            Defaults to ``1.0``. These wrapper backends accept scalar Python
            floats, not scale tensors.

        Notes
        -----
        The CuTe DSL monolithic implementation supports LSE output; the
        modular implementation does not. XQA does not support LSE output.
        """
        self._generated_fa_workspace.raise_if_invalid()
        has_fused_scale = bmm1_scale is not None or bmm2_scale is not None
        has_per_tensor_scale = ckv_scale is not None or kpe_scale is not None
        if has_fused_scale and has_per_tensor_scale:
            raise ValueError(
                "fused bmm scales and ckv_scale / kpe_scale are mutually exclusive."
            )
        bmm1_is_tensor = isinstance(bmm1_scale, torch.Tensor)
        bmm2_is_tensor = isinstance(bmm2_scale, torch.Tensor)
        if bmm1_is_tensor != bmm2_is_tensor and self._selected_backend != "xqa":
            raise ValueError(
                "bmm1_scale and bmm2_scale must be supplied together as a tensor pair."
            )

        if self._selected_backend is None and self._backend == "cutlass":
            result = self._run_cutlass(
                q_nope=q_nope,
                q_pe=q_pe,
                ckv_cache=ckv_cache,
                kpe_cache=kpe_cache,
                out=out,
                lse=lse,
                return_lse=return_lse,
                profiler_buffer=profiler_buffer,
                kv_len=kv_len,
                page_table=page_table,
                return_lse_base_on_e=return_lse_base_on_e,
                o_scale=o_scale,
                ckv_scale=ckv_scale,
                kpe_scale=kpe_scale,
                sinks=sinks,
                skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
            )
        elif self._selected_backend in (
            "fa2",
            "fa3",
            "cutlass",
            "trtllm-gen",
            "cute-dsl",
            "xqa",
        ):
            run_adapter = getattr(self, self._run_adapter_names[self._selected_backend])
            result = run_adapter(
                q_nope=q_nope,
                q_pe=q_pe,
                ckv_cache=ckv_cache,
                kpe_cache=kpe_cache,
                out=out,
                lse=lse,
                return_lse=return_lse,
                profiler_buffer=profiler_buffer,
                kv_len=kv_len,
                page_table=page_table,
                return_lse_base_on_e=return_lse_base_on_e,
                o_scale=o_scale,
                ckv_scale=ckv_scale,
                kpe_scale=kpe_scale,
                sinks=sinks,
                skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
            )
        else:
            raise RuntimeError(
                f"BatchMLAPagedAttentionWrapper.run() received unexpected selected backend {self._selected_backend!r}"
                "\nDid you forget to call BatchMLAPagedAttentionWrapper.plan()?"
            )
        return result
