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
import functools
import logging
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
    make_bucket_mapper,
)
from flashinfer.trace.templates.attention import (
    xqa_batch_decode_mla_trace,
)
from flashinfer.utils import (
    _check_block_tables_shape,
    _resolve_trtllm_gen_multi_ctas_kv_counter_buffer,
    check_shape_dtype_device,
    device_support_pdl,
    get_device_sm_count,
    log2e,
)

from ._backends.cute_dsl_backend import (
    CuteDslMlaDecodeRunner,
    _BatchMLAPagedAttentionCuteDslBackend,
    _cute_dsl_incompatibility_reason,
    _cute_dsl_max_supported_batch,
)

from ._backends.cutlass_backend import (
    get_mla_module as _get_mla_module,
)
from ._backends._fa_common import (
    get_batch_mla_module as _get_batch_mla_module,
)
from ._backends.trtllm_gen_backend import (
    _TRTLLM_GEN_MLA_MAX_BATCH,
    _BatchMLAPagedAttentionTrtllmGenBackend,
    TrtllmGenMlaDecodeRunner,
    get_trtllm_gen_fmha_module as _get_trtllm_gen_fmha_module,
)
from ._backends.xqa_backend import _BatchMLAPagedAttentionXqaBackend

logger = logging.getLogger(__name__)


get_mla_module = _get_mla_module
get_batch_mla_module = _get_batch_mla_module
get_trtllm_gen_fmha_module = _get_trtllm_gen_fmha_module


_MLAHeadDimensions = namedtuple(
    "_MLAHeadDimensions",
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
    runner_names: Sequence[str],
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


@functools.cache
def _mla_decode_tuning_config(
    buckets: Tuple[int, ...],
    num_pages: int,
    profile_seq_len: int,
) -> TuningConfig:
    """Return a stable per-shape tuning config for the batch sweep.

    ``AutoTuner._find_nearest_profile`` caches by the tuning config. Reusing
    the config and its initializer closures prevents distinct but equivalent
    per-call closures from retaining an unbounded profile cache.
    """

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
                tensor_initializers=(None, init_block_tables, init_seq_lens, None),
            ),
        ),
        use_cuda_graph=True,
        use_cold_l2_cache=True,
    )


def _build_mla_decode_tuning_config(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    workspace_buffer: torch.Tensor,
    runner_names: Sequence[str],
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_seq_len: int,
    device: torch.device,
    cute_dsl_max_batch: Optional[int] = None,
) -> TuningConfig:
    """Reduce one dispatch request to the stable tuning-config cache key."""
    page_size = kv_cache.shape[-2]
    provisioned_max_seq_len = block_tables.shape[-1] * page_size
    profile_seq_len = min(max_seq_len, provisioned_max_seq_len)
    buckets = _compute_mla_decode_buckets(
        workspace_buffer,
        runner_names,
        q_len,
        num_heads,
        kv_lora_rank,
        device,
        cute_dsl_max_batch,
    )
    return _mla_decode_tuning_config(buckets, kv_cache.shape[0], profile_seq_len)


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
        multi_ctas_kv_counter_buffer = _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
            multi_ctas_kv_counter_buffer,
            query.size(0),
            query.size(2),
            sm_count,
            query.device,
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


def _run_mla_decode_planned_dense_backend(
    *,
    backend: Literal["trtllm-gen", "cute-dsl"],
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: Optional[torch.Tensor],
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[List[torch.Tensor]],
    skip_softmax_threshold_scale_factor: Optional[float],
    enable_pdl: Optional[bool],
    is_var_seq: bool,
    lse: Optional[torch.Tensor],
    return_lse: bool,
    cute_dsl_impl: str,
    cum_seq_lens_q: Optional[torch.Tensor],
    max_q_len: Optional[int],
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor],
    sparse_mla_top_k: int = 0,
    uses_shared_paged_kv_idx: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Lower one explicit dense functional request through a planned backend."""
    if cum_seq_lens_q is None:
        if query.ndim != 4:
            raise ValueError(
                "query must have shape [batch_size, q_len, num_heads, head_dim_qk]"
            )
        batch_size, q_len = query.shape[:2]
        cum_seq_lens_q = torch.arange(
            batch_size + 1, device=query.device, dtype=torch.int32
        ).mul_(q_len)
        query_flat = query.flatten(0, 1)
        expected_out_shape = query.shape[:-1] + (kv_lora_rank,)
        resolved_max_q_len = max_q_len or q_len
    else:
        if backend == "cute-dsl":
            raise ValueError("cute-dsl MLA does not support cum_seq_lens_q")
        if query.ndim != 3:
            raise ValueError(
                "query must have shape [total_q, num_heads, head_dim_qk] "
                "when cum_seq_lens_q is provided"
            )
        query_flat = query
        expected_out_shape = query.shape[:-1] + (kv_lora_rank,)
        resolved_max_q_len = max_q_len or query.size(0)

    kv_cache = _check_trtllm_gen_mla_shape(
        query,
        kv_cache,
        kv_lora_rank,
        qk_rope_head_dim,
        sparse_mla_top_k,
        block_tables,
        kv_cache.size(-2),
        uses_shared_paged_kv_idx,
        batch_size=cum_seq_lens_q.size(0) - 1,
        max_q_len=resolved_max_q_len,
        require_aligned_block_table=backend == "cute-dsl",
    )
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=query.device)
    else:
        check_shape_dtype_device(
            out, expected_out_shape, torch.bfloat16, query.device, "out"
        )

    user_lse = lse
    if lse is not None and lse.ndim == 3:
        lse = lse.flatten(0, 1)
    if return_lse and lse is None:
        lse = torch.empty(
            (query_flat.size(0), query_flat.size(1)),
            dtype=torch.float32,
            device=query.device,
        )
        user_lse = lse

    sink = sinks
    if isinstance(sinks, (list, tuple)):
        if len(sinks) != 1:
            raise ValueError(f"{backend} MLA expects sinks to contain one tensor")
        sink = sinks[0]

    if backend == "trtllm-gen":
        trtllm_gen_backend = _BatchMLAPagedAttentionTrtllmGenBackend(workspace_buffer)
        trtllm_gen_backend.plan(
            cum_seq_lens_q=cum_seq_lens_q,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=resolved_max_q_len,
            max_seq_len=max_seq_len,
            num_heads=query_flat.size(1),
            head_dim_ckv=kv_lora_rank,
            head_dim_kpe=qk_rope_head_dim,
            page_size=kv_cache.size(-2),
            causal=False,
            sm_scale=1.0 / (qk_nope_head_dim + qk_rope_head_dim) ** 0.5,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
            use_profiler=False,
            qk_nope_head_dim=qk_nope_head_dim,
            enable_pdl=enable_pdl,
            is_var_seq=is_var_seq,
            use_sinks=sink is not None,
            sparse_mla_top_k=sparse_mla_top_k,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            has_ragged_query=cum_seq_lens_q is not None and query.ndim == 3,
            allow_fp8=True,
        )
        result = trtllm_gen_backend.run(
            q_nope=query_flat[..., :kv_lora_rank],
            q_pe=query_flat[..., kv_lora_rank:],
            ckv_cache=kv_cache.squeeze(1)[..., :kv_lora_rank],
            kpe_cache=kv_cache.squeeze(1)[..., kv_lora_rank:],
            out=out.flatten(0, 1) if out.ndim == 4 else out,
            lse=lse,
            return_lse=return_lse,
            sinks=sink,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )
    else:
        cute_dsl_backend = _BatchMLAPagedAttentionCuteDslBackend(workspace_buffer)
        cute_dsl_backend.plan(
            cum_seq_lens_q=cum_seq_lens_q,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=resolved_max_q_len,
            num_heads=query_flat.size(1),
            head_dim_ckv=kv_lora_rank,
            head_dim_kpe=qk_rope_head_dim,
            page_size=kv_cache.size(-2),
            causal=False,
            sm_scale=bmm1_scale,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
            use_profiler=False,
            is_var_seq=is_var_seq,
            cute_dsl_impl=cute_dsl_impl,
            use_sinks=sink is not None,
            enable_pdl=enable_pdl,
        )
        result = cute_dsl_backend.run(
            q_nope=query_flat[..., :kv_lora_rank],
            q_pe=query_flat[..., kv_lora_rank:],
            ckv_cache=kv_cache.squeeze(1)[..., :kv_lora_rank],
            kpe_cache=kv_cache.squeeze(1)[..., kv_lora_rank:],
            out=out.flatten(0, 1) if out.ndim == 4 else out,
            lse=lse,
            return_lse=return_lse,
            sinks=sink,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    if return_lse:
        _, backend_lse = result
        return out, user_lse if user_lse is not None else backend_lse
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
            "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
        )
    if skip_softmax_threshold_scale_factor is not None:
        raise ValueError("skip_softmax is not supported for XQA backend")
    if not uses_shared_paged_kv_idx:
        raise ValueError(
            "XQA MLA does not support separate KV page indices "
            "(uses_shared_paged_kv_idx=False)"
        )
    if sinks is not None:
        raise ValueError("XQA MLA does not support sinks")
    del (
        qk_nope_head_dim,
        max_seq_len,
        cute_dsl_impl,
        kv_scale_format,
        is_var_seq,
    )
    if kv_cache.ndim == 4:
        if kv_cache.size(1) != 1:
            raise ValueError(
                "XQA MLA expects a single KV cache head, "
                f"got kv_cache.shape[1] == {kv_cache.size(1)}"
            )
        kv_cache = kv_cache.squeeze(1)
    elif kv_cache.ndim != 3:
        raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")

    backend_impl = _BatchMLAPagedAttentionXqaBackend(workspace_buffer)
    backend_impl.plan(
        cum_seq_lens_q=None,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=query.size(1),
        num_heads=query.size(2),
        head_dim_ckv=kv_lora_rank,
        head_dim_kpe=qk_rope_head_dim,
        page_size=kv_cache.size(1),
        causal=False,
        sm_scale=bmm1_scale if isinstance(bmm1_scale, float) else 1.0,
        q_data_type=query.dtype,
        kv_data_type=kv_cache.dtype,
        use_profiler=False,
        enable_pdl=enable_pdl,
        initialize_semaphore=False,
    )
    backend_out = backend_impl.run(
        q_nope=query[..., :kv_lora_rank].squeeze(1),
        q_pe=query[..., kv_lora_rank:].squeeze(1),
        ckv_cache=kv_cache[..., :kv_lora_rank],
        kpe_cache=kv_cache[..., kv_lora_rank:],
        out=None if out is None else out.squeeze(1),
        lse=lse,
        return_lse=return_lse,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    if out is not None:
        return out
    return backend_out.unsqueeze(1)


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
    if backend == "trtllm-gen":
        if seq_lens is None:
            raise ValueError("seq_lens is required for trtllm-gen MLA")
        if cum_seq_lens_q is not None and (return_lse or lse is not None):
            raise NotImplementedError(
                "trtllm-gen MLA does not support return_lse/lse with cum_seq_lens_q"
            )
        return _run_mla_decode_planned_dense_backend(
            backend="trtllm-gen",
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            out=out,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=sinks,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            enable_pdl=enable_pdl,
            is_var_seq=is_var_seq,
            lse=lse,
            return_lse=return_lse,
            cute_dsl_impl="auto",
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
            sparse_mla_top_k=sparse_mla_top_k,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
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
    if cum_seq_lens_q is not None or max_q_len is not None:
        raise ValueError("cute-dsl MLA does not support cum_seq_lens_q / max_q_len")
    if sparse_mla_top_k > 0:
        raise ValueError("cute-dsl MLA does not support sparse_mla_top_k")
    if skip_softmax_threshold_scale_factor is not None:
        raise ValueError(
            "cute-dsl MLA does not support skip_softmax_threshold_scale_factor"
        )
    if not uses_shared_paged_kv_idx:
        raise ValueError("cute-dsl MLA does not support separate KV page indices")
    if multi_ctas_kv_counter_buffer is not None:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
        )
    if seq_lens is None:
        raise ValueError("seq_lens is required for cute-dsl MLA")
    return _run_mla_decode_planned_dense_backend(
        backend="cute-dsl",
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        enable_pdl=enable_pdl,
        is_var_seq=is_var_seq,
        lse=lse,
        return_lse=return_lse,
        cute_dsl_impl=cute_dsl_impl,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        multi_ctas_kv_counter_buffer=None,
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
    return _run_mla_decode_xqa(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        enable_pdl=enable_pdl,
        backend="xqa",
    )
