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

from dataclasses import replace
import functools
import threading
from typing import Callable, List, Optional, Sequence, Tuple, Union
import warnings

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TuningConfig,
    make_bucket_mapper,
)
from flashinfer.trace.templates.attention import xqa_batch_decode_mla_trace

from ._backends._cute_dsl_functional_common import (
    _cute_dsl_max_supported_batch,
)
from ._backends.cutlass_backend import CutlassMlaRunner
from ._backends.cute_dsl_modular_backend import CuteDslModularMlaDecodeRunner
from ._backends.cute_dsl_monolithic_backend import (
    CuteDslMonolithicMlaDecodeRunner,
)
from ._backends.fa2_backend import Fa2MlaRunner
from ._backends.fa3_backend import Fa3MlaRunner
from ._backends.trtllm_gen_backend import (
    _TRTLLM_GEN_MLA_MAX_BATCH,
    _trtllm_gen_mla_incompatibility_reason,
    TrtllmGenMlaDecodeRunner,
)
from ._backends.xqa_backend import XqaMlaDecodeRunner
from ._contracts import (
    _FunctionalBackendUnsupportedError,
    _FunctionalMLARequest,
    _FunctionalMLARunner,
)


_xqa_batch_decode_with_kv_cache_mla_warning_emitted = False
_xqa_batch_decode_with_kv_cache_mla_warning_lock = threading.Lock()


def _warn_xqa_batch_decode_with_kv_cache_mla_once() -> None:
    global _xqa_batch_decode_with_kv_cache_mla_warning_emitted
    if _xqa_batch_decode_with_kv_cache_mla_warning_emitted:
        return
    with _xqa_batch_decode_with_kv_cache_mla_warning_lock:
        if _xqa_batch_decode_with_kv_cache_mla_warning_emitted:
            return
        warnings.warn(
            "xqa_batch_decode_with_kv_cache_mla is deprecated; use "
            'batch_mla_paged_attention(..., backend="xqa") instead.',
            DeprecationWarning,
            stacklevel=3,
        )
        _xqa_batch_decode_with_kv_cache_mla_warning_emitted = True


_FUNCTIONAL_MLA_RUNNERS: dict[
    str, Callable[[_FunctionalMLARequest], _FunctionalMLARunner]
] = {
    "fa2": Fa2MlaRunner,
    "fa3": Fa3MlaRunner,
    "cutlass": CutlassMlaRunner,
    "trtllm-gen": TrtllmGenMlaDecodeRunner,
    "cute-dsl-monolithic": CuteDslMonolithicMlaDecodeRunner,
    "cute-dsl-modular": CuteDslModularMlaDecodeRunner,
    "xqa": XqaMlaDecodeRunner,
}


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
    sparse_top_k_width: int = 0,
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

    def init_sparse_top_k_lens(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.fill_(sparse_top_k_width)
        return tensor

    has_sparse_lens = sparse_top_k_width > 0
    input_idx = (0, 1, 2, 3, 4) if has_sparse_lens else (0, 1, 2, 3)
    initializers = (
        (None, init_block_tables, init_seq_lens, None, init_sparse_top_k_lens)
        if has_sparse_lens
        else (None, init_block_tables, init_seq_lens, None)
    )

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=input_idx,
                dim_idx=(0,) * len(input_idx),
                gen_tuning_buckets=buckets,
                map_to_tuning_buckets=make_bucket_mapper(buckets, round_map=False),
                tensor_initializers=initializers,
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
    sparse_top_k_width: int = 0,
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
    return _mla_decode_tuning_config(
        buckets, kv_cache.shape[0], profile_seq_len, sparse_top_k_width
    )


def _run_functional_mla(
    request: _FunctionalMLARequest,
    backend: str,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run one validated functional MLA request through concrete runner classes."""
    if backend not in ("auto", "cute-dsl", *_FUNCTIONAL_MLA_RUNNERS):
        raise ValueError(f"Backend {backend} not supported by functional MLA")

    def run_explicit(runner: _FunctionalMLARunner):
        runner = prepare_candidate(runner)
        return runner(inputs=runner.inputs, tactic=-1)

    def prepare_candidate(
        runner: _FunctionalMLARunner,
    ) -> _FunctionalMLARunner:
        runner.prepare_for_dispatch()
        return runner

    def run_cute_direct(
        direct_request: _FunctionalMLARequest,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run CuTe features whose dynamic metadata is owned by its public adapter."""
        if direct_request.seq_lens is None:
            raise ValueError("seq_lens is required for cute-dsl MLA")
        if direct_request.multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
            )
        from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

        return cute_dsl_mla_decode(
            query=direct_request.query,
            kv_cache=direct_request.kv_cache,
            workspace_buffer=direct_request.workspace_buffer,
            kv_lora_rank=direct_request.kv_lora_rank,
            qk_rope_head_dim=direct_request.qk_rope_head_dim,
            block_tables=direct_request.block_tables,
            seq_lens=direct_request.seq_lens,
            max_seq_len=direct_request.max_seq_len,
            softmax_scale=direct_request.bmm1_scale,
            output_scale=direct_request.bmm2_scale,
            out=direct_request.out,
            out_dtype=torch.bfloat16,
            is_var_seq=direct_request.is_var_seq,
            enable_pdl=direct_request.enable_pdl,
            lse=direct_request.lse,
            return_lse=direct_request.return_lse,
            sinks=direct_request.sinks,
            cute_dsl_impl=direct_request.cute_dsl_impl,
            cum_seq_lens_q=direct_request.cum_seq_lens_q,
            max_q_len=direct_request.max_q_len,
            enable_dcp=direct_request.enable_dcp,
            cp_world=direct_request.cp_world,
            cp_rank=direct_request.cp_rank,
            causal_seqlens_kv_global=direct_request.causal_seqlens_kv_global,
        )

    def make_cute_runner(
        cute_request: _FunctionalMLARequest, *, for_auto: bool
    ) -> _FunctionalMLARunner:
        implementation = cute_request.cute_dsl_impl
        if implementation not in ("auto", "monolithic", "modular"):
            raise ValueError(
                "cute_dsl_impl must be 'auto', 'monolithic', or 'modular', "
                f"got {implementation!r}"
            )

        candidate_request = (
            replace(cute_request, cute_dsl_impl="auto") if for_auto else cute_request
        )
        if implementation == "monolithic":
            runner = _FUNCTIONAL_MLA_RUNNERS["cute-dsl-monolithic"](candidate_request)
            return prepare_candidate(runner) if for_auto else runner
        if implementation == "modular":
            runner = _FUNCTIONAL_MLA_RUNNERS["cute-dsl-modular"](candidate_request)
            return prepare_candidate(runner) if for_auto else runner
        modular_request = (
            candidate_request
            if for_auto
            else replace(cute_request, cute_dsl_impl="modular")
        )
        if cute_request.sinks is not None:
            runner = _FUNCTIONAL_MLA_RUNNERS["cute-dsl-modular"](modular_request)
            return prepare_candidate(runner) if for_auto else runner
        try:
            return prepare_candidate(
                _FUNCTIONAL_MLA_RUNNERS["cute-dsl-monolithic"](candidate_request)
            )
        except _FunctionalBackendUnsupportedError:
            return prepare_candidate(
                _FUNCTIONAL_MLA_RUNNERS["cute-dsl-modular"](modular_request)
            )

    if request.enable_dcp:
        return run_cute_direct(replace(request, cute_dsl_impl="monolithic"))

    if request.cum_seq_lens_q is not None:
        if backend in ("cute-dsl", "cute-dsl-monolithic", "cute-dsl-modular"):
            return run_cute_direct(request)
        if backend == "trtllm-gen":
            return run_explicit(_FUNCTIONAL_MLA_RUNNERS["trtllm-gen"](request))
        if backend == "auto":
            num_heads = request.query.size(-2)
            trt_gap = 64 < num_heads < 128
            needs_cute = trt_gap or request.return_lse or request.lse is not None
            if not needs_cute:
                return run_explicit(_FUNCTIONAL_MLA_RUNNERS["trtllm-gen"](request))
            if request.sparse_mla_top_k > 0:
                reason = "head-count gap" if trt_gap else "LSE with cum_seq_lens_q"
                raise ValueError(
                    "auto: no backend supports this variable-Q sparse "
                    f"configuration ({reason})"
                )
            return run_cute_direct(request)

    if backend == "cute-dsl":
        return run_explicit(make_cute_runner(request, for_auto=False))
    if backend != "auto":
        return run_explicit(_FUNCTIONAL_MLA_RUNNERS[backend](request))

    runners: List[_FunctionalMLARunner] = []
    runner_names: List[str] = []
    trtllm_reason = _trtllm_gen_mla_incompatibility_reason(request.kv_cache)
    if 64 < request.query.size(-2) < 128:
        trtllm_reason = "trtllm-gen MLA decode does not support 64 < num_heads_q < 128"
    if trtllm_reason is None:
        try:
            trtllm_runner = prepare_candidate(
                _FUNCTIONAL_MLA_RUNNERS["trtllm-gen"](request)
            )
        except _FunctionalBackendUnsupportedError as error:
            trtllm_reason = str(error)
        else:
            runners.append(trtllm_runner)
            runner_names.append("trtllm-gen")

    cute_reason = None
    trtllm_only = (
        request.sparse_mla_top_k > 0 or request.multi_ctas_kv_counter_buffer is not None
    )
    if not trtllm_only:
        cute_request = request
        if runners:
            cute_request = replace(request, out=runners[0].inputs[3])
        try:
            cute_runner = make_cute_runner(cute_request, for_auto=True)
        except _FunctionalBackendUnsupportedError as error:
            cute_reason = str(error)
        if cute_reason is None:
            runners.append(cute_runner)
            runner_names.append("cute-dsl")

    if not runners:
        if trtllm_only:
            raise ValueError(
                f"auto: trtllm-gen does not support this configuration: {trtllm_reason}"
            )
        raise ValueError(
            "auto: no backend supports this configuration "
            f"(trtllm-gen: {trtllm_reason}; cute-dsl: {cute_reason})"
        )

    tuning_kv_cache = request.kv_cache
    _, q_len, num_heads, _ = request.query.shape
    tuning_config = _build_mla_decode_tuning_config(
        kv_cache=tuning_kv_cache,
        block_tables=request.block_tables,
        workspace_buffer=request.workspace_buffer,
        runner_names=runner_names,
        q_len=q_len,
        num_heads=num_heads,
        kv_lora_rank=request.kv_lora_rank,
        max_seq_len=request.max_seq_len,
        device=request.query.device,
        sparse_top_k_width=(
            request.block_tables.shape[-1]
            if request.sparse_mla_top_k_lens is not None
            else 0
        ),
    )
    inputs = [
        request.query,
        request.block_tables,
        request.seq_lens,
        runners[0].inputs[3],
    ]
    if request.sparse_mla_top_k_lens is not None:
        inputs.append(request.sparse_mla_top_k_lens)
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm_batch_decode_mla",
        runners,
        tuning_config,
        inputs,
    )
    return runner(
        inputs=inputs,
        tactic=tactic,
        multi_ctas_kv_counter_buffer=request.multi_ctas_kv_counter_buffer,
    )


@flashinfer_api(trace=xqa_batch_decode_mla_trace)
def xqa_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
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
    _warn_xqa_batch_decode_with_kv_cache_mla_once()
    request = _FunctionalMLARequest(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=0,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=None,
        enable_pdl=enable_pdl,
        is_var_seq=True,
        uses_shared_paged_kv_idx=True,
        lse=None,
        return_lse=False,
        cute_dsl_impl="auto",
        kv_scale_format="auto",
        cum_seq_lens_q=None,
        max_q_len=None,
        multi_ctas_kv_counter_buffer=None,
        sparse_mla_top_k_lens=None,
        enable_dcp=False,
        cp_world=1,
        cp_rank=0,
        causal_seqlens_kv_global=None,
    )
    return _run_functional_mla(request, "xqa")


_xqa_batch_decode_with_kv_cache_mla_fi_trace = (
    xqa_batch_decode_with_kv_cache_mla.fi_trace
)


@functools.wraps(_xqa_batch_decode_with_kv_cache_mla_fi_trace)
def _warn_once_xqa_batch_decode_with_kv_cache_mla_fi_trace(*args, **kwargs):
    _warn_xqa_batch_decode_with_kv_cache_mla_once()
    return _xqa_batch_decode_with_kv_cache_mla_fi_trace(*args, **kwargs)


xqa_batch_decode_with_kv_cache_mla.fi_trace = (  # type: ignore[attr-defined]
    _warn_once_xqa_batch_decode_with_kv_cache_mla_fi_trace
)
