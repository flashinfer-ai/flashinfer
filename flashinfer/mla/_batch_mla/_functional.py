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
from typing import cast, List, Optional, Sequence, Tuple, Union
import warnings

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TuningConfig,
    make_bucket_mapper,
)
from flashinfer.trace.templates.attention import (
    trtllm_batch_decode_mla_trace_dispatch,
    xqa_batch_decode_mla_trace,
)
from flashinfer.utils import check_shape_dtype_device, get_compute_capability

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
    MLAChosenRepresentation,
    _adjacent_last_dim_view,
    _choose_mla_references,
    _classify_mla_references,
    _split_mla_tensor_references,
)


_xqa_batch_decode_with_kv_cache_mla_warning_emitted = False
_xqa_batch_decode_with_kv_cache_mla_warning_lock = threading.Lock()
_trtllm_batch_decode_with_kv_cache_mla_warning_emitted = False
_trtllm_batch_decode_with_kv_cache_mla_warning_lock = threading.Lock()
_functional_split_materialization_warning_emitted = False
_functional_split_materialization_warning_lock = threading.Lock()


def _warn_functional_split_materialization_once() -> None:
    global _functional_split_materialization_warning_emitted
    if _functional_split_materialization_warning_emitted:
        return
    with _functional_split_materialization_warning_lock:
        if _functional_split_materialization_warning_emitted:
            return
        warnings.warn(
            "Independent split MLA inputs require materialization for a "
            "packed-native backend. Pass packed, adjacent-split, or trusted "
            "redundant inputs to avoid a per-call allocation.",
            UserWarning,
            stacklevel=4,
        )
        _functional_split_materialization_warning_emitted = True


def _warn_trtllm_batch_decode_with_kv_cache_mla_once() -> None:
    global _trtllm_batch_decode_with_kv_cache_mla_warning_emitted
    if _trtllm_batch_decode_with_kv_cache_mla_warning_emitted:
        return
    with _trtllm_batch_decode_with_kv_cache_mla_warning_lock:
        if _trtllm_batch_decode_with_kv_cache_mla_warning_emitted:
            return
        warnings.warn(
            "trtllm_batch_decode_with_kv_cache_mla is deprecated; "
            "use batch_mla_paged_attention instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        _trtllm_batch_decode_with_kv_cache_mla_warning_emitted = True


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


_FUNCTIONAL_MLA_RUNNERS: dict[str, type[_FunctionalMLARunner]] = {
    "fa2": cast(type[_FunctionalMLARunner], Fa2MlaRunner),
    "fa3": cast(type[_FunctionalMLARunner], Fa3MlaRunner),
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

    Reusing the config and its initializer closures avoids rebuilding the
    batch-sweep description on the decode hot path.
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
    initializers = [(1, init_block_tables), (2, init_seq_lens)]
    if has_sparse_lens:
        initializers.append((4, init_sparse_top_k_lens))

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=input_idx,
                dim_idx=(0,) * len(input_idx),
                gen_tuning_buckets=buckets,
                map_to_tuning_buckets=make_bucket_mapper(buckets, round_map=False),
            ),
        ),
        tensor_initializers=tuple(initializers),
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


def _materialize_functional_packed_reference(
    *,
    packed: Optional[torch.Tensor],
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
    representation: MLAChosenRepresentation,
    name: str,
) -> torch.Tensor:
    """Return a packed functional reference, materializing only when required."""
    if representation == "packed":
        assert packed is not None
        return packed
    assert left is not None and right is not None
    adjacent = _adjacent_last_dim_view(left, right)
    if adjacent is not None:
        return adjacent
    _warn_functional_split_materialization_once()
    return torch.cat((left, right), dim=-1)


def _select_functional_request(
    request: _FunctionalMLARequest,
    runner_factory: type[_FunctionalMLARunner],
) -> _FunctionalMLARequest:
    """Reduce raw references to one complete native form for one runner."""
    query, q_nope, q_pe, query_representation = _choose_mla_references(
        packed=request.query,
        split_1=request.q_nope,
        split_2=request.q_pe,
        availability=request.query_availability,
        preferred=runner_factory.native_query_representation,
    )
    kv_cache, ckv_cache, kpe_cache, kv_representation = _choose_mla_references(
        packed=request.kv_cache,
        split_1=request.ckv_cache,
        split_2=request.kpe_cache,
        availability=request.kv_availability,
        preferred=runner_factory.native_kv_representation,
    )
    if query_representation != runner_factory.native_query_representation:
        if runner_factory.native_query_representation == "packed":
            query = _materialize_functional_packed_reference(
                packed=query,
                left=q_nope,
                right=q_pe,
                representation=query_representation,
                name="query",
            )
            q_nope = q_pe = None
        else:
            q_nope, q_pe = _split_mla_tensor_references(
                packed=query,
                left=q_nope,
                right=q_pe,
                representation=query_representation,
                widths=(request.kv_lora_rank, request.qk_rope_head_dim),
                name="query",
            )
            query = None
        query_representation = runner_factory.native_query_representation
    if kv_representation != runner_factory.native_kv_representation:
        if runner_factory.native_kv_representation == "packed":
            kv_cache = _materialize_functional_packed_reference(
                packed=kv_cache,
                left=ckv_cache,
                right=kpe_cache,
                representation=kv_representation,
                name="KV cache",
            )
            ckv_cache = kpe_cache = None
        else:
            ckv_cache, kpe_cache = _split_mla_tensor_references(
                packed=kv_cache,
                left=ckv_cache,
                right=kpe_cache,
                representation=kv_representation,
                widths=(request.kv_lora_rank, request.qk_rope_head_dim),
                name="KV cache",
            )
            kv_cache = None
        kv_representation = runner_factory.native_kv_representation
    return replace(
        request,
        query=query,
        q_nope=q_nope,
        q_pe=q_pe,
        kv_cache=kv_cache,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        query_availability=query_representation,
        kv_availability=kv_representation,
    )


def _make_functional_runner(
    runner_factory: type[_FunctionalMLARunner],
    request: _FunctionalMLARequest,
) -> _FunctionalMLARunner:
    selected = _select_functional_request(request, runner_factory)
    return runner_factory(selected)


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
        runner_name = (
            "cute-dsl-modular"
            if direct_request.cute_dsl_impl == "modular"
            else "cute-dsl-monolithic"
        )
        direct_request = _select_functional_request(
            direct_request, _FUNCTIONAL_MLA_RUNNERS[runner_name]
        )
        assert direct_request.query is not None and direct_request.kv_cache is not None
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
            runner_factory = _FUNCTIONAL_MLA_RUNNERS["cute-dsl-monolithic"]
            runner = _make_functional_runner(runner_factory, candidate_request)
            return prepare_candidate(runner) if for_auto else runner
        if implementation == "modular":
            runner_factory = _FUNCTIONAL_MLA_RUNNERS["cute-dsl-modular"]
            runner = _make_functional_runner(runner_factory, candidate_request)
            return prepare_candidate(runner) if for_auto else runner
        modular_request = (
            candidate_request
            if for_auto
            else replace(cute_request, cute_dsl_impl="modular")
        )
        if cute_request.sinks is not None:
            runner_factory = _FUNCTIONAL_MLA_RUNNERS["cute-dsl-modular"]
            runner = _make_functional_runner(runner_factory, modular_request)
            return prepare_candidate(runner) if for_auto else runner
        try:
            return prepare_candidate(
                _make_functional_runner(
                    _FUNCTIONAL_MLA_RUNNERS["cute-dsl-monolithic"], candidate_request
                )
            )
        except _FunctionalBackendUnsupportedError:
            return prepare_candidate(
                _make_functional_runner(
                    _FUNCTIONAL_MLA_RUNNERS["cute-dsl-modular"], modular_request
                )
            )

    if request.enable_dcp:
        return run_cute_direct(replace(request, cute_dsl_impl="monolithic"))

    if request.cum_seq_lens_q is not None:
        if backend in ("cute-dsl", "cute-dsl-monolithic", "cute-dsl-modular"):
            return run_cute_direct(request)
        if backend == "trtllm-gen":
            return run_explicit(
                _make_functional_runner(_FUNCTIONAL_MLA_RUNNERS["trtllm-gen"], request)
            )
        if backend == "auto":
            trt_request = _select_functional_request(
                request, _FUNCTIONAL_MLA_RUNNERS["trtllm-gen"]
            )
            assert trt_request.query is not None
            num_heads = trt_request.query.size(-2)
            trt_gap = 64 < num_heads < 128
            needs_cute = trt_gap or request.return_lse or request.lse is not None
            if not needs_cute:
                return run_explicit(
                    _make_functional_runner(
                        _FUNCTIONAL_MLA_RUNNERS["trtllm-gen"], request
                    )
                )
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
        return run_explicit(
            _make_functional_runner(_FUNCTIONAL_MLA_RUNNERS[backend], request)
        )

    runners: List[_FunctionalMLARunner] = []
    runner_names: List[str] = []
    trt_request = _select_functional_request(
        request, _FUNCTIONAL_MLA_RUNNERS["trtllm-gen"]
    )
    assert trt_request.query is not None and trt_request.kv_cache is not None
    trtllm_reason = _trtllm_gen_mla_incompatibility_reason(trt_request.kv_cache)
    if 64 < trt_request.query.size(-2) < 128:
        trtllm_reason = "trtllm-gen MLA decode does not support 64 < num_heads_q < 128"
    if trtllm_reason is None:
        try:
            trtllm_runner = prepare_candidate(
                _make_functional_runner(_FUNCTIONAL_MLA_RUNNERS["trtllm-gen"], request)
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

    tuning_request = runners[0].request
    assert tuning_request.query is not None and tuning_request.kv_cache is not None
    tuning_kv_cache = tuning_request.kv_cache
    _, q_len, num_heads, _ = tuning_request.query.shape
    tuning_config = _build_mla_decode_tuning_config(
        kv_cache=tuning_kv_cache,
        block_tables=request.block_tables,
        workspace_buffer=request.workspace_buffer,
        runner_names=runner_names,
        q_len=q_len,
        num_heads=num_heads,
        kv_lora_rank=request.kv_lora_rank,
        max_seq_len=request.max_seq_len,
        device=tuning_request.query.device,
        sparse_top_k_width=(
            request.block_tables.shape[-1]
            if request.sparse_mla_top_k_lens is not None
            else 0
        ),
    )
    inputs = [
        tuning_request.query,
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


def _batch_mla_paged_attention_impl(
    *,
    query: Optional[torch.Tensor] = None,
    q_nope: Optional[torch.Tensor] = None,
    q_pe: Optional[torch.Tensor] = None,
    kv_cache: Optional[torch.Tensor] = None,
    ckv_cache: Optional[torch.Tensor] = None,
    kpe_cache: Optional[torch.Tensor] = None,
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
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Validate public options and dispatch raw tensor references."""
    query_availability = _classify_mla_references(
        packed=query, split_1=q_nope, split_2=q_pe, name="query"
    )
    kv_availability = _classify_mla_references(
        packed=kv_cache, split_1=ckv_cache, split_2=kpe_cache, name="KV cache"
    )
    if backend not in (
        "auto",
        "xqa",
        "trtllm-gen",
        "cute-dsl",
        "cute-dsl-monolithic",
        "cute-dsl-modular",
        "sparse",
        "fa2",
        "fa3",
        "cutlass",
    ):
        raise ValueError(f"Backend {backend} not supported")

    preference_backend = backend
    if preference_backend == "auto":
        preference_backend = "trtllm-gen"
    elif preference_backend == "cute-dsl":
        preference_backend = (
            "cute-dsl-modular" if cute_dsl_impl == "modular" else "cute-dsl-monolithic"
        )
    elif preference_backend == "sparse":
        preference_backend = "xqa"
    runner_factory = _FUNCTIONAL_MLA_RUNNERS[preference_backend]
    metadata_query, metadata_q_nope, _, _ = _choose_mla_references(
        packed=query,
        split_1=q_nope,
        split_2=q_pe,
        availability=query_availability,
        preferred=runner_factory.native_query_representation,
    )
    query_reference = metadata_query if metadata_query is not None else metadata_q_nope
    assert query_reference is not None

    if isinstance(bmm1_scale, torch.Tensor) and bmm1_scale.dtype != torch.float32:
        raise TypeError("bmm1_scale tensor must have dtype torch.float32")
    if isinstance(bmm2_scale, torch.Tensor) and bmm2_scale.dtype != torch.float32:
        raise TypeError("bmm2_scale tensor must have dtype torch.float32")
    if max_q_len is not None and cum_seq_lens_q is None:
        raise ValueError("max_q_len is only supported when cum_seq_lens_q is provided")
    is_nope_mla = kv_lora_rank == 512 and qk_rope_head_dim == 0
    if is_nope_mla and sparse_mla_top_k <= 0:
        raise ValueError(
            "Native qk_rope_head_dim=0 TRTLLM-GEN MLA requires sparse_mla_top_k > 0"
        )
    if is_nope_mla and sparse_mla_top_k_lens is None:
        raise ValueError(
            "Native qk_rope_head_dim=0 TRTLLM-GEN MLA requires sparse_mla_top_k_lens"
        )
    if sparse_mla_top_k_lens is not None:
        if not is_nope_mla:
            raise ValueError(
                "sparse_mla_top_k_lens is currently only supported by the "
                "native qk_rope_head_dim=0 TRTLLM-GEN MLA path"
            )
        expected_num_query_tokens = (
            query_reference.size(0) * query_reference.size(1)
            if query_reference.ndim == 4
            else query_reference.size(0)
        )
        check_shape_dtype_device(
            sparse_mla_top_k_lens,
            (expected_num_query_tokens,),
            torch.int32,
            query_reference.device,
            "sparse_mla_top_k_lens",
        )
        sparse_mla_top_k_lens = sparse_mla_top_k_lens.contiguous()

    if not isinstance(enable_dcp, bool):
        raise TypeError(f"enable_dcp must be a bool, got {type(enable_dcp).__name__}")
    if not isinstance(cp_world, int) or isinstance(cp_world, bool) or cp_world <= 0:
        raise ValueError(f"cp_world must be a positive integer, got {cp_world!r}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool):
        raise TypeError(f"cp_rank must be an integer, got {type(cp_rank).__name__}")
    if not enable_dcp:
        nondefault_dcp_args = []
        if cp_world != 1:
            nondefault_dcp_args.append(f"cp_world={cp_world}")
        if cp_rank != 0:
            nondefault_dcp_args.append(f"cp_rank={cp_rank}")
        if causal_seqlens_kv_global is not None:
            nondefault_dcp_args.append("causal_seqlens_kv_global")
        if nondefault_dcp_args:
            raise ValueError(
                "DCP arguments require enable_dcp=True; got "
                + ", ".join(nondefault_dcp_args)
            )
    else:
        if query_reference.ndim != 4:
            raise ValueError(
                "DCP requires a dense query with shape "
                "[batch_size, q_len_per_request, num_heads, head_dim_qk]"
            )
        if not 0 <= cp_rank < cp_world:
            raise ValueError(
                "cp_rank must satisfy 0 <= cp_rank < cp_world, got "
                f"cp_rank={cp_rank}, cp_world={cp_world}"
            )
        if backend not in ("auto", "cute-dsl", "cute-dsl-monolithic"):
            raise ValueError(
                "enable_dcp=True is only supported by backend='cute-dsl', got "
                f"backend={backend!r}"
            )
        if not return_lse:
            raise ValueError(
                "enable_dcp=True requires return_lse=True so rank-local "
                "attention states can be merged"
            )
        if sinks is not None:
            raise ValueError("DCP cannot be combined with sinks")
        if cum_seq_lens_q is not None or max_q_len is not None:
            raise ValueError("DCP does not support cum_seq_lens_q / max_q_len")
        if not isinstance(causal_seqlens_kv_global, torch.Tensor):
            raise TypeError(
                "causal_seqlens_kv_global must be a torch.Tensor, got "
                f"{type(causal_seqlens_kv_global).__name__}"
            )
        check_shape_dtype_device(
            causal_seqlens_kv_global,
            (query_reference.shape[0],),
            torch.int32,
            query_reference.device,
            "causal_seqlens_kv_global",
        )
        if not causal_seqlens_kv_global.is_contiguous():
            raise ValueError("causal_seqlens_kv_global must be contiguous")
        backend = "cute-dsl"

    if backend in ("cute-dsl-monolithic", "cute-dsl-modular"):
        cute_dsl_impl = backend.removeprefix("cute-dsl-")
    if backend == "auto":
        cc = get_compute_capability(query_reference.device)
        if cc[0] == 12 and sparse_mla_top_k > 0:
            backend = "sparse"
        elif cc[0] != 10:
            backend = "xqa"

    request = _FunctionalMLARequest(
        query=query,
        q_nope=q_nope,
        q_pe=q_pe,
        kv_cache=kv_cache,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        query_availability=query_availability,
        kv_availability=kv_availability,
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
        is_var_seq=is_var_seq,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        lse=lse,
        return_lse=return_lse,
        cute_dsl_impl=cute_dsl_impl,
        kv_scale_format=kv_scale_format,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
    )
    if backend == "sparse":
        packed_request = _select_functional_request(
            request, _FUNCTIONAL_MLA_RUNNERS["xqa"]
        )
        assert packed_request.query is not None and packed_request.kv_cache is not None
        from flashinfer.mla._core import _run_mla_decode_sparse

        return _run_mla_decode_sparse(
            query=packed_request.query,
            kv_cache=packed_request.kv_cache,
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
    return _run_functional_mla(request, backend)


@flashinfer_api(trace=trtllm_batch_decode_mla_trace_dispatch)
def batch_mla_paged_attention(
    *,
    query: Optional[torch.Tensor] = None,
    q_nope: Optional[torch.Tensor] = None,
    q_pe: Optional[torch.Tensor] = None,
    kv_cache: Optional[torch.Tensor] = None,
    ckv_cache: Optional[torch.Tensor] = None,
    kpe_cache: Optional[torch.Tensor] = None,
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
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Decode MLA with tensor-first packed, split, or trusted redundant inputs.

    This one-shot API accepts only raw tensor references. Query tensors may be
    provided as packed ``query`` or complete split ``q_nope`` / ``q_pe``
    tensors. KV cache tensors may be provided as packed ``kv_cache`` or
    complete split ``ckv_cache`` / ``kpe_cache`` tensors. Supplying both packed
    and complete split tensors for either group is a trusted redundant form:
    only the backend-native form is selected, and the redundant references are
    not inspected or validated. Partial split pairs are rejected.

    Packed query and KV tensors concatenate the no-position and RoPE parts on
    the last dimension. Split pairs must have widths ``kv_lora_rank`` and
    ``qk_rope_head_dim`` respectively. Backends whose native form differs use a
    view when the split tensors are adjacent. In this one-shot functional API,
    independent splits selected by a packed-native backend are materialized by
    concatenation; for KV input this is a per-call KV-cache-sized copy. Planned
    wrapper execution remains zero-copy, so a packed plan rejects independent
    non-adjacent splits.

    With ``backend="auto"``, SM100/SM103 devices use TRTLLM-GEN for sparse MLA
    when ``sparse_mla_top_k > 0``. SM120/SM121 devices use the packed sparse
    backend for ``sparse_mla_top_k > 0`` and XQA for dense decode. Dense
    backend selection uses the backend-native tensor representation for each
    candidate.

    Parameters
    ----------
    query : Optional[torch.Tensor]
        Packed query tensor with shape
        ``[batch_size, q_len_per_request, num_heads, head_dim_qk]`` or, when
        ``cum_seq_lens_q`` is provided, ``[total_q, num_heads, head_dim_qk]``.
        ``head_dim_qk = kv_lora_rank + qk_rope_head_dim``. For the SM120/SM121
        v32/GLM sparse backend, this must be BF16 with ``head_dim_qk == 576``.
    q_nope : Optional[torch.Tensor]
        Split query tensor containing the non-RoPE channels. Its last dimension
        must be ``kv_lora_rank``.
    q_pe : Optional[torch.Tensor]
        Split query tensor containing the RoPE channels. Its last dimension
        must be ``qk_rope_head_dim``.
    kv_cache : Optional[torch.Tensor]
        Packed paged KV cache. For dense backends, the accepted shapes are
        ``[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]`` and
        ``[num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim]``. The
        tensor uses the query-compatible dense dtype. For the SM120/SM121
        v32/GLM sparse backend, this is a packed uint8 cache with 656 bytes per
        token, shaped ``[num_pages, page_size, 656]`` or
        ``[num_pages, 1, page_size, 656]``.
    ckv_cache : Optional[torch.Tensor]
        Split paged KV cache containing compressed latent KV channels. Its
        last dimension must be ``kv_lora_rank``.
    kpe_cache : Optional[torch.Tensor]
        Split paged KV cache containing RoPE channels. Its last dimension must
        be ``qk_rope_head_dim``.
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
        ``[total_q, sparse_mla_top_k]``. For SM120/SM121 sparse v32/GLM, it is
        the sparse index matrix and must have shape
        ``[batch_size, q_len_per_request, sparse_mla_top_k]`` with int32
        physical token indices.
    seq_lens : Optional[torch.Tensor]
        Per-request KV sequence lengths for dense and TRTLLM-GEN paths. For
        SM120/SM121 sparse v32/GLM, pass per-request ``[batch_size]``,
        per-token ``[batch_size, q_len_per_request]``, or flattened
        ``[batch_size * q_len_per_request]`` active top-k lengths; if ``None``,
        every column in ``block_tables`` is active.
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
        CuteDSL, XQA, and SM120/SM121 sparse v32/GLM require a float. CUTLASS
        requires its fixed ``1 / sqrt(128 + qk_rope_head_dim)`` scale.
    bmm2_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM2. TRTLLM-GEN accepts a FP32 tensor or float.
        CuteDSL and XQA require a float. SM120/SM121 sparse v32/GLM requires
        ``1.0``.
    sinks : Optional[List[torch.Tensor]]
        Additional value per head in the denominator of the softmax.
        Supported by ``trtllm-gen``, ``cute-dsl``, and ``sparse``. On
        ``cute-dsl`` this requires the modular implementation;
        ``cute_dsl_impl="auto"`` (the default) promotes to modular
        automatically, and ``cute_dsl_impl="monolithic"`` with sinks set raises
        :class:`ValueError`.
    skip_softmax_threshold_scale_factor : Optional[float]
        Threshold scale factor for skipping softmax operations. Providing a
        value enables skip-softmax sparsity. The actual threshold equals the
        provided scale factor divided by the context length.
    enable_pdl : Optional[bool]
        Programmatic Dependent Launch toggle. When ``None`` (default), support
        is auto-detected from the query device. Honored by the ``trtllm-gen``,
        ``cute-dsl``, and ``xqa`` functional backends.
    backend : str = "auto"
        Implementation backend. Valid values are ``"auto"``, ``"xqa"``,
        ``"trtllm-gen"``, ``"cute-dsl"``, ``"cute-dsl-monolithic"``,
        ``"cute-dsl-modular"``, ``"sparse"``, ``"fa2"``, ``"fa3"``, and
        ``"cutlass"``. ``"auto"`` chooses ``"trtllm-gen"`` for SM100/SM103
        sparse MLA and chooses ``"sparse"`` for SM120/SM121 when
        ``sparse_mla_top_k > 0``; otherwise SM120/SM121 dense decode uses
        ``"xqa"``. ``"cute-dsl"`` preserves family-local selection. The
        concrete CuTe names require the selected implementation and do not fall
        back to its sibling. ``"fa2"`` and ``"fa3"`` are one-shot generated-FA
        paths. ``"cutlass"`` is limited to its SM100/SM110 single-query,
        128-head DeepSeek MLA launch envelope.
    is_var_seq : bool
        Whether the sequence length is variable.
    uses_shared_paged_kv_idx : bool = True
        Whether K and V page indices are shared as a unified index. ``False``
        uses TRT-LLM layout with a 3D page table
        ``[batch_size, 2, max_num_pages_per_seq]`` and is supported only by
        TRTLLM-GEN.
    lse : Optional[torch.Tensor] = None
        Optional pre-allocated Log-Sum-Exp buffer. Supported by
        ``trtllm-gen``, ``cute-dsl``, and ``sparse`` backends and required to
        have dtype ``torch.float32``. Accepted shapes are
        ``[batch_size * q_len_per_request, num_qo_heads]`` and
        ``[batch_size, q_len_per_request, num_qo_heads]``.
    return_lse : bool = False
        Whether to return LSE values. When true, returns ``(out, lse)``.
    cute_dsl_impl : str = "auto"
        Which CuTe DSL implementation to use for the ``cute-dsl`` family and
        its auto candidate. ``"auto"`` picks monolithic by default and promotes
        to modular for modular-only features such as sinks. ``"modular"`` and
        ``"monolithic"`` are strict.
    kv_scale_format : str = "auto"
        Scale semantics for the SM120/SM121 packed v32/GLM sparse backend.
        ``"auto"`` and ``"pow2_fp32"`` select DSv3.2 power-of-2 FP32 inline
        scales; ``"arbitrary_fp32"`` selects arbitrary FP32 inline scales.
    cum_seq_lens_q : Optional[torch.Tensor] = None
        Cumulative query sequence lengths for variable-length query support,
        shape ``[batch_size + 1]`` and dtype ``torch.int32``. TRTLLM-GEN owns
        the native variable-Q path; CuTe DSL handles configurations that require
        its compact variable-Q fallback, including LSE and unsupported TRTLLM-GEN
        head counts. When provided, packed ``query`` must have shape
        ``[total_q, num_heads, head_dim_qk]`` and split ``q_nope`` / ``q_pe``
        must have matching flattened leading dimensions.
    max_q_len : Optional[int] = None
        Maximum query sequence length represented by ``cum_seq_lens_q``.
        Providing it avoids host-side metadata validation.
    multi_ctas_kv_counter_buffer : Optional[torch.Tensor] = None
        Optional caller-owned counter buffer for TRTLLM-GEN. It must remain
        alive for every launch or CUDA graph replay that uses it and be
        zero-initialized once. Autotune profiling uses runner-owned storage;
        this buffer is used only for the selected TRTLLM-GEN final request.
    sparse_mla_top_k_lens : Optional[torch.Tensor] = None
        Flattened active sparse top-k lengths, one INT32 value per query token.
        Required by native ``kv_lora_rank=512, qk_rope_head_dim=0``
        TRTLLM-GEN MLA.
    enable_dcp : bool = False
        Enable cyclic decode context parallelism in monolithic CuTe DSL MLA.
        Requires ``return_lse=True`` so callers can merge rank-local states.
    cp_world : int = 1
        Context-parallel world size.
    cp_rank : int = 0
        Context-parallel rank for this launch.
    causal_seqlens_kv_global : Optional[torch.Tensor] = None
        Contiguous CUDA int32 tensor ``[batch_size]`` containing each request's
        global exclusive causal KV bound. Required when DCP is enabled.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        The MLA output tensor, or ``(out, lse)`` when ``return_lse=True``.

    Note
    ----
    In MLA, the actual BMM1 and BMM2 scales applied are fused as::

        bmm1_scale = q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5)
        bmm2_scale = v_scale * o_scale

    The two scale factors should be static constants for CUDA graph capture.
    On-device fused scale tensors may be used for dynamically changing FP8
    scale factors.

    Autotune
    --------
    On SM100/SM103 dense MLA, calling under ``flashinfer.autotune(True)`` with
    ``backend="auto"`` profiles both ``trtllm-gen`` and ``cute-dsl`` across a
    bucketed batch sweep and caches the winning runner per shape signature.
    Subsequent calls under ``autotune(False)`` dispatch to the cached choice.
    The autotune bucket range and cache key do not depend on
    ``kv_cache.shape[0]``; however, the page-aliasing ratio during profiling
    does depend on the pool size, so profile with a representative KV pool.

    Warning
    -------
    ``trtllm_batch_decode_with_kv_cache_mla`` remains as a deprecated alias for
    packed tensor callers. New code should call this function directly with the
    tensor form that matches its source buffers.
    """
    return _batch_mla_paged_attention_impl(
        query=query,
        q_nope=q_nope,
        q_pe=q_pe,
        kv_cache=kv_cache,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
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
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
    )


@flashinfer_api(trace=trtllm_batch_decode_mla_trace_dispatch)
def trtllm_batch_decode_with_kv_cache_mla(
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
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Deprecated compatibility alias for Batch MLA attention.

    See :func:`batch_mla_paged_attention` for the parameter and return-value
    contract.
    """
    _warn_trtllm_batch_decode_with_kv_cache_mla_once()
    return _batch_mla_paged_attention_impl(
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
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
    )


_trtllm_batch_decode_with_kv_cache_mla_fi_trace = (
    trtllm_batch_decode_with_kv_cache_mla.fi_trace
)


@functools.wraps(_trtllm_batch_decode_with_kv_cache_mla_fi_trace)
def _warn_once_trtllm_batch_decode_with_kv_cache_mla_fi_trace(*args, **kwargs):
    _warn_trtllm_batch_decode_with_kv_cache_mla_once()
    return _trtllm_batch_decode_with_kv_cache_mla_fi_trace(*args, **kwargs)


trtllm_batch_decode_with_kv_cache_mla.fi_trace = (  # type: ignore[attr-defined]
    _warn_once_trtllm_batch_decode_with_kv_cache_mla_fi_trace
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
        q_nope=None,
        q_pe=None,
        kv_cache=kv_cache,
        ckv_cache=None,
        kpe_cache=None,
        query_availability="packed",
        kv_availability="packed",
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
