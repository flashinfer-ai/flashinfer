"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

"""Automatic MLA selection through heuristics or current-workload autotuning.

The auto policy orders backends from request metadata and returns the first
supported plan. The autotune policy prepares eligible candidates and retains
an executable selected through profiling. Concrete backends own eligibility,
planning and execution. Autotune snapshots metadata at plan time; replan to
change lengths or page mappings.
"""

import hashlib
from dataclasses import fields, replace
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

import torch

from ...api_logging import experimental_auto_backends_allowed
from ...autotuner import AutoTuner, TuningConfig
from ...utils import (
    get_compute_capability as _get_compute_capability,
    is_sm90a_supported,
    is_sm120a_supported,
    is_sm121a_supported,
)
from ._backends._capabilities import _BackendPlanUnsupportedError, MLAPlanCapabilities
from ._backends._fa_common import _BatchMLAPagedAttentionFaBackendBase
from ._backends.cute_dsl_modular_backend import (
    _BatchMLAPagedAttentionCuteDslModularBackend,
)
from ._backends.cute_dsl_monolithic_backend import (
    _BatchMLAPagedAttentionCuteDslMonolithicBackend,
)
from ._backends.cutlass_backend import _BatchMLAPagedAttentionCutlassBackend
from ._backends.cutile_backend import _BatchMLAPagedAttentionCutileBackend
from ._backends.fa2_backend import _BatchMLAPagedAttentionFa2Backend
from ._backends.fa3_backend import _BatchMLAPagedAttentionFa3Backend
from ._backends.trtllm_gen_backend import _BatchMLAPagedAttentionTrtllmGenBackend
from ._backends.xqa_backend import _BatchMLAPagedAttentionXqaBackend
from ._contracts import MLAPlanMetadata, _are_adjacent_last_dim_views
from ._planning import _MLAPlanArguments

if TYPE_CHECKING:
    from ._wrapper import _ConcreteWrapperBackendType, _PlannedBackend


class _BatchMLAPagedAttentionAutoBackend:
    """Resolve the automatic policy to a prepared concrete backend."""

    @classmethod
    def plan_from_wrapper(cls, plan_args: _MLAPlanArguments) -> "_PlannedBackend":
        # The wrapper registers this class during import; resolve its shared
        # machinery only when planning starts, after initialization is complete.
        from ._wrapper import _BACKEND_TYPES

        plan_args.csr()  # Malformed metadata is never a candidate support refusal.

        # Retain the executable family used by the existing graph plan.
        if plan_args._use_cuda_graph and plan_args._previous_backend_name is not None:
            candidates: tuple[str, ...] = (plan_args._previous_backend_name,)
        else:
            device = plan_args._float_workspace_buffer.device
            if _get_compute_capability(device) == (8, 0):
                candidates = ordered_sm80_backends(plan_args)
            elif is_sm90a_supported(device):
                candidates = ordered_sm90_backends(plan_args)
            elif _get_compute_capability(device) in ((10, 0), (10, 3)):
                candidates = ordered_sm100_backends(plan_args)
            elif _get_compute_capability(device) == (10, 7):
                candidates = _ordered_sm107_backends(plan_args)
            elif is_sm120a_supported(device) or is_sm121a_supported(device):
                candidates = ordered_sm12x_backends(plan_args)
            else:
                candidates = _AUTO_BACKEND_CANDIDATES
        rejections: list[str] = []
        for candidate in candidates:
            try:
                backend_type = cast(
                    "type[_ConcreteWrapperBackendType]", _BACKEND_TYPES[candidate]
                )
                if (
                    backend_type._plan_capabilities.is_experimental
                    and not experimental_auto_backends_allowed()
                ):
                    raise _BackendPlanUnsupportedError(
                        "experimental backend requires "
                        "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1"
                    )
                return backend_type.plan_from_wrapper(plan_args)
            except _BackendPlanUnsupportedError as error:
                if len(candidates) == 1:
                    raise
                rejections.append(f"{candidate}: {error}")
        raise _BackendPlanUnsupportedError(
            "No supported planned MLA backend: " + "; ".join(rejections)
        )


# Default fallback groups: FA, TRT/CuTe, narrow adapters, then experimental.
# Architecture policies move their measured preferences first.
_AUTO_BACKEND_CANDIDATES = (
    "fa2",
    "fa3",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "cutlass",
    "xqa",
    "cutile",
)


def _prefer(*backends):
    return backends + tuple(b for b in _AUTO_BACKEND_CANDIDATES if b not in backends)


def ordered_sm80_backends(args: _MLAPlanArguments) -> tuple[str, ...]:
    """Prefer FA2 on SM80."""
    return _prefer("fa2")


def ordered_sm90_backends(args: _MLAPlanArguments) -> tuple[str, ...]:
    """Prefer FA3 on supported SM90 toolchains, with FA2 as fallback."""
    return _prefer("fa3", "fa2")


# SM100 warmed planned-wrapper measurements favor monolithic CuTe at larger
# work sizes. Keep FA2 before cuTile for native independent-split/page-1 fallback.
def ordered_sm100_backends(args):
    """Order every candidate from actual lengths, without deciding support.

    These coarse work boundaries approximate measured launch/throughput
    crossovers, not exact optima. Metadata transfers are plan-time setup only;
    re-read on each plan because callers may mutate the metadata in place.
    """
    csr = args.csr()
    offsets = csr.qo_indptr.cpu().tolist()
    kv_lens = csr.kv_len_arr.cpu().tolist()
    q_lens = [end - begin for begin, end in zip(offsets, offsets[1:], strict=False)]
    # Legacy flat metadata can have extra query offsets. Leave its interpretation
    # to the backend instead of imposing a new validity restriction here.
    if not q_lens or len(q_lens) != len(kv_lens):
        return _prefer("cute-dsl-monolithic", "trtllm-gen")
    total_q, total_kv = sum(q_lens), sum(kv_lens)
    if not total_q or not total_kv:
        return _prefer("cute-dsl-monolithic", "trtllm-gen")
    heads = args.num_heads
    work = heads * sum(q * kv for q, kv in zip(q_lens, kv_lens, strict=True))
    output_work = heads * total_q
    max_q, min_kv, max_kv = max(q_lens), min(kv_lens), max(kv_lens)
    batch = len(q_lens)
    small_work = work <= 2**21 and output_work <= 32768
    if not args._use_cuda_graph:
        # Packed tensors and adjacent split views share the same plan facts.
        # Retain FA2 where conversion/launch costs erase the packed-input gain.
        if small_work and max_kv <= 4096:
            return _prefer("fa2", "trtllm-gen")
        within_eager_limit = (
            work < 5 * 2**20
            if heads >= 128 and (max_q > 1 or output_work <= 2048 or max_kv <= 2048)
            else work <= 2**22
        )
        # Larger aggregate KV volume amortizes conversion even for adjacent
        # views. Share this exemption across both secondary eager guards.
        if (
            within_eager_limit
            and output_work <= 32768
            and max_kv < 32768
            and total_kv < 5 * 2**14
        ):
            return _prefer("fa2", "trtllm-gen")
        if (
            max_q == 1
            and work < 5 * 2**20
            and max_kv <= 4096
            and output_work <= 2048
            and batch < 32
            and total_kv < 5 * 2**14
        ):
            return _prefer("fa2", "trtllm-gen")
    # Base-2 LSE needs a prefix-aware preference in both execution modes.
    # The per-request margin is empirical, not a native support restriction.
    query_factor = 4 if heads <= 8 else 2
    near_prefill = query_factor * total_q + 32 * batch >= total_kv
    if (
        args.lse_mode == "base2"
        and work <= 2**23
        and output_work <= 32768
        and max_q > 1
        and max_kv <= 1024
        and near_prefill
    ):
        return _prefer("fa2", "trtllm-gen")
    if args._use_cuda_graph:
        # Large short-context prefill amortizes TRT's launch cost. Preserve
        # CuTe for longer prefixes and lower head counts with different crossovers.
        if heads >= 64 and max_q >= 128 and output_work >= 65536 and max_kv <= 512:
            return _prefer("trtllm-gen", "cute-dsl-monolithic")
        # Batched short prefill amortizes TRT's launch cost at larger head
        # counts. Lower-head requests retain CuTe even at the same output work.
        if heads >= 64 and output_work >= 24576 and 16 <= max_q <= 32 and max_kv <= 32:
            return _prefer("trtllm-gen", "cute-dsl-monolithic")
        # TRT wins small short-context query tiles; CuTe wins as those tiles
        # grow. Preserve the older narrow tile rule for larger query counts.
        query_tile = max_q * heads
        if work <= 2**20 and max_kv <= 512 and (query_tile <= 128 or max_q <= 16):
            return (
                _prefer("trtllm-gen", "cute-dsl-monolithic")
                if query_tile <= 576
                else _prefer("cute-dsl-monolithic", "trtllm-gen")
            )
        # Larger short-context decode batches favor TRT. Count actual queries
        # so empty metadata requests cannot trigger the throughput preference.
        if max_q == 1 and heads >= 64 and total_q >= 96 and max_kv <= 512:
            return _prefer("trtllm-gen", "cute-dsl-monolithic")
        if heads >= 128:
            # Distinguish long KV from short-KV multi-query work with the same
            # aggregate size. Their measured TRT/CuTe crossovers differ.
            if min_kv >= 2048:
                graph_limit = 6 * 2**20 if max_kv <= 4096 else 2**22
                if 1 < max_q <= 8 and min_kv > 4096:
                    graph_limit = 2**23
                if work <= graph_limit:
                    return _prefer("cute-dsl-monolithic", "trtllm-gen")
            return _prefer("trtllm-gen", "cute-dsl-monolithic")
        if max_q == 1 and (
            min_kv >= 8192
            or (output_work > 384 and max_kv > 512)
            or (16 < heads < 32 and min_kv > 512 and max_kv <= 2048)
        ):
            return _prefer("cute-dsl-monolithic", "trtllm-gen")
        if work <= 2**20 and max_q == 1:
            return _prefer("trtllm-gen", "cute-dsl-monolithic")
        return _prefer("cute-dsl-monolithic", "trtllm-gen")
    if heads >= 128:
        order = _prefer("trtllm-gen", "cute-dsl-monolithic")
    elif small_work:
        # Long-context decode keeps CuTe before FA2 after typed TRT rejection.
        order = (
            _prefer("trtllm-gen", "cute-dsl-monolithic")
            if max_q == 1
            else _prefer("trtllm-gen", "fa2")
        )
    else:
        order = _prefer("cute-dsl-monolithic", "trtllm-gen")
    # Larger multi-query work or aggregate KV volume can favor modular CuTe
    # after earlier implementations reject. Keep the eager guards authoritative.
    # Small-head Q4 requests favor FA2 after earlier native candidates reject.
    # Keep modular's measured Q2/Q3 and larger-head advantages.
    prefer_fa2_q4 = heads <= 16 and max_q == 4 and min(q_lens) == 4 and not args.causal
    if max_q > 1 and not prefer_fa2_q4 and (work >= 5 * 2**20 or total_kv >= 5 * 2**14):
        remaining = tuple(backend for backend in order if backend != "cute-dsl-modular")
        position = remaining.index("fa2")
        return remaining[:position] + ("cute-dsl-modular",) + remaining[position:]
    return order


def _ordered_sm107_backends(args: _MLAPlanArguments) -> tuple[str, ...]:
    # The SM100 policy performed well in SM107 development measurements.
    # Further investigation may identify useful SM107-specific ordering rules.
    return ordered_sm100_backends(args)


def ordered_sm12x_backends(args: _MLAPlanArguments) -> tuple[str, ...]:
    """Prefer FA2 on SM120/SM121, then native XQA and opt-in cuTile."""
    return _prefer("fa2", "xqa", "cutile")


# Measured selection for the current planned workload.

_OP = "batch_mla_paged_attention"
_CONFIGS = {
    graph: TuningConfig(use_cuda_graph=graph, use_cold_l2_cache=True)
    for graph in (False, True)
}


def _tensor_signature(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        adjacent = (
            len(value) == 2
            and all(isinstance(t, torch.Tensor) for t in value)
            and _are_adjacent_last_dim_views(*value)
        )
        return ("split", adjacent, tuple(_tensor_signature(t) for t in value))
    return (
        tuple(value.shape),
        tuple(value.stride()),
        value.dtype,
        str(value.device),
        value.data_ptr() % 16,
    )


def _input_shapes(inputs):
    return tuple(
        tuple(t.shape) if isinstance(t, torch.Tensor) else (0,) for t in inputs
    )


def _effective_config(tuner, use_cuda_graph):
    # Match choose_one's measurement policy for the strict cache checks. There
    # are no dynamic dimensions: tuning_buckets never expands a wrapper plan.
    config = tuner._apply_tuning_overrides(_CONFIGS[use_cuda_graph])
    if tuner._effective_measure_policy is not None:
        config = tuner._apply_measure_policy(config, tuner._effective_measure_policy)
    if config.use_cuda_graph and not use_cuda_graph:
        # choose_one reapplies the policy: reject instead of silently clamping.
        raise ValueError(
            "MLA wrapper use_cuda_graph=False does not allow CUDA graph profiling; "
            "use an eager autotune measurement policy."
        )
    return config


class _TunablePlannedBackend(Protocol):
    _backend: str
    _plan_capabilities: ClassVar[MLAPlanCapabilities]

    @classmethod
    def plan_from_wrapper(cls, args: _MLAPlanArguments) -> "_TunablePlannedBackend": ...

    def configure_tuning(self, *, cache_key: tuple, run_options: dict) -> None: ...


class _BatchMLAPagedAttentionAutotuneBackend:
    """A planned candidate set with a lazily selected, retained executable."""

    _candidate_types: ClassVar[tuple[type[_TunablePlannedBackend], ...]] = (
        _BatchMLAPagedAttentionTrtllmGenBackend,
        _BatchMLAPagedAttentionCuteDslMonolithicBackend,
        _BatchMLAPagedAttentionCuteDslModularBackend,
        _BatchMLAPagedAttentionXqaBackend,
        _BatchMLAPagedAttentionCutlassBackend,
        _BatchMLAPagedAttentionFa2Backend,
        _BatchMLAPagedAttentionFa3Backend,
        _BatchMLAPagedAttentionCutileBackend,
    )
    _backend = "autotune"
    _use_cuda_graph: bool
    _candidates: list[_TunablePlannedBackend]
    _workload_key: tuple
    _plan_capabilities: MLAPlanCapabilities

    @classmethod
    def plan_from_wrapper(cls, args: _MLAPlanArguments):
        if (
            args._float_workspace_buffer.is_cuda
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "MLA autotune plan() must finish before CUDA graph capture."
            )
        # Validate all supplied representations before candidate fallback. Keep
        # native CSR metadata for FA instead of materializing a dense table.
        if args.metadata.qo_indptr is not None:
            args.csr()
        else:
            args.native_dense()
        snapshot = {}
        digest = hashlib.sha256()
        for field in fields(args.metadata):
            value = getattr(args.metadata, field.name)
            if isinstance(value, torch.Tensor):
                value = value.clone()
                digest.update(field.name.encode())
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(value.cpu().numpy().tobytes())
            snapshot[field.name] = value
        metadata = MLAPlanMetadata(**snapshot)
        plan_facts = tuple(
            (f.name, getattr(args, f.name))
            for f in fields(args)
            if not f.name.startswith("_") and f.name != "metadata"
        )
        candidates, rejections = [], []
        # Candidate preparation must treat the shared workspace as scratch and
        # the snapshot as read-only. Persistent state belongs to each candidate;
        # backends with workspace-resident plans need isolated storage here.
        for backend_type in cls._candidate_types:
            try:
                if (
                    backend_type._plan_capabilities.is_experimental
                    and not experimental_auto_backends_allowed()
                ):
                    raise _BackendPlanUnsupportedError(
                        "experimental backend requires "
                        "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1"
                    )
                candidate_args = replace(args, metadata=metadata)
                if issubclass(backend_type, _BatchMLAPagedAttentionFaBackendBase):
                    fa_type = cast(
                        type[_BatchMLAPagedAttentionFaBackendBase], backend_type
                    )
                    # Each FA plan writes persistent integer storage. Graph
                    # metadata must also be private: preparing another candidate
                    # or replanning must not overwrite a retained executable.
                    changes = {"_graph_plan_int_workspace_buffer": None}
                    if args._use_cuda_graph:
                        csr = candidate_args.csr()
                        device = args._float_workspace_buffer.device
                        names = ("qo_indptr", "kv_indptr", "kv_indices", "kv_len_arr")
                        fa_type._validate_graph_metadata_buffers(
                            device,
                            tuple(
                                (
                                    name,
                                    getattr(args, f"_{name}_buf"),
                                    getattr(csr, name),
                                    name == "kv_indices",
                                )
                                for name in names
                            ),
                        )
                        for name in names:
                            source = getattr(csr, name)
                            changes[f"_{name}_buf"] = torch.empty_like(
                                source, device=device
                            )
                        if args._graph_plan_int_workspace_buffer is not None:
                            changes["_graph_plan_int_workspace_buffer"] = (
                                torch.empty_like(args._graph_plan_int_workspace_buffer)
                            )
                    candidate_args = replace(candidate_args, **changes)
                candidate = backend_type.plan_from_wrapper(candidate_args)
                if isinstance(candidate, _BatchMLAPagedAttentionXqaBackend):
                    candidate.prepare_for_tuning()
                candidates.append(candidate)
            except _BackendPlanUnsupportedError as error:
                rejections.append(
                    f"{backend_type._plan_capabilities.backend_name}: {error}"
                )
        if not candidates:
            raise _BackendPlanUnsupportedError(
                "No supported autotune MLA backend: " + "; ".join(rejections)
            )
        result = cls()
        result._candidates = candidates
        result._use_cuda_graph = args._use_cuda_graph
        result._workload_key = (
            "planned-v2",
            args._use_cuda_graph,
            digest.hexdigest(),
            metadata.max_q_len,
            plan_facts,
            args._float_workspace_buffer.numel()
            * args._float_workspace_buffer.element_size(),
            str(args._float_workspace_buffer.device),
            tuple(c._backend for c in candidates),
        )
        # The policy is not a concrete layout decision. The wrapper preserves
        # its declared input layout; each candidate lowers it in run_from_wrapper.
        caps = [c._plan_capabilities for c in candidates]
        result._plan_capabilities = MLAPlanCapabilities(
            backend_name="autotune",
            lse_modes=frozenset.union(*(c.lse_modes for c in caps)),
            kv_layouts=frozenset.union(*(c.kv_layouts for c in caps)),
            output_scales=frozenset.union(*(c.output_scales for c in caps)),
            scale_modes=frozenset.union(*(c.scale_modes for c in caps)),
            supports_skip_softmax=any(c.supports_skip_softmax for c in caps),
            supports_skip_softmax_with_lse=any(
                c.supports_skip_softmax_with_lse for c in caps
            ),
            supports_enable_pdl=any(c.supports_enable_pdl for c in caps),
            supports_sinks=any(c.supports_sinks for c in caps),
        )
        result._selection = None
        result._capture_signature = None
        return result

    def run_from_wrapper(
        self,
        *,
        query,
        kv_cache,
        out,
        lse,
        return_lse,
        profiler_buffer,
        kv_len,
        page_table,
        return_lse_base_on_e,
        o_scale,
        ckv_scale,
        ckv_scale_arr,
        kpe_scale,
        sinks,
        skip_softmax_threshold_scale_factor,
        bmm1_scale,
        bmm2_scale,
    ):
        for name, value in (
            ("profiler_buffer", profiler_buffer),
            ("kv_len", kv_len),
            ("page_table", page_table),
            ("ckv_scale_arr", ckv_scale_arr),
        ):
            if value is not None:
                raise ValueError(
                    f"{name} is not supported with MLA backend='autotune'."
                )
        if out is None:
            raise ValueError("out must be provided for MLA backend='autotune'.")
        inputs = [query, kv_cache, out, lse, sinks]
        if isinstance(bmm1_scale, torch.Tensor):
            # Dynamic scales belong to this invocation, including cache hits and
            # graph capture. Retain their structure in the signature, not values
            # or tensor objects in a selected candidate's immutable options.
            inputs.extend((bmm1_scale, bmm2_scale))
            bmm1_scale = bmm2_scale = None
        options = dict(
            return_lse=return_lse,
            return_lse_base_on_e=return_lse_base_on_e,
            o_scale=o_scale,
            ckv_scale=ckv_scale,
            kpe_scale=kpe_scale,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
        capturing = out.is_cuda and torch.cuda.is_current_stream_capturing()
        if capturing and not self._use_cuda_graph:
            raise RuntimeError(
                "MLA wrapper use_cuda_graph=False cannot run inside CUDA graph capture."
            )
        # A graph retains launch options and tensor layouts. Ordinary eager
        # calls retain only the backend choice and use current invocation data.
        if capturing or self._capture_signature is not None:
            signature = (
                tuple(_tensor_signature(t) for t in inputs),
                tuple(options.items()),
            )
            if (
                self._capture_signature is not None
                and signature != self._capture_signature
            ):
                raise RuntimeError(
                    "MLA autotune execution is bound to a CUDA graph; use a new wrapper for different run options."
                )
        if self._selection is not None:
            runner, tactic = self._selection
            result = runner(inputs, tactic=tactic, run_options=options)
            if capturing:
                self._capture_signature = signature
            return result
        if capturing:
            raise RuntimeError(
                "MLA backend='autotune' requires a prepared selection before CUDA graph capture; warm up run() inside flashinfer.autotune(True)."
            )
        return self._select_and_run(inputs, options)

    def _select_and_run(self, inputs, options):
        # First-run cache identity only. A successful choice lasts until replan.
        signature = (
            tuple(_tensor_signature(t) for t in inputs),
            tuple(options.items()),
        )
        tuner = AutoTuner.get()
        config = _effective_config(tuner, self._use_cuda_graph)
        candidates = self._candidates
        for candidate in candidates:
            candidate.configure_tuning(
                cache_key=(self._workload_key, signature), run_options=options
            )
        # Some kernels require compact split views while FA supports more
        # general strides. Admission belongs to each candidate, before any
        # launch; unexpected failures from admitted kernels still propagate.
        candidates = [
            candidate
            for candidate in candidates
            if candidate.get_valid_tactics(inputs, None)
        ]
        if not candidates:
            raise _BackendPlanUnsupportedError(
                "No autotune MLA backend supports the current tensor layout."
            )
        shapes = _input_shapes(inputs)
        hit, index, tactic, _ = tuner.search_cache(
            _OP, candidates, shapes, config, inputs=inputs
        )
        if not hit:
            if not tuner.is_tuning_mode:
                raise RuntimeError(
                    "No cached MLA autotune result for this planned workload and run options. Warm up run() inside flashinfer.autotune(True)."
                )
            # Profile real Q/KV and plan metadata, with private output storage.
            # No synthetic batch shapes or caller-visible output writes.
            # Keep split Q/KV tuples intact: their leaves can be adjacent
            # views. Static profiling reuses inputs and flushes L2, so no
            # flatten/clone step is needed (or allowed to break aliases).
            profile_inputs = list(inputs)
            for index in (2, 3):
                if inputs[index] is not None:
                    tensor = inputs[index]
                    profile_inputs[index] = torch.empty_strided(
                        tensor.shape,
                        tensor.stride(),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
            # Surface validation/launch failures before the generic tuner can
            # classify a failed tactic and fall back to its default runner.
            for candidate in candidates:
                candidate(profile_inputs, tactic=-1)
            tuner.choose_one(_OP, candidates, config, profile_inputs)
            hit, index, tactic, _ = tuner.search_cache(
                _OP, candidates, shapes, config, inputs=inputs
            )
            if not hit:
                raise RuntimeError(
                    "MLA autotuning produced no measured selection; profiling may have been skipped or failed."
                )
        runner = candidates[index]
        result = runner(inputs, tactic=tactic)
        self._selection = runner, tactic
        return result
