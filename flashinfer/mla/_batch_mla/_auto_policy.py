"""Backend-selection policies for :class:`BatchMLAPagedAttentionWrapper`.

Deterministic ``backend='auto'`` ranking and context-enabled autotune
orchestration share one selection lifecycle here.
"""

from dataclasses import dataclass, replace
import functools
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, cast

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
    make_bucket_mapper,
)
from flashinfer.autotuner.initializers import (
    autotuner_initializer_empty,
    autotuner_initializer_zeros,
)

from ._backends._fa_common import _BatchMLAGeneratedFaWorkspace
from ._contracts import MLAKVCache, MLAPlanMetadata, MLAQuery
from ._planning import _MLAPlanArguments


SM80_PREFERRED_BACKENDS = (
    "fa2",
    "fa3",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
)

SM90_PREFERRED_BACKENDS = (
    "fa3",
    "fa2",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
)

SM100_PREFERRED_BACKENDS = (
    "trtllm-gen",
    "fa2",
    "fa3",
    "cutlass",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
    "xqa",
)

SM120_PREFERRED_BACKENDS = (
    "xqa",
    "fa2",
    "fa3",
    "cutlass",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
)


_ARCHITECTURE_PREFERRED_BACKENDS = {
    None: SM80_PREFERRED_BACKENDS,
    (8, 0): SM80_PREFERRED_BACKENDS,
    (8, 9): SM80_PREFERRED_BACKENDS,
    (9, 0): SM90_PREFERRED_BACKENDS,
    (10, 0): SM100_PREFERRED_BACKENDS,
    (10, 3): SM100_PREFERRED_BACKENDS,
    (12, 0): SM120_PREFERRED_BACKENDS,
    (12, 1): SM120_PREFERRED_BACKENDS,
}


@dataclass(frozen=True, slots=True)
class MLAAutoSelectionTrace:
    """Immutable result of one automatic MLA planning attempt."""

    candidates: tuple[str, ...]
    rejections: tuple[tuple[str, str], ...]
    mode: Literal["deterministic", "tuning", "cache-only", "bypass"]
    bypass_reason: str | None
    resolved_backend: str
    profile_rejections: tuple[tuple[int, str, str], ...] = ()


def rank_auto_backend_candidates(
    compute_capability: Optional[tuple[int, int]],
) -> tuple[str, ...]:
    """Return the complete architecture-preferred automatic planning order."""
    return _ARCHITECTURE_PREFERRED_BACKENDS.get(
        compute_capability, _ARCHITECTURE_PREFERRED_BACKENDS[None]
    )


@dataclass(frozen=True, slots=True)
class _MLAAutotuneProfile:
    """Tensor-identity-free facts that define one wrapper tuning workload."""

    batch_size: int
    q_len: int
    max_q_len: int
    max_kv_len: int
    min_kv_len_bucket: int
    max_kv_len_bucket: int
    page_size: int
    table_width: int
    page_reuse: bool
    num_heads: int
    head_dim_ckv: int
    head_dim_kpe: int
    query_dtype: torch.dtype
    kv_dtype: torch.dtype
    kv_layout: str
    lse_mode: str
    output_dtype: torch.dtype
    output_scale: str
    scale_mode: str
    skip_softmax: bool
    use_sinks: bool
    causal: bool
    sm_scale: float
    qk_nope_head_dim: int | None
    enable_pdl: bool | None
    is_var_seq: bool | None
    use_profiler: bool
    use_cuda_graph: bool
    workspace_page_capacity: int
    policy_version: int = 1

    def cache_extras(self, backend_name: str) -> tuple[object, ...]:
        """Return synthesis-invariant cache extras for one backend runner."""
        return (
            backend_name,
            self.q_len,
            self.max_q_len,
            self.min_kv_len_bucket,
            self.max_kv_len_bucket,
            self.page_size,
            self.table_width,
            self.page_reuse,
            self.num_heads,
            self.head_dim_ckv,
            self.head_dim_kpe,
            self.query_dtype,
            self.kv_dtype,
            self.kv_layout,
            self.lse_mode,
            self.output_dtype,
            self.output_scale,
            self.scale_mode,
            self.skip_softmax,
            self.use_sinks,
            self.causal,
            self.sm_scale,
            self.qk_nope_head_dim,
            self.enable_pdl,
            self.is_var_seq,
            self.use_profiler,
            self.use_cuda_graph,
            self.workspace_page_capacity,
            self.policy_version,
        )


@dataclass(slots=True)
class _SyntheticMLAPlan:
    """Temporary value objects and plan arguments for one tuning bucket."""

    args: _MLAPlanArguments
    query: MLAQuery
    kv: MLAKVCache
    output: torch.Tensor
    lse: torch.Tensor | None
    profiler_buffer: torch.Tensor | None
    scalar_tensor: torch.Tensor
    sinks: torch.Tensor | None
    num_pages: int


class _PlannedBackend(Protocol):
    def run_from_wrapper(self, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _MLAAutoPlanResult:
    backend_name: str
    backend_impl: object
    trace: MLAAutoSelectionTrace


def _query_lengths(args: _MLAPlanArguments) -> torch.Tensor:
    dense = args.native_dense
    indptr = dense.cum_seq_lens_q.to(dtype=torch.int64)
    return indptr[1:] - indptr[:-1]


def autotune_bypass_reason(args: _MLAPlanArguments) -> str | None:
    """Return why this plan cannot use the initial batch-only sweep."""
    query_lengths = _query_lengths(args)
    if query_lengths.numel() > 1 and bool(
        torch.any(query_lengths != query_lengths[0]).item()
    ):
        return "nonuniform query lengths"
    return None


def summarize_for_autotune(args: _MLAPlanArguments) -> _MLAAutotuneProfile:
    """Reduce normalized plan metadata to immutable logical tuning facts."""
    dense = args.native_dense
    query_lengths = _query_lengths(args)
    batch_size = int(dense.seq_lens.numel())
    q_len = int(query_lengths[0].item()) if query_lengths.numel() else 0
    max_kv_len = int(dense.seq_lens.max().item()) if batch_size else 0
    min_kv_len = int(dense.seq_lens.min().item()) if batch_size else 0

    def sequence_bucket(value: int) -> int:
        return 1 if value <= 1 else 1 << (value - 1).bit_length()

    live_page_ids: list[int] = []
    for request_index, seq_len in enumerate(dense.seq_lens.tolist()):
        live_page_count = (int(seq_len) + args.page_size - 1) // args.page_size
        live_page_ids.extend(
            int(page_id)
            for page_id in dense.block_tables[request_index, :live_page_count].tolist()
        )
    unique_pages = set(live_page_ids)
    bytes_per_page = (
        args.page_size
        * (args.head_dim_ckv + args.head_dim_kpe)
        * torch.empty((), dtype=args.kv_data_type).element_size()
    )
    workspace_bytes = (
        args._float_workspace_buffer.numel()
        * args._float_workspace_buffer.element_size()
    )
    workspace_page_capacity = max(1, workspace_bytes // bytes_per_page)

    return _MLAAutotuneProfile(
        batch_size=batch_size,
        q_len=q_len,
        max_q_len=dense.max_q_len,
        max_kv_len=max_kv_len,
        min_kv_len_bucket=sequence_bucket(min_kv_len),
        max_kv_len_bucket=sequence_bucket(max_kv_len),
        page_size=args.page_size,
        table_width=int(dense.block_tables.shape[1]),
        page_reuse=len(unique_pages) < len(live_page_ids),
        num_heads=args.num_heads,
        head_dim_ckv=args.head_dim_ckv,
        head_dim_kpe=args.head_dim_kpe,
        query_dtype=args.q_data_type,
        kv_dtype=args.kv_data_type,
        kv_layout=args.kv_layout,
        lse_mode=args.lse_mode,
        output_dtype=args.output_dtype,
        output_scale=args.output_scale,
        scale_mode=args.scale_mode,
        skip_softmax=args.skip_softmax,
        use_sinks=args.use_sinks,
        causal=args.causal,
        sm_scale=args.sm_scale,
        qk_nope_head_dim=args.qk_nope_head_dim,
        enable_pdl=args.enable_pdl,
        is_var_seq=args.is_var_seq,
        use_profiler=args.use_profiler,
        use_cuda_graph=args._use_cuda_graph,
        workspace_page_capacity=workspace_page_capacity,
    )


def _seq_len_initializer(max_kv_len: int):
    def initialize(
        shape: Sequence[int], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.full(tuple(shape), max_kv_len, dtype=dtype, device=device)

    return initialize


def _block_table_initializer(profile: _MLAAutotuneProfile):
    def initialize(
        shape: Sequence[int], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        numel = 1
        for dimension in shape:
            numel *= dimension
        values = torch.arange(numel, dtype=dtype, device=device).reshape(tuple(shape))
        return torch.remainder(values, profile.workspace_page_capacity)

    return initialize


@functools.cache
def build_wrapper_tuning_config(
    profile: _MLAAutotuneProfile, *, buckets: tuple[int, ...]
) -> TuningConfig:
    """Build the same single linked-batch sweep used by functional MLA."""
    normalized_buckets = tuple(sorted(set(buckets)))
    if not normalized_buckets:
        raise ValueError("wrapper MLA autotuning requires at least one batch bucket")
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 2, 3),
                dim_idx=(0, 0, 0, 0),
                gen_tuning_buckets=normalized_buckets,
                map_to_tuning_buckets=make_bucket_mapper(
                    normalized_buckets, round_map=False
                ),
                tensor_initializers=(
                    autotuner_initializer_zeros,
                    _block_table_initializer(profile),
                    _seq_len_initializer(profile.max_kv_len),
                    autotuner_initializer_empty,
                ),
            ),
        ),
        use_cuda_graph=True,
        use_cold_l2_cache=False,
    )


def wrapper_tuning_inputs(
    args: _MLAPlanArguments, profile: _MLAAutotuneProfile
) -> list[torch.Tensor]:
    """Return shape/dtype templates for tuner lookup and synthesis."""
    dense = args.native_dense
    device = args._float_workspace_buffer.device
    return [
        torch.empty(
            (
                profile.batch_size,
                profile.q_len,
                profile.num_heads,
                profile.head_dim_ckv + profile.head_dim_kpe,
            ),
            dtype=profile.query_dtype,
            device=device,
        ),
        dense.block_tables,
        dense.seq_lens,
        torch.empty(
            (
                profile.batch_size,
                profile.q_len,
                profile.num_heads,
                profile.head_dim_ckv,
            ),
            dtype=profile.output_dtype,
            device=device,
        ),
    ]


def _compact_block_tables(
    block_tables: torch.Tensor, *, num_pages: int
) -> torch.Tensor:
    """Map arbitrary caller page IDs into one bounded canonical page pool."""
    _, inverse = torch.unique(block_tables, sorted=True, return_inverse=True)
    return (
        torch.remainder(inverse, num_pages)
        .reshape_as(block_tables)
        .to(dtype=block_tables.dtype)
    )


def build_synthetic_plan(
    template: _MLAPlanArguments,
    linked_inputs: list[torch.Tensor],
    *,
    profile: _MLAAutotuneProfile,
) -> _SyntheticMLAPlan:
    """Construct bounded canonical query/KV/metadata for one batch profile."""
    query_4d, block_tables, seq_lens, output_4d = linked_inputs
    batch_size = int(query_4d.shape[0])
    q_len = int(query_4d.shape[1])
    if block_tables.shape[0] != batch_size or seq_lens.shape[0] != batch_size:
        raise ValueError("synthetic MLA linked batch dimensions do not agree")

    unique_page_count = max(1, int(torch.unique(block_tables).numel()))
    num_pages = min(unique_page_count, profile.workspace_page_capacity)
    compact_tables = _compact_block_tables(block_tables, num_pages=num_pages)
    cum_seq_lens_q = torch.arange(
        0,
        (batch_size + 1) * q_len,
        q_len,
        dtype=torch.int32,
        device=query_4d.device,
    )
    metadata = MLAPlanMetadata.dense(
        cum_seq_lens_q,
        compact_tables.to(dtype=torch.int32),
        seq_lens.to(dtype=torch.int32),
        max_q_len=q_len,
    )
    synthetic_args = replace(
        template,
        metadata=metadata,
        _generated_fa_workspace=_BatchMLAGeneratedFaWorkspace(query_4d.device),
        _use_cuda_graph=False,
        _qo_indptr_buf=None,
        _kv_indptr_buf=None,
        _kv_indices_buf=None,
        _kv_len_arr_buf=None,
    )

    kv_shape = (
        num_pages,
        template.page_size,
        template.head_dim_ckv + template.head_dim_kpe,
    )
    if template.kv_layout == "combined":
        kv = MLAKVCache.packed(
            torch.zeros(kv_shape, dtype=template.kv_data_type, device=query_4d.device)
        )
    elif template.kv_layout == "adjacent-split":
        storage = torch.zeros(
            kv_shape, dtype=template.kv_data_type, device=query_4d.device
        )
        kv = MLAKVCache.split(
            storage[..., : template.head_dim_ckv],
            storage[..., template.head_dim_ckv :],
        )
    else:
        kv = MLAKVCache.split(
            torch.zeros(
                (num_pages, template.page_size, template.head_dim_ckv),
                dtype=template.kv_data_type,
                device=query_4d.device,
            ),
            torch.zeros(
                (num_pages, template.page_size, template.head_dim_kpe),
                dtype=template.kv_data_type,
                device=query_4d.device,
            ),
        )

    output = output_4d.reshape(-1, *output_4d.shape[2:])
    return _SyntheticMLAPlan(
        args=synthetic_args,
        query=MLAQuery.packed(query_4d.reshape(-1, *query_4d.shape[2:])),
        kv=kv,
        output=output,
        lse=(
            torch.empty(
                (output.shape[0], template.num_heads),
                dtype=torch.float32,
                device=query_4d.device,
            )
            if template.lse_mode != "none"
            else None
        ),
        profiler_buffer=(
            torch.empty(1 << 20, dtype=torch.uint64, device=query_4d.device)
            if template.use_profiler
            else None
        ),
        scalar_tensor=torch.ones((), dtype=torch.float32, device=query_4d.device),
        sinks=(
            torch.zeros(template.num_heads, dtype=torch.float32, device=query_4d.device)
            if template.use_sinks
            else None
        ),
        num_pages=num_pages,
    )


def _synthetic_run_kwargs(
    synthetic: _SyntheticMLAPlan,
) -> dict[str, object]:
    args = synthetic.args
    return {
        "query": synthetic.query,
        "kv": synthetic.kv,
        "out": synthetic.output,
        "lse": synthetic.lse,
        "return_lse": args.lse_mode != "none",
        "profiler_buffer": synthetic.profiler_buffer,
        "kv_len": None,
        "page_table": None,
        "return_lse_base_on_e": args.lse_mode == "basee",
        "o_scale": 1.0 if args.output_scale == "per-tensor" else None,
        "ckv_scale": 1.0 if args.scale_mode == "kv-per-tensor" else None,
        "kpe_scale": 1.0 if args.scale_mode == "kv-per-tensor" else None,
        "sinks": synthetic.sinks,
        "skip_softmax_threshold_scale_factor": (1.0 if args.skip_softmax else None),
        "bmm1_scale": (
            synthetic.scalar_tensor
            if args.scale_mode == "bmm-tensor"
            else (1.0 if args.scale_mode == "bmm-scalar" else None)
        ),
        "bmm2_scale": (
            synthetic.scalar_tensor
            if args.scale_mode == "bmm-tensor"
            else (1.0 if args.scale_mode == "bmm-scalar" else None)
        ),
    }


class _PlannedMLABackendRunner(TunableRunner):
    """Adapt one planned wrapper backend to the generic tuning protocol."""

    def __init__(
        self,
        backend_name: str,
        backend_type: Any,
        template: _MLAPlanArguments,
        profile: _MLAAutotuneProfile,
    ) -> None:
        self.backend_name = backend_name
        self.backend_type = backend_type
        self.template = template
        self.profile = profile
        # The generic runner hash deliberately ignores fields ending in
        # ``_cache``. Preparation must not change the cache identity.
        self._prepared_cache: tuple[_SyntheticMLAPlan, object] | None = None
        self._rejections_cache: dict[int, str] = {}

    @property
    def rejection(self) -> str | None:
        return (
            next(reversed(self._rejections_cache.values()))
            if self._rejections_cache
            else None
        )

    @property
    def profile_rejections(self) -> tuple[tuple[int, str, str], ...]:
        return tuple(
            (batch_size, self.backend_name, reason)
            for batch_size, reason in sorted(self._rejections_cache.items())
        )

    def __hash__(self) -> int:
        """Keep in-memory cache keys stable across equivalent wrappers."""
        return hash(
            (self.__class__.__name__, self.profile.cache_extras(self.backend_name))
        )

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple[object, ...]:
        return self.profile.cache_extras(self.backend_name)

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile | None,
    ) -> list[int]:
        del profile
        synthetic = build_synthetic_plan(
            self.template,
            inputs,
            profile=self.profile,
        )
        try:
            backend_impl = self.backend_type.plan_from_wrapper(synthetic.args)
            planned_backend = cast(_PlannedBackend, backend_impl)
            planned_backend.run_from_wrapper(**_synthetic_run_kwargs(synthetic))
            if synthetic.output.device.type == "cuda":
                torch.cuda.synchronize(synthetic.output.device)
        except _BackendPlanUnsupportedError as error:
            self._prepared_cache = None
            self._rejections_cache[int(inputs[0].shape[0])] = str(error)
            return []
        self._prepared_cache = (synthetic, backend_impl)
        self._rejections_cache.pop(int(inputs[0].shape[0]), None)
        return [-1]

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> object:
        del tactic, kwargs
        if self._prepared_cache is None:
            valid = self.get_valid_tactics(inputs, None)
            if not valid:
                raise _BackendPlanUnsupportedError(
                    self.rejection or f"{self.backend_name} unsupported"
                )
        if do_preparation:
            return None
        assert self._prepared_cache is not None
        synthetic, backend_impl = self._prepared_cache
        planned_backend = cast(_PlannedBackend, backend_impl)
        return planned_backend.run_from_wrapper(**_synthetic_run_kwargs(synthetic))


def _default_tuning_buckets(
    args: _MLAPlanArguments,
    profile: _MLAAutotuneProfile,
    candidates: tuple[str, ...],
) -> tuple[int, ...]:
    # Import lazily to keep functional and planned controllers lifecycle-local.
    # This helper is a pure shared bucket policy and does not instantiate a
    # functional runner.
    from ._functional import _compute_mla_decode_buckets

    runner_families: list[str] = []
    if "trtllm-gen" in candidates:
        runner_families.append("trtllm-gen")
    if any(name.startswith("cute-dsl-") for name in candidates):
        runner_families.append("cute-dsl")
    return _compute_mla_decode_buckets(
        args._float_workspace_buffer,
        runner_families,
        profile.q_len,
        profile.num_heads,
        profile.head_dim_ckv,
        args._float_workspace_buffer.device,
    )


def _plan_ranked_fallback(
    args: _MLAPlanArguments,
    candidates: tuple[str, ...],
    backend_types: Mapping[str, Any],
    *,
    initial_rejections: list[tuple[str, str]] | None = None,
    skip: frozenset[str] = frozenset(),
) -> tuple[str, object, list[tuple[str, str]]]:
    rejections = [] if initial_rejections is None else list(initial_rejections)
    last_rejection: _BackendPlanUnsupportedError | None = None
    for candidate in candidates:
        if candidate in skip:
            continue
        try:
            implementation = backend_types[candidate].plan_from_wrapper(args)
        except _BackendPlanUnsupportedError as error:
            last_rejection = error
            rejections.append((candidate, str(error)))
            continue
        return candidate, implementation, rejections
    summary = "; ".join(f"{name}: {reason}" for name, reason in rejections)
    raise _BackendPlanUnsupportedError(
        f"backend='auto' rejected all candidates [{', '.join(candidates)}]: {summary}"
    ) from last_rejection


def plan_auto_backend(
    args: _MLAPlanArguments,
    *,
    candidates: tuple[str, ...],
    backend_types: Mapping[str, Any],
    autotune_mode: bool | None,
    tuner: AutoTuner | Any | None = None,
    buckets: tuple[int, ...] | None = None,
) -> _MLAAutoPlanResult:
    """Resolve and plan one backend through the unified automatic policy."""
    bypass_reason = autotune_bypass_reason(args) if autotune_mode is not None else None
    if autotune_mode is None or bypass_reason is not None:
        name, implementation, rejections = _plan_ranked_fallback(
            args, candidates, backend_types
        )
        return _MLAAutoPlanResult(
            name,
            implementation,
            MLAAutoSelectionTrace(
                candidates,
                tuple(rejections),
                "bypass" if bypass_reason is not None else "deterministic",
                bypass_reason,
                name,
            ),
        )

    profile = summarize_for_autotune(args)
    profile = replace(
        profile,
        workspace_page_capacity=max(
            1, profile.workspace_page_capacity // max(1, len(candidates))
        ),
    )
    resolved_buckets = (
        _default_tuning_buckets(args, profile, candidates)
        if buckets is None
        else buckets
    )
    tuning_config = build_wrapper_tuning_config(profile, buckets=resolved_buckets)
    inputs = wrapper_tuning_inputs(args, profile)
    runners = [
        _PlannedMLABackendRunner(name, backend_types[name], args, profile)
        for name in candidates
    ]
    active_tuner = AutoTuner.get() if tuner is None else tuner
    custom_op = "batch_mla_wrapper_backend"
    selected_runner, _ = active_tuner.choose_one(
        custom_op,
        runners,
        tuning_config,
        inputs,
    )
    selected_name = selected_runner.backend_name

    try:
        implementation = backend_types[selected_name].plan_from_wrapper(args)
    except _BackendPlanUnsupportedError as error:
        reason = str(error)
        name, implementation, rejections = _plan_ranked_fallback(
            args,
            candidates,
            backend_types,
            initial_rejections=[(selected_name, reason)],
            skip=frozenset((selected_name,)),
        )
    else:
        name = selected_name
        rejections = [
            (runner.backend_name, runner.rejection)
            for runner in runners
            if runner.rejection is not None
        ]

    profile_rejections = tuple(
        rejection for runner in runners for rejection in runner.profile_rejections
    )
    return _MLAAutoPlanResult(
        name,
        implementation,
        MLAAutoSelectionTrace(
            candidates,
            tuple((candidate, str(reason)) for candidate, reason in rejections),
            "tuning" if autotune_mode else "cache-only",
            None,
            name,
            profile_rejections,
        ),
    )
