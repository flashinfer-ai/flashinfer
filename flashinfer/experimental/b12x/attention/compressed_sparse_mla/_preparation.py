"""Prepared declaration and native compiler extraction for compressed sparse MLA."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from b12x._lib.compile_plan import (
    attach_programs,
    compile_only_launches,
    load_programs,
    program_keys,
)
from b12x._lib.compiler import observe_launchers
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.preparation.types import require_prepared

from ._scratch import (
    B12XCompressedSparseMLABinding,
    B12XCompressedSparseMLAScratchCaps,
    plan_compressed_sparse_mla_scratch,
)
from ._tuning import SparseMlaConfig, SparseMlaQuery, TUNING


def _alignment(tensor: torch.Tensor) -> int:
    pointer = tensor.data_ptr()
    return min(16, pointer & -pointer) if pointer else 16


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": tuple(int(value) for value in tensor.shape),
        "stride": tuple(int(value) for value in tensor.stride()),
        "alignment": _alignment(tensor),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def invocation_from_descriptors(
    *,
    q: FrozenMapping,
    swa_cache: FrozenMapping,
    indexed_cache: FrozenMapping | None = None,
    attn_sink_present: bool = False,
    return_lse: bool = False,
    lse_scale: str = "base2",
    output_mode: str = "internal",
) -> FrozenMapping:
    """Freeze the complete caller-owned storage ABI without allocating tensors."""
    if lse_scale not in ("base2", "natural"):
        raise ValueError("lse_scale must be 'base2' or 'natural'")
    if output_mode not in ("internal", "provided"):
        raise ValueError("output_mode must be 'internal' or 'provided'")

    def descriptor(value: FrozenMapping, name: str) -> FrozenMapping:
        value = FrozenMapping(value)
        if set(value) != {"shape", "stride", "alignment", "dtype"}:
            raise ValueError(f"compressed MLA {name} descriptor is incomplete")
        shape = value["shape"]
        stride = value["stride"]
        alignment = value["alignment"]
        dtype = value["dtype"]
        if (
            not isinstance(shape, tuple)
            or not isinstance(stride, tuple)
            or len(shape) != len(stride)
            or any(type(size) is not int or size < 0 for size in shape)
            or any(type(value) is not int for value in stride)
            or type(alignment) is not int
            or alignment <= 0
            or not isinstance(dtype, str)
        ):
            raise TypeError(f"compressed MLA {name} descriptor has invalid metadata")
        return value

    return FrozenMapping(
        {
            "q": descriptor(q, "q"),
            "swa_k_cache": descriptor(swa_cache, "swa_cache"),
            "indexed_k_cache": (
                None
                if indexed_cache is None
                else descriptor(indexed_cache, "indexed_cache")
            ),
            "attn_sink_present": bool(attn_sink_present),
            "return_lse": bool(return_lse),
            "lse_scale": lse_scale,
            "output_mode": output_mode,
        }
    )


def invocation_from_tensors(
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    indexed_k_cache: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    lse_scale: str = "base2",
) -> FrozenMapping:
    """Capture storage ABI from the actual caller-owned tensor owners."""
    return invocation_from_descriptors(
        q=FrozenMapping(_tensor_metadata(q)),
        swa_cache=FrozenMapping(_tensor_metadata(swa_k_cache)),
        indexed_cache=(
            None
            if indexed_k_cache is None
            else FrozenMapping(_tensor_metadata(indexed_k_cache))
        ),
        attn_sink_present=attn_sink is not None,
        return_lse=return_lse,
        lse_scale=lse_scale,
        output_mode="provided" if out is not None else "internal",
    )


def _metadata(invocation: FrozenMapping, name: str) -> FrozenMapping:
    value = invocation[name]
    if not isinstance(value, FrozenMapping):
        raise TypeError(
            f"compressed MLA invocation field {name!r} must be tensor metadata"
        )
    expected = {"shape", "stride", "alignment", "dtype"}
    if set(value) != expected:
        raise ValueError(
            f"compressed MLA invocation field {name!r} has incomplete tensor metadata"
        )
    return value


def _query(
    caps: B12XCompressedSparseMLAScratchCaps, invocation: FrozenMapping
) -> SparseMlaQuery:
    expected = {
        "q",
        "swa_k_cache",
        "indexed_k_cache",
        "attn_sink_present",
        "return_lse",
        "lse_scale",
        "output_mode",
    }
    if set(invocation) != expected:
        raise ValueError(
            "compressed MLA plan requires invocation_from_descriptors metadata"
        )
    q = _metadata(invocation, "q")
    swa = _metadata(invocation, "swa_k_cache")
    indexed_value = invocation["indexed_k_cache"]
    indexed = (
        None
        if indexed_value is None
        else _metadata(FrozenMapping({"indexed": indexed_value}), "indexed")
    )
    if bool(indexed is not None) != bool(caps.indexed_width):
        raise ValueError(
            "compressed MLA indexed-cache presence differs from declared capacity"
        )
    q_shape = tuple(q["shape"])
    q_stride = tuple(q["stride"])
    if q["dtype"] != "bfloat16":
        raise TypeError("compressed MLA q metadata must be bfloat16")
    if tuple(q_shape[-2:]) != (caps.num_q_heads, caps.head_dim):
        raise ValueError(
            "compressed MLA q shape differs from declared local-head layout"
        )
    if len(q_shape) == 3:
        q_rows = int(q_shape[0])
    elif len(q_shape) == 4 and int(q_shape[1]) == 1:
        q_rows = int(q_shape[0])
    else:
        raise ValueError(
            "compressed MLA q metadata must be [rows, heads, 512] or [rows, 1, heads, 512]"
        )
    if q_rows > caps.max_q_rows:
        raise ValueError("compressed MLA q metadata exceeds declared row capacity")
    return SparseMlaQuery(
        cache_format=caps.cache_format,
        layout=caps.layout,
        mode=caps.mode,
        q_dtype=str(q["dtype"]),
        kv_dtype=str(caps.kv_dtype).removeprefix("torch."),
        num_q_heads=caps.num_q_heads,
        qk_head_dim=caps.head_dim,
        v_head_dim=caps.v_head_dim,
        swa_width=caps.swa_width,
        swa_page_size=caps.swa_page_size,
        indexed_width=caps.indexed_width,
        indexed_page_size=caps.indexed_page_size,
        query_rows=caps.max_q_rows,
        max_batch=caps.max_batch,
        max_kv_rows=caps.max_kv_rows,
        max_page_table_width=caps.max_page_table_width,
        max_q_chunks=caps.max_q_chunks,
        decode_row_capacity=caps.decode_row_capacity,
        use_cuda_graph=caps.use_cuda_graph,
        q_shape=q_shape,
        q_stride=q_stride,
        q_alignment=int(q["alignment"]),
        swa_cache_shape=tuple(swa["shape"]),
        swa_cache_stride=tuple(swa["stride"]),
        swa_cache_alignment=int(swa["alignment"]),
        indexed_cache_present=indexed is not None,
        indexed_cache_shape=None if indexed is None else tuple(indexed["shape"]),
        indexed_cache_stride=None if indexed is None else tuple(indexed["stride"]),
        indexed_cache_alignment=None if indexed is None else int(indexed["alignment"]),
        attn_sink_present=bool(invocation["attn_sink_present"]),
        return_lse=bool(invocation["return_lse"]),
        lse_scale=str(invocation["lse_scale"]),
        output_mode=str(invocation["output_mode"]),
    )


def _caps(
    query: SparseMlaQuery, config: SparseMlaConfig, ordinal: int
) -> B12XCompressedSparseMLAScratchCaps:
    return B12XCompressedSparseMLAScratchCaps(
        device=torch.device("cuda", ordinal),
        num_q_heads=query.num_q_heads,
        max_q_rows=query.query_rows,
        max_width=query.swa_width + query.indexed_width,
        max_page_table_width=query.max_page_table_width,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=query.qk_head_dim,
        v_head_dim=query.v_head_dim,
        max_batch=query.max_batch,
        max_kv_rows=query.max_kv_rows,
        max_chunks_per_row=config.max_chunks_per_row,
        max_q_chunks=query.max_q_chunks,
        decode_row_capacity=query.decode_row_capacity,
        page_size=query.swa_page_size,
        layout=query.layout,
        cache_format=query.cache_format,
        mode=query.mode,
        swa_width=query.swa_width,
        indexed_width=query.indexed_width,
        swa_page_size=query.swa_page_size,
        indexed_page_size=query.indexed_page_size,
        use_cuda_graph=query.use_cuda_graph,
    )


def _fake(shape, *, dtype, device, stride=None):
    if stride is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return torch.empty_strided(shape, stride, dtype=dtype, device=device)


def _fake_binding(query: SparseMlaQuery, config: SparseMlaConfig, ordinal: int):
    caps = _caps(query, config, ordinal)
    device = torch.device("cuda", ordinal)
    scratch_plan = plan_compressed_sparse_mla_scratch(caps, execution_config=config)
    (spec,) = scratch_plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    q = _fake(
        (query.query_rows, *query.q_shape[1:]),
        dtype=torch.bfloat16,
        device=device,
        stride=query.q_stride,
    )
    # The public ABI also admits [rows, 1, heads, D]; native MLA launch
    # lowering always consumes its non-copying [rows, heads, D] view.
    if q.ndim == 4:
        q = q.squeeze(1)
    indices = _fake(
        (query.query_rows, query.swa_width), dtype=torch.int32, device=device
    )
    lengths = _fake((query.query_rows,), dtype=torch.int32, device=device)
    indexed_indices = (
        _fake((query.query_rows, query.indexed_width), dtype=torch.int32, device=device)
        if query.indexed_cache_present
        else None
    )
    indexed_lengths = (
        _fake((query.query_rows,), dtype=torch.int32, device=device)
        if query.indexed_cache_present
        else None
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=indices,
        swa_lengths=lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
    )
    swa = _fake(
        query.swa_cache_shape,
        dtype=torch.uint8,
        device=device,
        stride=query.swa_cache_stride,
    )
    indexed = (
        _fake(
            query.indexed_cache_shape,
            dtype=torch.uint8,
            device=device,
            stride=query.indexed_cache_stride,
        )
        if query.indexed_cache_present
        else None
    )
    return device, binding, swa, indexed


def _lower_launch(
    query: SparseMlaQuery,
    config: SparseMlaConfig,
    ordinal: int,
    *,
    sm_count: int | None = None,
):
    """Resolve and retain the exact native route selected by this declaration."""
    if config.single_pass:
        return _lower_prefill_launch(query, config, ordinal)

    from b12x.attention._shared.mla.kernel import (
        compile_unified_decode_launch,
        prepare_unified_decode_launch,
    )
    from b12x.attention._shared.mla.traits import (
        ComputeMode,
        ModelType,
        ScaleFormat,
        make_unified_traits,
    )

    traits = (
        replace(
            make_unified_traits(
                ModelType.DSV41,
                ComputeMode.FP8
                if config.v41_compute_mode == "fp8"
                else ComputeMode.BF16,
                ScaleFormat.NVFP4_E4M3,
                fp8_rope=False,
            ),
            fp8_internal=config.v41_compute_mode == "fp8",
        )
        if query.cache_format == "deepseek_v41"
        else None
    )

    with torch.cuda.device(ordinal), FakeTensorMode():
        device, binding, swa, indexed = _fake_binding(query, config, ordinal)
        sink = (
            _fake((query.num_q_heads,), dtype=torch.float32, device=device)
            if query.attn_sink_present
            else None
        )
        out = (
            _fake(
                (query.query_rows, query.num_q_heads, query.v_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            if query.output_mode == "provided"
            else None
        )
        launch = prepare_unified_decode_launch(
            q_all=binding.q,
            q_alignment=query.q_alignment,
            swa_k_cache=swa,
            swa_indices=binding.swa_indices,
            swa_page_size=query.swa_page_size,
            workspace=binding.scratch,
            sm_count=(
                torch.cuda.get_device_properties(device).multi_processor_count
                if sm_count is None
                else sm_count
            ),
            indexed_k_cache=indexed,
            indexed_indices=binding.indexed_indices,
            indexed_page_size=query.indexed_page_size if indexed is not None else None,
            attn_sink=sink,
            return_lse=query.return_lse,
            v41_compute_mode=config.v41_compute_mode,
            traits_override=traits,
            v41_heads_per_block=config.v41_heads_per_block
            if query.cache_format == "deepseek_v41"
            else None,
        )
        return compile_unified_decode_launch(
            prepared=launch,
            q_all=binding.q,
            swa_k_cache=swa,
            swa_indices=binding.swa_indices,
            swa_topk_lengths=binding.swa_lengths,
            workspace=binding.scratch,
            sm_scale=1.0,
            latent_scale=1.0,
            swa_page_size=query.swa_page_size,
            indexed_k_cache=indexed,
            indexed_indices=binding.indexed_indices,
            indexed_topk_lengths=binding.indexed_lengths,
            indexed_page_size=query.indexed_page_size if indexed is not None else None,
            attn_sink=sink,
            return_lse=query.return_lse,
            lse_scale=query.lse_scale,
            out=out,
            v41_compute_mode=config.v41_compute_mode,
            v41_heads_per_block=config.v41_heads_per_block
            if query.cache_format == "deepseek_v41"
            else None,
        )


def _lower_prefill_launch(
    query: SparseMlaQuery, config: SparseMlaConfig, ordinal: int
) -> tuple[object, ...]:
    """Compile and retain every MG head partition for a single-pass route."""
    from b12x.attention._shared.mla.prefill import run_unified_prefill
    from b12x.attention._shared.mla.traits import (
        ComputeMode,
        ModelType,
        ScaleFormat,
        make_unified_traits,
    )

    traits = (
        replace(
            make_unified_traits(
                ModelType.DSV41,
                ComputeMode.BF16,
                ScaleFormat.NVFP4_E4M3,
                fp8_rope=False,
            ),
            fp8_internal=config.v41_compute_mode == "fp8",
        )
        if query.cache_format == "deepseek_v41"
        else None
    )

    with torch.cuda.device(ordinal), FakeTensorMode(), compile_only_launches():
        device, binding, swa, indexed = _fake_binding(query, config, ordinal)
        sink = (
            _fake((query.num_q_heads,), dtype=torch.float32, device=device)
            if query.attn_sink_present
            else None
        )
        out = (
            _fake(
                (query.query_rows, query.num_q_heads, query.v_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            if query.output_mode == "provided"
            else None
        )
        retained = run_unified_prefill(
            q=binding.q,
            kv_cache=swa,
            topk_indices=binding.swa_indices,
            sm_scale=1.0,
            page_block_size=query.swa_page_size,
            topk_length=binding.swa_lengths,
            attn_sink=sink,
            output=out,
            extra_kv_cache=indexed,
            extra_indices=binding.indexed_indices,
            extra_topk_length=binding.indexed_lengths,
            extra_page_block_size=(
                query.indexed_page_size if indexed is not None else None
            ),
            mg_enabled=True,
            traits_override=traits,
        )
    if not isinstance(retained, tuple) or any(
        isinstance(value, torch.Tensor) for value in retained
    ):
        raise RuntimeError("compressed MLA prefill did not retain native launchers")
    return tuple(retained)


def _lower_indexed_mapper(
    query: SparseMlaQuery, config: SparseMlaConfig, ordinal: int
) -> object | None:
    if query.cache_format != "deepseek_v41" or not query.indexed_cache_present:
        return None
    from b12x.attention.compressed_sparse_mla._metadata import map_indexed_pages

    with (
        torch.cuda.device(ordinal),
        FakeTensorMode(),
        compile_only_launches(),
        observe_launchers() as observed,
    ):
        device, binding, _swa, indexed = _fake_binding(query, config, ordinal)
        if indexed is None or binding.scratch.mapped_indices is None:
            raise RuntimeError(
                "V4.1 indexed mapper is missing prepared carrier storage"
            )
        page_table = _fake(
            (query.query_rows, query.max_page_table_width),
            dtype=torch.int32,
            device=device,
        )
        map_indexed_pages(
            binding.indexed_indices,
            binding.indexed_lengths,
            page_table,
            binding.scratch.mapped_indices,
            page_size=query.indexed_page_size,
            num_pages=int(indexed.shape[0]),
        )
    if len(observed) != 1:
        raise RuntimeError("V4.1 indexed mapper did not resolve exactly one launcher")
    return observed[0]


@dataclass(frozen=True)
class _CompressedSparseMlaPrograms:
    """Actual native launch owners returned by the compile factory.

    The compiler pool uses their attached program identities for admission, while
    materialization retains the same executable launch objects for replay.
    """

    launch: object
    mapper: object | None = None

    @property
    def __b12x_dependencies__(self):
        if isinstance(self.launch, tuple):
            return (*self.launch, self.mapper)
        return (
            *self.launch.grid_programs,
            self.launch.merge_program,
            self.launch.lse_program,
            self.mapper,
        )

    @property
    def __b12x_programs__(self):
        launch = self.launch
        programs = (
            program_keys(launch)
            if isinstance(launch, tuple)
            else program_keys(
                (*launch.grid_programs, launch.merge_program, launch.lse_program)
            )
        )
        return (
            *programs,
            *(() if self.mapper is None else program_keys(self.mapper)),
        )


@program_cache(scope="preparation")
def compile_compressed_sparse_mla(query_payload, config_payload, ordinal):
    """Compile and retain the selected native decode or MG prefill launchers."""
    query = SparseMlaQuery(**dict(query_payload))
    config = SparseMlaConfig.from_config(FrozenMapping(config_payload))
    TUNING.validate_query(query, None)
    launch = _lower_launch(query, config, ordinal)
    mapper = _lower_indexed_mapper(query, config, ordinal)
    programs = _CompressedSparseMlaPrograms(launch, mapper)
    if not programs.__b12x_programs__:
        raise RuntimeError("compressed MLA preparation produced no native programs")
    return programs


@dataclass(frozen=True)
class _PreparedCompressedSparseMla:
    query: SparseMlaQuery
    config: SparseMlaConfig
    scratch_plan: object
    launch: object
    mapper: object | None

    def scratch_specs(self):
        return self.scratch_plan.scratch_specs()

    def bind_for_preparation(self, **kwargs) -> B12XCompressedSparseMLABinding:
        binding = self.scratch_plan.bind(**kwargs)
        if binding.q.numel() and _alignment(binding.q) < self.query.q_alignment:
            raise ValueError(
                "compressed MLA Q alignment differs from the prepared plan"
            )
        return binding

    def bind(self, plan: Plan, **kwargs) -> B12XCompressedSparseMLABinding:
        binding = self.bind_for_preparation(**kwargs)
        return replace(binding, plan=plan)

    def run(self, binding: B12XCompressedSparseMLABinding, **kwargs):
        from b12x.attention._shared.mla.compressed_api import (
            compressed_sparse_mla_decode_forward,
        )

        kwargs.setdefault("swa_page_size", self.query.swa_page_size)
        if self.query.indexed_cache_present:
            kwargs.setdefault("indexed_page_size", self.query.indexed_page_size)
        if (
            kwargs.pop("cache_format", self.query.cache_format)
            != self.query.cache_format
        ):
            raise ValueError(
                "compressed MLA cache_format differs from the prepared plan"
            )

        if bool(kwargs.get("return_lse", False)) != self.query.return_lse:
            raise ValueError("compressed MLA return_lse differs from the prepared plan")
        if kwargs.get("lse_scale", "base2") != self.query.lse_scale:
            raise ValueError("compressed MLA lse_scale differs from the prepared plan")
        if (kwargs.get("out") is not None) != (self.query.output_mode == "provided"):
            raise ValueError(
                "compressed MLA output ownership differs from the prepared plan"
            )
        if (kwargs.get("attn_sink") is not None) != self.query.attn_sink_present:
            raise ValueError(
                "compressed MLA attention-sink route differs from the prepared plan"
            )
        indexed = kwargs.get("indexed_k_cache") is not None
        if indexed != self.query.indexed_cache_present:
            raise ValueError(
                "compressed MLA indexed-cache route differs from the prepared plan"
            )
        if (
            int(kwargs.get("swa_page_size", self.query.swa_page_size))
            != self.query.swa_page_size
        ):
            raise ValueError(
                "compressed MLA SWA page size differs from the prepared plan"
            )
        if (
            indexed
            and int(kwargs.get("indexed_page_size", 0)) != self.query.indexed_page_size
        ):
            raise ValueError(
                "compressed MLA indexed page size differs from the prepared plan"
            )
        return compressed_sparse_mla_decode_forward(
            binding=binding,
            prepared=self.launch,
            mapper_launcher=self.mapper,
            cache_format=self.query.cache_format,
            **kwargs,
        )


def plan(
    caps: B12XCompressedSparseMLAScratchCaps,
    *,
    invocation=FrozenMapping(),
    override=None,
) -> Plan:
    if not isinstance(caps, B12XCompressedSparseMLAScratchCaps):
        raise TypeError("caps must be compressed_sparse_mla.Caps")
    invocation = FrozenMapping(invocation)
    query = _query(caps, invocation)
    TUNING.validate_query(query, None)

    def state_caps(config, device):
        return _caps(query, config, device.ordinal)

    def scratch_plan(config, device):
        return plan_compressed_sparse_mla_scratch(
            state_caps(config, device),
            execution_config=config,
        )

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.attention.compressed_sparse_mla._preparation:compile_compressed_sparse_mla",
                TUNING.encode_query(query),
                TUNING.encode_config(config),
                device.ordinal,
            ),
        )

    def memory(config, device):
        return MemoryRequirements(scratch=scratch_plan(config, device).scratch_specs())

    def materialize(selection, device):
        # Resolve process-local executable owners only after the pool has made
        # their artifacts available.  The returned launch is the sole runtime
        # dispatcher; program keys are never used as an execution surrogate.
        programs = compile_compressed_sparse_mla(
            TUNING.encode_query(query),
            TUNING.encode_config(selection.config),
            device.ordinal,
        )
        load_programs(programs)
        return attach_programs(
            _PreparedCompressedSparseMla(
                query=query,
                config=selection.config,
                scratch_plan=scratch_plan(selection.config, device),
                launch=programs.launch,
                mapper=programs.mapper,
            ),
            programs,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
    )


def bind(plan: Plan, **kwargs) -> B12XCompressedSparseMLABinding:
    state = require_prepared(plan, "attention.compressed_sparse_mla", None)
    return state.bind(plan, **kwargs)


def run(
    *,
    plan: Plan | None = None,
    binding: B12XCompressedSparseMLABinding,
    **kwargs,
):
    if plan is None:
        plan = binding.plan
    if plan is None:
        raise TypeError("compressed MLA run requires a prepared Plan")
    if binding.plan is not plan:
        raise ValueError("compressed MLA binding belongs to another prepared plan")
    state = require_prepared(
        plan,
        "attention.compressed_sparse_mla",
        binding.q.device,
    )
    return state.run(binding, **kwargs)


__all__ = [
    "bind",
    "compile_compressed_sparse_mla",
    "invocation_from_descriptors",
    "invocation_from_tensors",
    "plan",
    "run",
]
