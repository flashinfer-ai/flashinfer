"""Prepared dense-MLA declaration and retained native launchers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation.types import FrozenMapping, MemoryRequirements, Plan, require_prepared

from ._scratch import Binding, Caps, _dense_mla_scratch_layout, _materialize, plan_dense_mla_scratch
from ._tuning import DenseMlaConfig, DenseMlaQuery, TUNING


_OPERANDS = (
    "q", "kv_cache", "output", "page_table", "cache_seqlens", "cu_seqlens_q",
    "kv_scale", "q_scale",
)


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported dense MLA dtype {name!r}")
    return value


def _alignment(tensor: torch.Tensor) -> int:
    pointer = int(tensor.data_ptr())
    return min(16, pointer & -pointer) if pointer else 16


def _descriptor(tensor: torch.Tensor) -> FrozenMapping:
    return FrozenMapping({
        "shape": tuple(int(value) for value in tensor.shape),
        "strides": tuple(int(value) for value in tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "alignment": _alignment(tensor),
    })

def _is_scalar_shape(shape: tuple[object, ...], strides: tuple[object, ...]) -> bool:
    return (shape == () and strides == ()) or (
        shape == (1,)
        and len(strides) == 1
        and isinstance(strides[0], int)
        and not isinstance(strides[0], bool)
        and strides[0] > 0
    )




def _cache_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        if int(tensor.shape[2]) != 1:
            raise ValueError("rank-4 dense MLA cache must have one KV head")
        return tensor[:, :, 0, :]
    return tensor


def invocation_from_descriptors(
    caps: Caps, *, operands: Mapping[str, Mapping[str, object] | None],
) -> FrozenMapping:
    """Describe the exact immutable tensor ABI without allocating device storage."""
    if not isinstance(caps, Caps):
        raise TypeError("caps must be dense_mla.Caps")
    normalized = {}
    for name in _OPERANDS:
        descriptor = operands.get(name)
        if descriptor is None:
            normalized[name] = None
            continue
        fields = FrozenMapping(descriptor)
        if set(fields) != {"shape", "strides", "dtype", "alignment"}:
            raise ValueError(f"dense MLA {name} ABI descriptor fields do not match schema")
        shape = tuple(fields["shape"])
        strides = tuple(fields["strides"])
        if (
            len(shape) != len(strides)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in (*shape, *strides)
            )
            or not isinstance(fields["dtype"], str)
            or not isinstance(fields["alignment"], int)
            or isinstance(fields["alignment"], bool)
            or int(fields["alignment"]) <= 0
        ):
            raise ValueError(f"dense MLA {name} ABI descriptor is invalid")
        if name in ("kv_scale", "q_scale"):
            if not _is_scalar_shape(shape, strides):
                raise ValueError(
                    f"dense MLA {name} ABI must be a float32 scalar or one-element view"
                )
        elif not shape:
            raise ValueError(f"dense MLA {name} ABI descriptor is invalid")
        if name == "kv_cache" and len(shape) == 4:
            if shape[2] != 1:
                raise ValueError("rank-4 dense MLA cache must have one KV head")
            fields = FrozenMapping({
                "shape": (shape[0], shape[1], shape[3]),
                "strides": (strides[0], strides[1], strides[3]),
                "dtype": fields["dtype"],
                "alignment": fields["alignment"],
            })
        normalized[name] = fields
    return FrozenMapping({"operands": FrozenMapping(normalized)})


def invocation_from_tensors(caps: Caps, **tensors: torch.Tensor | None) -> FrozenMapping:
    """Capture the actual ABI of caller-owned tensors for one declaration."""
    return invocation_from_descriptors(
        caps,
        operands={
            name: None if (tensor := tensors.get(name)) is None else _descriptor(
                _cache_view(tensor) if name == "kv_cache" else tensor
            )
            for name in _OPERANDS
        },
    )



def _canonical_invocation(caps: Caps) -> FrozenMapping:
    def descriptor(shape, dtype):
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= size
        return {
            "shape": tuple(shape),
            "strides": tuple(reversed(strides)),
            "dtype": str(dtype).removeprefix("torch."),
            "alignment": 16,
        }

    fp8 = caps.kv_dtype == torch.float8_e4m3fn
    return invocation_from_descriptors(caps, operands={
        "q": descriptor(
            (caps.max_total_q, caps.num_q_heads, caps.head_dim), caps.q_dtype
        ),
        "kv_cache": descriptor(
            (caps.num_cache_pages, caps.page_size, caps.physical_record_width),
            caps.kv_dtype,
        ),
        "output": descriptor(
            (caps.max_total_q, caps.num_q_heads, caps.v_head_dim), torch.bfloat16
        ),
        "page_table": descriptor(
            (caps.max_batch, caps.max_page_table_width), torch.int32
        ),
        "cache_seqlens": descriptor((caps.max_batch,), torch.int32),
        "cu_seqlens_q": descriptor((caps.max_batch + 1,), torch.int32),
        "kv_scale": descriptor((1,), torch.float32) if fp8 else None,
        "q_scale": descriptor((1,), torch.float32) if fp8 else None,
    })
def _abi(invocation: FrozenMapping) -> FrozenMapping:
    if set(invocation) != {"operands"}:
        raise ValueError("dense MLA declarations require invocation_from_tensors metadata")
    operands = invocation["operands"]
    if not isinstance(operands, FrozenMapping) or set(operands) != set(_OPERANDS):
        raise ValueError("dense MLA invocation operands must be complete")
    for name in _OPERANDS:
        value = operands[name]
        if name in ("kv_scale", "q_scale"):
            if value is not None and not isinstance(value, FrozenMapping):
                raise TypeError(f"dense MLA {name} ABI must be metadata or None")
        elif not isinstance(value, FrozenMapping):
            raise TypeError(f"dense MLA {name} ABI metadata is required")
    return operands

def _matches_abi(expected: FrozenMapping, actual: FrozenMapping) -> bool:
    """Check retained-launch compatibility after each side passed ABI validation.

    Only the native forward entries' static tensor layouts constrain reuse.
    Sequence metadata and scalar scale tensors are dynamic launch inputs: their
    live lengths, page-table width/stride, scalar rank, and scalar-view stride
    must not force recompilation of an otherwise prepared plan.
    """
    for name in ("q", "kv_cache", "output"):
        expected_value = expected[name]
        actual_value = actual[name]
        assert isinstance(expected_value, FrozenMapping)
        assert isinstance(actual_value, FrozenMapping)
        if (
            expected_value["dtype"] != actual_value["dtype"]
            or expected_value["strides"] != actual_value["strides"]
        ):
            return False
        expected_shape = tuple(expected_value["shape"])
        actual_shape = tuple(actual_value["shape"])
        if len(expected_shape) != len(actual_shape):
            return False
        if expected_shape[1:] != actual_shape[1:]:
            return False
    return True

def _validate_abi(caps: Caps, abi: FrozenMapping) -> None:
    def metadata(name: str) -> FrozenMapping | None:
        value = abi[name]
        return value if isinstance(value, FrozenMapping) else None

    q = metadata("q")
    cache = metadata("kv_cache")
    output = metadata("output")
    page_table = metadata("page_table")
    lengths = metadata("cache_seqlens")
    cu = metadata("cu_seqlens_q")
    assert all(value is not None for value in (q, cache, output, page_table, lengths, cu))
    if (
        q["dtype"] != str(caps.q_dtype).removeprefix("torch.")
        or tuple(q["shape"][1:]) != (caps.num_q_heads, caps.head_dim)
        or not 1 <= int(q["shape"][0]) <= caps.max_total_q
        or int(q["strides"][2]) != 1
        or int(q["strides"][1]) != caps.head_dim
        or int(q["strides"][0]) < caps.num_q_heads * caps.head_dim
        or int(q["alignment"]) < 16
        or int(q["strides"][0]) * caps.q_dtype.itemsize % 16
    ):
        raise ValueError("dense MLA q ABI differs from declared capacity")
    if (
        cache["dtype"] != str(caps.kv_dtype).removeprefix("torch.")
        or tuple(cache["shape"][1:]) != (caps.page_size, caps.physical_record_width)
        or not 1 <= int(cache["shape"][0]) <= caps.num_cache_pages
        or int(cache["strides"][2]) != 1
        or int(cache["strides"][1]) < caps.physical_record_width
        or int(cache["strides"][0]) < caps.page_size * int(cache["strides"][1])
        or int(cache["alignment"]) < 16
        or int(cache["strides"][0]) * caps.kv_dtype.itemsize % 16
    ):
        raise ValueError("dense MLA cache ABI is not a legal declared page layout")
    if (
        output["dtype"] != "bfloat16"
        or tuple(output["shape"]) != (
            int(q["shape"][0]), caps.num_q_heads, caps.v_head_dim,
        )
        or int(output["strides"][2]) != 1
        or int(output["strides"][1]) != caps.v_head_dim
        or int(output["strides"][0]) < caps.num_q_heads * caps.v_head_dim
        or int(output["alignment"]) < 16
        or int(output["strides"][0]) * torch.bfloat16.itemsize % 16
    ):
        raise ValueError("dense MLA output ABI differs from declared capacity")
    batch = int(lengths["shape"][0])
    if (
        lengths["dtype"] != "int32"
        or tuple(lengths["strides"]) != (1,)
        or not 1 <= batch <= caps.max_batch
        or cu["dtype"] != "int32"
        or tuple(cu["shape"]) != (batch + 1,)
        or tuple(cu["strides"]) != (1,)
        or page_table["dtype"] != "int32"
        or tuple(page_table["shape"])[0] != batch
        or not 1 <= int(page_table["shape"][1]) <= caps.max_page_table_width
        or tuple(page_table["strides"]) != (int(page_table["shape"][1]), 1)
    ):
        raise ValueError("dense MLA sequence metadata ABI differs from declared capacity")
    fp8 = caps.kv_dtype == torch.float8_e4m3fn
    for name, required in (("kv_scale", fp8), ("q_scale", fp8)):
        scale = metadata(name)
        if (scale is None) != (not required):
            raise ValueError(f"dense MLA {name} ABI presence differs from cache precision")
        if scale is not None and (
            scale["dtype"] != "float32"
            or not _is_scalar_shape(
                tuple(scale["shape"]), tuple(scale["strides"])
            )
            or int(scale["alignment"]) < 4
        ):
            raise ValueError(
                f"dense MLA {name} ABI must be an aligned float32 scalar"
            )


def _query(caps: Caps, abi: FrozenMapping) -> DenseMlaQuery:
    return DenseMlaQuery(
        mode=caps.mode, q_dtype=str(caps.q_dtype).removeprefix("torch."),
        kv_dtype=str(caps.kv_dtype).removeprefix("torch."),
        num_q_heads=caps.num_q_heads, qk_head_dim=caps.head_dim,
        v_head_dim=caps.v_head_dim, page_size=caps.page_size,
        query_rows=caps.max_total_q, max_batch=caps.max_batch,
        cache_tokens=caps.max_cache_tokens,
        physical_record_width=caps.physical_record_width,
        window_size=caps.window_size, use_cuda_graph=caps.use_cuda_graph,
        max_page_table_width=caps.max_page_table_width,
        num_cache_pages=caps.num_cache_pages, abi=abi,
    )


def _caps(query: DenseMlaQuery, config: DenseMlaConfig, device: torch.device) -> Caps:
    return Caps(
        device=device, mode=query.mode, q_dtype=_dtype(query.q_dtype),
        kv_dtype=_dtype(query.kv_dtype), num_q_heads=query.num_q_heads,
        page_size=query.page_size, max_total_q=query.query_rows,
        max_batch=query.max_batch, max_cache_tokens=query.cache_tokens,
        max_page_table_width=query.max_page_table_width,
        num_cache_pages=query.num_cache_pages,
        head_dim=query.qk_head_dim, v_head_dim=query.v_head_dim,
        physical_record_width=query.physical_record_width,
        window_size=query.window_size, use_cuda_graph=query.use_cuda_graph,
        budget=__import__("b12x.attention.dense_mla.planner", fromlist=["Budget"]).Budget(max_splits=config.max_splits),
    )


@program_cache(scope="preparation")
def compile_dense_mla(query_payload, config_payload, ordinal):
    """Compile selected forward/merge/quantizer entries from FakeTensor metadata."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x._lib.compile_plan import compile_only_launches
    from . import _kernel
    from ._scratch import Binding

    query = DenseMlaQuery.from_dict(query_payload)
    config = DenseMlaConfig.from_config(FrozenMapping(config_payload))
    caps = _caps(query, config, torch.device("cuda", ordinal))
    layout = _dense_mla_scratch_layout(caps)
    abi = query.abi
    with FakeTensorMode(), compile_only_launches():
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=caps.device)

        def operand(name: str) -> torch.Tensor | None:
            metadata = abi[name]
            if metadata is None:
                return None
            assert isinstance(metadata, FrozenMapping)
            return torch.empty_strided(
                tuple(int(value) for value in metadata["shape"]),
                tuple(int(value) for value in metadata["strides"]),
                dtype=_dtype(str(metadata["dtype"])),
                device=caps.device,
            )

        storage = empty((layout.nbytes,), torch.uint8)
        scratch = _materialize(caps, storage, layout)
        source_q = operand("q")
        cache = operand("kv_cache")
        output = operand("output")
        page_table = operand("page_table")
        lengths = operand("cache_seqlens")
        cu = operand("cu_seqlens_q")
        kv_scale = operand("kv_scale")
        q_scale = operand("q_scale")
        assert all(value is not None for value in (
            source_q, cache, output, page_table, lengths, cu,
        ))
        native_q = source_q
        if caps.q_dtype != caps.kv_dtype:
            assert scratch.quantized_q is not None
            native_q = scratch.quantized_q[: int(source_q.shape[0])]
        binding = Binding(
            scratch=scratch, q=native_q, kv_cache=cache, output=output,
            page_table=page_table, cache_seqlens=lengths, cu_seqlens_q=cu,
            kv_scale=kv_scale, q_scale=q_scale, sm_scale=1.0,
            active_splits=layout.num_splits, query_quant=None,
        )
        forward, merge = _kernel._compile_entries(binding)
        programs = {"forward": forward}
        if merge is not None:
            programs["merge"] = merge
        if caps.q_dtype != caps.kv_dtype:
            from .._shared import static_fp8_quant

            assert source_q is not None and q_scale is not None
            assert scratch.quantized_q is not None
            quant = static_fp8_quant.bind(
                source=source_q,
                output=scratch.quantized_q[: int(source_q.shape[0])],
                scale=q_scale,
                max_numel=caps.max_total_q * caps.num_q_heads * caps.head_dim,
            )
            static_fp8_quant.compile(binding=quant)
            programs["quantizer"] = static_fp8_quant.resolve_static_fp8_quant_launcher(
                binding=quant
            ).compiled
    return programs


@dataclass(frozen=True)
class _DenseMlaState:
    caps: Caps
    layout: object
    abi: FrozenMapping
    launchers: object | None = None

    def scratch_specs(self):
        return self.layout.scratch_specs()

    def bind(self, **kwargs) -> Binding:
        actual = invocation_from_tensors(
            self.caps, **{name: kwargs.get(name) for name in _OPERANDS}
        )
        actual_abi = actual["operands"]
        assert isinstance(actual_abi, FrozenMapping)
        _validate_abi(self.caps, actual_abi)
        if not _matches_abi(self.abi, actual_abi):
            raise ValueError("dense MLA binding ABI differs from its prepared plan")
        return self.layout.bind(**kwargs)

    def prime(self, binding: Binding) -> None:
        from ._kernel import resolve_dense_mla_launchers
        from .._shared.static_fp8_quant import resolve_static_fp8_quant_launcher

        launchers = resolve_dense_mla_launchers(binding=binding)
        quantizer = (
            resolve_static_fp8_quant_launcher(binding=binding.query_quant)
            if binding.query_quant is not None else None
        )
        object.__setattr__(self, "launchers", MappingProxyType({
            "dense": launchers,
            "quantizer": quantizer,
        }))

    def run(self, binding: Binding):
        if self.launchers is None:
            raise RuntimeError("dense MLA plan was not prepared")
        quantizer = self.launchers["quantizer"]
        if quantizer is not None:
            quantizer.run(binding.query_quant)
        return self.launchers["dense"].run(binding)


def plan(
    caps: Caps, *, invocation: FrozenMapping = FrozenMapping(),
    override: DenseMlaConfig | None = None,
) -> Plan:
    if not isinstance(caps, Caps):
        raise TypeError("caps must be dense_mla.Caps")
    invocation = FrozenMapping(invocation)
    if not invocation:
        invocation = _canonical_invocation(caps)
    abi = _abi(invocation)
    _validate_abi(caps, abi)
    query = _query(caps, abi)

    def lowered(config):
        return _caps(query, config, caps.device)

    def memory(config, device):
        native = plan_dense_mla_scratch(lowered(config))
        return MemoryRequirements(scratch=native.scratch_specs())

    def materialize(selection, device):
        native = plan_dense_mla_scratch(lowered(selection.config))
        return _DenseMlaState(lowered(selection.config), native, query.abi)

    return Plan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.attention.dense_mla._preparation:compile_dense_mla",
            TUNING.encode_query(replace(query, exhaustive=False)), TUNING.encode_config(config), device.ordinal,
        ),),
        _memory_requirements=memory, _materialize=materialize, _device=caps.device,
    )


def state(plan, *, device=None) -> _DenseMlaState:
    return require_prepared(plan, TUNING.component_id, device)


__all__ = [
    "compile_dense_mla", "invocation_from_descriptors", "invocation_from_tensors",
    "plan", "state",
]
