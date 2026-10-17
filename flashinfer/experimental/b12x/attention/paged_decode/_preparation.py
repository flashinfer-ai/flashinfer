"""Prepared paged decode/verify attention: declaration, compilation, execution."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import ScratchBufferSpec
from b12x._lib.utils.device import current_cuda_stream
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan

from ._tuning import TUNING, PagedDecodeConfig, PagedDecodeQuery

_FORWARD_ID = "attention.paged_decode.forward"
_MERGE_ID = "attention.paged_decode.merge"
_KV_WRITE_ID = "attention.paged_decode.kv_write"
_COMPILE_VERSION = 1
_KV_CUTE_DTYPE = {"bfloat16": cutlass.BFloat16, "float8_e4m3fn": cutlass.Uint8}

# (name, rank, cutlass dtype or kv marker, assumed alignment) in launch order.
_FORWARD_OPERANDS = (
    ("q", 3, cutlass.BFloat16, 16),
    ("k_cache", 4, "kv", 16),
    ("v_cache", 4, "kv", 16),
    ("page_table", 2, cutlass.Int32, 4),
    ("cache_seqlens", 1, cutlass.Int32, 4),
    ("cu_seqlens_q", 1, cutlass.Int32, 4),
    ("output", 3, cutlass.BFloat16, 16),
    ("o_part", 4, cutlass.Float32, 16),
    ("lse_part", 3, cutlass.Float32, 4),
    ("sinks", 1, cutlass.Float32, 4),
    ("k_descale", "descale", cutlass.Float32, 4),
    ("v_descale", "descale", cutlass.Float32, 4),
)
_MERGE_OPERANDS = (
    ("o_part", 4, cutlass.Float32, 16),
    ("lse_part", 3, cutlass.Float32, 4),
    ("cache_seqlens", 1, cutlass.Int32, 4),
    ("cu_seqlens_q", 1, cutlass.Int32, 4),
    ("output", 3, cutlass.BFloat16, 16),
    ("sinks", 1, cutlass.Float32, 4),
)


# The K/V cache append (``write_kv``), in launch order.
_KV_WRITE_OPERANDS = (
    ("key", 3, cutlass.BFloat16, 16),
    ("value", 3, cutlass.BFloat16, 16),
    ("k_cache", 4, "kv", 16),
    ("v_cache", 4, "kv", 16),
    ("slot_mapping", 1, cutlass.Int64, 8),
    ("k_scale", 1, cutlass.Float32, 4),
    ("v_scale", 1, cutlass.Float32, 4),
)


def _operand_dtype(dtype, query: PagedDecodeQuery):
    return _KV_CUTE_DTYPE[query.kv_dtype] if dtype == "kv" else dtype


def _operand_rank(rank, query: PagedDecodeQuery) -> int:
    if rank == "descale":
        return 2 if query.descale_layout == "head" else 1
    return rank


def _fake_operand(dtype, rank: int, align: int):
    """Compile-time stand-in: every extent dynamic, innermost stride 1."""
    from cutlass.cute.runtime import make_fake_tensor

    shape = tuple(cute.sym_int(32) for _ in range(rank))
    strides = tuple(1 if idx == rank - 1 else cute.sym_int(64) for idx in range(rank))
    return make_fake_tensor(dtype, shape, strides, assumed_align=align)


def _kernel_operand(tensor: torch.Tensor, dtype, align: int):
    """Launch-time operand matching ``_fake_operand``'s dynamic layout."""
    if tensor.stride(-1) != 1:
        raise ValueError(
            f"paged_decode operands must be contiguous in the last dim, got {tensor.stride()}"
        )
    converted = from_dlpack(tensor, assumed_align=align)
    converted.element_type = dtype
    return converted.mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def _validate_operand(name, tensor, *, device, rank, align, aligned_strides=False):
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.ndim != rank or tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have rank {rank} and a unit inner stride")
    if tensor.data_ptr() % align:
        raise ValueError(f"{name} must be {align}-byte aligned")
    if aligned_strides and any(
        stride * tensor.element_size() % align for stride in tensor.stride()[:-1]
    ):
        raise ValueError(f"{name} outer strides must be {align}-byte aligned")


def _forward_kernel(query: PagedDecodeQuery, config: PagedDecodeConfig):
    from ._kernel import PagedDecodeKernel

    return PagedDecodeKernel(
        group_size=query.group_size,
        max_q_per_req=query.max_q_per_req,
        window_left=query.window_left,
        has_sinks=query.has_sinks,
        split_kv=config.split_kv,
        stage_rows=config.stage_rows,
        num_stages=config.num_stages,
        min_tiles_per_split=config.min_tiles_per_split,
        head_dim_qk=query.head_dim_qk,
        head_dim_vo=query.head_dim_vo,
        page_size=query.page_size,
        head_splits=config.head_splits,
        causal=query.causal,
        kv_fp8=query.kv_dtype == "float8_e4m3fn",
        descale_layout=query.descale_layout,
    )


def _kv_write_kernel(query: PagedDecodeQuery):
    from ._kv_write import PagedKVWriteKernel

    return PagedKVWriteKernel(
        num_kv_heads=query.num_kv_heads,
        head_dim_k=query.head_dim_qk,
        head_dim_v=query.head_dim_vo,
        page_size=query.page_size,
        kv_fp8=query.kv_dtype == "float8_e4m3fn",
    )


def _kv_write_key(query: PagedDecodeQuery) -> tuple:
    return (
        query.kv_dtype,
        query.num_kv_heads,
        query.head_dim_qk,
        query.head_dim_vo,
        query.page_size,
    )


def _merge_kernel(query: PagedDecodeQuery, config: PagedDecodeConfig):
    from ._kernel import PagedDecodeMergeKernel

    heads_per_cta = next(h for h in (4, 2, 1) if query.num_q_heads % h == 0)
    return PagedDecodeMergeKernel(
        num_q_heads=query.num_q_heads,
        max_q_per_req=query.max_q_per_req,
        min_tiles_per_split=config.min_tiles_per_split,
        stage_rows=config.stage_rows,
        head_dim_vo=query.head_dim_vo,
        window_left=query.window_left,
        heads_per_cta=heads_per_cta,
        has_sinks=query.has_sinks,
    )


# Batch capacity, SM count and the split budget size scratch and the launch
# grid only (the kernels read the split extent from the partial buffers), so
# every batch bucket of a layer family shares one pair of programs.
_LAUNCH_ONLY_FIELDS = frozenset(("max_batch", "sm_count", "max_splits"))


def _compile_key(query: PagedDecodeQuery, config: PagedDecodeConfig) -> tuple:
    fields = {**TUNING.encode_query(query), **config.to_dict()}
    return tuple(
        sorted((k, v) for k, v in fields.items() if k not in _LAUNCH_ONLY_FIELDS)
    )


@program_cache(scope="preparation")
def compile_paged_decode(query_payload, config_payload, ordinal):
    """Compile the selected forward kernel (and split merge) from metadata only."""
    query = PagedDecodeQuery(**dict(query_payload))
    config = PagedDecodeConfig.from_config(FrozenMapping(config_payload))
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, config, None)
    key = _compile_key(query, config)
    programs = {}
    with torch.cuda.device(ordinal):
        stream = current_cuda_stream()
        fwd_args = [
            _fake_operand(_operand_dtype(dt, query), _operand_rank(rank, query), align)
            for _, rank, dt, align in _FORWARD_OPERANDS
        ]
        programs["forward"] = b12x_compile(
            _forward_kernel(query, config),
            *fwd_args,
            Float32(1.0),
            stream,
            compile_spec=KernelCompileSpec.from_key(_FORWARD_ID, _COMPILE_VERSION, key),
        )
        if config.split_kv:
            merge_args = [
                _fake_operand(dt, rank, align) for _, rank, dt, align in _MERGE_OPERANDS
            ]
            programs["merge"] = b12x_compile(
                _merge_kernel(query, config),
                *merge_args,
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    _MERGE_ID, _COMPILE_VERSION, key
                ),
            )
        write_args = [
            _fake_operand(_operand_dtype(dt, query), rank, align)
            for _, rank, dt, align in _KV_WRITE_OPERANDS
        ]
        programs["kv_write"] = b12x_compile(
            _kv_write_kernel(query),
            *write_args,
            cutlass.Int32(1),
            stream,
            compile_spec=KernelCompileSpec.from_key(
                _KV_WRITE_ID, _COMPILE_VERSION, _kv_write_key(query)
            ),
        )
    return programs


def scratch_specs(query: PagedDecodeQuery, config: PagedDecodeConfig, device):
    """FP32 split partials for the full row capacity (one set per device,
    reused by every layer that shares the geometry)."""
    if not config.split_kv:
        return ()
    rows, heads, splits = query.max_total_q, query.num_q_heads, config.max_splits
    return (
        ScratchBufferSpec(
            name="paged_decode.o_part",
            shape=(rows, heads, splits, query.head_dim_vo),
            dtype=torch.float32,
            device=device,
        ),
        ScratchBufferSpec(
            name="paged_decode.lse_part",
            shape=(rows, heads, splits),
            dtype=torch.float32,
            device=device,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class Binding:
    plan: Plan
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    output: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    o_part: torch.Tensor
    lse_part: torch.Tensor
    sinks: torch.Tensor
    k_descale: torch.Tensor
    v_descale: torch.Tensor
    softmax_scale: float


@dataclass(frozen=True)
class PreparedPagedDecode:
    query: PagedDecodeQuery
    config: PagedDecodeConfig
    device: torch.device
    programs: dict
    dummy_part: tuple[torch.Tensor, torch.Tensor]
    dummy_sinks: torch.Tensor
    dummy_descale: torch.Tensor

    def bind(
        self,
        *,
        plan: Plan,
        scratch,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        attention_sink_bias: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        softmax_scale: float | None = None,
    ) -> Binding:
        """Validate caller tensors against the prepared contract (no allocation,
        so binding inside CUDA graph capture is safe)."""
        query, cfg = self.query, self.config
        tensors = {
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "output": output,
            "page_table": page_table,
            "cache_seqlens": cache_seqlens,
            "cu_seqlens_q": cu_seqlens_q,
        }
        for name, rank, _, align in _FORWARD_OPERANDS:
            if name in tensors:
                _validate_operand(
                    name,
                    tensors[name],
                    device=self.device,
                    rank=rank,
                    align=align,
                    aligned_strides=name in ("q", "k_cache", "v_cache"),
                )
        kv_dtype = (
            torch.bfloat16 if query.kv_dtype == "bfloat16" else torch.float8_e4m3fn
        )
        batch = int(cache_seqlens.shape[0])
        if not 0 < batch <= query.max_batch:
            raise ValueError(
                f"batch {batch} exceeds the planned capacity {query.max_batch}"
            )
        if tuple(cu_seqlens_q.shape) != (batch + 1,) or page_table.shape[0] != batch:
            raise ValueError("page_table / cu_seqlens_q disagree with cache_seqlens")
        for name, tensor in (
            ("page_table", page_table),
            ("cache_seqlens", cache_seqlens),
            ("cu_seqlens_q", cu_seqlens_q),
        ):
            if tensor.dtype != torch.int32 or tensor.stride(-1) != 1:
                raise ValueError(f"{name} must be int32 with a unit inner stride")
        total_q = int(q.shape[0])
        if total_q > query.max_total_q or q.dtype != torch.bfloat16:
            raise ValueError("q exceeds the planned rows or is not BF16")
        if tuple(q.shape[1:]) != (query.num_q_heads, query.head_dim_qk):
            raise ValueError(f"q must be [T, {query.num_q_heads}, {query.head_dim_qk}]")
        if tuple(output.shape) != (total_q, query.num_q_heads, query.head_dim_vo):
            raise ValueError("output must be [T, Hq, head_dim_vo]")
        if output.dtype != torch.bfloat16 or output.stride(-1) != 1:
            raise ValueError("output must be BF16 with a unit inner stride")
        for name, cache, dim in (
            ("k_cache", k_cache, query.head_dim_qk),
            ("v_cache", v_cache, query.head_dim_vo),
        ):
            if cache.ndim != 4 or tuple(cache.shape[1:]) != (
                query.page_size,
                query.num_kv_heads,
                dim,
            ):
                raise ValueError(
                    f"{name} must be [pages, {query.page_size}, {query.num_kv_heads}, {dim}]"
                )
            allowed = (
                (kv_dtype, torch.uint8)
                if query.kv_dtype == "float8_e4m3fn"
                else (kv_dtype,)
            )
            if cache.dtype not in allowed:
                raise ValueError(f"{name} dtype {cache.dtype} differs from {kv_dtype}")
            if any(stride <= 0 for stride in cache.stride()[:-1]) or (
                cache.stride(0) % (cfg.stage_rows * cache.stride(1))
            ):
                raise ValueError(
                    f"{name} must have positive outer strides and a page stride "
                    "divisible by the KV stage stride"
                )
        if k_cache.shape[0] != v_cache.shape[0]:
            raise ValueError("k_cache and v_cache must have the same page count")
        if query.kv_dtype == "float8_e4m3fn":
            k_cache, v_cache = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
            if k_descale is None or v_descale is None:
                raise ValueError("FP8 KV requires k_descale and v_descale")
            want = {
                "tensor": (1,),
                "request": (batch,),
                "head": (batch, query.num_kv_heads),
            }[query.descale_layout]
            scales = []
            for name, scale in (("k_descale", k_descale), ("v_descale", v_descale)):
                if query.descale_layout == "tensor":
                    scale = scale.reshape(1)
                _validate_operand(
                    name,
                    scale,
                    device=self.device,
                    rank=len(want),
                    align=4,
                )
                if (
                    scale.dtype != torch.float32
                    or tuple(scale.shape) != want
                    or (scale.stride(-1) != 1)
                ):
                    raise ValueError(
                        f"{name} must be float32 {want} with a unit inner stride"
                    )
                scales.append(scale)
            k_descale, v_descale = scales
        else:
            k_descale = v_descale = self.dummy_descale
        if query.has_sinks:
            if (
                attention_sink_bias is None
                or tuple(attention_sink_bias.shape) != (query.num_q_heads,)
                or attention_sink_bias.dtype != torch.float32
            ):
                raise ValueError("attention_sink_bias must be float32 [num_q_heads]")
            sinks = attention_sink_bias
            _validate_operand(
                "attention_sink_bias", sinks, device=self.device, rank=1, align=4
            )
        elif attention_sink_bias is not None:
            raise ValueError("the plan was declared without attention sinks")
        else:
            sinks = self.dummy_sinks
        if cfg.split_kv:
            # Full-capacity partial buffers: the split extent is read from them.
            buffers = tuple(scratch)
            if len(buffers) != 2:
                raise ValueError("split paged_decode binds two scratch buffers")
            shapes = [spec.shape for spec in scratch_specs(query, cfg, self.device)]
            for name, buf in zip(("o_part", "lse_part"), buffers, strict=True):
                if buf.device != self.device or not buf.is_contiguous():
                    raise ValueError(
                        f"{name} scratch must be contiguous and on {self.device}"
                    )
                if buf.data_ptr() % (16 if name == "o_part" else 4):
                    raise ValueError(f"{name} scratch is not aligned")
            o_part, lse_part = (
                buf.view(torch.float32).view(shape)
                for buf, shape in zip(buffers, shapes, strict=True)
            )
        else:
            o_part, lse_part = self.dummy_part
        scale = (
            float(softmax_scale)
            if softmax_scale is not None
            else query.head_dim_qk**-0.5
        )
        return Binding(
            plan=plan,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            o_part=o_part,
            lse_part=lse_part,
            sinks=sinks,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=scale,
        )

    def write_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> None:
        """Append each token's K/V rows to its cache slot (capture-safe)."""
        query = self.query
        tensors = {
            "key": key,
            "value": value,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "slot_mapping": slot_mapping,
        }
        for name, rank, _, align in _KV_WRITE_OPERANDS:
            if name in tensors:
                _validate_operand(
                    name,
                    tensors[name],
                    device=self.device,
                    rank=rank,
                    align=align,
                    aligned_strides=name != "slot_mapping",
                )
        num_tokens = int(slot_mapping.shape[0])
        if num_tokens == 0:
            return
        fp8 = query.kv_dtype == "float8_e4m3fn"
        kv_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        heads = query.num_kv_heads
        for name, rows, dim in (
            ("key", key, query.head_dim_qk),
            ("value", value, query.head_dim_vo),
        ):
            if (
                rows.dtype != torch.bfloat16
                or rows.ndim != 3
                or tuple(rows.shape[1:]) != (heads, dim)
            ):
                raise ValueError(f"{name} must be BF16 [T, {heads}, {dim}]")
            if int(rows.shape[0]) < num_tokens:
                raise ValueError(f"{name} has fewer rows than slot_mapping")
        for name, cache, dim in (
            ("k_cache", k_cache, query.head_dim_qk),
            ("v_cache", v_cache, query.head_dim_vo),
        ):
            if cache.ndim != 4 or tuple(cache.shape[1:]) != (
                query.page_size,
                heads,
                dim,
            ):
                raise ValueError(
                    f"{name} must be [pages, {query.page_size}, {heads}, {dim}]"
                )
            if cache.dtype not in ((kv_dtype, torch.uint8) if fp8 else (kv_dtype,)):
                raise ValueError(f"{name} dtype {cache.dtype} differs from {kv_dtype}")
        if k_cache.shape[0] != v_cache.shape[0]:
            raise ValueError("k_cache and v_cache must have the same page count")
        if (
            slot_mapping.dtype != torch.int64
            or slot_mapping.ndim != 1
            or (slot_mapping.stride(0) != 1)
        ):
            raise ValueError("slot_mapping must be a contiguous int64 [T] tensor")
        if fp8:
            if k_scale is None or v_scale is None:
                raise ValueError("FP8 KV requires k_scale and v_scale")
            k_scale, v_scale = k_scale.reshape(1), v_scale.reshape(1)
            if k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
                raise ValueError("k_scale and v_scale must be float32")
            for name, scale in (("k_scale", k_scale), ("v_scale", v_scale)):
                _validate_operand(name, scale, device=self.device, rank=1, align=4)
            k_cache, v_cache = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
        else:
            k_scale = v_scale = self.dummy_descale.reshape(-1)[:1]
        tensors = {
            "key": key,
            "value": value,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "slot_mapping": slot_mapping,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }
        with torch.cuda.device(self.device):
            run_compiled(
                self.programs["kv_write"],
                (
                    *(
                        _kernel_operand(tensors[name], _operand_dtype(dt, query), align)
                        for name, _, dt, align in _KV_WRITE_OPERANDS
                    ),
                    cutlass.Int32(num_tokens),
                    current_cuda_stream(),
                ),
            )

    def run(self, binding: Binding) -> torch.Tensor:
        query, cfg = self.query, self.config
        tensors = {
            "q": binding.q,
            "k_cache": binding.k_cache,
            "v_cache": binding.v_cache,
            "page_table": binding.page_table,
            "cache_seqlens": binding.cache_seqlens,
            "cu_seqlens_q": binding.cu_seqlens_q,
            "output": binding.output,
            "o_part": binding.o_part,
            "lse_part": binding.lse_part,
            "sinks": binding.sinks,
            "k_descale": binding.k_descale,
            "v_descale": binding.v_descale,
        }
        with torch.cuda.device(self.device):
            stream = current_cuda_stream()
            run_compiled(
                self.programs["forward"],
                (
                    *(
                        _kernel_operand(tensors[name], _operand_dtype(dt, query), align)
                        for name, _, dt, align in _FORWARD_OPERANDS
                    ),
                    Float32(binding.softmax_scale),
                    stream,
                ),
            )
            if cfg.split_kv:
                run_compiled(
                    self.programs["merge"],
                    (
                        *(
                            _kernel_operand(tensors[name], dt, align)
                            for name, _, dt, align in _MERGE_OPERANDS
                        ),
                        stream,
                    ),
                )
        return binding.output


def make_plan(caps, *, invocation=None, override=None) -> Plan:
    from .api import Caps

    if not isinstance(caps, Caps):
        raise TypeError("caps must be paged_decode.Caps")
    invocation = FrozenMapping(invocation or {})
    if invocation:
        raise ValueError("paged_decode declarations carry no invocation metadata")
    device = torch.device(caps.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    query = PagedDecodeQuery(
        kv_dtype=str(caps.kv_dtype).removeprefix("torch."),
        num_q_heads=int(caps.num_q_heads),
        num_kv_heads=int(caps.num_kv_heads),
        head_dim_qk=int(caps.head_dim_qk),
        head_dim_vo=int(caps.head_dim_vo),
        page_size=int(caps.page_size),
        max_batch=int(caps.max_batch),
        max_q_per_req=int(caps.max_q_per_req),
        window_left=int(caps.window_left),
        causal=bool(caps.causal),
        has_sinks=bool(caps.has_sinks),
        descale_layout=str(caps.descale_layout),
        sm_count=int(sm_count),
    )
    TUNING.validate_query(query, None)

    def compile_jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.attention.paged_decode._preparation:compile_paged_decode",
                TUNING.encode_query(query),
                config.to_dict(),
                detected.ordinal,
            ),
        )

    def memory(config, detected):
        TUNING.validate_config(query, config, None)
        return MemoryRequirements(
            scratch=scratch_specs(query, config, torch.device("cuda", detected.ordinal))
        )

    def materialize(selection, detected):
        config = selection.config
        dev = torch.device("cuda", detected.ordinal)
        programs = compile_paged_decode(
            TUNING.encode_query(query), config.to_dict(), detected.ordinal
        )
        load_programs(programs)
        dummy_part = (
            torch.zeros(
                (1, query.num_q_heads, 1, query.head_dim_vo),
                dtype=torch.float32,
                device=dev,
            ),
            torch.zeros((1, query.num_q_heads, 1), dtype=torch.float32, device=dev),
        )
        descale_shape = (
            (1, query.num_kv_heads) if query.descale_layout == "head" else (1,)
        )
        state = PreparedPagedDecode(
            query=query,
            config=config,
            device=dev,
            programs=programs,
            dummy_part=dummy_part,
            dummy_sinks=torch.zeros(
                (query.num_q_heads,), dtype=torch.float32, device=dev
            ),
            dummy_descale=torch.ones(descale_shape, dtype=torch.float32, device=dev),
        )
        return attach_programs(state, *programs.values())

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=device,
    )


__all__ = [
    "Binding",
    "PreparedPagedDecode",
    "compile_paged_decode",
    "make_plan",
    "scratch_specs",
]
