"""Prepared sparse-MLA declarations and resolved prepared state."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation.types import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    require_prepared,
)
from b12x._lib.compile_plan import compile_only_launches
from b12x.attention._shared.mla.traits import KVARN_RECORD_VARIANTS

from ._scratch import (
    B12XSparseMLABinding,
    B12XSparseMLAScratchCaps as Caps,
    plan_sparse_mla_scratch,
)
from ._tuning import SparseMlaConfig, SparseMlaQuery, TUNING


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@program_cache(scope="preparation")
def compile_sparse_mla(caps, ordinal, prefill_mg_enabled=True):
    """Extract real decode or prefill native programs from FakeTensor metadata."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from ._scratch import (
        B12XSparseMLABinding,
        _materialize_sparse_mla_scratch,
        _sparse_mla_scratch_layout,
    )

    if not isinstance(caps, Caps):
        raise TypeError("sparse MLA compiler requires B12XSparseMLAScratchCaps")
    device = torch.device("cuda", ordinal)
    caps = Caps(
        **{
            name: getattr(caps, name)
            for name in Caps.__dataclass_fields__
            if name not in {"device", "cache_traits"}
        }
        | {"device": device}
    )
    layout = _sparse_mla_scratch_layout(caps)
    with FakeTensorMode(), compile_only_launches():

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        scratch = _materialize_sparse_mla_scratch(
            caps, empty((layout.nbytes,), torch.uint8), layout
        )
        rows = caps.max_q_rows
        q = empty((rows, caps.num_q_heads, caps.head_dim), caps.dtype)
        cache = empty(
            (
                max(1, caps.max_page_table_width),
                caps.page_size,
                caps.cache_record_bytes,
            ),
            caps.kv_dtype,
        )
        binding = B12XSparseMLABinding(
            scratch=scratch,
            q=q,
            selected_indices=empty((rows, caps.max_width), torch.int32),
            cache_seqlens_int32=empty((caps.max_batch,), torch.int32),
            nsa_cache_seqlens_int32=empty((rows,), torch.int32),
            model_type=caps.model_type,
            scale_format=caps.scale_format,
            cache_record_bytes=caps.cache_record_bytes,
            fp8_rope=caps.fp8_rope,
            latent_scale_per_token=caps.latent_scale_per_token,
            kv_cache=cache,
        )
        state = _SparseMlaState(
            caps,
            type(
                "_Layout",
                (),
                {"bind": lambda *_a, **_k: binding, "scratch_specs": lambda _s: ()},
            )(),
            _query_from_caps(
                caps, FrozenMapping({"prefill_mg_enabled": prefill_mg_enabled})
            ),
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        state.prime(
            binding,
            kv_cache=cache,
            attention_sink=(
                empty((caps.num_q_heads,), torch.float32)
                if caps.has_attention_sink
                else None
            ),
        )
    if state.prepared is None:
        raise RuntimeError(
            "sparse MLA compile factory did not retain its decode launchers"
        )
    return state.prepared


@program_cache(scope="preparation")
def compile_cache_writer(payload, ordinal):
    """Compile the exact cache writer from shape-faithful FakeTensor metadata."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x._lib.compile_plan import compile_only_launches
    from b12x.attention._shared.mla import kv_cache as writer

    device = torch.device("cuda", ordinal)
    with FakeTensorMode(), compile_only_launches():

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        kv_c = empty(
            tuple(payload["kv_c_shape"]), getattr(torch, payload["kv_c_dtype"])
        )
        cache = empty(
            tuple(payload["cache_shape"]), getattr(torch, payload["cache_dtype"])
        )
        slots = empty(
            tuple(payload["slots_shape"]), getattr(torch, payload["slots_dtype"])
        )
        if payload.get("kvarn"):
            from b12x.attention._shared.mla._kvarn_writer import (
                _compile_kvarn_writer,
            )

            return {
                "writer": _compile_kvarn_writer(
                    kv_c,
                    cache,
                    slots,
                    int(payload["k_bits"]),
                    int(payload["v_bits"]),
                    int(payload["group"]),
                )
            }
        if payload.get("tail_fp8"):
            return {
                "writer": writer._compile_glm_next_mla_cache_writer(kv_c, cache, slots)
            }
        if bool(payload["nvfp4"]):
            return {
                "writer": writer._compile_nvfp4_mla_writer(
                    kv_c, kv_c, cache, slots, True, False
                )
            }
        return {"writer": writer._compile_glm_next_mla_cache_writer(kv_c, cache, slots)}


def _query_from_caps(caps: Caps, invocation: FrozenMapping) -> SparseMlaQuery:
    unknown = set(invocation) - {
        "operation",
        "cache_layout",
        "slot_dtype",
        "prefill_mg_enabled",
    }
    if unknown:
        raise ValueError(f"unsupported sparse MLA invocation fields: {sorted(unknown)}")
    raw_prefill_mg_enabled = invocation.get(
        "prefill_mg_enabled",
        os.environ.get("B12X_MLA_SM120_PREFILL_MG", "1")
        not in ("0", "false", "False", "off"),
    )
    if type(raw_prefill_mg_enabled) is not bool:
        raise TypeError("prefill_mg_enabled must be bool")
    prefill_mg_enabled = raw_prefill_mg_enabled
    return SparseMlaQuery(
        mode=caps.mode,
        dtype=_dtype_name(caps.dtype),
        kv_dtype=_dtype_name(caps.kv_dtype),
        num_q_heads=caps.num_q_heads,
        qk_head_dim=caps.head_dim,
        v_head_dim=caps.v_head_dim,
        max_q_rows=caps.max_q_rows,
        max_width=caps.max_width,
        page_size=caps.page_size,
        model_type=caps.model_type,
        head_major_output=caps.head_major_output,
        scale_format=caps.scale_format,
        cache_record_bytes=caps.cache_record_bytes,
        fp8_rope=bool(caps.fp8_rope),
        latent_scale_per_token=bool(caps.latent_scale_per_token),
        has_attention_sink=bool(caps.has_attention_sink),
        cache_layout=str(invocation.get("cache_layout", "paged")),
        operation=str(invocation.get("operation", "attention")),
        slot_dtype=invocation.get("slot_dtype"),
        prefill_mg_enabled=prefill_mg_enabled,
        max_batch=caps.max_batch,
        max_kv_rows=caps.max_kv_rows,
        max_page_table_width=caps.max_page_table_width,
        max_chunks_per_row=caps.max_chunks_per_row,
        max_q_chunks=0 if caps.max_q_chunks is None else caps.max_q_chunks,
        physical_block_size=caps.page_size,
        physical_record_width=caps.cache_record_bytes,
    )


@dataclass(frozen=True)
class _SparseMlaState:
    caps: Caps
    layout: object
    query: SparseMlaQuery
    sm_count: int
    prepared: object | None = None

    def scratch_specs(self):
        return self.layout.scratch_specs()

    def bind_for_preparation(self, **kwargs) -> B12XSparseMLABinding:
        return self.layout.bind(**kwargs)

    def bind(self, **kwargs) -> B12XSparseMLABinding:
        return self.bind_for_preparation(**kwargs)

    def prime(self, binding: B12XSparseMLABinding, *, kv_cache, attention_sink=None):
        """Resolve every declared launcher from this state's exact live ABI."""
        if self.caps.mode == "decode":
            from b12x.attention._shared.mla.kernel import (
                compile_unified_decode_launch,
                prepare_unified_decode_launch,
            )

            prepared = prepare_unified_decode_launch(
                q_all=binding.q,
                swa_k_cache=kv_cache,
                swa_indices=binding.selected_indices,
                workspace=binding.scratch,
                swa_page_size=self.caps.page_size,
                sm_count=self.sm_count,
                attn_sink=attention_sink,
                scale_format_override=self.caps.scale_format,
                model_type_override=self.caps.model_type,
                fp8_rope_override=self.caps.fp8_rope,
                latent_scale_per_token=self.caps.latent_scale_per_token,
                return_lse=self.caps.return_lse,
                controls=FrozenMapping(),
            )
            prepared = compile_unified_decode_launch(
                prepared=prepared,
                q_all=binding.q,
                swa_k_cache=kv_cache,
                swa_indices=binding.selected_indices,
                swa_topk_lengths=binding.nsa_cache_seqlens_int32,
                workspace=binding.scratch,
                sm_scale=self.caps.softmax_scale,
                swa_page_size=self.caps.page_size,
                attn_sink=attention_sink,
                return_lse=self.caps.return_lse,
                lse_scale=self.caps.lse_scale,
            )
        else:
            from b12x._lib.compile_plan import compile_only_launches
            from b12x.attention._shared.mla.prefill import run_unified_prefill

            with compile_only_launches():
                prepared = run_unified_prefill(
                    q=binding.q,
                    kv_cache=kv_cache,
                    topk_indices=binding.selected_indices,
                    sm_scale=self.caps.softmax_scale,
                    page_block_size=self.caps.page_size,
                    topk_length=binding.nsa_cache_seqlens_int32,
                    model_type=self.caps.model_type,
                    scale_format=self.caps.scale_format,
                    fp8_rope=self.caps.fp8_rope,
                    latent_scale_per_token=self.caps.latent_scale_per_token,
                    traits_override=self.caps.cache_traits,
                    mg_enabled=self.query.prefill_mg_enabled,
                )
        object.__setattr__(self, "prepared", prepared)

    def run(self, binding: B12XSparseMLABinding, *, kv_cache, attention_sink=None):
        if self.prepared is None:
            # A plan materialized on first use resolves its launchers from its
            # first binding; under a kernel resolution guard this raises.
            self.prime(binding, kv_cache=kv_cache, attention_sink=attention_sink)
        if self.caps.mode == "decode":
            from b12x.attention._shared.mla.kernel import run_prepared_unified_decode

            return run_prepared_unified_decode(
                q_all=binding.q,
                swa_k_cache=kv_cache,
                swa_indices=binding.selected_indices,
                swa_topk_lengths=binding.nsa_cache_seqlens_int32,
                workspace=binding.scratch,
                sm_scale=self.caps.softmax_scale,
                swa_page_size=self.caps.page_size,
                attn_sink=attention_sink,
                return_lse=self.caps.return_lse,
                lse_scale=self.caps.lse_scale,
                prepared=self.prepared,
            )
        from b12x.attention._shared.mla.prefill import run_unified_prefill

        return run_unified_prefill(
            q=binding.q,
            kv_cache=kv_cache,
            topk_indices=binding.selected_indices,
            sm_scale=self.caps.softmax_scale,
            page_block_size=self.caps.page_size,
            topk_length=binding.nsa_cache_seqlens_int32,
            output=(
                binding.scratch.output_buffer[: binding.q.shape[0]]
                if getattr(binding.scratch, "output_buffer", None) is not None
                else None
            ),
            lse_out=(
                binding.scratch.final_lse[: binding.q.shape[0], : binding.q.shape[1]]
                if self.caps.return_lse
                and getattr(binding.scratch, "final_lse", None) is not None
                else None
            ),
            model_type=self.caps.model_type,
            scale_format=self.caps.scale_format,
            fp8_rope=self.caps.fp8_rope,
            latent_scale_per_token=self.caps.latent_scale_per_token,
            traits_override=self.caps.cache_traits,
            prepared=self.prepared,
            mg_enabled=self.query.prefill_mg_enabled,
        )


def plan(
    caps: Caps,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: SparseMlaConfig | None = None,
) -> Plan:
    """Declare one complete sparse-MLA route without allocating or compiling."""

    if not isinstance(caps, Caps):
        raise TypeError("plan requires sparse MLA Caps")
    invocation = FrozenMapping(invocation)
    query = _query_from_caps(caps, invocation)
    layout = plan_sparse_mla_scratch(caps)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.attention.sparse_mla._preparation:compile_sparse_mla",
                caps,
                device.ordinal,
                query.prefill_mg_enabled,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(
            scratch=layout.scratch_specs()
        ),
        _materialize=lambda selection, device: _SparseMlaState(
            caps, layout, query, device.identity.sm_count
        ),
        _device=caps.device,
    )


def _validate_tail_fp8_writer_args(kv_c, kv_cache, slot_mapping):
    """Validate the fp8-tail writer contract (fp8 content, kvarn pool stride).

    Mirrors ``_validate_glm_next_mla_cache_writer_args`` but accepts the
    kvarn pool's wider record stride (>= 528 B) instead of the fixed
    528/304 whitelist: the stock GLM writer addresses records with a
    runtime entry_stride, so only the whitelist rejected the 536 B pool.
    """
    from b12x.attention._shared.mla import kv_cache as writer

    if kv_c.ndim != 2 or int(kv_c.shape[1]) != writer._GLM_NEXT_LATENT_DIM:
        raise ValueError(
            "kv_c must be (num_tokens, "
            f"{writer._GLM_NEXT_LATENT_DIM}), got {tuple(kv_c.shape)}"
        )
    if kv_c.dtype != torch.bfloat16:
        raise TypeError(f"kv_c must be BF16, got {kv_c.dtype}")
    record_bytes = int(kv_cache.shape[2]) if kv_cache.ndim == 3 else -1
    if kv_cache.ndim != 3 or record_bytes < writer._GLM_NEXT_RECORD_BYTES:
        raise ValueError(
            "fp8-tail kv_cache must be (num_pages, page_size, "
            f">={writer._GLM_NEXT_RECORD_BYTES}) uint8, got {tuple(kv_cache.shape)}"
        )
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if slot_mapping.ndim != 1 or slot_mapping.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "slot_mapping must be a 1-D int32 or int64 tensor, got "
            f"{tuple(slot_mapping.shape)} {slot_mapping.dtype}"
        )
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
    if int(kv_c.shape[0]) < int(slot_mapping.shape[0]):
        raise ValueError(
            f"kv_c must cover slot_mapping's {int(slot_mapping.shape[0])} tokens, got "
            f"{int(kv_c.shape[0])} rows"
        )
    if kv_c.stride(1) != 1:
        raise ValueError("kv_c rows must be innermost-contiguous")
    if kv_cache.stride(2) != 1 or kv_cache.stride(1) != record_bytes:
        raise ValueError("kv_cache must have packed semantic records")
    if not (kv_c.is_cuda and kv_cache.is_cuda and slot_mapping.is_cuda):
        raise ValueError("all tensors must be on CUDA")
    if len({kv_c.device, kv_cache.device, slot_mapping.device}) != 1:
        raise ValueError("all tensors must be on the same device")


@dataclass(frozen=True)
class _WriterState:
    kv_cache: torch.Tensor
    slot_mapping: torch.Tensor
    compiled: object
    nvfp4: bool
    k_bits: int = 0
    v_bits: int = 0
    tail_fp8: bool = False

    def run(self, kv_c, kv_cache, slot_mapping):
        from b12x.attention._shared.mla import kv_cache as writer
        from b12x.attention._shared.mla._kvarn_writer import (
            concat_and_cache_glm_next_mla_kvarn,
        )

        if self.k_bits:
            concat_and_cache_glm_next_mla_kvarn(
                kv_c,
                kv_cache,
                slot_mapping,
                k_bits=self.k_bits,
                v_bits=self.v_bits,
                group=256,
            )
            return
        if self.tail_fp8:
            # fp8 content into the kvarn pool: the stock GLM writer takes
            # entry_stride as a runtime Int64, so only the record-width
            # whitelist in _validate_* rejects the wider pool.
            _validate_tail_fp8_writer_args(kv_c, kv_cache, slot_mapping)
            if int(kv_cache.shape[1]) != int(self.kv_cache.shape[1]):
                raise ValueError("cache-writer page size differs from preparation")
            if slot_mapping.dtype != self.slot_mapping.dtype:
                raise ValueError("cache-writer slot dtype differs from preparation")
            _, args, _ = writer._glm_next_cache_writer_launch(
                kv_c, kv_cache, slot_mapping
            )
            writer.run_compiled(self.compiled, args)
            return
        writer._validate_glm_next_mla_cache_writer_args(kv_c, kv_cache, slot_mapping)
        if (int(kv_cache.shape[-1]) == 304) != self.nvfp4:
            raise ValueError("cache-writer record format differs from preparation")
        if int(kv_cache.shape[1]) != int(self.kv_cache.shape[1]):
            raise ValueError("cache-writer page size differs from preparation")
        if slot_mapping.dtype != self.slot_mapping.dtype:
            raise ValueError("cache-writer slot dtype differs from preparation")
        if self.nvfp4:
            _, args, _ = writer._nvfp4_mla_writer_launch(
                kv_c, kv_c, kv_cache, slot_mapping, True, False
            )
        else:
            _, args, _ = writer._glm_next_cache_writer_launch(
                kv_c, kv_cache, slot_mapping
            )
        writer.run_compiled(self.compiled, args)


def plan_cache_writer(
    kv_c,
    kv_cache,
    slot_mapping,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: SparseMlaConfig | None = None,
) -> Plan:
    """Declare the exact GLM_NEXT paged cache-writer specialization."""
    from b12x.attention._shared.mla import kv_cache as writer

    record_bytes = int(kv_cache.shape[-1])
    # KVarN dual-rate / single-plane records: 536/648/600/400/272B per token
    # are unambiguous vs the fp8 (528) / nvfp4 (304) GLM_NEXT records; k/v
    # bits follow the variant (shared map with traits, incl. v_bits=0 for
    # the single-plane record). The VarN tile is 256 records; the manager
    # page may hold several tiles (page_size = k * 256).
    kvarn = (
        KVARN_RECORD_VARIANTS.get(record_bytes)
        if int(kv_cache.shape[1]) % 256 == 0
        else None
    )
    if not kvarn:
        writer._validate_glm_next_mla_cache_writer_args(kv_c, kv_cache, slot_mapping)
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("cache-writer invocation is encoded by its tensor formats")
    query = SparseMlaQuery(
        mode="writer",
        dtype=_dtype_name(kv_c.dtype),
        kv_dtype=_dtype_name(kv_cache.dtype),
        num_q_heads=0,
        qk_head_dim=int(kv_c.shape[1]),
        v_head_dim=0,
        max_q_rows=int(kv_c.shape[0]),
        max_width=0,
        page_size=int(kv_cache.shape[1]),
        model_type=2,
        head_major_output=False,
        scale_format=2 if record_bytes == 304 else 1,
        cache_record_bytes=record_bytes,
        fp8_rope=False,
        latent_scale_per_token=record_bytes == 304,
        has_attention_sink=False,
        cache_layout="paged_strided",
        operation="cache_writer",
        slot_dtype=_dtype_name(slot_mapping.dtype),
        prefill_mg_enabled=False,
        max_batch=int(kv_c.shape[0]),
        max_page_table_width=int(kv_cache.shape[0]),
        physical_block_size=int(kv_cache.shape[1]),
        physical_record_width=record_bytes,
        num_cache_blocks=int(kv_cache.shape[0]),
        max_physical_records=int(kv_cache.shape[0]) * int(kv_cache.shape[1]),
    )
    payload = {
        "kv_c_shape": tuple(kv_c.shape),
        "kv_c_dtype": _dtype_name(kv_c.dtype),
        "cache_shape": tuple(kv_cache.shape),
        "cache_dtype": _dtype_name(kv_cache.dtype),
        "slots_shape": tuple(slot_mapping.shape),
        "slots_dtype": _dtype_name(slot_mapping.dtype),
        "nvfp4": record_bytes == 304,
    }
    if kvarn:
        import os as _os

        if _os.environ.get("B12X_KVARN_DEBUG"):
            print(
                "[kvarn] writer plan: kv_cache",
                tuple(kv_cache.shape),
                "strides",
                tuple(kv_cache.stride()),
                "record",
                record_bytes,
                "slots",
                tuple(slot_mapping.shape),
                flush=True,
            )
        payload["kvarn"] = True
        payload["k_bits"], payload["v_bits"] = kvarn
        payload["group"] = 256

    def materialize(selection, device):
        if kvarn:
            from b12x.attention._shared.mla._kvarn_writer import (
                _compile_kvarn_writer,
            )

            compiled = _compile_kvarn_writer(
                kv_c, kv_cache, slot_mapping, kvarn[0], kvarn[1], 256
            )
            return _WriterState(
                kv_cache,
                slot_mapping,
                compiled,
                False,
                k_bits=kvarn[0],
                v_bits=kvarn[1],
            )
        if record_bytes == 304:
            compiled = writer._compile_nvfp4_mla_writer(
                kv_c, kv_c, kv_cache, slot_mapping, True, False
            )
        else:
            compiled = writer._compile_glm_next_mla_cache_writer(
                kv_c, kv_cache, slot_mapping
            )
        return _WriterState(kv_cache, slot_mapping, compiled, record_bytes == 304)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.attention.sparse_mla._preparation:compile_cache_writer",
                payload,
                device.ordinal,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
        _device=kv_c.device,
    )


def plan_cache_writer_fp8_tail(
    kv_c,
    kv_cache,
    slot_mapping,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: SparseMlaConfig | None = None,
) -> Plan:
    """Declare the fp8-tail cache-writer: fp8 content into a kvarn pool.

    The stock GLM fp8 writer takes entry_stride as a runtime Int64, so it
    writes correct 528 B-content records into a 536 B-stride kvarn pool with
    no kernel change. The plan identity (query operation + payload) is
    distinct from both the kvarn body plan and the stock fp8 plan, so a
    kvarn-pool tail plan and a kvarn-pool body plan coexist as separate
    prepared plans without compile-cache collisions.
    """
    from b12x.attention._shared.mla import kv_cache as writer

    record_bytes = int(kv_cache.shape[-1])
    if record_bytes < writer._GLM_NEXT_RECORD_BYTES:
        raise ValueError(
            "fp8-tail pool records must be at least "
            f"{writer._GLM_NEXT_RECORD_BYTES} bytes, got {record_bytes}"
        )
    _validate_tail_fp8_writer_args(kv_c, kv_cache, slot_mapping)
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("cache-writer invocation is encoded by its tensor formats")
    query = SparseMlaQuery(
        mode="writer",
        dtype=_dtype_name(kv_c.dtype),
        kv_dtype=_dtype_name(kv_cache.dtype),
        num_q_heads=0,
        qk_head_dim=int(kv_c.shape[1]),
        v_head_dim=0,
        max_q_rows=int(kv_c.shape[0]),
        max_width=0,
        page_size=int(kv_cache.shape[1]),
        model_type=2,
        head_major_output=False,
        scale_format=1,
        cache_record_bytes=record_bytes,
        fp8_rope=False,
        latent_scale_per_token=False,
        has_attention_sink=False,
        cache_layout="paged_strided",
        operation="cache_writer_fp8_tail",
        slot_dtype=_dtype_name(slot_mapping.dtype),
        prefill_mg_enabled=False,
        max_batch=int(kv_c.shape[0]),
        max_page_table_width=int(kv_cache.shape[0]),
        physical_block_size=int(kv_cache.shape[1]),
        physical_record_width=record_bytes,
        num_cache_blocks=int(kv_cache.shape[0]),
        max_physical_records=int(kv_cache.shape[0]) * int(kv_cache.shape[1]),
    )
    payload = {
        "kv_c_shape": tuple(kv_c.shape),
        "kv_c_dtype": _dtype_name(kv_c.dtype),
        "cache_shape": tuple(kv_cache.shape),
        "cache_dtype": _dtype_name(kv_cache.dtype),
        "slots_shape": tuple(slot_mapping.shape),
        "slots_dtype": _dtype_name(slot_mapping.dtype),
        "tail_fp8": True,
        "nvfp4": False,
    }

    def materialize(selection, device):
        compiled = writer._compile_glm_next_mla_cache_writer(
            kv_c, kv_cache, slot_mapping
        )
        return _WriterState(kv_cache, slot_mapping, compiled, False, tail_fp8=True)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.attention.sparse_mla._preparation:compile_cache_writer",
                payload,
                device.ordinal,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
        _device=kv_c.device,
    )


def state(plan, *, device=None) -> _SparseMlaState:
    return require_prepared(plan, TUNING.component_id, device)


def bind(plan, **kwargs) -> B12XSparseMLABinding:
    """Bind caller-owned storage to a published plan only."""
    return state(plan).bind_for_preparation(**kwargs)


def run(
    *,
    plan,
    binding: B12XSparseMLABinding,
    kv_cache=None,
    attention_sink=None,
):
    """Run the resident launchers held by a published plan."""
    prepared_state = state(plan, device=binding.q.device)
    return prepared_state.run(
        binding,
        kv_cache=binding.kv_cache if kv_cache is None else kv_cache,
        attention_sink=attention_sink,
    )


def writer_state(plan, *, device=None) -> _WriterState:
    return require_prepared(plan, TUNING.component_id, device)


__all__ = [
    "bind",
    "compile_sparse_mla",
    "plan",
    "plan_cache_writer",
    "plan_cache_writer_fp8_tail",
    "run",
    "state",
    "writer_state",
]
