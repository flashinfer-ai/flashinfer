"""Prepared native Trellis dense declarations and dispatch."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    current_plan,
)
from b12x.moe._shared.kernels.w4a16.prepare import PreparedTrellis256DenseWeight
from ._tuning import TUNING, TrellisConfig, TrellisQuery

_INTERNAL_C_TMPS = weakref.WeakValueDictionary()


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise TypeError("Trellis only supports fp16 and bf16 activations")


def query_from_weight(
    weight: PreparedTrellis256DenseWeight,
    *,
    max_rows: int,
    input_dtype: torch.dtype,
    output_mode: str = "functional",
    c_tmp_mode: str = "internal",
    hadamard_128=None,
    gemm_output_provided: bool = False,
    input_f16_provided: bool = False,
    rotated_f16_provided: bool = False,
    rotated_compute_provided: bool = False,
    gemm_output_f16_provided: bool = False,
    output_f16_provided: bool = False,
) -> TrellisQuery:
    """Describe every fixed dense-Trellis execution semantic before preparation."""
    if not isinstance(weight, PreparedTrellis256DenseWeight):
        raise TypeError("Trellis declarations require PreparedTrellis256DenseWeight")
    pair_kind = weight.trellis_pair_kind
    rate_axis = weight.trellis_rate_axis
    layout = "native" if pair_kind is None else f"{str(pair_kind).lower()}_{rate_axis}"
    dtype = _dtype_name(input_dtype)
    return TrellisQuery(
        max_rows=int(max_rows),
        in_features=int(weight.in_features),
        out_features=int(weight.out_features),
        input_dtype=dtype,
        compute_dtype=_dtype_name(weight.params_dtype),
        output_dtype=dtype,
        codebook=str(weight.trellis_codebook),
        bits=int(weight.trellis_bits),
        weight_layout=layout,
        pair_kind=pair_kind,
        rate_axis=rate_axis,
        output_mode=output_mode,
        c_tmp_mode=c_tmp_mode,
        transform_mode="supplied" if hadamard_128 is not None else "extension",
        gemm_output_provided=gemm_output_provided,
        input_f16_provided=input_f16_provided,
        rotated_f16_provided=rotated_f16_provided,
        rotated_compute_provided=rotated_compute_provided,
        gemm_output_f16_provided=gemm_output_f16_provided,
        output_f16_provided=output_f16_provided,
    )


@program_cache(scope="preparation")
def _compile_trellis(query_payload, config_payload, ordinal, sm_count):
    """Resolve exactly the native dense GEMM program selected by the session."""
    from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_gemm

    query = TrellisQuery(**dict(query_payload))
    config = TrellisConfig.from_config(FrozenMapping(config_payload))
    element_dtype = "fp16" if query.compute_dtype == "float16" else "bf16"
    route_blocks = (query.max_rows + config.block_rows - 1) // config.block_rows
    with torch.cuda.device(ordinal):
        return {
            "gemm": compile_w4a16_gemm(
                size_m=query.max_rows,
                size_n=query.out_features,
                size_k=query.in_features,
                num_experts=1,
                top_k=1,
                mul_topk_weights=False,
                tile_n=config.tile_n,
                tile_k=config.tile_k,
                moe_block_size=config.block_rows,
                max_m_blocks=route_blocks,
                element_dtype=element_dtype,
                weight_layout="trellis_t256",
                scale_format="e4m3_k32",
                w13_layout="packed",
                trellis_bits=query.bits,
                trellis_codebook=query.codebook,
                trellis_pair_kind=query.pair_kind,
                trellis_rate_axis=query.rate_axis,
                dense_route_fast_path=True,
            )
        }


def _c_tmp_elements(query: TrellisQuery, config: TrellisConfig, sm_count: int) -> int:
    from b12x.moe._shared.kernels.w4a16.kernel import packed_gemm_scratch_elements

    route_slots = (
        (query.max_rows + config.block_rows - 1) // config.block_rows
    ) * config.block_rows
    return packed_gemm_scratch_elements(
        size_n=query.out_features,
        route_slots=route_slots,
        moe_block_size=config.block_rows,
        sms=sm_count,
    )


def _lut_memory(query: TrellisQuery, device) -> PersistentMemory | None:
    if query.codebook == "mcg":
        return None
    resolved = torch.device("cuda", device.ordinal)
    if query.codebook == "lut_e4m3":
        from b12x._lib.quant.lut_e4m3 import lut_e4m3_value_table_resident

        resident = lut_e4m3_value_table_resident(resolved)
        key = ("lut_e4m3_value_table", resolved.type, resolved.index)
        required = 1 << 12
    else:
        from b12x._lib.quant.lut_fp16 import (
            LUT_FP16_SEGMENT_TABLE_BYTES,
            lut_fp16_segment_table_resident,
        )

        resident = lut_fp16_segment_table_resident(resolved)
        key = ("lut_fp16_segment_table", resolved.type, resolved.index)
        required = LUT_FP16_SEGMENT_TABLE_BYTES
    resident_nbytes = (
        resident.numel() * resident.element_size() if resident is not None else 0
    )
    return PersistentMemory(key, required, resident_nbytes)


@dataclass(frozen=True)
class _TrellisExecutionState:
    query: TrellisQuery
    weight: PreparedTrellis256DenseWeight
    device: torch.device
    launch: object
    execution_lut: torch.Tensor | None
    hadamard_128: object
    grid_cap: int
    c_tmp_owner: tuple[object, ...] | None

    def run(
        self,
        x,
        *,
        output=None,
        gemm_output=None,
        input_f16=None,
        rotated_f16=None,
        rotated_compute=None,
        gemm_output_f16=None,
        output_f16=None,
    ):
        from b12x.moe._shared.kernels.w4a16.kernel import run_trellis256_dense

        if not isinstance(x, torch.Tensor) or x.device != self.device:
            raise ValueError("Trellis source device differs from prepared plan")
        if _dtype_name(x.dtype) != self.query.input_dtype:
            raise ValueError("Trellis source dtype differs from preparation")
        if (
            x.ndim != 2
            or int(x.shape[0]) != self.query.max_rows
            or int(x.shape[1]) != self.query.in_features
        ):
            raise ValueError("Trellis plan requires its exact planned [M, K] input")
        if (output is None) != (self.query.output_mode == "functional"):
            raise ValueError("Trellis output mode differs from preparation")
        for name in (
            "gemm_output",
            "input_f16",
            "rotated_f16",
            "rotated_compute",
            "gemm_output_f16",
            "output_f16",
        ):
            if (locals()[name] is not None) != getattr(self.query, f"{name}_provided"):
                raise ValueError(
                    f"Trellis {name} workspace form differs from preparation"
                )
        return run_trellis256_dense(
            x,
            self.weight,
            launch=self.launch,
            execution_lut=self.execution_lut,
            grid_cap=self.grid_cap,
            output=output,
            gemm_output=gemm_output,
            input_f16=input_f16,
            rotated_f16=rotated_f16,
            rotated_compute=rotated_compute,
            gemm_output_f16=gemm_output_f16,
            output_f16=output_f16,
            hadamard_128=self.hadamard_128,
        )


def plan(
    query: TrellisQuery,
    *,
    weight: PreparedTrellis256DenseWeight,
    invocation=FrozenMapping(),
    override=None,
    c_tmp=None,
    hadamard_128=None,
) -> Plan:
    """Declare an exact-M Trellis plan; all code is loaded by the session."""
    if not isinstance(query, TrellisQuery):
        raise TypeError("Trellis plan requires TrellisQuery")
    if not isinstance(weight, PreparedTrellis256DenseWeight):
        raise TypeError("Trellis plan requires PreparedTrellis256DenseWeight")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("Trellis invocation semantics belong in TrellisQuery")
    if hadamard_128 is not None:
        supplied = (
            hadamard_128
            if callable(hadamard_128)
            else getattr(hadamard_128, "had_r_128", None)
        )
        if not callable(supplied):
            raise TypeError("hadamard_128 must be callable or expose had_r_128")
    if query.transform_mode != (
        "supplied" if hadamard_128 is not None else "extension"
    ):
        raise ValueError(
            "Trellis transform metadata differs from the supplied transform"
        )
    if query.c_tmp_mode != ("provided" if c_tmp is not None else "internal"):
        raise ValueError("Trellis c_tmp metadata differs from the supplied workspace")
    if (
        query.in_features,
        query.out_features,
        query.compute_dtype,
        query.codebook,
        query.bits,
        query.pair_kind,
        query.rate_axis,
    ) != (
        weight.in_features,
        weight.out_features,
        _dtype_name(weight.params_dtype),
        weight.trellis_codebook,
        weight.trellis_bits,
        weight.trellis_pair_kind,
        weight.trellis_rate_axis,
    ):
        raise ValueError("Trellis declaration metadata differs from prepared weight")

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.gemm.trellis_linear._preparation:_compile_trellis",
                TUNING.encode_query(query),
                TUNING.encode_config(config),
                device.ordinal,
                device.identity.sm_count,
            ),
        )

    def memory(config, device):
        elements = _c_tmp_elements(query, config, device.identity.sm_count)
        specs = []
        persistent = []
        if c_tmp is not None:
            if (
                c_tmp.device != torch.device("cuda", device.ordinal)
                or c_tmp.dtype != torch.float32
                or not c_tmp.is_contiguous()
                or int(c_tmp.data_ptr()) % 16
                or c_tmp.numel() < elements
            ):
                raise ValueError(
                    "provided Trellis c_tmp cannot cover the selected launch"
                )
            specs.append(
                scratch_buffer_spec(
                    "trellis_linear.c_tmp", nbytes=elements * 4, device=c_tmp.device
                )
            )
        else:
            owner = ("gemm.trellis_linear.c_tmp", current_plan(), device.ordinal)
            resident = _INTERNAL_C_TMPS.get(owner)
            resident_nbytes = (
                resident.numel() * resident.element_size()
                if resident is not None
                else 0
            )
            persistent.append(PersistentMemory(owner, elements * 4, resident_nbytes))
        lut_memory = _lut_memory(query, device)
        if lut_memory is not None:
            persistent.append(lut_memory)
        sizes = {
            "gemm_output": query.max_rows * query.out_features * 2,
            "input_f16": query.max_rows * query.in_features * 2,
            "rotated_f16": query.max_rows * query.in_features * 2,
            "rotated_compute": query.max_rows * query.in_features * 2,
            "gemm_output_f16": query.max_rows * query.out_features * 2,
            "output_f16": query.max_rows * query.out_features * 2,
        }
        for name, nbytes in sizes.items():
            if getattr(query, f"{name}_provided"):
                specs.append(
                    scratch_buffer_spec(
                        f"trellis_linear.{name}",
                        nbytes=nbytes,
                        device=torch.device("cuda", device.ordinal),
                    )
                )
        return MemoryRequirements(tuple(specs), tuple(persistent))

    def materialize(selection, device):
        from b12x.moe._shared.kernels.w4a16.kernel import (
            _W4A16GemmLaunch,
            _resolve_exl3_hadamard_128,
            _trellis256_execution_lut,
        )

        config = selection.config
        programs = _compile_trellis(
            TUNING.encode_query(query),
            TUNING.encode_config(config),
            device.ordinal,
            device.identity.sm_count,
        )
        elements = _c_tmp_elements(query, config, device.identity.sm_count)
        resolved_device = torch.device("cuda", device.ordinal)
        c_tmp_owner = None
        if c_tmp is None:
            c_tmp_owner = (
                "gemm.trellis_linear.c_tmp",
                current_plan(),
                device.ordinal,
            )
            scratch = _INTERNAL_C_TMPS.get(c_tmp_owner)
            if scratch is None or scratch.numel() < elements:
                scratch = torch.empty(
                    elements, dtype=torch.float32, device=resolved_device
                )
                _INTERNAL_C_TMPS[c_tmp_owner] = scratch
        else:
            scratch = c_tmp
        launch = _W4A16GemmLaunch(kernel=programs["gemm"], c_tmp=scratch)
        lut = (
            None
            if query.codebook == "mcg"
            else _trellis256_execution_lut(resolved_device, query.codebook)
        )
        return _TrellisExecutionState(
            query=query,
            weight=weight,
            device=resolved_device,
            launch=launch,
            execution_lut=lut,
            hadamard_128=_resolve_exl3_hadamard_128(hadamard_128),
            grid_cap=device.identity.sm_count * int(programs["gemm"].blocks_per_sm),
            c_tmp_owner=c_tmp_owner,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=weight.trellis.device,
    )
