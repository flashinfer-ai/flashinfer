"""Metadata-only compiler extraction and immutable HyperConnection lowering."""
from __future__ import annotations

from functools import partial

import torch

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.compiler import run_compiled
from b12x._lib.utils import current_cuda_stream
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._tuning import HyperConnectionConfig, HyperConnectionQuery, TUNING


def _decode_query(payload):
    values = dict(payload)
    if values["limit"] == "+inf":
        values["limit"] = float("inf")
    return HyperConnectionQuery(**values)


@program_cache(scope="preparation")
def compile_hyperconnection(query_payload, config_payload, ordinal):
    """Return the exact production programs without allocating tensor storage."""
    from . import _cute as native
    from . import _kernels as kernels

    query = _decode_query(query_payload)
    config = HyperConnectionConfig.from_config(FrozenMapping(config_payload))
    role, streams, hidden = query.operation, query.streams, query.hidden_size
    device = torch.device("cuda", ordinal)
    if role == "grouped_rmsnorm" and not query.zero_centered:
        weight_dtype = getattr(torch, query.weight_dtype)
        return native._compile(
            ("rmsnorm", ordinal, streams, hidden, str(weight_dtype)),
            native._GroupedRmsNorm(streams, hidden, False), 3, device,
            has_eps=True, runtime_ints=1,
            pointer_dtypes=(torch.bfloat16, weight_dtype, torch.bfloat16),
        )
    if role == "grouped_rmsnorm":
        return kernels._grouped_rmsnorm_kernel.warmup(
            torch.bfloat16, torch.bfloat16, torch.bfloat16, query.eps,
            HIDDEN_SIZE=hidden, STREAMS=streams, BLOCK_H=config.reduction_block_h,
            num_warps=config.reduction_num_warps, grid=(query.max_tokens * streams,),
        )
    if role == "gate_mean":
        import triton
        return kernels._gate_mean_kernel.warmup(
            torch.bfloat16, torch.bfloat16, torch.bfloat16,
            HIDDEN_SIZE=hidden, STREAMS=streams, BLOCK_H=config.pointwise_block,
            num_warps=4, grid=(query.max_tokens, triton.cdiv(hidden, config.pointwise_block)),
        )
    if role == "scaled_silu":
        return native._compile(
            ("silu", ordinal, streams, query.lowrank), native._ScaledSilu(streams, query.lowrank),
            2, device, runtime_ints=1, runtime_int64s=1,
        )
    if role == "combine":
        return native._compile(
            ("combine", ordinal, streams, hidden), native._Combine(streams, hidden),
            4, device, runtime_ints=1, runtime_int64s=1,
        )
    if role == "combine_norm":
        packed = hidden % native._PackedCombineNorm._PACK_ELEMENTS == 0
        cls = native._PackedCombineNorm if packed else native._CombineNorm
        return native._compile(
            ("combine_norm_packed" if packed else "combine_norm", ordinal, streams, hidden),
            cls(streams, hidden), 6, device, has_eps=True, runtime_ints=1, runtime_int64s=1,
        )
    if role == "engram_mix":
        return native._compile(
            ("engram_mix", ordinal, streams, hidden, query.token_mask),
            native._EngramMix(streams, hidden, query.token_mask), 5, device,
            has_eps=True, runtime_ints=1,
            pointer_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32,
                            torch.bool if query.token_mask else torch.bfloat16, torch.bfloat16),
        )
    if role in ("swiglu", "add", "sigmoid"):
        dtypes = tuple(getattr(torch, name) for name in (query.left_dtype, query.right_dtype, query.output_dtype))
        width = hidden if role == "swiglu" else 1
        key = (role, ordinal, *(str(dtype) for dtype in dtypes), width,
               None if query.limit == float("inf") else query.limit, query.round_silu)
        return native._compile(
            key, native._Pointwise(role, native._POINTER_DTYPES[dtypes[2]], width, query.limit, query.round_silu),
            3, device, runtime_int64s=1, pointer_dtypes=dtypes,
        )
    raise ValueError(f"unknown HyperConnection operation {role!r}")


def _launcher(query, config, compiled):
    from . import _cute as native
    from . import _kernels as kernels

    role = query.operation
    if role == "grouped_rmsnorm" and query.zero_centered:
        def invoke(*tensors, eps=None):
            kernels._norm_launch(*tensors, query.eps, query.streams, query.hidden_size,
                                 config.reduction_block_h, config.reduction_num_warps)
    elif role == "gate_mean":
        def invoke(*tensors, eps=None):
            kernels._gate_mean_launch(*tensors, query.streams, query.hidden_size, config.pointwise_block)
    else:
        if role == "scaled_silu":
            runtime = lambda tensors: (tensors[0].numel(), tensors[0].stride(0))
        elif role in ("combine", "combine_norm"):
            runtime = lambda tensors: (tensors[0].shape[0], tensors[2].stride(0))
        elif role == "grouped_rmsnorm":
            runtime = lambda tensors: (tensors[0].shape[0] * query.streams,)
        elif role == "engram_mix":
            runtime = lambda tensors: (tensors[0].shape[0],)
        else:
            runtime = lambda tensors: (tensors[-1].numel(),)
        has_eps = role in ("grouped_rmsnorm", "engram_mix", "combine_norm")

        def invoke(*tensors, eps=None):
            args = [native._pointer(tensor) for tensor in tensors]
            if has_eps:
                args.append(query.eps)
            args.extend(int(value) for value in runtime(tensors))
            args.append(current_cuda_stream())
            run_compiled(compiled, tuple(args))
    return attach_programs(invoke, compiled)


def _materialize(caps, query, selection, device):
    from ._impl import _HyperConnectionState
    compiled = compile_hyperconnection(TUNING.encode_query(query), selection.config.to_dict(), device.ordinal)
    return _HyperConnectionState(
        caps=caps, query=query, config=selection.config,
        launch=_launcher(query, selection.config, compiled),
    )


def plan_hyperconnection(caps, *, invocation=FrozenMapping(), override=None):
    from ._impl import HyperConnectionCaps
    if not isinstance(caps, HyperConnectionCaps):
        raise TypeError("caps must be HyperConnectionCaps")
    values = dict(invocation)
    if values.get("limit") == float("inf"):
        values["limit"] = "+inf"
    invocation = FrozenMapping(values)
    fields = {
        "operation", "output_mode", "left_dtype", "right_dtype", "output_dtype",
        "token_mask", "eps", "limit", "round_silu", "zero_centered", "weight_dtype",
    }
    if set(invocation) - fields or "operation" not in invocation:
        raise ValueError("HyperConnection declarations require one explicit operation and known invocation fields")
    options = invocation.to_dict()
    if options.get("limit") == "+inf":
        options["limit"] = float("inf")
    options.setdefault("output_mode", "functional" if options["operation"] in ("combine", "combine_norm") else "provided")
    query = HyperConnectionQuery(
        dtype=str(caps.dtype).removeprefix("torch."), max_tokens=caps.max_tokens,
        hidden_size=caps.hidden_size, streams=caps.streams, lowrank=caps.lowrank, **options,
    )
    return Plan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        shared=True, _device=caps.device,
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.norm.hyperconnection._preparation:compile_hyperconnection",
            TUNING.encode_query(query), config.to_dict(), device.ordinal,
        ),),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=partial(_materialize, caps, query),
    )
