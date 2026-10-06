"""Supporting activation packing into the dense GEMM block-scaled layout."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen


@triton.jit
def _scale_offset(row, group, GROUPS: tl.constexpr):
    return (
        (row // 128 * triton.cdiv(GROUPS, 4) + group // 4) * 512
        + row % 32 * 16
        + row // 32 % 4 * 4
        + group % 4
    )


@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
def _quantize(
    X, Q, S, AG, WG, ALPHA, M,
    INPUT_K: tl.constexpr, K: tl.constexpr,
    FP4: tl.constexpr, RECIPROCAL: tl.constexpr,
    GROUP: tl.constexpr, CHUNKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * CHUNKS + tl.arange(0, CHUNKS)
    k = group[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(X + row * INPUT_K + k, (row < M) & (k < INPUT_K), 0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    if FP4:
        g = tl.load(AG)
        sf = (amax * (1.0 / 6.0) * g).to(tl.float8e4nv)
        scale = sf.to(tl.float32)
        effective_scale = scale * tl.div_rn(1.0, g)
        inverse = tl.div_rn(1.0, effective_scale)
        inverse = tl.where(effective_scale == 0, 0.0, inverse)
        z = tl.minimum(tl.abs(x * inverse[:, None]), 6.0)
        # Round to nearest, ties to even E2M1 code.
        code = (z > 0.25).to(tl.uint8)
        code += (z >= 0.75).to(tl.uint8)
        code += (z > 1.25).to(tl.uint8)
        code += (z >= 1.75).to(tl.uint8)
        code += (z > 2.5).to(tl.uint8)
        code += (z >= 3.5).to(tl.uint8)
        code += (z > 5.0).to(tl.uint8)
        code |= tl.where((x < 0) & (code != 0), 8, 0).to(tl.uint8)
        pair = tl.reshape(code, (CHUNKS, GROUP // 2, 2))
        lo, hi = tl.split(pair)
        packed = lo | (hi << 4)
        byte = group[:, None] * (GROUP // 2) + tl.arange(0, GROUP // 2)[None, :]
        tl.store(Q + row * (K // 2) + byte, packed, (row < M) & (byte < K // 2))
        sf_byte = sf.to(tl.uint8, bitcast=True)
        if (row == 0) & (tl.program_id(1) == 0):
            weight_scale = tl.load(WG)
            if RECIPROCAL:
                weight_scale = tl.div_rn(1.0, weight_scale)
            tl.store(ALPHA, tl.div_rn(weight_scale, g))
    else:
        ratio = tl.where(amax > 0, amax / 448.0, 1.0)
        exponent = tl.minimum(tl.maximum(tl.ceil(tl.log2(ratio)), -127.0), 127.0)
        scale = tl.exp2(exponent)
        sf_byte = (exponent + 127).to(tl.uint8)
        values = (x / scale[:, None]).to(tl.float8e4nv)
        tl.store(Q + row * K + k, values, (row < M) & (k < K))
        if (row == 0) & (tl.program_id(1) == 0):
            tl.store(ALPHA, 1.0)
    tl.store(S + _scale_offset(row, group, K // GROUP), sf_byte,
             group < K // GROUP)
    if row == 0:
        pad = (M // 128).to(tl.int64) * 128 + tl.arange(0, 128)
        tl.store(S + _scale_offset(pad[:, None], group[None, :], K // GROUP),
                 0 if FP4 else 127,
                 (pad[:, None] >= M) & (pad[:, None] < ((M + 127) // 128) * 128)
                 & (group[None, :] < K // GROUP))


_COMPILED: dict[tuple, object] = {}


def launch(kernel, args, constants, grid, *, device, num_warps=4, num_stages=3):
    """Resolve by static geometry, then launch the exact compiled callable."""
    key = (kernel.__name__, device.index, tuple(
        arg.dtype if isinstance(arg, torch.Tensor) else type(arg) for arg in args
    ), tuple(constants.items()), num_warps, num_stages)
    compiled = _COMPILED.get(key)
    all_args = (*args, *constants.values())
    if compiled is None:
        raise_if_kernel_resolution_frozen("triton.compile", target=kernel, cache_key=key)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("blockscaled kernel must be prewarmed before CUDA graph capture")
        compiled = kernel[grid](*all_args, num_warps=num_warps, num_stages=num_stages,
                                enable_fp_fusion=False)
        if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
            raise RuntimeError("blockscaled kernels require zero implicit Triton launch scratch")
        _COMPILED[key] = compiled
    else:
        compiled[grid](*all_args)
