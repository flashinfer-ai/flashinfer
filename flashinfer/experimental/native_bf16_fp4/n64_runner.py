# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile 64-column canonical weight staging for small-M W4A16."""

from functools import cache
from typing import Any

import torch

from ...autotuner import AutoTuner, TunableRunner, TuningConfig
from ...utils import get_compute_capability, get_device_index

_COMPILED: dict[tuple, Any] = {}
_TUNING_CONFIG = TuningConfig(use_cuda_graph=True, use_cold_l2_cache=True)


def _compile(m, n, k, enable_pdl, tactic):
    import cutlass
    import cutlass.cute as cute

    from ...cute_dsl import fp4_common
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import n64_staged_kernel, kernel

    tk, warps, splits, stages = tactic
    sf_size = ((n + 127) // 128) * ((k // 16 + 3) // 4) * 512
    operands = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (m, k // 2), stride_order=(1, 0), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n, k // 8), stride_order=(1, 0), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (sf_size,), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4),
        cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=2
        ),
        (
            cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (splits, m, n),
                stride_order=(2, 1, 0),
                assumed_align=4,
            )
            if splits > 1
            else None
        ),
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    op = n64_staged_kernel.NativeN64Bf16Fp4Kernel(splits, warps, enable_pdl, tk, stages)
    name = f"m{m}_n{n}_k{k}_p{int(enable_pdl)}_tk{tk}_w{warps}_s{splits}_st{stages}"
    return build_and_load_cute_dsl_kernel(
        "native_n64_bf16_fp4_sm12x",
        name,
        lambda: cute.compile(op, *operands, stream, options="--enable-tvm-ffi"),
        extra_key_files=(
            __file__,
            n64_staged_kernel.__file__,
            kernel.__file__,
            fp4_common.__file__,
        ),
    )


class NativeN64Bf16Fp4Runner(TunableRunner):
    def get_cache_key_extras(self, inputs):
        return get_compute_capability(inputs[0].device), inputs[-1]

    def get_valid_tactics(self, inputs, profile):
        m, k = inputs[0].shape
        a_rows = min(m + 1, 16)
        return [
            (tile_k, warps, splits, stages)
            for tile_k in (128, 256)
            if k % tile_k == 0
            for warps in (4, 8)
            for splits in (1, 2, 3, 4, 6, 8)
            if splits <= k // tile_k
            for stages in (2, 3)
            if stages
            * (
                a_rows * (tile_k // 2 + 4) * 4
                + 64 * (tile_k // 8 + 4) * 4
                + (tile_k // 64) * 512
            )
            <= 49152
        ]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, weight, scales, alpha, out, enable_pdl = inputs
        m, k = a.shape
        n = weight.shape[0]
        if tactic == -1:
            tactic = (128, 8, min(4, k // 128), 3)
        if tactic not in self.get_valid_tactics(inputs, None):
            raise ValueError("Invalid N64 W4A16 tactic")
        key = get_device_index(a.device), m, n, k, enable_pdl, tactic
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm N64 W4A16 before graph capture")
            with torch.cuda.device(a.device):
                compiled = _compile(m, n, k, enable_pdl, tactic)
            _COMPILED[key] = compiled
        splits = tactic[2]
        partial = (
            torch.empty((splits, m, n), dtype=torch.float32, device=a.device)
            if splits > 1
            else None
        )
        compiled(
            a.view(torch.int32),
            weight.view(torch.int32),
            scales.view(torch.float8_e4m3fn).view(-1),
            alpha,
            out,
            partial,
        )
        return out


@cache
def get_runner():
    return NativeN64Bf16Fp4Runner()


def run(a, weight, scales, alpha, *, enable_pdl=True):
    tensors = (a, weight, scales, alpha)
    if not all(tensor.is_cuda and tensor.device == a.device for tensor in tensors):
        raise ValueError("N64 W4A16 requires one CUDA device")
    if get_compute_capability(a.device) not in ((12, 0), (12, 1)):
        raise ValueError("N64 W4A16 requires an SM12x GPU")
    if a.ndim != 2 or a.dtype != torch.bfloat16:
        raise ValueError("A must be a 2D BF16 tensor")
    m, k = a.shape
    if not 1 <= m <= 16 or k < 128 or k % 128:
        raise ValueError("N64 W4A16 requires M in 1..16 and K divisible by 128")
    if weight.ndim != 2 or weight.dtype != torch.uint8:
        raise ValueError("weight must be a 2D packed uint8 NVFP4 tensor")
    n = weight.shape[0]
    if not n or n % 128 or weight.shape[1] * 2 != k:
        raise ValueError("N64 W4A16 requires matching K and N divisible by 128")
    if scales.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("scales must contain canonical E4M3 bytes")
    if scales.numel() != n * k // 16:
        raise ValueError("scales must use the canonical 128x4 layout")
    if not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors[:3]):
        raise ValueError("Compact operands must be contiguous and 16-byte aligned")
    if alpha.dtype != torch.float32 or alpha.numel() != 1:
        raise ValueError("alpha must be a scalar FP32 CUDA tensor")
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    inputs = [a, weight, scales, alpha.reshape(1), out, enable_pdl]
    runner, tactic = AutoTuner.get().choose_one(
        "native_n64_bf16_fp4_sm12x", [get_runner()], _TUNING_CONFIG, inputs
    )
    return runner(inputs, tactic=tactic)
