# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Cached CuTe DSL launch and bounded small-M tuning."""

from functools import cache
from typing import Any

import torch

from ....autotuner import AutoTuner, TunableRunner, TuningConfig
from ....utils import get_compute_capability, get_device_index

_COMPILED: dict[tuple, Any] = {}
_TUNING_CONFIG = TuningConfig(use_cuda_graph=True, use_cold_l2_cache=True)


def _compile(m, n, k, dtype, has_alpha, enable_pdl, tactic):
    import cutlass
    import cutlass.cute as cute

    from ....cute_dsl import fp4_common
    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import kernel, staged_kernel

    kind, extent, warps, splits = tactic[:4]
    if kind == "n64":
        from .n64_runner import _compile as compile_n64

        if dtype != torch.bfloat16 or not has_alpha:
            raise ValueError("The 64-column tactic needs BF16 output and alpha")
        return compile_n64(m, n, k, enable_pdl, tactic[1:])
    staged = kind == "staged"
    out_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    a = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32 if staged else cutlass.BFloat16,
        (m, k // 2 if staged else k),
        stride_order=(1, 0),
        assumed_align=16 if staged else 2,
    )
    b = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n, k // 8),
        stride_order=(1, 0),
        assumed_align=16 if staged else 4,
    )
    sf_size = ((n + 127) // 128) * ((k // 16 + 3) // 4) * 512
    sf = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sf_size,), assumed_align=16 if staged else 1
    )
    alpha = (
        cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        if has_alpha
        else None
    )
    out = cute.runtime.make_fake_compact_tensor(
        out_type, (m, n), stride_order=(1, 0), assumed_align=2
    )
    if staged:
        op = staged_kernel.NativeBf16Fp4StagedKernel(
            splits, warps, enable_pdl, extent, tactic[4]
        )
    elif kind == "mma":
        op = kernel.NativeBf16Fp4MmaKernel(splits, warps, enable_pdl, extent)
    else:
        op = kernel.NativeBf16Fp4Kernel(extent, warps, enable_pdl, splits)
    partial = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (splits, m, n),
            stride_order=(2, 1, 0),
            assumed_align=4,
        )
        if splits > 1
        else None
    )
    operands = (a, b, sf, alpha, out, partial)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    dtype_key = "bf16" if dtype == torch.bfloat16 else "fp16"
    name = f"m{m}_n{n}_k{k}_{dtype_key}_a{int(has_alpha)}_p{int(enable_pdl)}_{kind}{extent}_w{warps}_s{splits}"
    if staged:
        name += f"_st{tactic[4]}"
    return build_and_load_cute_dsl_kernel(
        "native_bf16_fp4_sm12x",
        name,
        lambda: cute.compile(op, *operands, stream, options="--enable-tvm-ffi"),
        extra_key_files=(
            __file__,
            kernel.__file__,
            staged_kernel.__file__,
            fp4_common.__file__,
        ),
    )


def _can_stage(inputs):
    a, b, sf = inputs[:3]
    return (
        a.shape[0] <= 16
        and a.shape[1] % 64 == 0
        and all(t.data_ptr() % 16 == 0 for t in (a, b, sf))
    )


class NativeBf16Fp4Runner(TunableRunner):
    def get_cache_key_extras(self, inputs):
        a, _, _, alpha, out, enable_pdl = inputs
        return (
            get_compute_capability(a.device),
            out.dtype,
            alpha is None,
            enable_pdl,
            _can_stage(inputs),
        )

    def get_valid_tactics(self, inputs, profile):
        m, k = inputs[0].shape
        n = inputs[1].shape[0]
        if m > 16:
            return [("mma", 4 if n >= 512 else 1, w, 1) for w in (4, 8)]
        rows = sorted({1, min(m, 2), min(m, 4)})
        simd_splits = (
            [s for s in (1, 2, 4, 8) if s <= (k + 255) // 256] if m == 1 else [1]
        )
        tactics = [("simd", r, w, s) for r in rows for w in (4, 8) for s in simd_splits]
        if _can_stage(inputs):
            if m > 1:
                tactics = []
            tactics += [
                ("staged", tk, w, s, stages)
                for tk in (128 if k % 128 == 0 else 64,)
                for w in (4, 8)
                for s in (1, 2, 4, 8)
                if s <= k // tk
                for stages in (2, 3)
            ]
            if m <= 4 and inputs[4].dtype == torch.bfloat16 and inputs[3] is not None:
                tactics += [
                    ("n64", tk, w, s, stages)
                    for tk in (128, 256)
                    if k % tk == 0
                    for w in (4, 8)
                    for s in (1, 2, 4, 8)
                    if s <= k // tk
                    for stages in (2, 3)
                    if stages
                    * (
                        (m + 1) * (tk // 2 + 4) * 4
                        + 64 * (tk // 8 + 4) * 4
                        + (tk // 64) * 512
                    )
                    <= 49152
                ]
            return tactics
        mma_splits = (
            (1, 2, 4, 8, 16, 32, 64) if n < 8192 and k >= 4096 else (1, 2, 4, 8)
        )
        tactics += [
            ("mma", 4 if n >= 512 else 1, w, s)
            for s in mma_splits
            if s <= (k + 63) // 64
            for w in (4, 8)
        ]
        return tactics

    def validate_tactic(self, inputs, tactic):
        return tactic == -1 or tactic in self.get_valid_tactics(inputs, None)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, sf, alpha, out, enable_pdl = inputs
        m, k = a.shape
        n = b.shape[0]
        if not self.validate_tactic(inputs, tactic):
            raise ValueError("Invalid native W4A16 tactic")
        if tactic == -1:
            if m > 16:
                tactic = ("mma", 4 if n >= 512 else 1, 4, 1)
            elif _can_stage(inputs):
                tk = 128 if k % 128 == 0 else 64
                splits = min(4 if n < 8192 else 1, k // tk)
                tactic = ("staged", tk, 8 if n < 8192 else 4, splits, 3)
            elif m == 1:
                tactic = ("simd", 1, 4, 2 if n < 8192 and k >= 512 else 1)
            else:
                splits = max(s for s in (1, 2, 4, 8) if s <= (k + 63) // 64)
                if n < 8192 and k >= 4096:
                    splits = 32
                tactic = ("mma", 4 if n >= 512 else 1, 4, splits)
        key = (
            get_device_index(a.device),
            m,
            n,
            k,
            out.dtype,
            alpha is None,
            enable_pdl,
            tactic,
        )
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm cute-dsl-native for this shape before graph capture"
                )
            with torch.cuda.device(a.device):
                compiled = _compile(
                    m, n, k, out.dtype, alpha is not None, enable_pdl, tactic
                )
            _COMPILED[key] = compiled
        operands = [
            a.view(torch.int32) if tactic[0] in ("staged", "n64") else a,
            b.view(torch.int32),
            sf.view(torch.float8_e4m3fn).view(-1),
            alpha,
            out,
        ]
        splits = tactic[3]
        partial = (
            torch.empty((splits, m, n), device=a.device, dtype=torch.float32)
            if splits > 1
            else None
        )
        operands.append(partial)
        compiled(*operands)
        return out


@cache
def get_runner():
    return NativeBf16Fp4Runner()


def run(a, b, sf, alpha, out_dtype, out, enable_pdl):
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)
    inputs = [a, b, sf, alpha, out, enable_pdl]
    runner, tactic = AutoTuner.get().choose_one(
        "native_bf16_fp4_sm12x", [get_runner()], _TUNING_CONFIG, inputs
    )
    return runner(inputs, tactic=tactic)
