# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile 64-column canonical weight staging for small-M W4A16."""


def _compile(m, n, k, enable_pdl, tactic):
    import cutlass
    import cutlass.cute as cute

    from ....cute_dsl import fp4_common
    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import kernel, n64_staged_kernel, staged_kernel

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
            staged_kernel.__file__,
            kernel.__file__,
            fp4_common.__file__,
        ),
    )
