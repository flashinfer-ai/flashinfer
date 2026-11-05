# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PowLU scalar/tensor epilogues and independent power/clamping reference."""

import pytest
import torch

from flashinfer.fused_moe import PowLU
from tests.moe.test_frost_low_latency_bf16 import _activation


@pytest.mark.parametrize(
    "activation",
    [PowLU(), PowLU(m=1.5, limit=2.0), PowLU(m=0.125), PowLU(m=3.0, limit=1.0)],
)
@pytest.mark.parametrize("vector", [False, True])
def test_powlu_epilogue_boundaries(activation, vector):
    from tests.moe.test_frost_low_latency import _require_gpu

    _require_gpu(allow_sm120=True, allow_sm90=True)
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.utils import get_compute_capability
    from flashinfer.fused_moe.backends.frost_low_latency.fma.activations import (
        apply_activation,
    )
    from flashinfer.fused_moe.backends.frost_low_latency.fma.activation_config import (
        lower_activation,
    )

    @cute.kernel
    def kernel(up, gate, out, parameters: cutlass.Constexpr, vector: cutlass.Constexpr):
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = bid * 128 + tid
        if row < cute.size(up) // 4:
            if cutlass.const_expr(vector):
                u = cute.make_rmem_tensor(4, cutlass.Float32)
                g = cute.make_rmem_tensor(4, cutlass.Float32)
                y = cute.make_rmem_tensor(4, cutlass.Float32)
                for j in cutlass.range_constexpr(4):
                    u[j], g[j] = up[row * 4 + j], gate[row * 4 + j]
                y.store(
                    apply_activation(u.load(), g.load(), "powlu", False, parameters)
                )
                for j in cutlass.range_constexpr(4):
                    out[row * 4 + j] = y[j]
            else:
                for j in cutlass.range_constexpr(4):
                    out[row * 4 + j] = apply_activation(
                        up[row * 4 + j], gate[row * 4 + j], "powlu", False, parameters
                    )

    @cute.jit
    def launch(
        up, gate, out, parameters: cutlass.Constexpr, vector: cutlass.Constexpr, stream
    ):
        kernel(up, gate, out, parameters, vector).launch(
            grid=((cute.size(up) // 4 + 127) // 128, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    device = torch.device("cuda")
    edges = torch.tensor(
        [
            -100.0,
            -1.0,
            -0.0,
            0.0,
            2.0**-149,
            2.0**-126,
            1.0 - 2.0**-23,
            1.0,
            1.0 + 2.0**-23,
            4.0,
            16.0,
            65504.0,
            2.0**120,
            float("inf"),
        ],
        device=device,
    )
    gate = torch.cat((edges, torch.logspace(-35, 35, 1000, device=device)))
    gate = torch.cat((gate, torch.zeros((-gate.numel()) % 4, device=device)))
    up = torch.linspace(-8.0, 8.0, gate.numel(), device=device)
    output = torch.full_like(up, float("nan"))
    major, minor = get_compute_capability(device)
    _, parameters = lower_activation(activation)
    compiled = cute.compile(
        launch,
        *(from_dlpack(v) for v in (up, gate, output)),
        parameters,
        vector,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch sm_{major}{minor}a --enable-tvm-ffi",
    )
    compiled(up, gate, output)
    torch.testing.assert_close(
        output, _activation(up, gate, activation), rtol=2e-5, atol=2e-6
    )


def test_powlu_parameters_are_part_of_lowering():
    from flashinfer.fused_moe.backends.frost_low_latency.fma.activation_config import (
        lower_activation,
    )

    for activation in (PowLU(), PowLU(m=1.5, limit=2.0), PowLU(m=0.125)):
        assert lower_activation(activation) == (
            "powlu",
            (activation.m, activation.limit),
        )
