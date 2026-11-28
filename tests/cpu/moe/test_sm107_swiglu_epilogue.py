# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host semantic checks of the production epilogue body with scalar intrinsics.

These checks do not validate CuTe compilation or GPU instruction behavior.
The end-to-end reference and CUDA graph tests live in test_cute_dsl_fused_moe.py.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def epilogue(request):
    float_type = getattr(request, "param", float)
    path = Path(__file__).resolve().parents[3] / (
        "flashinfer/fused_moe/cute_dsl/rubin/"
        "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py"
    )
    module = ast.parse(path.read_text(), str(path))
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel"
    )
    method = next(
        node
        for node in kernel.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_swiglu_epilogue"
    )
    method.decorator_list = []
    for arg in method.args.args:
        arg.annotation = None

    def sigmoid(value, fastmath=True):
        exp = math.exp(-abs(value))
        return float_type((1.0 if value >= 0 else exp) / (1.0 + exp))

    namespace = {
        "cutlass": SimpleNamespace(
            const_expr=bool, Float32=float_type, range_constexpr=range
        ),
        "cute": SimpleNamespace(
            size=math.prod,
            arch=SimpleNamespace(
                mul_packed_f32x2=lambda a, b: tuple(
                    x * y for x, y in zip(a, b, strict=True)
                ),
                add_packed_f32x2=lambda a, b: tuple(
                    x + y for x, y in zip(a, b, strict=True)
                ),
                rcp_approx=lambda x: 1.0 / x,
            ),
            math=SimpleNamespace(exp2=lambda x, fastmath: 2.0**x),
        ),
        "fmin": lambda a, b, nan: min(a, b),
        "fclip_xorsign": lambda x, limit: math.copysign(min(abs(x), limit), x),
        "sigmoid_f32": sigmoid,
        "silu_f32": lambda x, fastmath: x * sigmoid(x),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method.name]


class Fragment(list):
    @property
    def shape(self):
        return (len(self),)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 0.0), (1.702, 0.0), (1.0, 1.0), (1.702, 1.0), (0.5, -0.75)]
)
@pytest.mark.parametrize("limit", [float("inf"), 7.0, 0.1])
@pytest.mark.parametrize("expert_scale", [0.5, 1.0, 2.0])
def test_scaled_clamped_swiglu(epilogue, packed, alpha, beta, limit, expert_scale):
    up = Fragment([-9.0, 9.0, -0.25, 0.25, 0.0, -2.0])
    gate = Fragment([-8.0, 8.0, 0.5, -0.5, 0.0, 2.0])
    output = Fragment([0.0] * len(up))
    config = SimpleNamespace(
        vectorized_f32=packed,
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
        has_swiglu_alpha=alpha != 1.0,
        has_swiglu_beta=beta != 0.0,
        has_swiglu_limit=math.isfinite(limit),
    )
    epilogue(config, up, gate, expert_scale, output)
    expected = []
    for u, g in zip(up, gate, strict=True):
        u = max(-limit, min(limit, expert_scale * u))
        g = min(limit, expert_scale * g)
        expected.append((u + beta) * g / (1.0 + math.exp(-alpha * g)))
    assert output == pytest.approx(expected, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("epilogue", [np.float32], indirect=True)
def test_custom_alpha_scalar_avoids_fp32_overflow(epilogue):
    config = SimpleNamespace(
        vectorized_f32=False,
        swiglu_alpha=1.702,
        has_swiglu_alpha=True,
        has_swiglu_beta=False,
        has_swiglu_limit=False,
    )
    up = Fragment([np.float32(2e19)])
    gate = Fragment([np.float32(-2e19)])
    output = Fragment([np.float32(0)])
    # Old ordering overflowed up * gate before multiplying by the zero sigmoid.
    with np.errstate(over="ignore", invalid="ignore"):
        epilogue(config, up, gate, 1.0, output)
    assert output[0] == 0.0
