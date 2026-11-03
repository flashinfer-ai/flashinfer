# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and route tests of the source-built GEMM backends on SM100/SM103."""

import re

import pytest
import torch

from flashinfer import mm_bf16_fp4, prepare_bf16_fp4_weights
from flashinfer.gemm.gemm_bf16_fp4_blackwell import (
    _cell_key,
    _route_plan,
    _select_route,
)
from flashinfer.jit.blackwell_bf16_fp4 import MODULES, ROUTES

# Smallest M whose 16-row activation-tile count exceeds the CUDA Y grid limit.
LARGE_M = 65535 * 16 + 16


@pytest.fixture(scope="module")
def blackwell_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("the source-built backends require SM100 or SM103")
    return device


def _make_weights(n, k, device, generator):
    # E2M1 stores a sign bit and the magnitudes below. Construct codes directly
    # so this GEMM test does not depend on a separate quantizer's correctness.
    codes = torch.randint(
        0, 16, (n, k), dtype=torch.uint8, device=device, generator=generator
    )
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device
    )
    values = magnitudes[(codes & 7).long()]

    # Every scale and dequantized value is exactly representable in FP16.
    scale_values = torch.tensor(
        [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0],
        dtype=torch.float32,
        device=device,
    )
    scale_indices = torch.randint(
        len(scale_values), (n, k // 16), device=device, generator=generator
    )
    scales = scale_values[scale_indices]
    reference_weights = torch.where((codes & 8) != 0, -values, values) * (
        scales.repeat_interleave(16, dim=1)
    )

    # Canonical 128x4 scale layout: [N/128, K_sf/4, N%32, N/32%4, K_sf%4].
    # Build it from padded logical rows without using backend layout helpers.
    n_blocks = (n + 127) // 128
    k_blocks = (k // 16 + 3) // 4
    padded_scales = torch.zeros(
        (n_blocks * 128, k_blocks * 4), dtype=torch.uint8, device=device
    )
    padded_scales[:n, : k // 16] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    canonical_scales = (
        padded_scales.reshape(n_blocks, 4, 32, k_blocks, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )
    return packed, canonical_scales, reference_weights


def _reference(a, reference_weights, alpha_value, out_dtype):
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        reference = a.float() @ reference_weights.T
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    if alpha_value is not None:
        reference = reference * alpha_value
    return reference.to(out_dtype)


def _case(device, backend, shape, out_dtype, alpha_value, seed):
    m, n, k = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    a = (
        torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.125
    )
    packed, scales, reference_weights = _make_weights(n, k, device, generator)
    alpha = (
        None
        if alpha_value is None
        else torch.tensor([alpha_value], dtype=torch.float32, device=device)
    )
    prepared = prepare_bf16_fp4_weights(packed, scales, alpha, backend=backend)
    return a, prepared, _reference(a, reference_weights, alpha_value, out_dtype)


@pytest.mark.parametrize(
    "backend,shape,out_dtype,alpha_value,caller_out,enable_pdl",
    [
        pytest.param(
            "blackwell-native",
            (16, 1024, 1024),
            torch.bfloat16,
            None,
            False,
            False,
            id="native-m16-bf16",
        ),
        pytest.param(
            "blackwell-tiled",
            (16, 1024, 1024),
            torch.bfloat16,
            None,
            False,
            False,
            id="tiled-m16-bf16",
        ),
        pytest.param(
            "blackwell-native",
            (16, 1024, 1024),
            torch.float16,
            0.75,
            True,
            True,
            id="native-m16-fp16-alpha-out-pdl",
        ),
        pytest.param(
            "blackwell-tiled",
            (16, 1024, 1024),
            torch.bfloat16,
            0.75,
            True,
            True,
            id="tiled-m16-alpha-out-pdl",
        ),
        pytest.param(
            "blackwell-native",
            (7, 65, 48),
            torch.bfloat16,
            -0.5,
            True,
            False,
            id="native-ragged-m-n-k48",
        ),
        pytest.param(
            "blackwell-native",
            (1, 129, 16),
            torch.float16,
            None,
            False,
            True,
            id="native-m1-ragged-n-k16",
        ),
        pytest.param(
            "blackwell-native",
            (129, 192, 128),
            torch.bfloat16,
            0.75,
            False,
            True,
            id="native-large-m-tail",
        ),
        pytest.param(
            "blackwell-tiled",
            (7, 4160, 192),
            torch.bfloat16,
            -0.5,
            True,
            False,
            id="tiled-short-k-ragged-m",
        ),
        pytest.param(
            "blackwell-tiled",
            (13, 192, 48),
            torch.bfloat16,
            None,
            False,
            True,
            id="tiled-ragged-m-k48",
        ),
        # Routes the earlier parametrization never reached.
        pytest.param(
            "blackwell-tiled",
            (33, 128, 16),
            torch.bfloat16,
            0.5,
            False,
            True,
            id="tiled-k16-persistent",
        ),
        pytest.param(
            "blackwell-tiled",
            (5, 4096, 32),
            torch.bfloat16,
            None,
            True,
            False,
            id="tiled-k32",
        ),
        pytest.param(
            "blackwell-tiled",
            (40, 256, 256),
            torch.bfloat16,
            -0.25,
            False,
            True,
            id="tiled-m64-route",
        ),
        pytest.param(
            "blackwell-tiled",
            (17, 64, 80),
            torch.bfloat16,
            None,
            False,
            False,
            id="tiled-generic-fallback-k80",
        ),
        # smk S=1 burst cell (tiled M <= 16, K % 64 == 0, K <= 256): the K=64 lower edge, the
        # alpha=None cell on the second contract row, and the caller-owned output with PDL off.
        pytest.param(
            "blackwell-tiled",
            (16, 64, 64),
            torch.bfloat16,
            0.5,
            False,
            True,
            id="tiled-smk-k64-edge",
        ),
        pytest.param(
            "blackwell-tiled",
            (13, 11008, 128),
            torch.bfloat16,
            None,
            False,
            True,
            id="tiled-smk-m13-n11008-k128",
        ),
        pytest.param(
            "blackwell-tiled",
            (9, 192, 192),
            torch.bfloat16,
            0.75,
            True,
            False,
            id="tiled-smk-caller-out-pdl-off",
        ),
        pytest.param(
            "blackwell-tiled",
            (17, 3072, 3072),
            torch.bfloat16,
            0.75,
            False,
            True,
            id="tiled-m32-long-k",
        ),
        # smk split-K planes cells (tiled M <= 32, K % 64 == 0 with a routed policy plan): the K1024
        # S=4 cell (alpha=None), the K4096 S=4 cell with caller-owned out and PDL off, the TILE_M 32
        # S=6 cell on a full M=32 tile with a K tail, and a K tail of the S=4 K1024 cell.
        pytest.param(
            "blackwell-tiled",
            (16, 1024, 1024),
            torch.bfloat16,
            None,
            False,
            True,
            id="tiled-smk-split-m16-k1024-alpha-none",
        ),
        pytest.param(
            "blackwell-tiled",
            (1, 4096, 4096),
            torch.bfloat16,
            0.75,
            True,
            False,
            id="tiled-smk-split-m1-k4096-caller-out-pdl-off",
        ),
        pytest.param(
            "blackwell-tiled",
            (32, 192, 2624),
            torch.bfloat16,
            -0.5,
            False,
            True,
            id="tiled-smk-split-m32-k2624-tail",
        ),
        pytest.param(
            "blackwell-tiled",
            (5, 128, 832),
            torch.bfloat16,
            0.5,
            False,
            True,
            id="tiled-smk-split-m5-k832-tail",
        ),
        # cuDNN-ABI smk burst cell (native M <= 16, K == 1024): F16 output, caller-owned out with
        # alpha=None and PDL off, ragged N.
        pytest.param(
            "blackwell-native",
            (16, 1024, 1024),
            torch.float16,
            0.5,
            False,
            True,
            id="native-smk-m16-k1024-fp16",
        ),
        pytest.param(
            "blackwell-native",
            (7, 65, 1024),
            torch.bfloat16,
            None,
            True,
            False,
            id="native-smk-m7-n65-k1024-caller-out-pdl-off",
        ),
        pytest.param(
            "blackwell-native",
            (33, 128, 192),
            torch.float16,
            0.5,
            False,
            False,
            id="native-fp16-cp-async-alpha",
        ),
        pytest.param(
            "blackwell-native",
            (768, 2112, 2048),
            torch.bfloat16,
            0.75,
            False,
            True,
            id="native-m768-n2112-k2048",
        ),
        # cuDNN-ABI smk split-K cluster DSMEM-mailbox cell `_z4c` (native BF16 M <= 16, K == 4096, >= 128 CTAs; no planes workspace): the R1 row and a
        # caller-owned-out / alpha-None / PDL-off row with ragged N tiles; the F16 twin stays on the shipped
        # route and is covered by the generic native rows.
        pytest.param(
            "blackwell-native",
            (1, 4096, 4096),
            torch.bfloat16,
            None,
            False,
            True,
            id="native-smk-split-m1-n4096-k4096",
        ),
        pytest.param(
            "blackwell-native",
            (16, 2112, 4096),
            torch.bfloat16,
            None,
            True,
            False,
            id="native-smk-split-m16-n2112-k4096-caller-out-pdl-off",
        ),
        pytest.param(
            "blackwell-native",
            (LARGE_M, 64, 16),
            torch.bfloat16,
            None,
            False,
            True,
            id="native-large-m-flat-grid",
        ),
        pytest.param(
            "blackwell-tiled",
            (LARGE_M, 64, 80),
            torch.bfloat16,
            0.5,
            False,
            False,
            id="tiled-large-m-flat-grid",
        ),
    ],
)
def test_blackwell_bf16_fp4_numerical(
    blackwell_device, backend, shape, out_dtype, alpha_value, caller_out, enable_pdl
):
    m, n, _k = shape
    a, prepared, reference = _case(
        blackwell_device, backend, shape, out_dtype, alpha_value, 42
    )
    out = (
        torch.full((m, n), float("nan"), dtype=out_dtype, device=blackwell_device)
        if caller_out
        else None
    )
    actual = mm_bf16_fp4(
        a,
        *prepared,
        backend=backend,
        out_dtype=out_dtype,
        out=out,
        enable_pdl=enable_pdl,
    )
    if caller_out:
        assert actual is out

    # Compare to the represented FP4 weights, not to unquantized source weights.
    # Thus the standard BF16/FP16 pointwise tolerance need not include FP4 error.
    torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("backend", ["blackwell-native", "blackwell-tiled"])
def test_blackwell_bf16_fp4_many_distinct_activations(blackwell_device, backend):
    """Distinct activation and output buffers each get freshly encoded descriptors."""

    m, n, k = 3, 64, 256
    generator = torch.Generator(device=blackwell_device).manual_seed(7)
    packed, scales, reference_weights = _make_weights(n, k, blackwell_device, generator)
    prepared = prepare_bf16_fp4_weights(packed, scales, None, backend=backend)
    activations = [
        torch.randn(
            (m, k), dtype=torch.bfloat16, device=blackwell_device, generator=generator
        )
        * 0.125
        for _ in range(8)
    ]
    outputs = [mm_bf16_fp4(a, *prepared, backend=backend) for a in activations]
    torch.cuda.synchronize()
    for a, out in zip(activations, outputs, strict=True):
        torch.testing.assert_close(
            out,
            _reference(a, reference_weights, None, torch.bfloat16),
            atol=1e-2,
            rtol=1e-2,
        )


def test_blackwell_bf16_fp4_planes_workspace_eviction(blackwell_device):
    """More distinct split-K planes workspaces than the host keeps cached (16) must evict cleanly and stay exact.

    M = 16, K = 4096 tiled rows take the eight-plane split cell, M = 17, K = 3072 rows the six-plane cell; every
    (M, N, S) is a distinct workspace key, so 22 keys exercise the least-recently-used eviction path once.  The
    calls run without PDL: the eight-plane cell's PDL programs reduce through a cluster and take no workspace.
    """

    backend = "blackwell-tiled"
    generator = torch.Generator(device=blackwell_device).manual_seed(11)
    cases = [(16, 128 * i, 4096) for i in range(1, 19)] + [
        (17, 64 * i, 3072) for i in range(1, 5)
    ]
    for m, n, k in cases:
        packed, scales, reference_weights = _make_weights(
            n, k, blackwell_device, generator
        )
        prepared = prepare_bf16_fp4_weights(packed, scales, None, backend=backend)
        a = (
            torch.randn(
                (m, k),
                dtype=torch.bfloat16,
                device=blackwell_device,
                generator=generator,
            )
            * 0.125
        )
        actual = mm_bf16_fp4(a, *prepared, backend=backend, enable_pdl=False)
        torch.testing.assert_close(
            actual,
            _reference(a, reference_weights, None, torch.bfloat16),
            atol=1e-2,
            rtol=1e-2,
        )
        # a second call with the same key must reuse the cached workspace and stay bitwise
        again = mm_bf16_fp4(a, *prepared, backend=backend, enable_pdl=False)
        torch.cuda.synchronize()
        assert torch.equal(actual, again)


@pytest.mark.parametrize(
    "backend,shape",
    [
        pytest.param("blackwell-native", (16, 1024, 1024), id="native-m16"),
        pytest.param("blackwell-tiled", (16, 1024, 1024), id="tiled-m16"),
        pytest.param("blackwell-native", (1, 4096, 4096), id="native-smk-split-k4096"),
        pytest.param("blackwell-native", (768, 2112, 2048), id="native-m768"),
        pytest.param("blackwell-tiled", (7, 4160, 192), id="tiled-smk-m7"),
        pytest.param("blackwell-tiled", (13, 11008, 128), id="tiled-smk-m13"),
        # smk split-K cells: the eight-plane PDL program reduces through a (1, 1, 8) cluster; the six-plane
        # program's workspace (planes + zeroed counters) is captured with the launch.
        pytest.param("blackwell-tiled", (1, 4096, 4096), id="tiled-smk-split-k4096"),
        pytest.param(
            "blackwell-tiled", (17, 3072, 3072), id="tiled-smk-split-m17-k3072"
        ),
    ],
)
def test_blackwell_bf16_fp4_cuda_graph_capture(blackwell_device, backend, shape):
    """A first call inside CUDA Graph capture is legal and replays bitwise."""

    m, n, _k = shape
    a, prepared, reference = _case(
        blackwell_device, backend, shape, torch.bfloat16, 0.75, 11
    )
    out = torch.full(
        (m, n), float("nan"), dtype=torch.bfloat16, device=blackwell_device
    )
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        mm_bf16_fp4(a, *prepared, backend=backend, out=out, enable_pdl=True)
    torch.cuda.synchronize()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
    replayed = out.clone()
    a.copy_(a * 0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(out, replayed)
    torch.testing.assert_close(
        out, (reference.float() * 0.5).to(torch.bfloat16), atol=1e-2, rtol=1e-2
    )


def _operands(device, backend, shape=(16, 64, 1024)):
    """Valid operands; for this shape the tiled route reads A through a raw pointer."""

    a, (b, b_descale, _alpha), _reference = _case(
        device, backend, shape, torch.bfloat16, None, 3
    )
    return a, b, b_descale


def _rejects(exc, message, a, b, b_descale, alpha=None, **kwargs):
    with pytest.raises(exc, match=re.escape(message)):
        mm_bf16_fp4(a, b, b_descale, alpha, **kwargs)


@pytest.mark.parametrize("backend", ["blackwell-native", "blackwell-tiled"])
def test_blackwell_bf16_fp4_rejects_invalid_operands(blackwell_device, backend):
    """Every invalid operand raises before any program is built or launched."""

    a, b, s = _operands(blackwell_device, backend)
    f32 = dict(dtype=torch.float32, device=blackwell_device)
    contiguous_b = f"{backend} requires contiguous b and b_descale"
    alpha_message = "alpha must be a contiguous float32 tensor with shape (1,)"
    cases = [
        ("a, b, and b_descale must be contiguous", a.t().contiguous().t(), b, s),
        (contiguous_b, a, b.t().contiguous().t(), s),
        (contiguous_b, a, b, s.t().contiguous().t()),
        ("a, b, and b_descale must be on the same device", a, b.cpu(), s),
        ("K=1016 must be a multiple of block_size=16", a[:, :1016], b, s),
        ("a.shape[1]=512 but prepared b encodes K=1024", a[:, :512], b, s),
        ("M and N must be positive; got M=0, N=64", a[:0], b, s),
        (alpha_message, a, b, s, torch.ones((1,), **f32).to(torch.float16)),
        (alpha_message, a, b, s, torch.ones((2,), **f32)),
        (alpha_message, a, b, s, torch.ones((), **f32)),
        ("alpha must be on the same device as a", a, b, s, torch.ones((1,))),
    ]
    for message, *operands in cases:
        _rejects(ValueError, message, *operands, backend=backend)
    _rejects(
        TypeError, "a must be bfloat16", a.to(torch.float16), b, s, backend=backend
    )
    out = torch.empty((16, 64), dtype=torch.bfloat16, device=blackwell_device)
    out_message = (
        "out must be contiguous, on the same device, and have shape/dtype "
        "(16, 64)/torch.bfloat16"
    )
    for bad_out in (out[:, :32], out.to(torch.float16), out.t().contiguous().t()):
        _rejects(ValueError, out_message, a, b, s, backend=backend, out=bad_out)


def test_blackwell_bf16_fp4_rejects_wrong_layout(blackwell_device):
    """Operands in the other backend's layout, N % 64 != 0, N = 0, bad out dtypes."""

    native, tiled = "blackwell-native", "blackwell-tiled"
    a, b_n, s_n = _operands(blackwell_device, native)
    _, b_t, s_t = _operands(blackwell_device, tiled)
    a_n0, b_n0, s_n0 = _operands(blackwell_device, native, (16, 0, 1024))
    s_n_u8, s_t_i8 = s_n.view(torch.uint8), s_t.view(torch.int8)
    cases = [
        (native, "blackwell-native expects the uint8 [N, K/2] weight", a, b_t, s_t),
        (tiled, "blackwell-tiled expects the int32 [K/16, N*2] weight", a, b_n, s_n),
        (native, "linear float8_e4m3fn scales with shape (64, 64)", a, b_n, s_n_u8),
        (tiled, "S0E5M3 uint8 scales with shape (64, 64)", a, b_t, s_t_i8),
        (tiled, "N to be a multiple of 64; got N=32", a, b_t[:, :64], s_t[:, :32]),
        (native, "M and N must be positive; got M=16, N=0", a_n0, b_n0, s_n0),
    ]
    for backend, message, *operands in cases:
        _rejects(ValueError, message, *operands, backend=backend)
    _rejects(
        ValueError,
        "blackwell-tiled requires bfloat16 output; got torch.float16",
        a,
        b_t,
        s_t,
        backend=tiled,
        out_dtype=torch.float16,
    )
    _rejects(
        ValueError,
        "out_dtype must be bfloat16 or float16; got torch.float32",
        a,
        b_n,
        s_n,
        backend=native,
        out_dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["blackwell-native", "blackwell-tiled"])
def test_blackwell_bf16_fp4_skip_check_still_validates(blackwell_device, backend):
    """``skip_check=True`` skips the API checker; the compute path rejects the same."""

    a, b, s = _operands(blackwell_device, backend)
    bad_alpha = torch.ones((2,), dtype=torch.float32, device=blackwell_device)
    bad_out = torch.empty((1, 64), dtype=torch.bfloat16, device=blackwell_device)
    for operands, kwargs in (
        ((a.t().contiguous().t(), b, s), {}),
        ((a[:0], b, s), {}),
        ((a[:, :512].contiguous(), b, s), {}),
        ((a, b.t().contiguous().t(), s), {}),
        ((a, b.cpu(), s), {}),
        ((a, b, s, bad_alpha), {}),
        ((a, b, s), dict(out_dtype=torch.float32)),
        ((a, b, s), dict(out=bad_out)),
    ):
        with pytest.raises(ValueError):
            mm_bf16_fp4(*operands, backend=backend, skip_check=True, **kwargs)


# Route of every test row, replayed without a GPU or a JIT build: the inputs
# ``(backend, (M, N, K), out dtype, alpha, PDL)`` and the stages
# ``((route cell, launch grid), ...)`` that ``_select_route`` returns at
# ``ROUTE_SM_COUNT`` SMs (the SM count only sizes persistent grids).  A route
# change updates the rows it moves.  The CUDA-graph rows ``tiled-m16`` and
# ``native-m768`` share their route with ``tiled-m16-alpha-out-pdl`` and
# ``native-m768-n2112-k2048`` and are not repeated.
ROUTE_SM_COUNT = 148
EXPECTED_ROUTES = {
    "native-m16-bf16": (
        ("blackwell-native", (16, 1024, 1024), torch.bfloat16, None, False),
        (("cudnn_smk_k4_n16_bf16:alpha0:pdl0", (64, 1, 1)),),
    ),
    "tiled-m16-bf16": (
        ("blackwell-tiled", (16, 1024, 1024), torch.bfloat16, None, False),
        (("cute_smk_s1_m16_z4_bf16:alpha0:pdl0", (16, 1, 4)),),
    ),
    "native-m16-fp16-alpha-out-pdl": (
        ("blackwell-native", (16, 1024, 1024), torch.float16, 0.75, True),
        (("cudnn_smk_k4_n16_f16:alpha1:pdl1", (64, 1, 1)),),
    ),
    "tiled-m16-alpha-out-pdl": (
        ("blackwell-tiled", (16, 1024, 1024), torch.bfloat16, 0.75, True),
        (("cute_smk_s1_m16_z4_bf16:alpha1:pdl1", (16, 1, 4)),),
    ),
    "native-ragged-m-n-k48": (
        ("blackwell-native", (7, 65, 48), torch.bfloat16, -0.5, False),
        (("cudnn_cp_async_bf16:alpha1:pdl0:grid2d", (2, 1, 1)),),
    ),
    "native-m1-ragged-n-k16": (
        ("blackwell-native", (1, 129, 16), torch.float16, None, True),
        (("cudnn_cp_async_f16:alpha0:pdl1:grid2d", (3, 1, 1)),),
    ),
    "native-large-m-tail": (
        ("blackwell-native", (129, 192, 128), torch.bfloat16, 0.75, True),
        (("cudnn_cp_async_bf16:alpha1:pdl1:grid2d", (3, 9, 1)),),
    ),
    "tiled-short-k-ragged-m": (
        ("blackwell-tiled", (7, 4160, 192), torch.bfloat16, -0.5, False),
        (("cute_smk_s1_bf16:alpha1:pdl0", (65, 1, 1)),),
    ),
    "tiled-ragged-m-k48": (
        ("blackwell-tiled", (13, 192, 48), torch.bfloat16, None, True),
        (("cute_warp_mma_m16_k48_bf16:alpha0:pdl1", (3, 1, 1)),),
    ),
    "tiled-k16-persistent": (
        ("blackwell-tiled", (33, 128, 16), torch.bfloat16, 0.5, True),
        (("cute_warp_mma_m16_k16_bf16:alpha1:pdl1", (6, 1, 1)),),
    ),
    "tiled-k32": (
        ("blackwell-tiled", (5, 4096, 32), torch.bfloat16, None, False),
        (("cute_warp_mma_m16_k32_bf16:alpha0:pdl0", (64, 1, 1)),),
    ),
    "tiled-m64-route": (
        ("blackwell-tiled", (40, 256, 256), torch.bfloat16, -0.25, True),
        (("cute_warp_mma_m64_bf16:alpha1:pdl1", (4, 1, 1)),),
    ),
    "tiled-generic-fallback-k80": (
        ("blackwell-tiled", (17, 64, 80), torch.bfloat16, None, False),
        (("cute_bf16:alpha0:pdl0:grid2d", (1, 2, 1)),),
    ),
    "tiled-m32-long-k": (
        ("blackwell-tiled", (17, 3072, 3072), torch.bfloat16, 0.75, True),
        (("cute_smk_s2_m32_z6_bf16:alpha1:pdl1", (48, 1, 6)),),
    ),
    "native-fp16-cp-async-alpha": (
        ("blackwell-native", (33, 128, 192), torch.float16, 0.5, False),
        (("cudnn_cp_async_f16:alpha1:pdl0:grid2d", (2, 3, 1)),),
    ),
    "native-m768-n2112-k2048": (
        ("blackwell-native", (768, 2112, 2048), torch.bfloat16, 0.75, True),
        (("cudnn_tcgen05_m128_cg2_bf16:alpha1:pdl1", (132, 1, 1)),),
    ),
    "native-m1-n4096-k4096": (
        ("blackwell-native", (1, 4096, 4096), torch.bfloat16, None, True),
        (("cudnn_smk_k4_z4c_bf16:alpha0:pdl1", (64, 1, 4)),),
    ),
    "native-large-m-flat-grid": (
        ("blackwell-native", (LARGE_M, 64, 16), torch.bfloat16, None, True),
        (("cudnn_cp_async_bf16:alpha0:pdl1:flat", (65536, 1, 1)),),
    ),
    "tiled-large-m-flat-grid": (
        ("blackwell-tiled", (LARGE_M, 64, 80), torch.bfloat16, 0.5, False),
        (("cute_bf16:alpha1:pdl0:flat", (65536, 1, 1)),),
    ),
    "many-distinct-native": (
        ("blackwell-native", (3, 64, 256), torch.bfloat16, None, True),
        (("cudnn_tma_bf16:alpha0:pdl1:grid2d", (1, 1, 1)),),
    ),
    "many-distinct-tiled": (
        ("blackwell-tiled", (3, 64, 256), torch.bfloat16, None, True),
        (("cute_smk_s1_bf16:alpha0:pdl1", (1, 1, 1)),),
    ),
    "graph-native-m16": (
        ("blackwell-native", (16, 1024, 1024), torch.bfloat16, 0.75, True),
        (("cudnn_smk_k4_n16_bf16:alpha1:pdl1", (64, 1, 1)),),
    ),
    "graph-native-split-k": (
        ("blackwell-native", (1, 4096, 4096), torch.bfloat16, 0.75, True),
        (("cudnn_smk_k4_z4c_bf16:alpha1:pdl1", (64, 1, 4)),),
    ),
    "tiled-m768-n2112-k2048-alpha-pdl": (
        ("blackwell-tiled", (768, 2112, 2048), torch.bfloat16, 1.0, True),
        (("cute_tcgen05_m128_cg2_n128_bf16:alpha1:pdl1", (102, 1, 1)),),
    ),
}
# Program (``MODULES`` key suffix) behind every cell the rows above reach.
EXPECTED_PROGRAMS = {
    "cudnn_cp_async_bf16:alpha0:pdl1:flat": "734a89d3a2dae4a7cb1e",
    "cudnn_cp_async_bf16:alpha1:pdl0:grid2d": "525782f3f3e0be42ccb5",
    "cudnn_cp_async_bf16:alpha1:pdl1:grid2d": "5231418224f82e216679",
    "cudnn_cp_async_f16:alpha0:pdl1:grid2d": "17f70c3774940e052131",
    "cudnn_cp_async_f16:alpha1:pdl0:grid2d": "931adc9ba7101cfcd57e",
    "cudnn_smk_k4_n16_bf16:alpha0:pdl0": "ffb4392d5fa01f008308",
    "cudnn_smk_k4_n16_bf16:alpha1:pdl1": "7468dc6cc46a70c1decc",
    "cudnn_smk_k4_n16_f16:alpha1:pdl1": "1d093cef4e9d03349cbd",
    "cudnn_smk_k4_z4c_bf16:alpha0:pdl1": "af124eba68ad18229e9e",
    "cudnn_smk_k4_z4c_bf16:alpha1:pdl1": "916900985e3576f137a2",
    "cudnn_tcgen05_m128_cg2_bf16:alpha1:pdl1": "f09a6b112cd48156248b",
    "cudnn_tma_bf16:alpha0:pdl1:grid2d": "245b04e342922bec79a0",
    "cute_bf16:alpha0:pdl0:grid2d": "1bf5950bdb68b3a00594",
    "cute_bf16:alpha1:pdl0:flat": "2f1f07fe72294a63de83",
    "cute_smk_s1_bf16:alpha0:pdl1": "1ca53a59dd02a007f622",
    "cute_smk_s1_bf16:alpha1:pdl0": "7ab95684b2efbc6cbe8e",
    "cute_smk_s1_m16_z4_bf16:alpha0:pdl0": "2cc4a32a57472117df41",
    "cute_smk_s1_m16_z4_bf16:alpha1:pdl1": "dfbf721a98d42045de27",
    "cute_smk_s2_m32_z6_bf16:alpha1:pdl1": "2b7eedebd8f3fb650580",
    "cute_tcgen05_m128_cg2_n128_bf16:alpha1:pdl1": "03663ccb5c3794adf73f",
    "cute_warp_mma_m16_k16_bf16:alpha1:pdl1": "964205318c4b32d14401",
    "cute_warp_mma_m16_k32_bf16:alpha0:pdl0": "6b97292c5497a02ffed5",
    "cute_warp_mma_m16_k48_bf16:alpha0:pdl1": "4d0cccec3b78ae116caa",
    "cute_warp_mma_m64_bf16:alpha1:pdl1": "bee1a475ab153a6e8530",
}


@pytest.mark.parametrize("case_id", list(EXPECTED_ROUTES))
def test_blackwell_bf16_fp4_route_selection(case_id):
    """Each row selects its pinned cells, grids and programs."""

    inputs, expected = EXPECTED_ROUTES[case_id]
    backend, (m, n, k), out_dtype, alpha_value, enable_pdl = inputs
    stages = _select_route(
        m,
        n,
        k,
        tiled=backend == "blackwell-tiled",
        output_bf16=out_dtype == torch.bfloat16,
        has_alpha=alpha_value is not None,
        enable_pdl=enable_pdl,
        sm_count=ROUTE_SM_COUNT,
    )
    # ``_cell_key`` takes every stage field but the grid; a signature change
    # updates this call and the cell spellings in the tables above.
    actual = tuple(
        (_cell_key(component, has_alpha, pdl, flat_grid), grid)
        for component, has_alpha, pdl, flat_grid, grid in stages
    )
    assert actual == expected
    for cell, _grid in actual:
        assert ROUTES[cell] == f"cake_blackwell_bf16_fp4_{EXPECTED_PROGRAMS[cell]}"


def test_blackwell_bf16_fp4_route_table():
    """Every route names a registered program and every program is routed."""

    assert set(ROUTES.values()) == set(MODULES)


def test_blackwell_bf16_fp4_split_workspace_follows_the_program():
    """The split-K planes workspace is bound per routed program, from its ``MODULES`` record.

    The eight-plane cell's PDL programs reduce through a (1, 1, 8) cluster and take no workspace; its non-PDL
    programs keep the FP32 planes and the u32 completion counters.
    """

    for enable_pdl, cluster, workspace in (
        (True, (1, 1, 8), None),
        (False, (1, 1, 1), (8, 4096, "completion_u32", torch.uint32)),
    ):
        (spec,) = _route_plan(
            16,
            4096,
            4096,
            tiled=True,
            output_bf16=True,
            has_alpha=False,
            enable_pdl=enable_pdl,
            sm_count=ROUTE_SM_COUNT,
        )
        assert spec.cell == f"cute_smk_s2_p2_n2_m16_z8_bf16:alpha0:pdl{int(enable_pdl)}"
        assert tuple(spec.cluster_dims) == cluster
        assert spec.workspace == workspace
