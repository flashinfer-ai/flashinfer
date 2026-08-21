# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from flashinfer import mm_bf16_swiglu, prepare_bf16_swiglu_weight
from flashinfer.utils import is_sm100a_supported


@pytest.mark.parametrize("input_order", ["gate_up", "up_gate"])
def test_prepare_bf16_swiglu_weight_layout_cpu(input_order: str):
    n, k = 128, 128
    gate = torch.arange(n, dtype=torch.bfloat16).view(n, 1).expand(n, k).clone()
    up = -torch.arange(1, n + 1, dtype=torch.bfloat16).view(n, 1).expand(n, k).clone()
    canonical = (
        torch.cat((gate, up), dim=0)
        if input_order == "gate_up"
        else torch.cat((up, gate), dim=0)
    )

    prepared = prepare_bf16_swiglu_weight(canonical, input_order=input_order)

    assert prepared.shape == (k, 2 * n)
    assert prepared.dtype == torch.bfloat16
    assert prepared.device == canonical.device
    assert prepared.T.is_contiguous()
    assert prepared.data_ptr() != canonical.data_ptr()

    physical = prepared.T.view(n // 16, 2, 16, k)
    torch.testing.assert_close(physical[:, 0].reshape(n, k), up, rtol=0, atol=0)
    torch.testing.assert_close(physical[:, 1].reshape(n, k), gate, rtol=0, atol=0)


@pytest.mark.parametrize(
    "weight,error,match",
    [
        (torch.empty(128, 128, dtype=torch.float32), TypeError, "bfloat16"),
        (torch.empty(128, 128, dtype=torch.bfloat16).T, ValueError, "row-major"),
        (torch.empty(127, 128, dtype=torch.bfloat16), ValueError, "positive even"),
        (torch.empty(192, 128, dtype=torch.bfloat16), ValueError, "divisible by 64"),
        (torch.empty(128, 64, dtype=torch.bfloat16), ValueError, "multiple of 128"),
    ],
)
def test_prepare_bf16_swiglu_weight_rejects_invalid_contract(
    weight: torch.Tensor, error: type[Exception], match: str
):
    with pytest.raises(error, match=match):
        prepare_bf16_swiglu_weight(weight)


def test_prepare_bf16_swiglu_weight_rejects_invalid_order():
    weight = torch.empty(128, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="input_order"):
        prepare_bf16_swiglu_weight(weight, input_order="gate-first")


def test_mm_bf16_swiglu_fi_trace_cpu():
    m, n, k = 3, 64, 128
    a = torch.zeros((m, k), dtype=torch.bfloat16)
    weight = torch.zeros((2 * n, k), dtype=torch.bfloat16)
    prepared = prepare_bf16_swiglu_weight(weight)

    definition = mm_bf16_swiglu.fi_trace(a=a, b=prepared, pdl=False)
    assert definition["op_type"] == "gemm_bf16_swiglu"
    assert definition["axes"]["M"]["type"] == "var"
    assert definition["axes"]["gate_up_size"]["value"] == 2 * n
    assert definition["axes"]["N"]["type"] == "var"
    assert definition["axes"]["K"]["value"] == k
    assert definition["outputs"]["C"]["shape"] == ["M", "N"]


def _require_bf16_swiglu_gpu(device: torch.device | int | None = None):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    major, minor = torch.cuda.get_device_capability(device)
    cc = major * 10 + minor
    if not mm_bf16_swiglu.is_compute_capability_supported(cc):
        pytest.skip(f"mm_bf16_swiglu does not support SM{cc}")
    # The API additionally requires CUDA 12.8+, which the compute-capability
    # check cannot express; without this every test below hard-fails on an
    # older toolkit instead of skipping.
    cuda_device = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if not is_sm100a_supported(cuda_device):
        pytest.skip("mm_bf16_swiglu requires SM100/SM103 with CUDA 12.8+")
    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is unavailable")


def _swiglu_reference(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Preserve the unfused BF16 GEMM output boundary."""
    gate_up = F.linear(a, weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


def _assert_swiglu_close(reference: torch.Tensor, actual: torch.Tensor) -> None:
    cosine = F.cosine_similarity(
        reference.float().flatten(), actual.float().flatten(), dim=0
    )
    assert cosine > 0.999
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)


def _swiglu_fp32_reference(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Skip the BF16 hand-off that :func:`_swiglu_reference` preserves."""
    gate, up = F.linear(a.float(), weight.float()).chunk(2, dim=-1)
    return (F.silu(gate) * up).to(torch.bfloat16)


def _bitwise_match_fraction(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return (actual == reference).float().mean().item()


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(1, 4096, 128), (3, 512, 6144), (16, 2560, 8192)],
    ids=["split_k1", "split_k4", "split_k2"],
)
def test_mm_bf16_swiglu_pins_bf16_intermediate_boundary(m: int, n: int, k: int):
    """Pin the ``FP32 -> BF16 -> FP32`` hand-off that defines this op.

    ``_assert_swiglu_close`` passes against both the documented BF16-boundary
    reference and a pure-FP32 SwiGLU, so it cannot catch an epilogue that drops
    the round trip.  Bitwise agreement separates them by a wide margin: the
    kernel reproduces the BF16 boundary on >99% of elements (split-K reduction
    order and the fast-math sigmoid explain the rest) while the FP32 boundary
    agrees on roughly two thirds.
    """
    _require_bf16_swiglu_gpu()
    torch.manual_seed(4096 + m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    ).to(torch.bfloat16)
    prepared = prepare_bf16_swiglu_weight(weight)

    bf16_boundary = _swiglu_reference(a, weight)
    assert _bitwise_match_fraction(mm_bf16_swiglu(a, prepared), bf16_boundary) > 0.95
    # Keep the bound above meaningful: an FP32-accumulate epilogue must not be
    # able to clear it at this shape.
    fp32_boundary = _swiglu_fp32_reference(a, weight)
    assert _bitwise_match_fraction(fp32_boundary, bf16_boundary) < 0.85


def test_mm_bf16_swiglu_glm_shared_expert_shape():
    """Correctness anchor for GLM-5.2's TP4 shared-expert gate/up GEMM."""
    _require_bf16_swiglu_gpu()
    torch.manual_seed(42)
    m, n, k = 3, 512, 6144
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    ).to(torch.bfloat16)
    prepared = prepare_bf16_swiglu_weight(weight)

    reference = _swiglu_reference(a, weight)
    outputs = [mm_bf16_swiglu(a, prepared, pdl=pdl) for pdl in (False, True)]
    for out in outputs:
        assert out.shape == (m, n)
        _assert_swiglu_close(reference, out)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("m", "n", "k", "expected_mma_n", "expected_split_k"),
    [
        pytest.param(1, 64, 128, 8, 1, id="mma_n8-split_k1-short_k"),
        pytest.param(7, 192, 768, 8, 3, id="mma_n8-split_k3"),
        pytest.param(31, 256, 1024, 8, 4, id="mma_n8-split_k4"),
        pytest.param(32, 1024, 1280, 8, 2, id="mma_n8-split_k2"),
        # Wide N exhausts the one-wave budget, so the selector widens the tile
        # instead of adding kernel-N tiles.
        pytest.param(32, 4096, 128, 16, 1, id="mma_n16-wave-bound"),
        pytest.param(32, 8192, 128, 32, 1, id="mma_n32-wave-bound"),
        # The same wide tiles with an M tail and split-K enabled: epilogue
        # predication plus a mailbox carrying 8/16 values per gate/up half
        # rather than the 4 that mma_n=8 produces.
        pytest.param(17, 2048, 256, 16, 2, id="mma_n16-split_k2-m_tail"),
        pytest.param(17, 4096, 256, 32, 2, id="mma_n32-split_k2-m_tail"),
        pytest.param(1, 4096, 128, 8, 1, id="wide_n-shallow_k"),
        pytest.param(32, 64, 8192, 8, 4, id="narrow_n-deep_k"),
    ],
)
def test_mm_bf16_swiglu_default_tactic_paths(
    m: int,
    n: int,
    k: int,
    expected_mma_n: int,
    expected_split_k: int,
):
    """Cover every default MMA-N tile and split-K dispatch path.

    The expected tiles depend on the device's SM count and L2 capacity, but
    are stable across every part this kernel supports.
    """
    _require_bf16_swiglu_gpu()
    from flashinfer.gemm.kernels.dense_bf16_swiglu_sm100_splitk import (
        default_swiglu_tactic,
    )

    tactic = default_swiglu_tactic(m, n, k)
    assert tactic.mma_n == expected_mma_n
    assert tactic.split_k == expected_split_k

    torch.manual_seed(2000 + m + n + k)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    ).to(torch.bfloat16)
    prepared = prepare_bf16_swiglu_weight(weight)

    actual = mm_bf16_swiglu(a, prepared)
    _assert_swiglu_close(_swiglu_reference(a, weight), actual)


@pytest.mark.parametrize(
    ("n", "k"),
    [(512, 6144), (1024, 6144), (2560, 8192), (8192, 2048), (256, 8192)],
)
def test_mm_bf16_swiglu_default_tactic_occupancy_does_not_collapse(n: int, k: int):
    """Guard the M>16 cliff: growing M must never shrink the launched grid.

    The paired epilogue locks mma_m to 128, so a selector that inherits a
    tile chosen for mma_m=64 halves grid-x exactly as M crosses 16 and makes
    the fused kernel slower than the unfused GEMM+SwiGLU it replaces.
    """
    _require_bf16_swiglu_gpu()
    from flashinfer.gemm.kernels.dense_bf16_swiglu_sm100_splitk import (
        _swiglu_tactic_footprint,
        default_swiglu_tactic,
    )

    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    # Bounded at 32 on purpose. Past that the selector legitimately trades
    # CTAs for weight traffic by widening ``mma_n``, so a monotonic grid is
    # not the right invariant there; the cliff this guards is the low-M one.
    ctas = [
        _swiglu_tactic_footprint(default_swiglu_tactic(m, n, k), m, n, k)[0]
        for m in range(1, 33)
    ]
    regressions = [
        (m, before, after)
        for m, (before, after) in enumerate(itertools.pairwise(ctas), start=1)
        if after < before
    ]
    assert not regressions, (
        f"default tactic loses parallelism as M grows for N={n}, K={k}: "
        f"(M, CTAs before, CTAs after) = {regressions}"
    )
    assert max(ctas) <= sm_count, (
        f"default tactic exceeds one wave for N={n}, K={k}: "
        f"max CTAs {max(ctas)} > {sm_count} SMs"
    )


@pytest.mark.parametrize("m", [1, 3, 8, 16, 32, 33, 64])
def test_mm_bf16_swiglu_decode_m_values_and_pdl(m: int):
    """Exercise dynamic-M tails and both PDL launch variants."""
    _require_bf16_swiglu_gpu()
    torch.manual_seed(1000 + m)
    n, k = 64, 128
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    ).to(torch.bfloat16)
    prepared = prepare_bf16_swiglu_weight(weight)
    reference = _swiglu_reference(a, weight)

    pdl_off = mm_bf16_swiglu(a, prepared, pdl=False)
    pdl_on = mm_bf16_swiglu(a, prepared, pdl=True)
    _assert_swiglu_close(reference, pdl_off)
    _assert_swiglu_close(reference, pdl_on)
    torch.testing.assert_close(pdl_off, pdl_on, rtol=0, atol=0)


@pytest.mark.parametrize("pdl", [False, True], ids=["pdl_off", "pdl_on"])
def test_mm_bf16_swiglu_cuda_graph_replay(pdl: bool):
    """Capture the public API and verify replay reads refreshed inputs."""
    _require_bf16_swiglu_gpu()
    torch.manual_seed(3141 + int(pdl))
    m, n, k = 3, 64, 128
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    ).to(torch.bfloat16)
    prepared = prepare_bf16_swiglu_weight(weight)
    graph_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    # Compile, allocate, and establish stream dependencies before capture.
    for _ in range(3):
        mm_bf16_swiglu(a, prepared, pdl=pdl, out=graph_out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = mm_bf16_swiglu(a, prepared, pdl=pdl, out=graph_out)
    assert captured_out is graph_out

    replay_a = torch.randn_like(a)
    a.copy_(replay_a)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replay_out = graph_out.clone()

    eager_out = mm_bf16_swiglu(a, prepared, pdl=pdl)
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_out, eager_out, rtol=0, atol=0)
    _assert_swiglu_close(_swiglu_reference(a, weight), replay_out)


def test_mm_bf16_swiglu_preallocated_out():
    _require_bf16_swiglu_gpu()
    m, n, k = 1, 64, 128
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16)
    prepared = prepare_bf16_swiglu_weight(weight)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    result = mm_bf16_swiglu(a, prepared, pdl=True, out=out)
    assert result is out
    _assert_swiglu_close(_swiglu_reference(a, weight), result)


@pytest.mark.parametrize(
    ("aliased_input", "skip_check"),
    [
        pytest.param("a", False, id="public-check-a"),
        pytest.param("b", True, id="runtime-check-b"),
    ],
)
def test_mm_bf16_swiglu_rejects_overlapping_out(aliased_input: str, skip_check: bool):
    _require_bf16_swiglu_gpu()
    m, n, k = 1, 128, 128
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16)
    prepared = prepare_bf16_swiglu_weight(weight)
    if aliased_input == "a":
        out = a
    else:
        out = prepared.T.view(-1)[: m * n].view(m, n)

    with pytest.raises(
        ValueError,
        match=rf"out must not overlap {aliased_input} storage",
    ):
        mm_bf16_swiglu(a, prepared, out=out, skip_check=skip_check)


def test_mm_bf16_swiglu_accepts_non_current_cuda_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    original_device = torch.cuda.current_device()
    target_device = next(
        (
            device
            for device in range(torch.cuda.device_count())
            if device != original_device
            and mm_bf16_swiglu.is_compute_capability_supported(
                10 * torch.cuda.get_device_capability(device)[0]
                + torch.cuda.get_device_capability(device)[1]
            )
        ),
        None,
    )
    if target_device is None:
        pytest.skip("no non-current CUDA device supports mm_bf16_swiglu")
    _require_bf16_swiglu_gpu(target_device)

    try:
        torch.cuda.set_device(original_device)
        m, n, k = 1, 64, 128
        device = torch.device("cuda", target_device)
        a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
        weight = torch.randn((2 * n, k), device=device, dtype=torch.bfloat16)
        prepared = prepare_bf16_swiglu_weight(weight)

        assert torch.cuda.current_device() != target_device
        actual = mm_bf16_swiglu(a, prepared)
        assert torch.cuda.current_device() == original_device
        torch.cuda.synchronize(target_device)
        assert actual.device == device
        _assert_swiglu_close(_swiglu_reference(a, weight), actual)
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("pdl_type", TypeError, "pdl must be bool"),
        ("rank", ValueError, "must be 2-D"),
        ("dtype", TypeError, "both be bfloat16"),
        ("a_layout", ValueError, "a must be contiguous"),
        ("b_layout", ValueError, "b must be column-major"),
        ("m_range", ValueError, "1 <= M <= 64"),
        ("shape", ValueError, "incompatible shapes"),
        ("n_alignment", ValueError, "N must be divisible by 64"),
        ("k_alignment", ValueError, "K must be a positive multiple of 128"),
        ("out_dtype", TypeError, "out must be bfloat16"),
        ("out_shape", ValueError, "out must have shape"),
        ("out_layout", ValueError, "out must be contiguous"),
        ("alignment", ValueError, "32-byte aligned"),
    ],
)
def test_mm_bf16_swiglu_rejects_invalid_cuda_contract(
    case: str, error: type[Exception], match: str
):
    _require_bf16_swiglu_gpu()
    a = torch.empty((1, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    b = prepare_bf16_swiglu_weight(weight)
    kwargs: dict[str, object] = {}

    if case == "pdl_type":
        kwargs["pdl"] = 1
    elif case == "rank":
        a = a.unsqueeze(0)
    elif case == "dtype":
        a = a.float()
    elif case == "a_layout":
        a = torch.empty((128, 2), device="cuda", dtype=torch.bfloat16).T[:1]
    elif case == "b_layout":
        b = b.contiguous()
    elif case == "m_range":
        a = a.expand(65, -1).clone()
    elif case == "shape":
        b = torch.empty((128, 256), device="cuda", dtype=torch.bfloat16).T
    elif case == "n_alignment":
        b = torch.empty((96, 128), device="cuda", dtype=torch.bfloat16).T
    elif case == "k_alignment":
        a = torch.empty((1, 64), device="cuda", dtype=torch.bfloat16)
        b = torch.empty((128, 64), device="cuda", dtype=torch.bfloat16).T
    elif case == "out_dtype":
        kwargs["out"] = torch.empty((1, 64), device="cuda", dtype=torch.float32)
    elif case == "out_shape":
        kwargs["out"] = torch.empty((1, 65), device="cuda", dtype=torch.bfloat16)
    elif case == "out_layout":
        a = a.expand(2, -1).clone()
        kwargs["out"] = torch.empty((2, 128), device="cuda", dtype=torch.bfloat16)[
            :, ::2
        ]
    elif case == "alignment":
        a = torch.empty((1, 129), device="cuda", dtype=torch.bfloat16)[:, 1:]
        assert a.is_contiguous() and a.data_ptr() % 32
    else:
        raise AssertionError(f"unhandled test case {case}")

    with pytest.raises(error, match=match):
        mm_bf16_swiglu(a, b, **kwargs)
