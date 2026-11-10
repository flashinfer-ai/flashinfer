"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Tests for the experimental VibeCUDA Kimi-K3 AttnRes backend (unfused).

Upstream-CAKE-test to vibecuda-test mapping (tests/experimental/
test_cake_kimi_k3_attn_res.py):
  test_plan_route_policy / test_plan_route_bootstrap_variants /
    test_plan_route_rejects_bad_shapes
      -> test_production_route_policy (the compiled frozen ladder) and
         test_prepare_rejects_unfused_violations
  test_registered_keys_cover_the_measured_grid_when_programs_exist
      -> test_route_id_covers_the_evaluation_grid
  test_matches_reference -> test_matches_reference (same FP32 reference
      callable, same input generator style, same tolerances)
  test_semantic_variants_take_the_bootstrap
      -> not applicable: the vibecuda backend implements the unfused variant
         only; fused semantics are rejected loudly (tested)
  test_row_padded_layout_takes_the_bootstrap
      -> test_row_padded_bank_layout (row padding is natively supported via
         the bank row stride)
  test_launch_makes_no_allocation_and_replays_in_a_graph
      -> test_launch_makes_no_allocation_and_replays_in_a_graph
  test_prepare_rejects_bad_bindings
      -> test_prepare_rejects_unfused_violations (+ read-only input checks)
The precision reference is the upstream tests' own
``reference_kimi_k3_attn_res`` (FP32, exact BF16 state mutations), called
with delta=None / write=-1 / qk_weight=ones and eps=output_norm_eps=1e-6.
"""

import pytest
import torch

from flashinfer.experimental.cake_kimi_k3_attn_res.cake_backend import (
    HIDDEN_SIZE,
    reference_kimi_k3_attn_res,
)
from flashinfer.experimental.vibecuda_kimi_k3_attn_res.vibecuda_backend import (
    prepare_kimi_k3_attn_res as vibecuda_prepare,
    registered_arches,
    vibecuda_supported_device,
)
from flashinfer.kimi_k3_attn_res import kimi_k3_attn_res, prepare_kimi_k3_attn_res

# Same evaluation contract as the upstream CAKE tests.
ATOL = 8e-2
RTOL = 3e-2
EPS = 1e-6


def _device_arch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    if not vibecuda_supported_device(device):
        pytest.skip("SM100 / SM103 required")
    arch = {
        (10, 0): "sm_100a",
        (10, 3): "sm_103a",
    }[tuple(torch.cuda.get_device_capability(device))]
    if arch not in registered_arches():
        pytest.skip("no VibeCUDA AttnRes module registered for this architecture")
    return device, arch


def _make_inputs(M, K, *, seed=40900000, row_padding=0):
    g = torch.Generator(device="cuda").manual_seed(seed + M * 131 + K)
    prefix = torch.empty((M, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    prefix.uniform_(-1.0, 1.0, generator=g)
    # base storage may carry row padding so blocks.stride(0) > K * H
    base = torch.empty(
        (M, K + row_padding, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
    )
    base.uniform_(-1.0, 1.0, generator=g)
    blocks = base[:, :K]
    cw = torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    cw.normal_(1.0, 0.05, generator=g)
    ow = torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    ow.normal_(1.0, 0.05, generator=g)
    out = torch.full(
        (M, HIDDEN_SIZE), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    return dict(prefix=prefix, blocks=blocks, cw=cw, ow=ow, out=out, num_blocks=K)


def _reference(inputs):
    """The upstream tests' FP32 reference on the unfused variant."""
    expected = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
    reference_kimi_k3_attn_res(
        expected["prefix"],
        None,  # delta
        expected["blocks"],
        expected.get("reference_norm_weight", expected["cw"]),
        expected["cw"]
        if "reference_norm_weight" in expected
        else torch.ones(
            HIDDEN_SIZE, device=expected["prefix"].device, dtype=torch.bfloat16
        ),
        expected["ow"],
        expected["out"],
        num_blocks=inputs["num_blocks"],
        block_write_idx=-1,
        eps=EPS,
        output_norm_eps=EPS,
    )
    return expected


def _check(inputs, expected, pristine):
    torch.testing.assert_close(
        inputs["out"].float(), expected["out"].float(), atol=ATOL, rtol=RTOL
    )
    # the output must be fully written (NaN sentinel cannot survive)
    assert torch.isfinite(inputs["out"].float()).all()
    # the unfused operator never mutates prefix or the bank
    assert torch.equal(
        inputs["prefix"].view(torch.int16), pristine["prefix"].view(torch.int16)
    )
    assert torch.equal(
        inputs["blocks"].view(torch.int16), pristine["blocks"].view(torch.int16)
    )
    assert torch.equal(inputs["cw"].view(torch.int16), pristine["cw"].view(torch.int16))
    assert torch.equal(inputs["ow"].view(torch.int16), pristine["ow"].view(torch.int16))


# ---------------------------------------------------------------------------
# Production route policy (compiled frozen ladder, on-device)
# ---------------------------------------------------------------------------

ROUTE_ROWS = [
    # (num_blocks, M, expected impl; nsm-dependent rows expanded below)
    (0, 1, 97),
    (0, 16384, 97),
    (1, 1, 97),
    (1, 56, 97),
    (1, 57, 87),
    (1, 1024, 87),
    (2, 8, 138),
    (2, 288, 138),
    (2, 289, 133),
    (2, 9364, 133),
    (3, 31, 85),
    (4, 16, 138),
    (4, 288, 138),
    (4, 289, 87),
    (5, 95, 85),
    (6, 3, 85),
    (7, 1, 136),
    (8, 56, 134),
    (8, 57, 112),
    (8, 64, 112),
    (8, 151, 112),
    (8, 152, 134),
    (8, 2048, 134),
]


@pytest.mark.parametrize("nvb,M,impl", ROUTE_ROWS)
def test_production_route_policy(nvb, M, impl):
    device, arch = _device_arch()
    from flashinfer.experimental.vibecuda_kimi_k3_attn_res.vibecuda_jit import (
        load_vibecuda_kimi_k3_attn_res_module,
    )

    module = load_vibecuda_kimi_k3_attn_res_module(arch)
    assert module.route_id(nvb, M) == impl


def test_sm_count_dependent_routes():
    device, arch = _device_arch()
    from flashinfer.experimental.vibecuda_kimi_k3_attn_res.vibecuda_jit import (
        load_vibecuda_kimi_k3_attn_res_module,
    )

    module = load_vibecuda_kimi_k3_attn_res_module(arch)
    nsm = torch.cuda.get_device_properties(device).multi_processor_count
    assert module.num_sms() == nsm
    # chpg CTA-per-token bands cut over at the runtime SM count
    for nvb, lo in ((3, 32), (5, 96), (6, 4), (7, 1)):
        assert module.route_id(nvb, lo) == 136
        assert module.route_id(nvb, nsm) == 136
        assert module.route_id(nvb, nsm + 1) == (
            85 if nvb != 7 or nsm + 1 <= 287 else 87
        )
    # nvb==8 T > 2048: chtclc (139) only on the 152-SM SM100 route, else 87
    assert module.route_id(8, 4096) == (139 if nsm == 152 else 87)
    assert module.route_id(8, 9364) == (139 if nsm == 152 else 87)


def test_nvb6_large_t_route_policy():
    # SM100 payload adopts the TMEM persistent route (87) for the nvb==6
    # large-T tail (measured crossover at T=2048: 87 wins ~1-3% at 2048..4096,
    # ties at 1024..1536, loses below; the SM103 payload keeps 85, its route
    # table is tuned independently).
    device, arch = _device_arch()
    from flashinfer.experimental.vibecuda_kimi_k3_attn_res.vibecuda_jit import (
        load_vibecuda_kimi_k3_attn_res_module,
    )

    module = load_vibecuda_kimi_k3_attn_res_module(arch)
    large_t_route = 87 if arch == "sm_100a" else 85
    assert module.route_id(6, 1024) == 85
    assert module.route_id(6, 2048) == large_t_route
    assert module.route_id(6, 4096) == large_t_route
    assert module.route_id(6, 16384) == large_t_route


@pytest.mark.parametrize(
    "M,K",
    [
        (1, 1),
        (8, 2),
        (16, 4),
        (32, 4),
        (64, 8),
        (128, 8),
        (256, 8),
        (512, 8),
        (1024, 8),
        (2048, 8),
        (4096, 8),
        (8192, 8),
        (9364, 8),
        (3, 3),
        (32, 3),
        (96, 5),
        (4, 6),
        (1, 7),
        (300, 7),
        (512, 2),
    ],
)
def test_route_id_covers_the_evaluation_grid(M, K):
    device, arch = _device_arch()
    from flashinfer.experimental.vibecuda_kimi_k3_attn_res.vibecuda_jit import (
        load_vibecuda_kimi_k3_attn_res_module,
    )

    module = load_vibecuda_kimi_k3_attn_res_module(arch)
    assert module.route_id(K, M) > 0


# ---------------------------------------------------------------------------
# Device tests
# ---------------------------------------------------------------------------

GPU_ROWS = [
    (1, 1),
    (8, 2),
    (16, 4),
    (32, 4),
    (64, 8),
    (128, 8),
    (256, 8),
    (512, 8),
    (1024, 8),
    (2048, 8),
    (4096, 8),
    (8192, 8),
    (9364, 8),
    (3, 3),
    (32, 3),
    (96, 5),
    (4, 6),
    (1, 7),
    (300, 7),
    (512, 2),
    (289, 4),
    (4, 2),
    (289, 2),
]

# Complete shape grid from the public CAKE AttnRes benchmark at PR6070.
# Keep the original SGLang rows and their distinct input distribution above.
CAKE_UNFUSED_ROWS = [
    (m, k)
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    for k in (0, 1, 4, 8)
] + [(m, k) for m in (1, 4096) for k in (2, 3, 5, 6, 7)]


def _make_cake_unfused_inputs(M, K):
    g = torch.Generator(device="cuda").manual_seed(31000)
    prefix = torch.empty((M, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    prefix.uniform_(-1, 1, generator=g)
    blocks = torch.empty((M, 8, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    blocks.uniform_(-1, 1, generator=g)
    norm = torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    norm.normal_(1, 0.05, generator=g)
    norm.fill_(1)
    cw = torch.empty_like(norm).normal_(0, 0.02, generator=g)
    ow = torch.empty_like(norm).normal_(1, 0.05, generator=g)
    out = torch.full_like(prefix, float("nan"))
    return dict(
        prefix=prefix,
        blocks=blocks,
        cw=cw,
        ow=ow,
        out=out,
        num_blocks=K,
        reference_norm_weight=norm,
    )


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("M,K", CAKE_UNFUSED_ROWS)
def test_complete_cake_unfused_grid(M, K, pdl):
    device, arch = _device_arch()
    inputs = _make_cake_unfused_inputs(M, K)
    expected = _reference(inputs)
    pristine = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
    runner = prepare_kimi_k3_attn_res(
        inputs["prefix"],
        None,
        inputs["blocks"],
        inputs["cw"],
        None,
        inputs["ow"],
        inputs["out"],
        num_blocks=K,
        eps=EPS,
        output_norm_eps=EPS,
        enable_pdl=pdl,
        backend="vibecuda",
    )
    runner.launch()
    torch.cuda.synchronize()
    _check(inputs, expected, pristine)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("M,K", GPU_ROWS)
def test_matches_reference(M, K, pdl):
    device, arch = _device_arch()
    inputs = _make_inputs(M, K)
    expected = _reference(inputs)
    pristine = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
    runner = prepare_kimi_k3_attn_res(
        inputs["prefix"],
        None,
        inputs["blocks"],
        inputs["cw"],
        None,
        inputs["ow"],
        inputs["out"],
        num_blocks=K,
        eps=EPS,
        output_norm_eps=EPS,
        enable_pdl=pdl,
        backend="vibecuda",
    )
    assert (
        runner.launch(
            inputs["prefix"],
            inputs["blocks"],
            inputs["cw"],
            inputs["ow"],
            inputs["out"],
        )
        is inputs["out"]
    )
    torch.cuda.synchronize()
    _check(inputs, expected, pristine)


@pytest.mark.parametrize("M,K,padding", [(7, 4, 2), (8, 1, 7), (56, 1, 2), (57, 1, 7)])
def test_row_padded_bank_layout(M, K, padding):
    device, arch = _device_arch()
    inputs = _make_inputs(M, K, row_padding=padding)
    assert inputs["blocks"].stride(0) == (K + padding) * HIDDEN_SIZE
    expected = _reference(inputs)
    pristine = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
    kimi_k3_attn_res(
        inputs["prefix"],
        None,
        inputs["blocks"],
        inputs["cw"],
        None,
        inputs["ow"],
        inputs["out"],
        num_blocks=K,
        eps=EPS,
        output_norm_eps=EPS,
        backend="vibecuda",
    )
    torch.cuda.synchronize()
    _check(inputs, expected, pristine)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    "family,M,K",
    [("sglang", m, k) for m, k in GPU_ROWS]
    + [("cake", m, k) for m, k in CAKE_UNFUSED_ROWS],
)
def test_launch_makes_no_allocation_and_replays_in_a_graph(family, M, K, pdl):
    device, arch = _device_arch()
    inputs = _make_cake_unfused_inputs(M, K) if family == "cake" else _make_inputs(M, K)
    runner = vibecuda_prepare(
        inputs["prefix"],
        None,
        inputs["blocks"],
        inputs["cw"],
        None,
        inputs["ow"],
        inputs["out"],
        num_blocks=K,
        eps=EPS,
        output_norm_eps=EPS,
        enable_pdl=pdl,
    )
    runner.launch(
        inputs["prefix"], inputs["blocks"], inputs["cw"], inputs["ow"], inputs["out"]
    )
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner.launch(
        inputs["prefix"], inputs["blocks"], inputs["cw"], inputs["ow"], inputs["out"]
    )
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    # Graph replay follows the device contents of the bound tensors.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        runner.launch(
            inputs["prefix"],
            inputs["blocks"],
            inputs["cw"],
            inputs["ow"],
            inputs["out"],
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner.launch(
                inputs["prefix"],
                inputs["blocks"],
                inputs["cw"],
                inputs["ow"],
                inputs["out"],
            )
    torch.cuda.synchronize()
    inputs["prefix"].neg_()
    inputs["blocks"].mul_(0.5)
    pristine = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
    inputs["out"].fill_(float("nan"))
    expected = _reference(inputs)
    graph.replay()
    torch.cuda.synchronize()
    _check(inputs, expected, pristine)


def test_prepare_rejects_unfused_violations():
    device, arch = _device_arch()
    inputs = _make_inputs(2, 4)
    base = dict(
        prefix=inputs["prefix"],
        blocks=inputs["blocks"],
        norm_weight=inputs["cw"],
        output_norm_weight=inputs["ow"],
        out=inputs["out"],
    )
    with pytest.raises(NotImplementedError):
        vibecuda_prepare(**base, delta=inputs["prefix"], qk_weight=None, num_blocks=4)
    with pytest.raises(NotImplementedError):
        vibecuda_prepare(
            **base, delta=None, qk_weight=None, num_blocks=4, block_write_idx=0
        )
    with pytest.raises(NotImplementedError):
        vibecuda_prepare(**base, delta=None, qk_weight=inputs["cw"], num_blocks=4)
    with pytest.raises(NotImplementedError):
        vibecuda_prepare(
            **{**base, "output_norm_weight": None},
            delta=None,
            qk_weight=None,
            num_blocks=4,
        )
    with pytest.raises(NotImplementedError):
        vibecuda_prepare(
            **base,
            delta=None,
            qk_weight=None,
            num_blocks=4,
            eps=1e-6,
            output_norm_eps=1e-5,
        )
    with pytest.raises(ValueError):
        vibecuda_prepare(**base, delta=None, qk_weight=None, num_blocks=9)
    with pytest.raises(ValueError):
        vibecuda_prepare(
            **{**base, "out": inputs["out"].float()},
            delta=None,
            qk_weight=None,
            num_blocks=4,
        )
    with pytest.raises(ValueError):
        vibecuda_prepare(
            **{**base, "prefix": inputs["prefix"].cpu()},
            delta=None,
            qk_weight=None,
            num_blocks=4,
        )


def test_prepare_rejects_unknown_backend():
    device, arch = _device_arch()
    inputs = _make_inputs(2, 4)
    with pytest.raises(ValueError):
        prepare_kimi_k3_attn_res(
            inputs["prefix"],
            None,
            inputs["blocks"],
            inputs["cw"],
            None,
            inputs["ow"],
            inputs["out"],
            num_blocks=4,
            backend="triton",
        )
