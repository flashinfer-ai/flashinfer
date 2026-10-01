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

import math

import pytest
import torch

import warnings

from flashinfer.fused_moe.bgmv_moe import (
    BGMVMoECakePlan,
    BGMVMoEPortablePlan,
    prepare_bgmv_moe,
)


_PERF_SHAPES = [
    (hidden_size, num_tokens, torch.bfloat16)
    for hidden_size in (3072, 2688)
    for num_tokens in (1, 4, 8, 32, 256, 512, 1024)
]
_FP16_SHAPES = [
    (3072, 8, torch.float16),
    (2688, 4, torch.float16),
]


_SUPPORTED_CAPABILITIES = ((9, 0), (10, 0), (10, 3))


def _require_cake_arch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() not in _SUPPORTED_CAPABILITIES:
        pytest.skip("generated Cake BGMV MoE tests require exact SM90, SM100 or SM103")


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def _make_inputs(
    hidden_size,
    num_tokens,
    dtype,
    *,
    arbitrary_routes=False,
    rank=32,
    num_slices=1,
    x_dtype=None,
    top_k=2,
    expert_sorted=False,
):
    torch.manual_seed(42)
    device = "cuda"
    num_experts = 128
    num_loras = 2
    num_pairs = num_tokens * top_k
    # Scales keep the shrink (~0.5) and the output (~1) at O(1) so the
    # 1e-2 tolerances are meaningful: an all-zero result must fail. The
    # previous 0.1 / 0.01 / 0.01 scales produced outputs below 1e-2.
    x = torch.randn(num_tokens, hidden_size, dtype=x_dtype or dtype, device=device)
    weight_scale = 0.5 / math.sqrt(hidden_size)
    lora_a_weights = [
        torch.randn(
            num_loras,
            num_experts,
            rank,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        * weight_scale
        for _ in range(num_slices)
    ]
    lora_b_weights = [
        torch.randn(
            num_loras,
            num_experts,
            hidden_size,
            rank,
            dtype=dtype,
            device=device,
        )
        * (2.0 / math.sqrt(rank))
        for _ in range(num_slices)
    ]
    sorted_token_ids = torch.arange(
        num_tokens, dtype=torch.int64, device=device
    ).repeat_interleave(top_k)
    expert_ids = torch.randint(
        0, num_experts, (num_pairs,), dtype=torch.int64, device=device
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    ).reshape(-1)
    lora_indices = torch.randint(
        0, num_loras, (num_tokens,), dtype=torch.int64, device=device
    )
    if num_tokens > 1:
        lora_indices[0] = -1
    if arbitrary_routes:
        order = (
            torch.arange(num_pairs, dtype=torch.int64, device=device)
            .reshape(num_tokens, top_k)
            .transpose(0, 1)
            .reshape(-1)
        )
        sorted_token_ids = sorted_token_ids[order].contiguous()
        expert_ids = expert_ids[order].contiguous()
        topk_weights = topk_weights[order].contiguous()
    if expert_sorted:
        # MoE dispatch order: pairs grouped by expert, so each token's routes
        # are spread over the whole pair list.
        order = torch.argsort(expert_ids, stable=True)
        sorted_token_ids = sorted_token_ids[order].contiguous()
        expert_ids = expert_ids[order].contiguous()
        topk_weights = topk_weights[order].contiguous()
    return (
        x,
        lora_a_weights,
        lora_b_weights,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        num_experts,
    )


def _reference(inputs):
    (
        x,
        lora_a_weights,
        lora_b_weights,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        _num_experts,
    ) = inputs
    num_tokens = x.shape[0]
    feat_outs = [int(w.shape[2]) for w in lora_b_weights]
    output = torch.zeros(
        num_tokens, sum(feat_outs), dtype=torch.float32, device=x.device
    )
    valid = (sorted_token_ids >= 0) & (sorted_token_ids < num_tokens)
    valid_pairs = torch.nonzero(valid, as_tuple=False).flatten()
    for start in range(0, valid_pairs.numel(), 64):
        pair_ids = valid_pairs[start : start + 64]
        tokens = sorted_token_ids[pair_ids]
        loras = lora_indices[tokens]
        active = loras >= 0
        if not bool(active.any()):
            continue
        pair_ids = pair_ids[active]
        tokens = tokens[active]
        loras = loras[active]
        experts = expert_ids[pair_ids]
        col = 0
        for lora_a, lora_b, feat_out in zip(
            lora_a_weights, lora_b_weights, feat_outs, strict=True
        ):
            a = lora_a[loras, experts].float()
            shrink = torch.bmm(
                x[tokens].float().unsqueeze(1), a.transpose(1, 2)
            ).squeeze(1)
            b = lora_b[loras, experts].float()
            delta = torch.bmm(b, shrink.unsqueeze(2)).squeeze(2)
            delta *= topk_weights[pair_ids].unsqueeze(1)
            output[:, col : col + feat_out].index_add_(0, tokens, delta)
            col += feat_out
    # Guard against vacuous comparisons: the reference must carry signal well
    # above the 1e-2 tolerances (10x atol) on every token that has a LoRA, so a kernel
    # that leaves its output zeroed cannot pass.
    active_tokens = (lora_indices >= 0).nonzero().flatten()
    assert bool((output[active_tokens].abs().amax(dim=1) > 0.1).all()), (
        "reference output too small for a meaningful comparison"
    )
    return output


@pytest.mark.parametrize(
    ("hidden_size", "num_tokens", "dtype"), _PERF_SHAPES + _FP16_SHAPES
)
def test_prepared_pipeline_matches_reference(hidden_size, num_tokens, dtype):
    _require_cake_arch()
    inputs = _make_inputs(hidden_size, num_tokens, dtype)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="cake")
    actual = plan.run()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    # Replays keep exact pointers but consume current tensor contents.
    inputs[0].mul_(0.75)
    expected_replay = _reference(inputs)
    actual_replay = plan.run()
    torch.testing.assert_close(actual_replay, expected_replay, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [4, 32])
def test_arbitrary_route_order_and_nondefault_stream(dtype, num_tokens):
    _require_cake_arch()
    inputs = _make_inputs(2688, num_tokens, dtype, arbitrary_routes=True)
    expected = _reference(inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        plan = prepare_bgmv_moe(*inputs, backend="cake")
        actual = plan.run()
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prepared_pipeline_is_bitwise_reproducible(dtype):
    _require_cake_arch()
    inputs = _make_inputs(2688, 4, dtype, arbitrary_routes=True)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="cake")
    first = plan.run().clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, expected, atol=1e-2, rtol=1e-2)
    for _ in range(7):
        replay = plan.run().clone()
        torch.cuda.synchronize()
        assert torch.equal(replay, first)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_invalid_pair_padding_and_outer_graph_capture(dtype):
    _require_cake_arch()
    inputs = list(_make_inputs(3072, 8, dtype))
    device = inputs[0].device
    inputs[3] = torch.cat(
        [inputs[3], torch.tensor([-1, 8], dtype=torch.int64, device=device)]
    )
    inputs[4] = torch.cat([inputs[4], torch.zeros(2, dtype=torch.int64, device=device)])
    inputs[6] = torch.cat(
        [inputs[6], torch.zeros(2, dtype=torch.float32, device=device)]
    )
    inputs = tuple(inputs)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="cake")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = plan.run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_extra_valid_route_uses_general_path(dtype):
    _require_cake_arch()
    inputs = list(_make_inputs(2688, 32, dtype))
    device = inputs[0].device
    inputs[3] = torch.cat(
        [inputs[3], torch.tensor([0], dtype=torch.int64, device=device)]
    )
    inputs[4] = torch.cat(
        [inputs[4], torch.tensor([0], dtype=torch.int64, device=device)]
    )
    inputs[6] = torch.cat(
        [inputs[6], torch.tensor([0.25], dtype=torch.float32, device=device)]
    )
    inputs = tuple(inputs)
    expected = _reference(inputs)
    actual = prepare_bgmv_moe(*inputs, backend="cake").run()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_cpu_input_reports_device_requirement():
    x = torch.empty((1, 2688), dtype=torch.bfloat16)
    empty_i64 = torch.empty((1,), dtype=torch.int64)
    empty_f32 = torch.empty((1,), dtype=torch.float32)
    with pytest.raises(ValueError, match="exact SM90, SM100 or SM103 CUDA device"):
        prepare_bgmv_moe(
            x,
            [],
            [],
            empty_i64,
            empty_i64,
            empty_i64,
            empty_f32,
            1,
            backend="cake",
        )


@pytest.mark.parametrize(
    ("tensor_index", "value", "message"),
    [
        (4, 128, "expert_ids values"),
        (5, 2, "lora_indices values"),
    ],
)
def test_invalid_routing_indices_rejected(tensor_index, value, message):
    _require_cake_arch()
    inputs = list(_make_inputs(2688, 4, torch.bfloat16))
    inputs[tensor_index][0] = value
    with pytest.raises(ValueError, match=message):
        prepare_bgmv_moe(*inputs, backend="cake")


def test_blackwell_backend_alias_and_plan_alias():
    _require_cake_arch()
    from flashinfer.fused_moe import BGMVMoEBlackwellPlan, BGMVMoECakePlan

    assert BGMVMoEBlackwellPlan is BGMVMoECakePlan
    inputs = _make_inputs(3072, 4, torch.bfloat16)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="blackwell")
    assert isinstance(plan, BGMVMoECakePlan)
    torch.testing.assert_close(plan.run(), expected, atol=1e-2, rtol=1e-2)


def test_unknown_backend_rejected():
    x = torch.empty((1, 2688), dtype=torch.bfloat16)
    empty_i64 = torch.empty((1,), dtype=torch.int64)
    empty_f32 = torch.empty((1,), dtype=torch.float32)
    with pytest.raises(ValueError, match="backend"):
        prepare_bgmv_moe(
            x, [], [], empty_i64, empty_i64, empty_i64, empty_f32, 1, backend="cuda"
        )


def test_cake_plan_reports_backend_used():
    _require_cake_arch()
    plan = prepare_bgmv_moe(*_make_inputs(2688, 4, torch.bfloat16), backend="cake")
    assert isinstance(plan, BGMVMoECakePlan)
    assert plan.backend_used == "cake"
    assert plan.schedule_id is not None


# (hidden, rank, tokens, dtype, arbitrary routes, top_k): runtime-hidden
# generic bundles. Tokens <= 8 take the 64-lane expand, more the 128-lane one;
# <= 32 pairs take the decode shrink (PPB=4), more the prefill shrink (PPB=1).
_GENERIC_CASES = [
    (3072, 16, 8, torch.bfloat16, False, 2),
    (3072, 8, 40, torch.float16, False, 2),
    (3072, 64, 4, torch.bfloat16, True, 2),
    (2048, 32, 32, torch.float16, False, 2),
    (736, 32, 16, torch.bfloat16, False, 2),
    (1344, 32, 512, torch.bfloat16, False, 2),
    (2880, 32, 1, torch.float16, False, 2),
    (2112, 64, 300, torch.bfloat16, True, 2),
    (1472, 8, 64, torch.float16, True, 2),
    (4096, 32, 8, torch.bfloat16, True, 4),
    (1856, 16, 96, torch.float16, True, 8),
    # top-k above the 16-slot route index: exact serial-scan fallback
    (2048, 8, 24, torch.bfloat16, True, 20),
    # hidden-split shrink: decode kernel x 4 splits, 1-pair kernel x 7 / 6 splits
    (7168, 64, 16, torch.float16, False, 2),
    (7168, 8, 4, torch.bfloat16, True, 2),
    (5888, 32, 4, torch.bfloat16, False, 2),
    (4096, 32, 16, torch.float16, True, 2),
]


# (hidden, tokens, dtype, top_k): expert-sorted pair order (MoE dispatch layout)
# through both variants' route index; the specialized rows are the S4 shapes.
_EXPERT_SORTED_CASES = [
    (3072, 4, torch.bfloat16, 2),
    (3072, 1024, torch.bfloat16, 2),
    (2688, 1024, torch.float16, 2),
    (3072, 4096, torch.bfloat16, 2),
    (2048, 512, torch.float16, 2),
    (1344, 64, torch.bfloat16, 4),
]


@pytest.mark.parametrize(
    ("hidden_size", "num_tokens", "dtype", "top_k"),
    _EXPERT_SORTED_CASES,
    ids=[
        f"h{h}_t{t}_{str(d).split('.')[-1]}_k{k}" for h, t, d, k in _EXPERT_SORTED_CASES
    ],
)
def test_expert_sorted_routes_match_reference_and_replay_bitwise(
    hidden_size, num_tokens, dtype, top_k
):
    inputs = _make_inputs(
        hidden_size, num_tokens, dtype, top_k=top_k, expert_sorted=True
    )
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="cake", fallback=False)
    assert isinstance(plan, BGMVMoECakePlan)
    assert plan.variant == ("specialized" if hidden_size in (2688, 3072) else "generic")
    first = plan.run().clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, expected, atol=1e-2, rtol=1e-2)
    for _ in range(3):
        replay = plan.run().clone()
        torch.cuda.synchronize()
        assert torch.equal(replay, first)
    # Launch counter advanced once per run (first run + 3 replays); the
    # parity recorded for the last shrink is (launches - 1) & 1.
    assert int(plan.route_index[0]) == 4
    assert int(plan.route_index[1]) == 1
    plan.close()


@pytest.mark.parametrize(
    ("hidden_size", "rank", "num_tokens", "dtype", "arbitrary_routes", "top_k"),
    _GENERIC_CASES,
    ids=[
        f"h{h}_r{r}_t{t}_{str(d).split('.')[-1]}{'_arb' if a else ''}_k{k}"
        for h, r, t, d, a, k in _GENERIC_CASES
    ],
)
def test_generic_variant_matches_reference_and_replays_bitwise(
    hidden_size, rank, num_tokens, dtype, arbitrary_routes, top_k
):
    _require_cake_arch()
    inputs = _make_inputs(
        hidden_size,
        num_tokens,
        dtype,
        rank=rank,
        arbitrary_routes=arbitrary_routes,
        top_k=top_k,
    )
    expected = _reference(inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plan = prepare_bgmv_moe(*inputs, backend="cake", fallback=False)
    assert isinstance(plan, BGMVMoECakePlan)
    assert plan.backend_used == "cake"
    assert plan.variant == "generic"
    first = plan.run().clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, expected, atol=1e-2, rtol=1e-2)
    for _ in range(3):
        replay = plan.run().clone()
        torch.cuda.synchronize()
        assert torch.equal(replay, first)
    # Replays consume current tensor contents through the same pointers.
    inputs[0].mul_(0.5)
    torch.testing.assert_close(plan.run(), _reference(inputs), atol=1e-2, rtol=1e-2)
    plan.close()


@pytest.mark.parametrize("hidden_size", [2688, 3072])
def test_specialized_variant_is_preferred_at_rank_32(hidden_size):
    _require_cake_arch()
    plan = prepare_bgmv_moe(
        *_make_inputs(hidden_size, 4, torch.bfloat16), backend="cake"
    )
    assert plan.variant == "specialized"
    plan.close()
    plan = prepare_bgmv_moe(
        *_make_inputs(hidden_size, 4, torch.bfloat16, rank=16), backend="cake"
    )
    assert plan.variant == "generic"
    plan.close()


def test_generic_variant_outer_graph_capture_and_padding():
    _require_cake_arch()
    inputs = list(_make_inputs(1344, 8, torch.bfloat16, rank=16))
    device = inputs[0].device
    inputs[3] = torch.cat(
        [inputs[3], torch.tensor([-1, 8], dtype=torch.int64, device=device)]
    )
    inputs[4] = torch.cat([inputs[4], torch.zeros(2, dtype=torch.int64, device=device)])
    inputs[6] = torch.cat(
        [inputs[6], torch.zeros(2, dtype=torch.float32, device=device)]
    )
    inputs = tuple(inputs)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="cake")
    assert plan.variant == "generic"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = plan.run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    plan.close()


_FALLBACK_CASES = [
    (
        "slices_fp16",
        dict(hidden_size=3072, num_tokens=4, dtype=torch.float16, num_slices=2),
        "exactly one LoRA slice",
    ),
    (
        "slices_bf16_generic_shape",
        dict(hidden_size=2048, num_tokens=40, dtype=torch.bfloat16, num_slices=2),
        "exactly one LoRA slice",
    ),
]

# Inputs no generated program serves; fallback=False must reject them with the
# listed reason. (The portable kernels do not compile these shapes either.)
_UNSUPPORTED_CASES = [
    (
        "rank",
        dict(hidden_size=3072, num_tokens=8, dtype=torch.bfloat16, rank=12),
        "rank in",
    ),
    (
        "hidden",
        dict(hidden_size=2052, num_tokens=8, dtype=torch.bfloat16),
        "positive multiple of 8",
    ),
    (
        "slices",
        dict(hidden_size=3072, num_tokens=4, dtype=torch.float16, num_slices=2),
        "exactly one LoRA slice",
    ),
]


def _fallback_inputs(case):
    kwargs = dict(case)
    return _make_inputs(
        kwargs.pop("hidden_size"),
        kwargs.pop("num_tokens"),
        kwargs.pop("dtype"),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("name", "case", "message"), _FALLBACK_CASES, ids=[c[0] for c in _FALLBACK_CASES]
)
def test_fallback_plan_matches_reference(name, case, message):
    _require_cuda()
    inputs = _fallback_inputs(case)
    expected = _reference(inputs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = prepare_bgmv_moe(*inputs, backend="cake")
    assert isinstance(plan, BGMVMoEPortablePlan)
    assert plan.backend_used == "portable"
    assert plan.schedule_id is None
    # The device check runs before the shape checks, so on a device without a
    # generated program (e.g. SM120, SM107) the reason names the capability.
    if torch.cuda.get_device_capability() in _SUPPORTED_CAPABILITIES:
        assert message in plan.fallback_reason
    else:
        assert "exact SM90, SM100 or SM103" in plan.fallback_reason
    fallback_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert all("portable" in str(w.message) for w in fallback_warnings)
    out = plan.run()
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    # Replay path (graph captured on the first eager run).
    torch.testing.assert_close(plan.run(), expected, atol=1e-2, rtol=1e-2)
    plan.close()


@pytest.mark.parametrize(
    ("name", "case", "message"),
    _UNSUPPORTED_CASES,
    ids=[c[0] for c in _UNSUPPORTED_CASES],
)
def test_strict_mode_raises_for_unsupported_inputs(name, case, message):
    # The listed reasons are only reached on a device with a generated program;
    # test_fallback_on_unsupported_capability covers the device rejection.
    _require_cake_arch()
    inputs = _fallback_inputs(case)
    with pytest.raises(ValueError, match=message):
        prepare_bgmv_moe(*inputs, backend="cake", fallback=False)


def test_fallback_on_unsupported_capability(monkeypatch):
    _require_cuda()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 0)
    )
    inputs = _make_inputs(3072, 4, torch.bfloat16)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="cake")
    assert isinstance(plan, BGMVMoEPortablePlan)
    assert "capability=(8, 0)" in plan.fallback_reason
    torch.testing.assert_close(plan.run(), expected, atol=1e-2, rtol=1e-2)
    with pytest.raises(ValueError, match="exact SM90, SM100 or SM103"):
        prepare_bgmv_moe(*inputs, backend="cake", fallback=False)


def test_fallback_warning_is_emitted_once_per_reason():
    _require_cuda()
    import sys

    bgmv_moe_module = sys.modules["flashinfer.fused_moe.bgmv_moe"]
    bgmv_moe_module._fallback_reasons_warned.clear()
    inputs = _make_inputs(2048, 4, torch.bfloat16, num_slices=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_bgmv_moe(*inputs, backend="cake")
        prepare_bgmv_moe(*inputs, backend="cake")
    fallback_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(fallback_warnings) == 1


def test_fallback_plan_outer_graph_capture_and_stream_check():
    _require_cuda()
    inputs = _make_inputs(2048, 32, torch.float16, arbitrary_routes=True, num_slices=2)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="cake")
    assert isinstance(plan, BGMVMoEPortablePlan)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        plan.run()
        with torch.cuda.graph(graph, stream=stream):
            out = plan.run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    with torch.cuda.stream(stream):
        torch.testing.assert_close(plan.run(), expected, atol=1e-2, rtol=1e-2)
    with pytest.raises(RuntimeError, match="original CUDA stream"):
        plan.run()
    plan.close()


def test_invalid_inputs_raise_even_with_fallback():
    _require_cuda()
    inputs = list(_make_inputs(2048, 4, torch.bfloat16, x_dtype=torch.float32))
    with pytest.raises(ValueError, match="share one dtype"):
        prepare_bgmv_moe(*inputs, backend="cake")
    inputs = list(_make_inputs(2048, 4, torch.bfloat16))
    inputs[4][0] = 128
    with pytest.raises(ValueError, match="expert_ids values"):
        prepare_bgmv_moe(*inputs, backend="cake")
    inputs = list(_make_inputs(2048, 4, torch.bfloat16, num_slices=2))
    inputs[2][1] = inputs[2][1][:, :, : inputs[2][1].shape[2] // 2].contiguous()
    with pytest.raises(ValueError, match="same feat_out"):
        prepare_bgmv_moe(*inputs, backend="cake")
