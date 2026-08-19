import pytest
import torch

from flashinfer.fused_moe.fused_routing_dsv3 import (
    _is_cake_dsv3_fused_routing_supported,
)


def _supported(**overrides):
    params = {
        "capability": (10, 0),
        "num_tokens": 1,
        "num_experts": 256,
        "n_group": 8,
        "topk_group": 4,
        "topk": 8,
        "score_dtype": torch.float32,
        "bias_dtype": torch.bfloat16,
    }
    params.update(overrides)
    return _is_cake_dsv3_fused_routing_supported(**params)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
@pytest.mark.parametrize("score_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cake_backend_accepts_contract_dtype_and_arch_union(
    capability, score_dtype, bias_dtype
):
    assert _supported(
        capability=capability, score_dtype=score_dtype, bias_dtype=bias_dtype
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"capability": (9, 0)},
        {"capability": (12, 0)},
        {"num_tokens": 0},
        {"num_experts": 257},
        {"n_group": 9, "num_experts": 252},
        {"topk_group": 5},
        {"n_group": 8, "num_experts": 8, "topk_group": 4},
        {"n_group": 8, "num_experts": 264},
        {"n_group": 8, "num_experts": 256, "topk": 9},
        {"n_group": 3, "num_experts": 96, "topk_group": 1, "topk": 5},
        {"score_dtype": torch.float64},
        {"bias_dtype": torch.float64},
    ],
)
def test_cake_backend_rejects_calls_outside_contract(overrides):
    assert not _supported(**overrides)


@pytest.mark.parametrize("num_experts", [2, 128, 256, 384])
def test_cake_backend_accepts_single_group_boundary(num_experts):
    assert _supported(
        num_experts=num_experts,
        n_group=1,
        topk_group=1,
        topk=min(8, num_experts),
    )


def test_cake_backend_rejects_single_group_above_boundary():
    assert not _supported(num_experts=385, n_group=1, topk_group=1)
