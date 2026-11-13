"""Per-token NVFP4 MoE: reject a GEMM1 K that is not 256-aligned (#9333).

With ``per_token_scale`` the trtllm-gen FP4 MoE ran on a ``hidden_size`` that
is 128- but not 256-aligned (Nemotron Nano's 2688) and silently returned wrong
output, NaN included.  Both public entry points now reject that shape before
any device work.  The compiled module is replaced by a recorder, so these tests
only check which calls reach it and run on CPU.
"""

import pytest
import torch

import flashinfer.fused_moe.core as core
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)

NUM_TOKENS = 7
NUM_EXPERTS = 8
TOP_K = 2
INTERMEDIATE_SIZE = 1920


class _RecordingModule:
    def __init__(self):
        self.calls = 0

    def trtllm_fp4_block_scale_moe(self, *args, **kwargs):
        self.calls += 1
        return [torch.empty(0)]


@pytest.fixture
def module(monkeypatch):
    recorder = _RecordingModule()
    monkeypatch.setattr(core, "get_trtllm_moe_sm100_module", lambda: recorder)
    return recorder


def _common_args(hidden_size, per_token, nvfp4=True):
    if nvfp4:
        hidden_states = torch.zeros(NUM_TOKENS, hidden_size // 2, dtype=torch.uint8)
        hidden_states_scale = torch.zeros(
            NUM_TOKENS, hidden_size // 16, dtype=torch.uint8
        )
    else:
        hidden_states = torch.zeros(NUM_TOKENS, hidden_size, dtype=torch.bfloat16)
        hidden_states_scale = None
    w1 = torch.zeros(
        NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, hidden_size // 2, dtype=torch.uint8
    )
    w1_scale = torch.zeros(
        NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, hidden_size // 16, dtype=torch.uint8
    )
    w2 = torch.zeros(
        NUM_EXPERTS, hidden_size, INTERMEDIATE_SIZE // 2, dtype=torch.uint8
    )
    w2_scale = torch.zeros(
        NUM_EXPERTS, hidden_size, INTERMEDIATE_SIZE // 16, dtype=torch.uint8
    )
    scalar = torch.ones(NUM_EXPERTS, dtype=torch.float32)
    return dict(
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=w1_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        gemm2_bias=None,
        output1_scale_scalar=scalar,
        output1_scale_gate_scalar=scalar,
        output2_scale_scalar=scalar,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=None,
        per_token_scale=(
            torch.ones(NUM_TOKENS, dtype=torch.float32) if per_token else None
        ),
    )


def _call(entry, hidden_size, per_token, nvfp4=True):
    kwargs = _common_args(hidden_size, per_token, nvfp4)
    if entry == "routed":
        topk_ids = torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int32)
        topk_weights = torch.ones(NUM_TOKENS, TOP_K, dtype=torch.bfloat16)
        return trtllm_fp4_block_scale_routed_moe((topk_ids, topk_weights), **kwargs)
    routing_logits = torch.zeros(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32)
    return trtllm_fp4_block_scale_moe(routing_logits, **kwargs)


ENTRIES = ["routed", "logits"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("hidden_size", [2688, 1152, 384])
def test_per_token_rejects_hidden_not_256_aligned(module, entry, hidden_size):
    with pytest.raises(ValueError, match="multiple of 256") as excinfo:
        _call(entry, hidden_size, per_token=True)
    assert str(hidden_size) in str(excinfo.value)
    assert "valid_hidden_size" in str(excinfo.value)
    assert module.calls == 0


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("hidden_size", [2816, 3072, 4096])
def test_per_token_accepts_256_aligned_hidden(module, entry, hidden_size):
    _call(entry, hidden_size, per_token=True)
    assert module.calls == 1


@pytest.mark.parametrize("entry", ENTRIES)
def test_without_per_token_scale_is_unchanged(module, entry):
    # Only the per-token path is known to miscompute at 2688; the other FP4
    # paths keep their current behaviour.
    _call(entry, 2688, per_token=False)
    assert module.calls == 1


@pytest.mark.parametrize("entry", ENTRIES)
def test_non_nvfp4_hidden_states_are_not_checked_here(module, entry):
    # per_token_scale with non-NVFP4 activations is rejected by the launcher
    # with its own message; this guard only covers the packed FP4 width.
    _call(entry, 2688, per_token=True, nvfp4=False)
    assert module.calls == 1


def test_padded_width_is_what_is_checked(module):
    # A 2688-wide model padded to 2816 passes; the real width goes through
    # valid_hidden_size.
    kwargs = _common_args(2816, per_token=True)
    topk_ids = torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int32)
    topk_weights = torch.ones(NUM_TOKENS, TOP_K, dtype=torch.bfloat16)
    trtllm_fp4_block_scale_routed_moe(
        (topk_ids, topk_weights), valid_hidden_size=2688, **kwargs
    )
    assert module.calls == 1
