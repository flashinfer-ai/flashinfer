"""NVFP4 MoE: reject a GEMM1 K that is not 256-aligned (#9333).

With NVFP4 activations the trtllm-gen FP4 MoE ran on a ``hidden_size`` that is
128- but not 256-aligned (Nemotron Nano's 2688) and silently returned wrong
output, NaN included, with static and per-token scales alike.  Both public
entry points, and the unified API's ``TrtllmFp4RoutedRunner``, now reject that
shape before any device work.  The compiled module is replaced by a recorder
(and the runner never gets as far as its weights), so these tests only check
which calls get through, and run on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

import flashinfer.fused_moe.core as core
from flashinfer.fused_moe import (
    MoEActivationPack,
    QuantFormat,
    RoutingInputMode,
    TrtllmFp4RoutedRunner,
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


def _common_args(hidden_size, per_token, act="nvfp4"):
    # NVFP4 uses one scale per 16 elements, MxFP4 one per 32.
    sf_vec = 32 if act == "mxfp4" else 16
    if act in ("nvfp4", "mxfp4"):
        hidden_states = torch.zeros(NUM_TOKENS, hidden_size // 2, dtype=torch.uint8)
        hidden_states_scale = torch.zeros(
            NUM_TOKENS, hidden_size // sf_vec, dtype=torch.uint8
        )
    else:
        hidden_states = torch.zeros(NUM_TOKENS, hidden_size, dtype=torch.bfloat16)
        hidden_states_scale = None
    w1 = torch.zeros(
        NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, hidden_size // 2, dtype=torch.uint8
    )
    w1_scale = torch.zeros(
        NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, hidden_size // sf_vec, dtype=torch.uint8
    )
    w2 = torch.zeros(
        NUM_EXPERTS, hidden_size, INTERMEDIATE_SIZE // 2, dtype=torch.uint8
    )
    w2_scale = torch.zeros(
        NUM_EXPERTS, hidden_size, INTERMEDIATE_SIZE // sf_vec, dtype=torch.uint8
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


def _call(entry, hidden_size, per_token, act="nvfp4"):
    kwargs = _common_args(hidden_size, per_token, act)
    if entry == "routed":
        topk_ids = torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int32)
        topk_weights = torch.ones(NUM_TOKENS, TOP_K, dtype=torch.bfloat16)
        return trtllm_fp4_block_scale_routed_moe((topk_ids, topk_weights), **kwargs)
    routing_logits = torch.zeros(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32)
    return trtllm_fp4_block_scale_moe(routing_logits, **kwargs)


ENTRIES = ["routed", "logits"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("per_token", [True, False], ids=["per_token", "static"])
@pytest.mark.parametrize("hidden_size", [2688, 1152, 384])
def test_nvfp4_rejects_hidden_not_256_aligned(module, entry, per_token, hidden_size):
    with pytest.raises(ValueError, match="multiple of 256") as excinfo:
        _call(entry, hidden_size, per_token=per_token)
    assert str(hidden_size) in str(excinfo.value)
    assert "valid_hidden_size" in str(excinfo.value)
    assert module.calls == 0


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("per_token", [True, False], ids=["per_token", "static"])
@pytest.mark.parametrize("hidden_size", [2816, 3072, 4096])
def test_nvfp4_accepts_256_aligned_hidden(module, entry, per_token, hidden_size):
    _call(entry, hidden_size, per_token=per_token)
    assert module.calls == 1


@pytest.mark.parametrize("entry", ENTRIES)
def test_mxfp4_hidden_states_are_not_checked_here(module, entry):
    # MxFP4 activations share the uint8 packing but are not the NVFP4 path
    # reported in #9333; the launcher's dtype deduction tells them apart.
    _call(entry, 2688, per_token=False, act="mxfp4")
    assert module.calls == 1


@pytest.mark.parametrize("entry", ENTRIES)
def test_non_fp4_hidden_states_are_not_checked_here(module, entry):
    # per_token_scale with non-NVFP4 activations is rejected by the launcher
    # with its own message; this guard only covers the packed FP4 width.
    _call(entry, 2688, per_token=True, act="bf16")
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


class _WeightsReached(Exception):
    """Raised when pack_inputs gets past the activation checks."""


class _UnreachableWeights:
    def get_view(self, backend_key):
        raise _WeightsReached


def _unified_runner():
    # Skip build(): it compiles the module. pack_inputs only needs these.
    runner = TrtllmFp4RoutedRunner.__new__(TrtllmFp4RoutedRunner)
    runner._built = True
    runner._pair = (QuantFormat.NVFP4, QuantFormat.NVFP4)
    runner.config = SimpleNamespace(quant=SimpleNamespace(per_token_scale=False))
    return runner


def _unified_pack(hidden_size):
    return MoEActivationPack(
        torch.zeros(NUM_TOKENS, hidden_size // 2, dtype=torch.uint8),
        torch.zeros(NUM_TOKENS, hidden_size // 16, dtype=torch.float8_e4m3fn),
        torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int32),
        torch.ones(NUM_TOKENS, TOP_K, dtype=torch.float32),
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
    )


@pytest.mark.parametrize("hidden_size", [2688, 1152, 384])
def test_unified_runner_rejects_hidden_not_256_aligned(hidden_size):
    with pytest.raises(ValueError, match="multiple of 256") as excinfo:
        _unified_runner().pack_inputs(_unified_pack(hidden_size), _UnreachableWeights())
    assert str(hidden_size) in str(excinfo.value)


@pytest.mark.parametrize("hidden_size", [2816, 3072, 4096])
def test_unified_runner_accepts_256_aligned_hidden(hidden_size):
    # The check passes, so pack_inputs goes on to read the weights.
    with pytest.raises(_WeightsReached):
        _unified_runner().pack_inputs(_unified_pack(hidden_size), _UnreachableWeights())
