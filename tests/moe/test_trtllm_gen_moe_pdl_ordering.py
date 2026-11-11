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

Back-to-back trtllm-gen MoE calls with PDL must match calls without PDL, bit for bit.

Regression test for routing outputs that were not visible yet when the kernels after
the routing kernel started. With PDL on, ``trtllm_fp8_block_scale_routed_moe`` returned
wrong rows for some calls with 2560 to 8192 tokens (DeepSeek-V3-style shapes).
The workspace is filled with 0xFF before every call, so a read of scratch that comes
too early returns garbage instead of a value left by an earlier call. With correct
ordering this does not change any result.
"""

import pytest
import torch

from flashinfer.fused_moe import (
    Fp8QuantizationType,
    trtllm_fp8_block_scale_routed_moe,
    trtllm_fused_moe_workspace_size,
)
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import get_compute_capability

# DeepSeek-V3-style block-FP8 MoE, intermediate size of one tensor-parallel shard.
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 7168
INTER = 128
BLOCK = 128
# The problem showed up at 2560 to 8192 tokens.
TOKEN_COUNTS = (8192, 4096, 2560, 1024)
NUM_LAYERS = 4
NUM_WAVES = 300


@pytest.fixture(autouse=True)
def _require_sm100_family():
    """Skip unless the GPU is SM100 or SM103, where trtllm-gen fused MoE runs."""
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm-gen fused MoE is only exercised on SM100 and SM103.")


def _rand_fp8(shape, gen):
    """Return a random FP8 E4M3 tensor of ``shape``, filled one expert at a time."""
    out = torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
    for e in range(shape[0]):  # one expert at a time keeps the bf16 temporary small
        # How often the race shows up depends on where buffers land in memory. The extra
        # "* 1.0" temporary gives a layout that hit it in every run on GB200 without the
        # fix (about 1.5% of the calls).
        out[e] = (
            torch.randn(shape[1:], generator=gen, device="cuda", dtype=torch.bfloat16)
            * 1.0
        ).to(torch.float8_e4m3fn)
    return out


def _make_layer(gen):
    """Return random block-FP8 expert weights and their per-block scales for one layer."""

    def scale(*shape):
        return 0.02 * (1 + torch.rand(*shape, generator=gen, device="cuda"))

    w13 = _rand_fp8((NUM_EXPERTS, 2 * INTER, HIDDEN), gen)
    w2 = _rand_fp8((NUM_EXPERTS, HIDDEN, INTER), gen)
    w13_s = scale(NUM_EXPERTS, 2 * INTER // BLOCK, HIDDEN // BLOCK)
    w2_s = scale(NUM_EXPERTS, HIDDEN // BLOCK, INTER // BLOCK)
    return dict(w13=w13, w2=w2, w13_s=w13_s, w2_s=w2_s)


def _make_inputs(num_tokens, gen):
    """Return FP8 hidden states, their scales and packed top-k routing for one call."""
    x = (
        torch.randn(num_tokens, HIDDEN, generator=gen, device="cuda")
        .clamp_(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    x_scale = (
        0.5 + torch.rand(HIDDEN // BLOCK, num_tokens, generator=gen, device="cuda")
    ).contiguous()
    ids = (
        torch.rand(num_tokens, NUM_EXPERTS, generator=gen, device="cuda")
        .topk(TOP_K, dim=1)
        .indices.int()
    )
    w = torch.rand(num_tokens, TOP_K, generator=gen, device="cuda") + 0.1
    w = (w / w.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    # Packed top-k: (expert_id << 16) | bf16 weight bits.
    topk = ((ids << 16) | (w.view(torch.int16).to(torch.int32) & 0xFFFF)).contiguous()
    return dict(x=x, x_scale=x_scale, topk=topk)


def _run(layer, inputs, workspace, out, enable_pdl):
    """Run one routed block-FP8 MoE call into ``out`` with the given PDL setting."""
    trtllm_fp8_block_scale_routed_moe(
        topk_ids=inputs["topk"],
        routing_bias=None,
        hidden_states=inputs["x"],
        hidden_states_scale=inputs["x_scale"],
        gemm1_weights=layer["w13"],
        gemm1_weights_scale=layer["w13_s"],
        gemm2_weights=layer["w2"],
        gemm2_weights_scale=layer["w2_s"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=2.5,
        routing_method_type=int(RoutingMethodType.TopK),
        use_shuffled_weight=False,
        weight_layout=0,
        enable_pdl=enable_pdl,
        output=out,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        workspace_buffer=workspace,
    )


def test_routed_fp8_block_scale_moe_pdl_back_to_back():
    """Back-to-back PDL calls must match PDL-off references bit for bit.

    References run one at a time with PDL off. Each wave then runs every call back to
    back on one stream with PDL on, refilling the workspace with 0xFF before each call.
    """
    gen = torch.Generator("cuda").manual_seed(0)
    layers = [_make_layer(gen) for _ in range(NUM_LAYERS)]
    calls = [
        (layer, num_tokens, _make_inputs(num_tokens, gen))
        for layer in range(NUM_LAYERS)
        for num_tokens in TOKEN_COUNTS
    ]
    workspace = torch.empty(
        trtllm_fused_moe_workspace_size(
            max(TOKEN_COUNTS),
            HIDDEN,
            INTER,
            NUM_EXPERTS,
            TOP_K,
            quant_mode="fp8_block_scale",
            use_shuffled_weight=False,
            weight_layout=0,
        ),
        dtype=torch.uint8,
        device="cuda",
    )

    refs = []
    for layer, num_tokens, inputs in calls:
        ref = torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        _run(layers[layer], inputs, workspace, ref, enable_pdl=False)
        torch.cuda.synchronize()
        refs.append(ref)
    assert all(torch.isfinite(r.float()).all() and r.abs().max() > 0 for r in refs)
    outs = [torch.empty_like(ref) for ref in refs]

    mismatches = []
    for wave in range(NUM_WAVES):
        for out in outs:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        # The calls of a wave run back to back on one stream, with no host sync.
        for (layer, _, inputs), out in zip(calls, outs, strict=True):
            workspace.fill_(0xFF)
            _run(layers[layer], inputs, workspace, out, enable_pdl=True)
        torch.cuda.synchronize()
        for i, (out, ref) in enumerate(zip(outs, refs, strict=True)):
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                mismatches.append((wave, calls[i][1]))
    assert not mismatches, (
        f"{len(mismatches)} of {NUM_WAVES * len(calls)} calls with enable_pdl=True "
        f"differ from enable_pdl=False; first (wave, num_tokens): {mismatches[:5]}"
    )
