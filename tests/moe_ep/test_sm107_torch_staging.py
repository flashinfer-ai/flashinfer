"""Portable CUDA tests for Rubin's Torch preprocessing and staging contracts."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep import FleetParams
from flashinfer.moe_ep.backends.mega.kernel.sm107.validation import (
    validate_forward_metadata,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    Sm107BlockScaledMoeConfig,
    compute_megamoe_reference_sm107_block_scaled,
    interleave_gate_up_16,
    preprocess_block_scaled_weights,
    quantize_mxfp8_block32,
    quantize_nvfp4_block16,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.correctness import (
    sampled_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required; any supported Torch GPU"
)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
def test_sampled_oracle_matches_full_raw_scale_oracle(kind):
    torch.manual_seed(317)
    cfg = Sm107BlockScaledMoeConfig(
        num_total_experts=4,
        max_tokens_per_rank=10,
        num_topk=2,
        hidden=128,
        intermediate=64,
        rank=0,
        world_size=1,
        quant_kind=kind,
    )

    def quantize(t):
        return (
            quantize_nvfp4_block16(t.float())
            if kind == "nvfp4"
            else quantize_mxfp8_block32(t.float(), cfg.torch_act_data_dtype)
        )

    w13 = (torch.randn(4, 128, 128, device="cuda") / 128**0.5).bfloat16()
    w2 = (torch.randn(4, 128, 64, device="cuda") / 64**0.5).bfloat16()
    l1, l2 = preprocess_block_scaled_weights(
        w13, w2, quant_kind=kind, intermediate_size=64
    )
    x, sf = quantize(torch.randn(10, 128, device="cuda").bfloat16())
    ids = torch.arange(20, device="cuda", dtype=torch.int32).view(10, 2) % 4
    ids[3, 0] = -1
    weights = torch.rand(10, 2, device="cuda")
    ws = SimpleNamespace(config=cfg, x=x, x_sf=sf, topk_idx=ids, topk_weights=weights)
    indices, expected = sampled_reference(ws, l1, l2, 10, sample_size=4)
    q1, sf1 = quantize(interleave_gate_up_16(w13, intermediate_size=64))
    q2, sf2 = quantize(w2)
    full = compute_megamoe_reference_sm107_block_scaled(
        x,
        sf,
        ids,
        weights,
        q1,
        sf1,
        q2,
        sf2,
        quant_kind=kind,
        local_expert_offset=0,
        gate_up_clamp=None,
        apply_topk_at_fc1=True,
    )
    torch.testing.assert_close(expected, full[indices], rtol=0.01, atol=0.001)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
def test_prequantized_metadata_rejects_wrong_encoding_and_oversized_scales(kind):
    fp = FleetParams(num_experts=4, max_tokens_per_rank=10, token_hidden_size=128)
    dtype = (
        torch.float4_e2m1fn_x2
        if kind == "nvfp4"
        else torch.float8_e4m3fn
        if kind == "mxfp8_e4m3"
        else torch.float8_e5m2
    )
    sf_dtype = torch.float8_e4m3fn if kind == "nvfp4" else torch.float8_e8m0fnu
    cols = 8 if kind == "nvfp4" else 4
    x = torch.empty(3, 64 if kind == "nvfp4" else 128, device="cuda", dtype=dtype)
    ids = torch.zeros(3, 2, device="cuda", dtype=torch.int64)
    weights = torch.ones(3, 2, device="cuda")
    sf = torch.empty(3, cols, device="cuda", dtype=sf_dtype)

    def validate(**overrides):
        validate_forward_metadata(
            **(
                dict(
                    hidden_states=x,
                    topk_ids=ids,
                    topk_weights=weights,
                    fleet_params=fp,
                    top_k=2,
                    quantize_input=False,
                    quant_kind=kind,
                    scales=sf,
                )
                | overrides
            )
        )

    validate()
    validate(scales=sf.view(torch.uint8))
    with pytest.raises(ValueError, match="scales must have dtype"):
        validate(scales=sf.view(torch.float8_e5m2))
    with pytest.raises(ValueError, match="columns"):
        validate(scales=torch.empty(3, cols + 4, device="cuda", dtype=sf_dtype))
    with pytest.raises(ValueError, match="int32 or int64"):
        validate(topk_ids=ids.float())
    with pytest.raises(ValueError, match="same CUDA device"):
        validate(topk_weights=weights.cpu())


def test_fp4_pack_unpack_graph_replay():
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        pack_f32_to_fp4,
        unpack_fp4_to_f32,
    )

    # Exercise every packed byte, including both signs and signed zero.
    raw = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    expected = unpack_fp4_to_f32(raw)
    staged = raw.cuda()

    def roundtrip():
        return unpack_fp4_to_f32(pack_f32_to_fp4(unpack_fp4_to_f32(staged)))

    roundtrip()  # The public layer also warms up before graph capture.
    torch.cuda.synchronize()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        output = roundtrip()
    for shift in (0, 1, 7):
        staged.copy_(raw.roll(shift, dims=0))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, expected.roll(shift, dims=0).cuda(), rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
def test_real_dlpack_binding_is_safe_in_a_new_capture_stream(kind):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        Sm107BlockScaledSymmBuffer,
    )

    cfg = Sm107BlockScaledMoeConfig(
        num_total_experts=4,
        max_tokens_per_rank=4,
        num_topk=2,
        hidden=128,
        intermediate=64,
        rank=0,
        world_size=1,
        quant_kind=kind,
    )
    ws = object.__new__(Sm107BlockScaledSymmBuffer)
    ws.config = cfg
    ws.device = torch.device("cuda", torch.cuda.current_device())
    ws._destroyed = False
    ws._launch_key = None
    ws._launch_kwargs = None
    ws.x = torch.empty(
        4, 64 if kind == "nvfp4" else 128, device="cuda", dtype=cfg.torch_act_data_dtype
    )
    ws.x_sf = torch.empty(
        4, 128 // cfg.sf_vec_size, device="cuda", dtype=cfg.torch_act_sf_dtype
    )
    ws.topk_idx = torch.zeros(4, 2, device="cuda", dtype=torch.int32)
    ws.topk_weights = torch.ones(4, 2, device="cuda")
    ws.output_activation = torch.empty(4, 128, device="cuda", dtype=torch.bfloat16)
    ws.shared_workspace = torch.zeros(4096, device="cuda", dtype=torch.uint8)
    ws.local_workspace = torch.zeros_like(ws.shared_workspace)
    ws._pre_reduced_activation = torch.empty(4, 2, 128, device="cuda")
    ws._symmetric_base = ws.shared_workspace.data_ptr()
    ws._peer_offsets = (0,)
    ws._compiled = lambda **kwargs: ws.output_activation.fill_(3)
    weights = preprocess_block_scaled_weights(
        torch.randn(4, 128, 128, device="cuda").bfloat16(),
        torch.randn(4, 128, 64, device="cuda").bfloat16(),
        quant_kind=kind,
        intermediate_size=64,
    )
    ws.launch(*weights)
    torch.cuda.synchronize()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        ws.launch(*weights)
    graph.replay()
    torch.cuda.synchronize()
    assert ws._launch_key[-1] == stream.cuda_stream
    assert (ws.output_activation == 3).all()
    del graph
