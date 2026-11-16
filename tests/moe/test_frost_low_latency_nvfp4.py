# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 FMA: independent block quantization and changing-input graph checks."""

from dataclasses import replace
import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _assert_near, _require_gpu
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS, _activation


def _linear_scale(scale):
    e, rows, groups = scale.shape
    return (
        scale.view(torch.uint8)
        .reshape(e, rows // 128, groups // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(e, rows, groups)
        .view(torch.float8_e4m3fn)
        .float()
    )


def _decode(q, scale):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=q.device
    )
    value = torch.stack((lut[(q & 15).long()], lut[(q >> 4).long()]), -1).flatten(-2)
    return (value.reshape(*value.shape[:-1], -1, 16) * scale[..., None]).flatten(-2)


def _round_mid(value, global_scale):
    groups = value.reshape(-1, 16)
    scale = (groups.abs().amax(-1) / 6 * global_scale).to(torch.float8_e4m3fn).float()
    scaled = torch.where(
        scale[:, None] > 0, groups * global_scale / scale[:, None], 0.0
    )
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=value.device)
    distance = (scaled.abs()[..., None] - levels).abs()
    code = distance.argmin(-1)
    # E2M1 ties round to the even low bit, not always the smaller magnitude.
    even_distance = distance[..., ::2]
    even_code = even_distance.argmin(-1) * 2
    code = torch.where(even_distance.amin(-1) == distance.amin(-1), even_code, code)
    return (levels[code] * scaled.sign() * scale[:, None]).flatten()


def reference(q, sf, view, ids, scores, activation, packed):
    x = _decode(q, sf.view(torch.float8_e4m3fn).float())
    w1 = _decode(
        view["fc1_expert_weights"], _linear_scale(view["fc1_weight_block_scale"])
    )
    w2 = _decode(
        view["fc2_expert_weights"], _linear_scale(view["fc2_weight_block_scale"])
    )
    rows = []
    for token in range(ids.shape[0]):
        for expert in ids[token].tolist():
            dot = (x[token].double() @ w1[expert].double().T).float() * view[
                "fc1_dequant_scale"
            ][expert]
            up, gate = (
                dot.chunk(2) if activation.is_gated else (dot, torch.zeros_like(dot))
            )
            mid = _round_mid(
                _activation(up, gate, activation).bfloat16().float(),
                view["fc2_act_global_scale"],
            )
            row = (mid.double() @ w2[expert].double().T).float() * view[
                "fc2_dequant_scale"
            ][expert]
            rows.append(row.bfloat16())
    raw = torch.stack(rows)
    rw = scores.bfloat16().float() if packed else scores.float()
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        result = result + raw[slot :: ids.shape[1]].float() * rw[:, slot, None]
    return raw, result.bfloat16()


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_nvfp4_activation_tactics_and_changing_graphs(index, activation):
    _require_gpu(allow_sm120=True)
    torch.manual_seed(7100 + index)
    t, h, i, e, k = (1, 2, 4, 8)[index % 4], 512, 128, 5, 3
    w1 = (
        torch.randn(e, (2 if activation.is_gated else 1) * i, h, device="cuda") * 0.08
    ).bfloat16()
    w2 = (torch.randn(e, h, i, device="cuda") * 0.08).bfloat16()
    view = FrostLowLatencyConfig.prepare_nvfp4_weights(w1, w2, activation=activation)
    view["fc1_dequant_scale"].copy_(torch.linspace(0.5, 1.5, e, device="cuda"))
    view["fc2_dequant_scale"].copy_(torch.linspace(0.25, 1.25, e, device="cuda"))
    view["fc2_act_global_scale"].fill_(2.0)
    wp = MoEWeightPack()
    wp.prepare_for("frost_low_latency", view)
    quant = QuantConfig(
        weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4, nvfp4_4over6=None
    )
    q, sf = TrtllmFp4Config.prepare_activations(
        torch.randn(t, h, device="cuda").bfloat16(), quant=quant
    )
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).view(t, k) % e
    scores = torch.rand(t, k, device="cuda")
    config = MoEConfig(
        RoutingConfig(e, k),
        quant,
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    for finalize in (True, False):
        packed = finalize or index % 2 == 0
        cfg = replace(config, finalize=MoEFinalizeConfig(do_finalize=finalize))
        act = MoEActivationPack(
            q,
            sf,
            ids,
            scores,
            routing_input_mode=(
                RoutingInputMode.PackedPrecomputed
                if packed
                else RoutingInputMode.UnpackedPrecomputed
            ),
        )
        runner = FrostLowLatencyRunner(cfg, q.device)
        runner.check_support()
        runner.build()
        inputs = runner.pack_inputs(act, wp)
        for tactic in runner.get_valid_tactics(inputs, None):
            runner.forward(inputs, tactic=tactic)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = runner.forward(inputs, tactic=tactic)
            for step in range(3):
                qn, sn = TrtllmFp4Config.prepare_activations(
                    torch.randn(t, h, device="cuda").bfloat16(), quant=quant
                )
                q.copy_(qn)
                sf.copy_(sn)
                ids.copy_(
                    (
                        torch.arange(t * k, device="cuda", dtype=torch.int32).view(t, k)
                        + step
                    )
                    % e
                )
                scores.copy_(torch.rand_like(scores) - 0.2)
                inputs[3].copy_(scores)
                graph.replay()
                raw, expected = reference(q, sf, view, ids, scores, activation, packed)
                _assert_near(
                    output if finalize else output[0], expected if finalize else raw
                )
                if not finalize:
                    torch.testing.assert_close(
                        output[1],
                        scores.bfloat16() if packed else scores,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        output[2], torch.arange(t * k, device="cuda", dtype=torch.int32)
                    )


def test_nvfp4_recipe_rejection():
    from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config

    cfg = MoEConfig(
        RoutingConfig(8, 2),
        QuantConfig(
            weight=QuantFormat.NVFP4,
            activation=QuantFormat.NVFP4,
            nvfp4_4over6=NVFP44Over6Config(),
        ),
        ExpertConfig(128),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    runner = FrostLowLatencyRunner(cfg, "cpu")
    with pytest.raises(NotImplementedError, match="4over6"):
        runner._check_nvfp4_recipe()


@pytest.mark.parametrize("negative", [False, True])
def test_nvfp4_exact_midpoints_round_to_even(negative):
    from flashinfer.fused_moe import Identity

    _require_gpu(allow_sm120=True)
    h, i = 512, 128
    device = torch.device("cuda", torch.cuda.current_device())
    w1 = torch.zeros((1, i, h // 2), dtype=torch.uint8, device=device)
    # With x=[1,1,0,...], rows 0/1 dot to +/-5 and 6. Alpha makes the
    # BF16 activation exactly +/-1.171875 and 1.40625, respectively.
    w1[0, 0, 0] = 0xDC if negative else 0x54
    w1[0, 1, 0] = 0x55
    view = {
        "fc1_expert_weights": w1,
        "fc2_expert_weights": torch.zeros(
            (1, h, i // 2), dtype=torch.uint8, device=device
        ),
        "fc1_weight_block_scale": torch.full(
            (1, i, h // 16), 0x38, dtype=torch.uint8, device=device
        ),
        "fc2_weight_block_scale": torch.full(
            (1, h, i // 16), 0x38, dtype=torch.uint8, device=device
        ),
        "fc1_dequant_scale": torch.full((1,), 0.234375, device=device),
        "fc2_dequant_scale": torch.ones(1, device=device),
        "fc1_act_global_scale": torch.ones((), device=device),
        "fc2_act_global_scale": torch.full((), 2.0, device=device),
    }
    q = torch.zeros((1, h // 2), dtype=torch.uint8, device=device)
    q[0, 0] = 0x22
    act = MoEActivationPack(
        q,
        torch.full((1, h // 16), 0x38, dtype=torch.uint8, device=device),
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.ones((1, 1), device=device),
    )
    config = MoEConfig(
        RoutingConfig(1, 1),
        QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4, nvfp4_4over6=None),
        ExpertConfig(i),
        activation=Identity(),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    runner = FrostLowLatencyRunner(config, device)
    runner.check_support()
    runner.build()
    weights = MoEWeightPack()
    weights.prepare_for("frost_low_latency", view)
    inputs = runner.pack_inputs(act, weights)
    expected = torch.zeros_like(inputs[11])
    # Scale=0.46875 makes row 0 exactly +/-5, halfway between +/-4 and
    # +/-6. E2M1 ties-to-even chooses the code for 4, not 6.
    expected[0, 0] = 0x7E if negative else 0x76
    for tactic in runner.get_valid_tactics(inputs, None):
        runner.forward(inputs, tactic=tactic)
        torch.testing.assert_close(inputs[11], expected, rtol=0, atol=0)
        assert inputs[12][0].item() == 0x2F
