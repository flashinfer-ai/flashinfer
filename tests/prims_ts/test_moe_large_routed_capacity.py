# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prims-TS MXFP4 x MXFP8 MoE tactics at a large routed token capacity.

A routed GEMM pads every expert's tokens to whole tiles, so its token capacity
times the hidden size can pass 2**28 FP8 elements well before the activations
themselves are large. Every tactic must still launch there and agree with the
others.
"""

import pytest
import torch

# Modules reaching cutlass.experimental are imported inside the tests, so this
# module also collects on a CuTe DSL without Prims-TS (tests/prims_ts/conftest.py).


def _prims_ts_gpu_available() -> bool:
    from flashinfer.prims_ts.utils import is_prims_ts_available
    from flashinfer.utils import get_compute_capability

    if not torch.cuda.is_available() or not is_prims_ts_available():
        return False
    return get_compute_capability(torch.device("cuda")) in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(
    not _prims_ts_gpu_available(),
    reason="Prims-TS MoE requires SM100/SM103 and nvidia-cutlass-dsl",
)


def _captured_runner_call(monkeypatch, layer, act, pack):
    """Run ``layer`` once and return the Prims-TS runner and its forward arguments."""
    from flashinfer.prims_ts.moe.runner import PrimsTsMxfp4Mxfp8MoERunner

    captured = {}
    forward = PrimsTsMxfp4Mxfp8MoERunner.forward

    def recording_forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        captured.update(runner=self, inputs=inputs, kwargs=kwargs)
        return forward(
            self, inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
        )

    monkeypatch.setattr(PrimsTsMxfp4Mxfp8MoERunner, "forward", recording_forward)
    layer(act, pack)
    monkeypatch.setattr(PrimsTsMxfp4Mxfp8MoERunner, "forward", forward)
    return captured["runner"], captured["inputs"], captured["kwargs"]


def test_ldgsts_gathered_fc1_matches_tma_gathered_fc1(monkeypatch):
    from flashinfer.fused_moe import MoELayer, MoEWeightPack, PrimsTsConfig
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        RouteImpl,
        compute_max_num_ctas_in_token_dim_for_moe,
    )
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_mxfp4_mxfp8_moe_tactic

    # Every token picks every expert: 65536 routed rows, which tile_n=128
    # pads to a capacity of 66432 rows.
    num_experts, hidden, intermediate, tokens = 8, 4096, 256, 8192
    top_k = num_experts
    capacity = 128 * compute_max_num_ctas_in_token_dim_for_moe(
        num_tokens=tokens,
        num_experts=num_experts,
        top_k=top_k,
        token_tile_size=128,
        cluster_dim_in_token=1,
    )
    assert capacity * hidden >= 2**28
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    quant = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts, 2 * intermediate, hidden, device=device, dtype=torch.bfloat16
    ) * (hidden**-0.5)
    w2 = torch.randn(
        num_experts, hidden, intermediate, device=device, dtype=torch.bfloat16
    ) * (intermediate**-0.5)
    ids = torch.arange(num_experts, device=device, dtype=torch.int32).repeat(tokens, 1)
    weights = torch.rand(tokens, top_k, device=device)
    x_q, x_scale = PrimsTsConfig.prepare_activations(x, quant=quant)
    pack = MoEWeightPack()
    pack.prepare_for(
        "prims_ts",
        PrimsTsConfig.prepare_weights(
            w1,
            w2,
            quant=quant,
            num_local_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation=SwiGLU(),
            device=device,
        ),
    )
    layer = MoELayer(
        MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=quant,
            experts=ExpertConfig(intermediate_size=intermediate),
            activation=SwiGLU(),
            backend=BackendOptions((PrimsTsConfig(),)),
            execution=ExecutionConfig(tune_max_num_tokens=tokens),
        )
    )
    act = MoEActivationPack(x_q, x_scale, ids, weights)
    runner, inputs, kwargs = _captured_runner_call(monkeypatch, layer, act, pack)

    def fc1_route(tactic):
        pair = map_trtllm_mxfp4_mxfp8_moe_tactic(
            tactic,
            num_tokens=tokens,
            top_k=top_k,
            num_local_experts=num_experts,
        )
        return pair.fc1.cfg.build().route_act

    tactics = [t for t in runner.get_valid_tactics(inputs, None) if t != -1]
    by_route = {}
    for tactic in tactics:
        if tactic[0] == 128:
            by_route.setdefault(fc1_route(tactic), tactic)
    assert {int(RouteImpl.LDGSTS), int(RouteImpl.TMA)} <= by_route.keys()

    outputs = []
    for route in (int(RouteImpl.TMA), int(RouteImpl.LDGSTS)):
        runner.forward(inputs, tactic=by_route[route], **kwargs)
        torch.cuda.synchronize()
        outputs.append(MoeRunnerInputs.from_list(inputs).output.float().clone())

    tma_out, ldgsts_out = outputs
    assert torch.isfinite(ldgsts_out).all()
    rel_err = (ldgsts_out - tma_out).norm() / tma_out.norm()
    assert rel_err < 1e-3, f"LDGSTS vs TMA gather relative error {rel_err:.3e}"
