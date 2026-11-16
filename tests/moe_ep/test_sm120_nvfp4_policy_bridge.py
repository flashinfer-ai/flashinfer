"""The FlashInfer GIN transport must select the native hybrid-topology policies."""

from types import SimpleNamespace
import pytest
from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
    bootstrap_paths,
)


@pytest.mark.parametrize(
    "model,topk,experts,sms",
    [("dsv", 6, 256, (56, 38, 8, 8)), ("glm", 8, 288, (64, 30, 8, 8))],
)
@pytest.mark.parametrize("explicit_old", [False, True])
def test_ep8_frontend_native_policy_bridge(
    monkeypatch, model, topk, experts, sms, explicit_old
):
    bootstrap_paths()
    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel.shim import (
        runtime,
    )
    from moe_sm120_nvfp4_split.runtime import green_context

    monkeypatch.setattr(green_context, "query_sm_resource_info", lambda: (110, 8, 8))
    monkeypatch.setattr(
        runtime.MegaMoESm120Nvfp4Frontend,
        "_green_resources_for",
        lambda self, s: SimpleNamespace(
            k1_sm_count=s.kernel.k1_sms, k2_sm_count=s.kernel.k2_sms
        ),
    )
    frontend = object.__new__(runtime.MegaMoESm120Nvfp4Frontend)
    knobs = dict(k1_sms=48, k2_sms=30, tx_sms=16, rx_sms=16) if explicit_old else None
    frontend.config = runtime.MegaMoESm120Nvfp4Config(
        0,
        8,
        512,
        topk,
        experts,
        4096,
        2048,
        gate_up_clamp=10.0,
        numerical_contract="owner_bf16",
        knobs=knobs,
    )
    frontend.workspace = SimpleNamespace(
        _topology=SimpleNamespace(same_numa_peer_count=3, cross_numa_peer_count=4)
    )
    spec = frontend._select_spec(32)
    k = spec.kernel
    assert k.comm_backend == k.kernel_comm_backend == "gin"
    assert (k.k1_sms, k.k2_sms, k.tx_sms, k.rx_sms) == (
        (48, 30, 16, 16) if explicit_old else sms
    )
    assert k.k1_tile == k.k2_tile == ((64, 32, 128) if explicit_old else (128, 32, 128))
    assert spec.jit.fc1_utility_postprocess == (model == "dsv")
