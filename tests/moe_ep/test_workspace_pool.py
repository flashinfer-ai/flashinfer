"""Shared mega workspace pool: one symm buffer per session key across layers.

Without pooling every ``MoEEpMegaLayer`` allocates its own symmetric-heap
workspace + compiled session (43x at DeepSeek-scale — the vLLM integration
carried this as a wrapper-side cache). CPU tests cover the pool/refcount
semantics and the base-class wiring; the GPU test proves two real nvfp4
layers share one buffer and stay numerically correct through destroy.
"""

from __future__ import annotations

from unittest import mock

import pytest


def _pool():
    from flashinfer.moe_ep.core.kernel import workspace_pool

    return workspace_pool


@pytest.fixture(autouse=True)
def _isolate_pool(monkeypatch):
    wp = _pool()
    monkeypatch.setattr(wp, "_POOL", {})
    monkeypatch.setattr(wp, "_KEY_BY_ID", {})


def test_acquire_shares_and_release_refcounts():
    wp = _pool()
    made = []

    def factory():
        ws = mock.MagicMock()
        made.append(ws)
        return ws

    a = wp.acquire_workspace(("k",), factory)
    b = wp.acquire_workspace(("k",), factory)
    assert a is b
    assert len(made) == 1
    assert wp.pooled_workspace_count() == 1

    assert wp.release_workspace(a) is False  # one ref left
    assert wp.pooled_workspace_count() == 1
    assert wp.release_workspace(b) is True  # last ref: caller destroys
    assert wp.pooled_workspace_count() == 0


def test_distinct_keys_do_not_share():
    wp = _pool()
    a = wp.acquire_workspace(("k1",), mock.MagicMock)
    b = wp.acquire_workspace(("k2",), mock.MagicMock)
    assert a is not b
    assert wp.pooled_workspace_count() == 2


def test_release_of_unpooled_workspace_says_destroy():
    wp = _pool()
    assert wp.release_workspace(object()) is True


def test_epilogue_pool_key_semantics():
    import torch

    wp = _pool()
    assert wp.epilogue_pool_key(None) is None
    assert wp.epilogue_pool_key(1.0) == wp.epilogue_pool_key(1)
    assert wp.epilogue_pool_key(1.0) != wp.epilogue_pool_key(2.0)
    t = torch.zeros(2)
    assert wp.epilogue_pool_key(t) == wp.epilogue_pool_key(t)
    # Equal values but different tensors must NOT share (values are baked
    # into the buffer at creation).
    assert wp.epilogue_pool_key(t) != wp.epilogue_pool_key(torch.zeros(2))


def test_knobs_pool_key_is_order_insensitive():
    wp = _pool()
    a = wp.knobs_pool_key({"flag_batch": 4, "mma_tiler_mnk": (256, 128, 256)})
    b = wp.knobs_pool_key({"mma_tiler_mnk": [256, 128, 256], "flag_batch": 4})
    assert a == b
    assert wp.knobs_pool_key(None) is None
    assert wp.knobs_pool_key("auto") == "auto"


def _fake_backend_cls():
    from flashinfer.moe_ep.core.kernel.base import MegaKernelBackend

    class _FakeMegaBackend(MegaKernelBackend):
        allocations = 0

        def __init__(self, config: object, pool_key) -> None:
            super().__init__(config)
            self._pool_key = pool_key

        @classmethod
        def kernel_name(cls) -> str:
            return "fake_mega"

        def _allocate_workspace(self, fleet_params):
            type(self).allocations += 1
            return mock.MagicMock(name=f"ws{type(self).allocations}")

        def _workspace_pool_key(self, fleet_params):
            return self._pool_key

        def compute(self, workspace, transformed_weights, *, output):
            return output

    return _FakeMegaBackend


def test_base_prepare_workspace_pools_and_destroy_refcounts():
    from flashinfer.moe_ep import BootstrapConfig, FleetParams

    cls = _fake_backend_cls()
    bootstrap = BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    fp = FleetParams(num_experts=2, max_tokens_per_rank=4, token_hidden_size=8)

    k1 = cls(object(), pool_key=("shared",))
    k2 = cls(object(), pool_key=("shared",))
    ws1 = k1.prepare_workspace(bootstrap, fp)
    ws2 = k2.prepare_workspace(bootstrap, fp)
    assert ws1 is ws2
    assert cls.allocations == 1

    k1.destroy(ws1)
    ws1.destroy.assert_not_called()  # still referenced by the other layer
    k2.destroy(ws2)
    ws2.destroy.assert_called_once()


def test_base_prepare_workspace_unpooled_when_key_none():
    from flashinfer.moe_ep import BootstrapConfig, FleetParams

    cls = _fake_backend_cls()
    bootstrap = BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    fp = FleetParams(num_experts=2, max_tokens_per_rank=4, token_hidden_size=8)

    k1 = cls(object(), pool_key=None)
    k2 = cls(object(), pool_key=None)
    ws1 = k1.prepare_workspace(bootstrap, fp)
    ws2 = k2.prepare_workspace(bootstrap, fp)
    assert ws1 is not ws2
    assert cls.allocations == 2
    k1.destroy(ws1)
    ws1.destroy.assert_called_once()


@pytest.mark.arch_blackwell
def test_two_nvfp4_layers_share_one_symm_buffer(monkeypatch):
    """Two same-geometry layers: one buffer, one compile, correct numerics."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from flashinfer.utils import get_compute_capability

    cap = get_compute_capability(torch.device("cuda"))
    if cap[0] != 10:
        pytest.skip(f"needs sm_100/sm_103; got sm_{cap[0]}{cap[1]}")
    pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        Nvfp4CutedslMegaMoeConfig,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, intermediate, num_experts, topk, max_tokens = 2048, 1024, 4, 4, 64

    def _layer(seed: int) -> MoEEpMegaLayer:
        g = torch.Generator(device="cuda").manual_seed(seed)
        w13 = torch.randn(
            num_experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=g,
        )
        w2 = torch.randn(
            num_experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device="cuda",
            generator=g,
        )
        return MoEEpMegaLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            fleet_params=FleetParams(
                num_experts=num_experts,
                max_tokens_per_rank=max_tokens,
                token_hidden_size=hidden,
            ),
            weights=MoEWeightPack(w13=w13, w2=w2),
            backend=MegaConfig(
                megakernel=Nvfp4CutedslMegaMoeConfig(
                    intermediate_size=intermediate, top_k=topk, gate_up_clamp=10.0
                ),
                quantize_input=True,
                preprocess_weights=True,
            ),
        )

    layer1 = _layer(seed=100)
    layer2 = _layer(seed=100)  # same weights -> same math
    try:
        layer1.warmup()
        layer2.warmup()
        assert layer1._workspace is layer2._workspace, "buffer not shared"

        g = torch.Generator(device="cuda").manual_seed(9)
        t = MoEEpTensors(
            hidden_states=torch.randn(
                32, hidden, dtype=torch.bfloat16, device="cuda", generator=g
            ),
            topk_ids=(torch.arange(32 * topk, device="cuda") % num_experts).view(
                32, topk
            ),
            topk_weights=torch.full(
                (32, topk), 0.25, dtype=torch.float32, device="cuda"
            ),
        )
        y1 = layer1.forward(t).clone()
        y2 = layer2.forward(t)
        torch.cuda.synchronize()
        assert torch.equal(y1, y2), "shared-buffer layers disagree"

        # First destroy must NOT free the shared buffer out from under layer2.
        layer1.destroy()
        y2b = layer2.forward(t)
        torch.cuda.synchronize()
        assert torch.equal(y2b, y2)
    finally:
        layer2.destroy()
