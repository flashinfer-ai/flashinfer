"""The FlashInfer GIN transport must select the native hybrid-topology policies."""

from types import SimpleNamespace
import pytest
from flashinfer.moe_ep.shims.sm120.nvfp4_split_cutedsl import (
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
    from flashinfer.moe_ep.shims.sm120.nvfp4_split_cutedsl import (
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


def _frontend(monkeypatch, *, world, topk, experts, hidden, intermediate, same, cross):
    bootstrap_paths()
    from flashinfer.moe_ep.shims.sm120.nvfp4_split_cutedsl import (
        runtime,
    )
    from moe_sm120_nvfp4_split.runtime import green_context

    monkeypatch.setattr(green_context, "query_sm_resource_info", lambda: (110, 8, 8))
    monkeypatch.setattr(
        green_context,
        "query_green_context_sm_counts",
        lambda k1_sm_count: (k1_sm_count, 110 - k1_sm_count),
    )
    monkeypatch.setattr(
        runtime.MegaMoESm120Nvfp4Frontend,
        "_green_resources_for",
        lambda self, s: SimpleNamespace(
            k1_sm_count=s.kernel.k1_sms, k2_sm_count=s.kernel.k2_sms
        ),
    )
    frontend = object.__new__(runtime.MegaMoESm120Nvfp4Frontend)
    frontend.config = runtime.MegaMoESm120Nvfp4Config(
        0,
        world,
        512,
        topk,
        experts,
        hidden,
        intermediate,
        gate_up_clamp=10.0,
        numerical_contract="owner_bf16",
    )
    frontend.workspace = SimpleNamespace(
        _topology=SimpleNamespace(
            same_numa_peer_count=same, cross_numa_peer_count=cross
        )
    )
    return frontend


def test_ep4_anchor_table_residency_reaches_frontend(monkeypatch):
    # DSV4.1-Flash EP4 T16 anchor: k1_ctas_per_sm=2, k2_stages=3.
    frontend = _frontend(
        monkeypatch,
        world=4,
        topk=6,
        experts=384,
        hidden=5120,
        intermediate=2304,
        same=3,
        cross=0,
    )
    k = frontend._select_spec(16).kernel
    assert k.comm_backend == "p2p_direct"
    assert (k.k1_ctas_per_sm, k.k2_launch_factor, k.dispatch_warps) == (2, 1, 1)
    assert k.k2_stages == 3


def test_ep8_frontend_skips_p2p_split_anchors(monkeypatch):
    # GLM5.3-Flash EP8: FlashInfer always runs gin (native gin_hybrid), so a
    # p2p_split anchor (T32) leaves the built-in selection while a gin_hybrid
    # anchor (T1024: k1_stages=3, k2_stages=3) applies.
    from moe_sm120_nvfp4_split import tuned_table

    kwargs = dict(
        world=8,
        topk=8,
        experts=288,
        hidden=4096,
        intermediate=2048,
        same=3,
        cross=4,
    )
    tabulated = {
        t: _frontend(monkeypatch, **kwargs)._select_spec(t) for t in (32, 1024)
    }
    monkeypatch.setattr(tuned_table, "_load", lambda: dict(schema=1, tables=[]))
    from moe_sm120_nvfp4_split import heuristic

    heuristic._select_with_profiles.cache_clear()
    builtin = {t: _frontend(monkeypatch, **kwargs)._select_spec(t) for t in (32, 1024)}
    heuristic._select_with_profiles.cache_clear()
    assert tabulated[32] == builtin[32]
    k = tabulated[1024].kernel
    assert k.comm_backend == "gin"
    assert (k.k1_stages, k.k2_stages) == (3, 3)
    assert (k.k1_ctas_per_sm, k.k2_launch_factor) == (1, 1)


def _peer(host, numa_node):
    return SimpleNamespace(host=host, numa_node=numa_node, p2p_domain=(host, numa_node))


def _ep8_topology():
    peers = tuple(_peer("h0", 0 if r < 4 else 1) for r in range(8))
    return SimpleNamespace(
        same_numa_peer_count=3,
        cross_numa_peer_count=4,
        force_ibgda_peer_mask=0b11110000,
        peers=peers,
    )


def _ep16_two_host_topology():
    # Two hosts of 2 x 4 GPUs; PCI and NUMA numbers repeat across hosts.
    peers = tuple(_peer(f"h{r // 8}", (r % 8) // 4) for r in range(16))
    return SimpleNamespace(
        same_numa_peer_count=3,
        cross_numa_peer_count=12,
        force_ibgda_peer_mask=0xFFF0,
        peers=peers,
    )


def test_ep8_p2p_split_bucket_plan(monkeypatch):
    bootstrap_paths()
    from flashinfer.moe_ep.shims.sm120.nvfp4_split_cutedsl import (
        runtime,
    )
    from moe_sm120_nvfp4_split.runtime import green_context

    monkeypatch.setattr(green_context, "query_sm_resource_info", lambda: (110, 8, 8))
    monkeypatch.delenv(runtime.P2P_SPLIT_MAX_TOKENS_ENV, raising=False)
    buckets = [*runtime.DECODE_GRAPH_COMPILE_BUCKETS, 512]

    def config(knobs=None):
        return runtime.MegaMoESm120Nvfp4Config(
            0, 8, 512, 6, 384, 5120, 2304, gate_up_clamp=10.0, knobs=knobs
        )

    topology = _ep8_topology()
    # DSV4.1-Flash EP8 anchors up to 512 select p2p_split; 512 exceeds the
    # default decode-bucket limit.
    assert runtime.plan_p2p_split_buckets(config(), topology, buckets) == set(
        runtime.DECODE_GRAPH_COMPILE_BUCKETS
    )
    monkeypatch.setenv(runtime.P2P_SPLIT_MAX_TOKENS_ENV, "512")
    assert 512 in runtime.plan_p2p_split_buckets(config(), topology, buckets)
    assert not runtime.plan_p2p_split_buckets(
        config({"comm_backend": "gin"}), topology, buckets
    )
    # Same-NUMA EP4 never shadows peers.
    ep4 = SimpleNamespace(
        same_numa_peer_count=3, cross_numa_peer_count=0, force_ibgda_peer_mask=0
    )
    assert not runtime.plan_p2p_split_buckets(config(), ep4, buckets)


def test_ep16_two_host_p2p_split_bucket_plan(monkeypatch):
    """The other host's GPUs are cross peers: EP16 over two hosts shadows them
    like a cross-NUMA EP8, following the EP16 anchors."""
    bootstrap_paths()
    from flashinfer.moe_ep.shims.sm120.nvfp4_split_cutedsl import (
        runtime,
    )
    from moe_sm120_nvfp4_split.runtime import green_context
    from moe_sm120_nvfp4_split.transport.shadow import ShadowLayout

    monkeypatch.setattr(green_context, "query_sm_resource_info", lambda: (110, 8, 8))
    monkeypatch.delenv(runtime.P2P_SPLIT_MAX_TOKENS_ENV, raising=False)
    topology = _ep16_two_host_topology()
    assert ShadowLayout.from_topology(topology, 0).num_cross == 12
    config = runtime.MegaMoESm120Nvfp4Config(
        0, 16, 512, 6, 384, 5120, 2304, gate_up_clamp=10.0
    )
    assert runtime.plan_p2p_split_buckets(
        config, topology, [*runtime.DECODE_GRAPH_COMPILE_BUCKETS, 512]
    ) == set(runtime.DECODE_GRAPH_COMPILE_BUCKETS)


@pytest.mark.parametrize(
    "selected, override, expected",
    [
        ([4, 4, 4, 2], None, 4),
        ([2, 4], None, 4),
        ([2, 2, 4], None, 2),
        ([], None, 2),
        ([4, 4], "1", 1),
    ],
)
def test_one_xnuma_fsplit_per_workspace(monkeypatch, selected, override, expected):
    """p2p_split buckets share cumulative commit words, so a workspace runs one
    forwarder split: the majority selection, ties to the larger value."""
    bootstrap_paths()
    from flashinfer.moe_ep.shims.sm120.nvfp4_split_cutedsl import (
        runtime,
    )

    if override is None:
        monkeypatch.delenv("MEGAMOE_XNUMA_FSPLIT", raising=False)
    else:
        monkeypatch.setenv("MEGAMOE_XNUMA_FSPLIT", override)
    assert runtime._uniform_xnuma_fsplit(selected) == expected


def test_ep8_p2p_split_bucket_uses_folded_selection(monkeypatch):
    frontend = _frontend(
        monkeypatch,
        world=8,
        topk=6,
        experts=384,
        hidden=5120,
        intermediate=2304,
        same=3,
        cross=4,
    )
    frontend.workspace._topology = _ep8_topology()
    frontend.workspace._bucket_backends = {32: "p2p_split"}
    split = frontend._select_spec(32).kernel
    # DSV4.1-Flash EP8 T32 anchor: p2p_split with K1/K2 N64.
    assert split.comm_backend == "p2p_direct"
    assert not split.rank_local_combine
    assert (split.k1_tile[1], split.k2_tile[1], split.k2_warps) == (64, 64, 8)
    assert (split.tx_sms, split.rx_sms) == (0, 0)
    gin = frontend._select_spec(1024).kernel
    assert gin.comm_backend == "gin"
