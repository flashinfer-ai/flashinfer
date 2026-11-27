"""Host-only contracts of the NVLinkOneSided MegaMoE backend.

``sm103_mxfp8_mxfp4_bf16_nvlink_one_sided`` without a GPU and without
importing the CuTe DSL: registration and exports, config and knob validation,
``sync_protocol`` and the engine protocols it selects,
the shapes and devices ``validate_init`` rejects, the CUDA graph capture guard,
the activation (SwiGLU or SiTU) and its engine configs, the shared expert's
config and weights, and the host mirror of the main kernel's token tile
choice. The engine path each capacity resolves to is asserted by the
layer-vs-oracle cases in
``tests/moe_ep/test_moe_ep_nvlink_one_sided_megamoe_multirank.py``.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MoEEpConfigError,
    Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSided_MegaMoeConfig as Config,
    preprocess_sm103_mxfp4_nvlink_one_sided_mega_weights,
)
from flashinfer.moe_ep.backends.mega.kernel.sm103.mxfp8_mxfp4_bf16_nvlink_one_sided import (
    Sm103Mxfp8Mxfp4NVLinkOneSidedMegaKernelBackend,
    preprocess_mega_weights,
)
from flashinfer.moe_ep.core.kernel.registry import (
    _available_mega_kernels,
    create_mega_kernel,
)

pytestmark = pytest.mark.cpu_only

NAME = "sm103_mxfp8_mxfp4_bf16_nvlink_one_sided"
HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS = 256
TOPK = 6
CLAMP = 10.0
SHARED = {"shared_expert": "fp8_in_kernel"}
# The protocols a config forces, and the engine fields each one resolves to.
SENTINEL = {"sync_protocol": "sentinel"}
FENCE = {"sync_protocol": "fence"}
SENTINEL_ENGINE = {"dispatch_protocol": "sentinel", "combine_protocol": "sentinel"}
FENCE_ENGINE = {"dispatch_protocol": "pull", "combine_protocol": "counter"}
POOL = {"fc1_token_pool": True}
JIT = {"just_in_time_claims": True}
NO_JIT = {"just_in_time_claims": False}
NO_HEAD = {"shared_expert_entry": False}
DIE_LOCAL = {"die_local_weights": True}
IN_KERNEL = {"reduce_topk_in_kernel": True}
# Kimi-K3's SiTU betas.
SITU = {"activation": "situ", "situ_beta": 4.0, "situ_linear_beta": 25.0}


def _config(**fields):
    return Config(intermediate_size=INTERMEDIATE, top_k=TOPK, **fields)


def _kernel(world: int, *, top_k: int = TOPK, **fields):
    kernel = create_mega_kernel(
        Config(
            intermediate_size=INTERMEDIATE,
            top_k=top_k,
            activation_clamp=CLAMP,
            **fields,
        )
    )
    kernel.bind_ep_bootstrap(
        BootstrapConfig(world_size=world, rank=world - 1, auto_bootstrap=False)
    )
    return kernel


def _fleet(capacity: int, hidden: int = HIDDEN) -> FleetParams:
    return FleetParams(
        num_experts=EXPERTS, max_tokens_per_rank=capacity, token_hidden_size=hidden
    )


def _engine_config(capacity: int, world: int = 8, knobs=None, **fields):
    return _kernel(world, knobs=knobs, **fields)._engine_config(_fleet(capacity))


def test_registration_and_exports():
    assert NAME in _available_mega_kernels()
    config = _config(gate_up_clamp=CLAMP)
    assert config.kernel_name == NAME
    backend = create_mega_kernel(config)
    assert isinstance(backend, Sm103Mxfp8Mxfp4NVLinkOneSidedMegaKernelBackend)
    assert backend.kernel_name() == NAME
    assert backend.supports_output_view
    assert (
        preprocess_sm103_mxfp4_nvlink_one_sided_mega_weights is preprocess_mega_weights
    )


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"knobs": "auto"}, "no autotuner"),
        ({"gate_up_clamp": 7.0, "activation_clamp": 10.0}, "same clamp"),
        ({"die_local_weights": 1}, "die_local_weights must be None or a bool"),
        ({"reduce_topk_in_kernel": 1}, "reduce_topk_in_kernel must be a bool"),
        ({"routing_weights_after_fc2": 1}, "routing_weights_after_fc2 must be a bool"),
        ({"fc1_token_pool_capacity": -1}, "fc1_token_pool_capacity"),
        ({"shared_expert": "bf16"}, "shared_expert must be"),
        ({**SHARED, "shared_gate_up_clamp": -1.0}, "shared_gate_up_clamp"),
        ({**SHARED, "shared_bf16_rounding": 1}, "shared_bf16_rounding must be a bool"),
        ({"shared_gate_up_clamp": 10.0}, "need shared_expert"),
        # The recipe's BF16 rounding of the routes' sum needs the separate reduce.
        (
            {**SHARED, "shared_bf16_rounding": True, "reduce_topk_in_kernel": True},
            "reduce_topk_in_kernel",
        ),
        # SiTU takes both betas, positive and finite real numbers, and no clamp;
        # SwiGLU takes neither beta; the shared expert runs SwiGLU only.
        ({"activation": "gelu"}, "activation must be 'swiglu' or 'situ'"),
        ({"activation": "situ"}, "needs situ_beta positive"),
        ({**SITU, "situ_linear_beta": None}, "needs situ_linear_beta positive"),
        ({**SITU, "situ_beta": 0.0}, "needs situ_beta positive"),
        ({**SITU, "situ_linear_beta": -25.0}, "needs situ_linear_beta positive"),
        ({**SITU, "situ_beta": float("inf")}, "needs situ_beta positive"),
        ({**SITU, "situ_beta": float("nan")}, "needs situ_beta positive"),
        ({**SITU, "situ_beta": True}, "needs situ_beta positive"),
        ({**SITU, "situ_linear_beta": "25"}, "needs situ_linear_beta positive"),
        ({"situ_beta": 4.0, "situ_linear_beta": 25.0}, "need activation='situ'"),
        ({"situ_linear_beta": 25.0}, "need activation='situ'"),
        ({**SITU, "gate_up_clamp": CLAMP}, "takes no clamp"),
        ({**SITU, "activation_clamp": CLAMP}, "takes no clamp"),
        ({**SITU, **SHARED}, "shared expert runs SwiGLU only"),
        ({**SITU, **SHARED, "shared_gate_up_clamp": CLAMP}, "SwiGLU only"),
        ({**SITU, "shared_gate_up_clamp": CLAMP}, "need shared_expert"),
        # One protocol selector, with the all-to-all's values; the engine's
        # protocol fields are not knobs.
        ({"sync_protocol": "pull"}, "sync_protocol must be one of"),
        ({"sync_protocol": "counter"}, "sync_protocol must be one of"),
        ({"sync_protocol": None}, "sync_protocol must be one of"),
        ({"knobs": {"dispatch_protocol": "pull"}}, "select the protocols with sync"),
        ({"knobs": {"combine_protocol": "counter"}}, "select the protocols with sync"),
        (
            {**SENTINEL, "knobs": {"dispatch_protocol": "sentinel"}},
            "cannot set \\['dispatch_protocol'\\]",
        ),
    ],
)
def test_config_validation(fields, message):
    with pytest.raises(ValueError, match=message):
        _config(**fields)


@pytest.mark.parametrize(
    "capacity, knobs, fields, message",
    [
        # Engine fields the host owns, names no engine field has, and the capacity.
        (128, {"hidden": 1024}, {}, "unknown"),
        (128, {"weight_layout": "kmajor"}, {}, "unknown"),
        (8200, None, {}, "at most 8192 tokens per rank"),
        (1024, {"dispatch_rounds": 4}, {}, "unknown"),
        # The pull dispatch's knobs need the pull dispatch, which the Sentinel
        # protocols do not run.
        (512, {"comm_pull_depth": 2}, SENTINEL, "comm_pull_depth"),
        (512, {"pull_warps": 4}, SENTINEL, "pull_warps"),
        (512, {"comm_exchange_ctas": 8}, SENTINEL, "comm_exchange_ctas"),
        (128, {"pull_warps": 4}, {}, "pull_warps"),
        (512, {"pull_warps": 5}, FENCE, "pull_warps"),
        # FC1's token pool takes the walking CTAs' rows and four Gather-B warps.
        (2048, {"fc1_token_pool": 1}, FENCE, "fc1_token_pool"),
        (2048, {**POOL, "pull_warps": 4}, FENCE, "fc1_token_pool"),
        (2048, {**POOL, "gather_b_warps": 8}, FENCE, "fc1_token_pool"),
        (128, POOL, SENTINEL, "fc1_token_pool"),
        (2048, {"fc1_token_pool_fallback": "expert"}, {}, "fc1_token_pool_fallback"),
        (2048, {"gather_b_warps": 6}, {}, "gather_b_warps must be"),
        # Rings, tiles, prefetch and die-local weights.
        (512, {"token_ab_stages": 1}, {}, "token_ab_stages must be"),
        (1024, {"alternate_token_tile_n": 64}, {}, "alternate_token_tile_n must be"),
        (1024, {"token_tile_choice": "alternate"}, {}, "token_tile_choice must be"),
        (256, {"entry_prefetch_bytes": -1}, {}, "entry_prefetch_bytes must be"),
        (1024, {"work_id_mode": "grid_stride"}, DIE_LOCAL, "die_local_weights req"),
        # Just-in-time claims and their lead.
        (128, {"just_in_time_claims": 1}, {}, "Python bool"),
        (128, {**JIT, "claim_lead_k_tiles": 0}, {}, "positive Python integer"),
        (128, {**NO_JIT, "claim_lead_k_tiles": 8}, {}, "only to just-in-time claims"),
        # Combine-slot knobs belong to the Fence combine's slots.
        (256, {"discard_combine_slots": True}, SENTINEL, "protocol='counter'"),
        (256, {"combine_warp_stores": True}, SENTINEL, "protocol='counter'"),
        (
            4096,
            {"discard_combine_slots": True},
            {**FENCE, **IN_KERNEL},
            "no combine slots",
        ),
        # Shared-expert knobs need a shared expert, and take only their values.
        (256, {"shared_expert_entry": True}, {}, "need shared_expert"),
        (256, {"shared_fc2_placement": "late"}, SHARED, "shared_fc2_placement must"),
        (2048, {"shared_head_token_tiles": 0}, SHARED, "shared_head_token_tiles must"),
        (2048, {"shared_head_token_tiles": 2, **NO_HEAD}, SHARED, "caps the entry"),
    ],
)
def test_knob_validation(dist_not_initialized, capacity, knobs, fields, message):
    with pytest.raises(MoEEpConfigError, match=message):
        _engine_config(capacity, 8, knobs, **fields)


def test_engine_config_validation():
    """The engine config rejects the combinations the backend never resolves to."""
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        Sm100GenphaseMoeConfig,
    )

    common = dict(
        num_total_experts=EXPERTS,
        max_tokens_per_rank=256,
        num_topk=TOPK,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        rank=0,
        world_size=8,
        quant_kind="mxfp4_mxfp8",
        cluster_shape_mn=(2, 1),
        mma_tiler_mnk=(256, 128, 128),
        **SENTINEL_ENGINE,
    )
    Sm100GenphaseMoeConfig(**SHARED, **common)
    Sm100GenphaseMoeConfig(**{**common, **FENCE_ENGINE})
    shared_rounding = {**SHARED, "shared_bf16_rounding": True}
    for fields, message in (
        # Two dispatch protocols; "counter" names the Fence combine only.
        ({"dispatch_protocol": "counter"}, "'sentinel' or 'pull'"),
        # The pull dispatch needs the Fence combine, whose tail barrier orders
        # the peers' fetches before the next staging.
        ({"dispatch_protocol": "pull"}, "requires combine_protocol='counter'"),
        (IN_KERNEL, "only under combine_protocol='counter'"),
        ({**SHARED, "cluster_shape_mn": (4, 1)}, "shared_expert needs GenPhase"),
        ({**shared_rounding, **IN_KERNEL, **FENCE_ENGINE}, "in-kernel top-k reduce"),
    ):
        with pytest.raises(ValueError, match=message):
            Sm100GenphaseMoeConfig(**{**common, **fields})


@pytest.mark.parametrize(
    "tokens, world, experts, topk, dispatch",
    [
        # DeepSeek-V4.1-Flash, top-6 of 384: 1.35 routes per reached rank at EP8.
        (192, 8, 384, 6, "sentinel"),
        (256, 8, 384, 6, "sentinel"),
        (257, 8, 384, 6, "pull"),
        (4096, 8, 384, 6, "pull"),
        # 1.8 routes at EP4 and 6 at EP1.
        (192, 4, 384, 6, "sentinel"),
        (193, 4, 384, 6, "pull"),
        (193, 1, 48, 6, "pull"),
        # Kimi-K3, top-16 of 896 (2.26), and HY4, top-8 of 256 (1.51), at EP8.
        (192, 8, 896, 16, "sentinel"),
        (193, 8, 896, 16, "pull"),
        (192, 8, 256, 8, "sentinel"),
        (256, 8, 256, 8, "pull"),
    ],
)
def test_dispatch_rule(tokens, world, experts, topk, dispatch):
    """The Sentinel dispatch to 192 tokens per rank, to 256 within 1.5 routes per reached rank; the combine follows."""
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        dispatch_protocol,
        nvlink_one_sided_tiling,
    )

    assert dispatch_protocol(tokens, world, experts, topk) == dispatch
    tiling = nvlink_one_sided_tiling(tokens, world, experts, topk)
    assert tiling["dispatch_protocol"] == dispatch
    combine = "sentinel" if dispatch == "sentinel" else "counter"
    assert tiling["combine_protocol"] == combine
    # "auto" is the rule, and forcing the protocol the rule picks changes nothing.
    chosen = "sentinel" if dispatch == "sentinel" else "fence"
    for protocol in ("auto", chosen):
        assert tiling == nvlink_one_sided_tiling(
            tokens, world, experts, topk, sync_protocol=protocol
        )


@pytest.mark.parametrize("capacity", [1, 16, 128, 256, 288, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("world", [2, 8])
def test_sync_protocol(dist_not_initialized, capacity, world):
    """``sync_protocol`` selects the dispatch and combine pair; the schedule follows it.

    "sentinel" runs the Sentinel dispatch and combine and "fence" the pull
    dispatch and the Fence (counter) combine at every capacity; "auto" resolves
    to the one of them the dispatch rule picks, field for field. The pull
    dispatch's pipelined groups and just-in-time claims, and the in-kernel
    top-k reduce, follow the protocol, not the capacity alone.
    """
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        dispatch_protocol,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.block_scaled import (
        JUST_IN_TIME_CLAIMS_MAX_TOKENS,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.genphase import (
        PIPELINED_GROUPS_MIN_TOKENS,
        PULL_JUST_IN_TIME_MIN_TOKENS,
        SINGLE_GROUP_MAX_TOKENS,
    )

    assert _config().sync_protocol == "auto"
    configs = {
        protocol: _engine_config(capacity, world, sync_protocol=protocol)
        for protocol in ("auto", "sentinel", "fence")
    }
    sentinel, fence = configs["sentinel"], configs["fence"]
    assert (sentinel.dispatch_protocol, sentinel.combine_protocol) == (
        "sentinel",
        "sentinel",
    )
    assert (fence.dispatch_protocol, fence.combine_protocol) == ("pull", "counter")
    rule = dispatch_protocol(capacity, world, EXPERTS, TOPK)
    assert configs["auto"] == (sentinel if rule == "sentinel" else fence)
    # The token tile and cluster follow the load alone.
    for name in ("mma_tiler_mnk", "cluster_shape_mn", "fc2_tma_stages"):
        assert getattr(sentinel, name) == getattr(fence, name)
    # Up to SINGLE_GROUP_MAX_TOKENS one schedule group holds every FC1 tile;
    # above it groups of 256 FC1 tiles, which the pull dispatch halves once they
    # are pipelined. Just-in-time claims: the pull dispatch's from
    # PULL_JUST_IN_TIME_MIN_TOKENS, the Sentinel dispatch's at the smallest
    # capacities.
    single = capacity <= SINGLE_GROUP_MAX_TOKENS
    pipelined = capacity >= PIPELINED_GROUPS_MIN_TOKENS
    assert sentinel.pipelined_groups == fence.pipelined_groups == pipelined
    assert sentinel.schedule_policy == ("grouped", 4096 if single else 256)
    assert fence.schedule_policy == (
        "grouped",
        4096 if single else 128 if pipelined else 256,
    )
    assert fence.resolved_just_in_time_claims == (
        capacity >= PULL_JUST_IN_TIME_MIN_TOKENS
    )
    assert sentinel.resolved_just_in_time_claims == (
        capacity <= JUST_IN_TIME_CLAIMS_MAX_TOKENS
    )
    # The in-kernel top-k reduce takes effect under the Fence combine only.
    for protocol, config in configs.items():
        inplace = _engine_config(capacity, world, sync_protocol=protocol, **IN_KERNEL)
        assert inplace.reduce_topk_in_kernel == (config.combine_protocol == "counter")
        assert inplace == dataclasses.replace(
            config, reduce_topk_in_kernel=inplace.reduce_topk_in_kernel
        )


def test_sync_protocol_tiling_rejects_other_names():
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        nvlink_one_sided_tiling,
    )

    for protocol in ("pull", "counter", "flag"):
        with pytest.raises(ValueError, match="sync_protocol must be one of"):
            nvlink_one_sided_tiling(128, 8, EXPERTS, TOPK, sync_protocol=protocol)


def test_forced_sentinel_build_failure_names_sync_protocol(
    dist_not_initialized, monkeypatch
):
    """An engine the forced Sentinel protocols cannot build fails at init and names ``sync_protocol``.

    Where "auto" would run the Fence protocols the message says so; where it
    would run the Sentinel ones too, or the protocol is not forced, the
    engine's own error stands.
    """
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm103.mxfp8_mxfp4_bf16_nvlink_one_sided import (
        backend,
    )

    def fail(config):
        raise MoEEpConfigError("cannot build its engine: SMEM plan needs more")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 3))
    monkeypatch.setattr(backend, "_build_engine", fail)
    bootstrap = BootstrapConfig(world_size=8, rank=7, auto_bootstrap=False)
    for capacity, fields, hint in (
        (4096, SENTINEL, True),
        (128, SENTINEL, False),
        (4096, FENCE, False),
        (4096, {}, False),
    ):
        with pytest.raises(MoEEpConfigError, match="SMEM plan") as error:
            _kernel(8, **fields).validate_init(bootstrap, _fleet(capacity))
        named = "sync_protocol='auto' runs the Fence protocols" in str(error.value)
        assert named == hint, (capacity, fields)


def test_init_rejections(dist_not_initialized, monkeypatch):
    """Rows not in runs of 32 x 16 bytes and top_k above 32 fail at init."""
    import torch

    # Host-only: skip the device checks and the engine build of validate_init.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    bootstrap = BootstrapConfig(world_size=4, rank=3, auto_bootstrap=False)
    for capacity in (128, 256):
        for hidden in (2304, 3840):
            with pytest.raises(MoEEpConfigError, match="multiple of 512"):
                _kernel(4).validate_init(bootstrap, _fleet(capacity, hidden))
        # Kimi-K3's 3584 among them.
        for hidden in (2560, 3072, 3584):
            _kernel(4).validate_init(bootstrap, _fleet(capacity, hidden))
    with pytest.raises(MoEEpConfigError, match="top_k up to 32"):
        _kernel(4, top_k=33).validate_init(bootstrap, _fleet(128))
    _kernel(4, top_k=32).validate_init(bootstrap, _fleet(128))


def test_init_admits_sm100_and_sm103(dist_not_initialized, monkeypatch):
    """``validate_init`` builds the engine on SM100 (B200) and SM103 (B300) devices only."""
    import torch

    from flashinfer.moe_ep import MoEEpArchError
    from flashinfer.moe_ep.backends.mega.kernel.sm103.mxfp8_mxfp4_bf16_nvlink_one_sided import (
        backend,
    )

    built = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_build_engine", built.append)
    bootstrap = BootstrapConfig(world_size=4, rank=3, auto_bootstrap=False)
    for capability in ((10, 0), (10, 3), (9, 0), (11, 0), (12, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
        built.clear()
        if capability in ((10, 0), (10, 3)):
            _kernel(4).validate_init(bootstrap, _fleet(128))
            assert len(built) == 1
        else:
            with pytest.raises(MoEEpArchError, match="sm_100 or sm_103"):
                _kernel(4).validate_init(bootstrap, _fleet(128))
            assert not built


def test_capture_requires_compiled_workspace():
    """The capture guard runs before staging, so nothing is recorded on failure."""
    backend = create_mega_kernel(_config())
    for workspace, message in (
        (SimpleNamespace(_destroyed=False, _compiled=None), "warmup"),
        (SimpleNamespace(_destroyed=True, _compiled=None), "destroyed"),
    ):
        with pytest.raises(MoEEpConfigError, match=message):
            backend.validate_capture_ready(workspace, None)
    backend.validate_capture_ready(
        SimpleNamespace(_destroyed=False, _compiled=object()), None
    )


def test_situ_config(dist_not_initialized):
    """SiTU reaches the engine config as its own configuration, one per pair of betas.

    The activation and its betas change no other engine field, so the tiling,
    protocols and workspace layout follow the capacity as under SwiGLU, while
    the engine configs, which key the built engines and the workspace pool,
    differ from SwiGLU's and from each other's.
    """
    config = Config(
        intermediate_size=INTERMEDIATE,
        top_k=TOPK,
        activation="situ",
        situ_beta=4,
        situ_linear_beta=25,
    )
    assert (config.situ_beta, config.situ_linear_beta) == (4.0, 25.0)
    assert all(
        type(value) is float for value in (config.situ_beta, config.situ_linear_beta)
    )
    assert config.resolved_gate_up_clamp is None
    swiglu = _config()
    assert (swiglu.activation, swiglu.situ_beta, swiglu.situ_linear_beta) == (
        "swiglu",
        None,
        None,
    )
    # SiTU combines with the options that do not touch the activation.
    for fields in (IN_KERNEL, {"routing_weights_after_fc2": True}, DIE_LOCAL):
        _config(**SITU, **fields)
    other = {**SITU, "situ_beta": 2.0, "situ_linear_beta": 8.0}
    for capacity in (16, 128, 256, 1024, 4096):
        for world in (2, 8):
            # _kernel sets the SwiGLU clamp, which SiTU rejects.
            plain = create_mega_kernel(_config())
            situ = create_mega_kernel(_config(**SITU))
            situ_other = create_mega_kernel(_config(**other))
            configs = []
            for kernel in (plain, situ, situ_other):
                kernel.bind_ep_bootstrap(
                    BootstrapConfig(world_size=world, rank=0, auto_bootstrap=False)
                )
                configs.append(kernel._engine_config(_fleet(capacity)))
            plain_config, situ_config, other_config = configs
            assert plain_config.activation == "swiglu"
            assert plain_config.situ_beta is plain_config.situ_linear_beta is None
            assert situ_config == dataclasses.replace(plain_config, **SITU)
            assert other_config == dataclasses.replace(plain_config, **other)
            assert len({plain_config, situ_config, other_config}) == 3


def test_situ_engine_config_validation():
    """The engine config takes SiTU with its betas, without a clamp or a shared expert."""
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        Sm100GenphaseMoeConfig,
    )

    common = dict(
        num_total_experts=EXPERTS,
        max_tokens_per_rank=256,
        num_topk=TOPK,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        rank=0,
        world_size=8,
        quant_kind="mxfp4_mxfp8",
        cluster_shape_mn=(2, 1),
        mma_tiler_mnk=(256, 128, 128),
        **SENTINEL_ENGINE,
    )
    Sm100GenphaseMoeConfig(**SITU, **common)
    for fields, message in (
        ({**SITU, "gate_up_clamp": CLAMP}, "SiTU does not support gate_up_clamp"),
        ({**SITU, "situ_linear_beta": None}, "positive, finite situ_linear_beta"),
        ({"situ_beta": 4.0}, "require activation='situ'"),
        ({**SITU, **SHARED}, "excludes shared_expert"),
    ):
        with pytest.raises(ValueError, match=message):
            Sm100GenphaseMoeConfig(**{**common, **fields})


def test_routing_weights_after_fc2_config(dist_not_initialized):
    """Off by default; on, it changes only the engine's order of the routing weight."""
    assert not _config().routing_weights_after_fc2
    for capacity in (128, 4096):
        plain = _engine_config(capacity)
        assert plain.apply_routing_weights_before_fc2
        after = _engine_config(capacity, routing_weights_after_fc2=True)
        assert after == dataclasses.replace(
            plain, apply_routing_weights_before_fc2=False
        )


def test_shared_expert_weights(dist_not_initialized):
    """The shared expert's weights become two more K-major E4M3 legs; packs and the config must agree."""
    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep import PrequantizedMoEWeightsWithSharedExpert as Pack
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.kernel_helpers import (
        swizzled_flat_sf_size,
    )

    experts, hidden, inter = 2, 512, 256
    generator = torch.Generator().manual_seed(5)

    def randint(low, high, *shape):
        return torch.randint(low, high, shape, dtype=torch.uint8, generator=generator)

    routed = (
        randint(0, 255, experts, 2 * inter, hidden // 2),
        randint(0, 255, experts, hidden, inter // 2),
        randint(100, 140, experts, 2 * inter, hidden // 32),
        randint(100, 140, experts, hidden, inter // 32),
    )
    shared = dict(
        shared_w13=randint(0, 126, 2 * inter, hidden),
        shared_w2=randint(0, 126, hidden, inter).view(torch.float8_e4m3fn),
        shared_w13_scale=randint(100, 140, 2 * inter, hidden // 32),
        shared_w2_scale=randint(100, 140, hidden, inter // 32).view(
            torch.float8_e8m0fnu
        ),
    )
    pack = Pack(*routed, **shared)
    sizes = dict(intermediate_size=inter, hidden_size=hidden, weight_layout="kmajor")
    legs = preprocess_mega_weights(pack, **SHARED, **sizes)
    plain = preprocess_mega_weights(MoEWeightPack(*routed), **sizes)
    assert len(legs) == 4 and len(plain) == 2
    for leg, plain_leg in zip(legs[:2], plain, strict=True):
        assert all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(leg, plain_leg, strict=True)
        )
    (w1, s1), (w2, s2) = legs[2:]
    assert tuple(w1.shape) == (1, hidden, 2 * inter) and w1.dtype == torch.float8_e4m3fn
    assert tuple(w2.shape) == (1, inter, hidden) and w2.dtype == torch.float8_e4m3fn
    assert w1.permute(0, 2, 1).is_contiguous() and w1._mega_weight_layout == "kmajor"
    assert tuple(s1.shape) == (1, swizzled_flat_sf_size(2 * inter, hidden // 32))
    assert tuple(s2.shape) == (1, swizzled_flat_sf_size(hidden, inter // 32))
    assert torch.equal(
        w2[0].permute(1, 0).view(torch.uint8), shared["shared_w2"].view(torch.uint8)
    )
    # The pack and the config must agree, and the shared expert has the routed width
    # and E4M3 weights.
    narrow = {**shared, "shared_w2": shared["shared_w2"][:, : inter // 2]}
    wide = {**shared, "shared_w13": shared["shared_w13"].float()}
    for weights, fields, message in (
        (MoEWeightPack(*routed), SHARED, "PrequantizedMoEWeightsWithSharedExpert"),
        (pack, {}, "PrequantizedMoEWeightsWithSharedExpert"),
        (Pack(*routed, **narrow), SHARED, "intermediate width"),
        (Pack(*routed, **wide), SHARED, "E4M3"),
    ):
        with pytest.raises(MoEEpConfigError, match=message):
            preprocess_mega_weights(weights, **fields, **sizes)


def test_token_tile_choice_rule():
    """The host mirror of the main kernel's choice: the alternate exactly when every plan tile fits it."""
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        chosen_token_tile_n,
    )

    # V4.1-Flash, EP8, perfect routing: every local expert takes T * 8 * 6 / 384
    # routes; at 1024 and 2048 tokens per rank every N192 token tile holds 128.
    assert chosen_token_tile_n([128] * 48, 192, 128) == 128
    assert chosen_token_tile_n([256] * 48, 192, 128) == 128
    assert chosen_token_tile_n([512] * 48, 192, 128) == 192
    # One expert past one N128 tile (129 routes, or 257 over two tiles) keeps N192.
    assert chosen_token_tile_n([128] * 47 + [129], 192, 128) == 192
    assert chosen_token_tile_n([256] * 47 + [257], 192, 128) == 192
    # Experts without routes fit any tile; no alternate keeps the plan's tile.
    assert chosen_token_tile_n([0] * 48, 192, 128) == 128
    assert chosen_token_tile_n([128] * 48, 192, 0) == 192
    # N240's slots hold two N128 tiles' scale factors as well.
    assert chosen_token_tile_n([120] * 32, 240, 128) == 128
    assert chosen_token_tile_n([200] * 32, 240, 128) == 240
