"""Host-only contracts of the NVLinkOneSided MegaMoE backend.

``sm103_mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl`` without a GPU and without
importing the CuTe DSL: registration and exports, config and knob validation,
the shapes and devices ``validate_init`` rejects, the CUDA graph capture guard,
the shared expert's config and weights, and the host mirror of the main
kernel's token tile choice. The engine path each capacity resolves to is
asserted by the layer-vs-oracle cases in
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
    Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSidedCutedsl_MegaMoeConfig as Config,
    preprocess_sm103_mxfp4_nvlink_one_sided_mega_weights,
)
from flashinfer.moe_ep.backends.mega.kernel.sm103.mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl import (
    Sm103Mxfp8Mxfp4NVLinkOneSidedMegaKernelBackend,
    preprocess_mega_weights,
)
from flashinfer.moe_ep.core.kernel.registry import (
    _available_mega_kernels,
    create_mega_kernel,
)

pytestmark = pytest.mark.cpu_only

NAME = "sm103_mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl"
HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS = 256
TOPK = 6
CLAMP = 10.0
SHARED = {"shared_expert": "fp8_in_kernel"}
SENTINEL = {"dispatch_protocol": "sentinel", "combine_protocol": "sentinel"}
COUNTER = {"dispatch_protocol": "counter", "combine_protocol": "counter"}
PULL = {"dispatch_protocol": "pull", "combine_protocol": "counter"}
COUNTER_DISPATCH = {"dispatch_protocol": "counter", "combine_protocol": "sentinel"}
POOL = {**PULL, "fc1_token_pool": True}
JIT = {"just_in_time_claims": True}
NO_JIT = {"just_in_time_claims": False}
NO_HEAD = {"shared_expert_entry": False}
DIE_LOCAL = {"die_local_weights": True}
IN_KERNEL = {"reduce_topk_in_kernel": True}


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
        # Dispatch rounds need the counter dispatch and the counter combine.
        (1024, {**COUNTER, "dispatch_rounds": 3}, {}, "dispatch_rounds must be"),
        (1024, {**COUNTER_DISPATCH, "dispatch_rounds": 4}, {}, "combine_protocol"),
        (1024, {**PULL, "dispatch_rounds": 4}, {}, "dispatch_protocol='counter'"),
        # The pull dispatch needs the counter combine; its knobs need it.
        (128, {**SENTINEL, "dispatch_protocol": "pull"}, {}, "protocol='counter'"),
        (512, {**COUNTER, "comm_pull_depth": 2}, {}, "comm_pull_depth"),
        (512, {**COUNTER, "pull_warps": 4}, {}, "pull_warps"),
        (512, {**PULL, "pull_warps": 5}, {}, "pull_warps"),
        # FC1's token pool takes the walking CTAs' rows and four Gather-B warps.
        (2048, {**PULL, "fc1_token_pool": 1}, {}, "fc1_token_pool"),
        (2048, {**POOL, "pull_warps": 4}, {}, "fc1_token_pool"),
        (2048, {**POOL, "gather_b_warps": 8}, {}, "fc1_token_pool"),
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
        # Combine-slot knobs belong to the counter combine's slots.
        (256, {**SENTINEL, "discard_combine_slots": True}, {}, "protocol='counter'"),
        (256, {**SENTINEL, "combine_warp_stores": True}, {}, "protocol='counter'"),
        (4096, {**PULL, "discard_combine_slots": True}, IN_KERNEL, "no combine slots"),
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
        **COUNTER_DISPATCH,
    )
    Sm100GenphaseMoeConfig(**SHARED, **common)
    shared_rounding = {**SHARED, "shared_bf16_rounding": True}
    for fields, message in (
        (IN_KERNEL, "only under combine_protocol='counter'"),
        ({**SHARED, "cluster_shape_mn": (4, 1)}, "shared_expert needs GenPhase"),
        ({**shared_rounding, **IN_KERNEL, **COUNTER}, "in-kernel top-k reduce"),
    ):
        with pytest.raises(ValueError, match=message):
            Sm100GenphaseMoeConfig(**{**common, **fields})


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
    from flashinfer.moe_ep.backends.mega.kernel.sm103.mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl import (
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
