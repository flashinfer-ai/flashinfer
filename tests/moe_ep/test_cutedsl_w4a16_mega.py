"""Public W4A16 MegaMoE correctness on one or two Blackwell ranks.

Run the single-rank cases with pytest, or the two-rank cases with::

    torchrun --master-addr=127.0.0.1 --master-port=29531 --nproc_per_node=2 -m pytest \
        tests/moe_ep/test_cutedsl_w4a16_mega.py -v -m gpu_2

The oracle decodes the original packed weights independently of the backend.
It retains both BF16 activation boundaries, FP32 post-MMA global scales, and
FP32 routing after the BF16 FC2 output. No activation or combine quantization
is present. Graph comparisons use exact eager/replay equality.
"""

from __future__ import annotations

import dataclasses
import math
import os

import pytest
import torch
import torch.distributed as dist

from .mega_oracle_compare import _assert_mega_oracle_term_band_close

_HIDDEN = 256
_INTERMEDIATE = 256
_EXPERTS = 4
_TOP_K = 2
_CAPACITY = 64


@pytest.fixture(autouse=True)
def _isolate_default_tactics(monkeypatch):
    # Keep default-N128/raw-ring coverage independent of a user's tuned cache.
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "0")


def _bootstrap(expected_world_size):
    from flashinfer.moe_ep import BootstrapConfig, ensure_moe_ep_cuda_device

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size:
        pytest.skip(f"requires {expected_world_size} ranks, got {world_size}")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    bootstrap = BootstrapConfig(
        world_size=world_size, rank=int(os.environ.get("RANK", "0"))
    )
    ensure_moe_ep_cuda_device(bootstrap)
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100 or SM103")
    return bootstrap


def _barrier():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _weights(*, sparse=False, hidden=_HIDDEN, intermediate=_INTERMEDIATE):
    from flashinfer.moe_ep import PrequantizedMoEWeights

    generator = torch.Generator(device="cuda").manual_seed(20260907)
    shapes = (
        (_EXPERTS, 2 * intermediate, hidden // 2),
        (_EXPERTS, hidden, intermediate // 2),
    )
    packed = [
        torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=generator
        )
        for shape in shapes
    ]
    scales = [
        torch.randint(
            -3,
            0,
            (*shape[:-1], shape[-1] // 8),
            device="cuda",
            generator=generator,
        )
        .float()
        .exp2()
        .to(torch.float8_e4m3fn)
        for shape in shapes
    ]
    alphas = [
        torch.linspace(start, end, _EXPERTS, device="cuda", dtype=torch.float32)
        for start, end in ((0.71013, 1.23017), (1.17019, 0.83023))
    ]
    if sparse:
        for tensor in packed:
            tensor.zero_()
        for tensor in scales:
            tensor.fill_(1.0)
        # Canonical FC1 is [gate; up]. Experts 0 and 3 have identical FC1
        # and opposite FC2, making FP32 routing precision observable after
        # cancellation. Codes 2/10 are +1/-1 in E2M1, in the low nibble.
        packed[0][:, 0, 0] = 2
        packed[0][:, intermediate, 0] = 2
        packed[1][0, 0, 0] = 2
        packed[1][-1, 0, 0] = 10
        alphas[0].fill_(1.00390625)
        alphas[1].fill_(1.001953125)
    return PrequantizedMoEWeights(
        w13=packed[0],
        w2=packed[1],
        w13_scale=scales[0],
        w2_scale=scales[1],
        w13_global_scale=alphas[0],
        w2_global_scale=alphas[1],
    )


def _decode_nvfp4(packed, block_scale):
    # Independent E2M1 decode: low nibble is the first logical K element.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    decoded = lut[codes.long()] * block_scale.float().repeat_interleave(16, dim=-1)
    return decoded.to(torch.bfloat16)


def _reference_terms(tensors, weights, *, gate_up_clamp=None):
    """Return weighted FP32 terms before the final BF16 output conversion."""
    x = tensors.hidden_states.float()
    w13 = _decode_nvfp4(weights.w13, weights.w13_scale).float()
    w2 = _decode_nvfp4(weights.w2, weights.w2_scale).float()
    terms = torch.zeros(
        (x.shape[0], tensors.topk_ids.shape[1], x.shape[1]),
        device=x.device,
        dtype=torch.float32,
    )
    # FP32 torch GEMMs must not substitute TF32 operands for the exact BF16
    # values. The fused kernel uses BF16 MMA with FP32 accumulation.
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in range(_EXPERTS):
            tokens, slots = torch.where(tensors.topk_ids == expert)
            if tokens.numel() == 0:
                continue
            fc1 = (x[tokens] @ w13[expert].T) * weights.w13_global_scale[expert]
            gate, up = fc1.chunk(2, dim=-1)
            if gate_up_clamp is not None:
                gate = gate.clamp(max=gate_up_clamp)
                up = up.clamp(min=-gate_up_clamp, max=gate_up_clamp)
            intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            fc2 = (intermediate.float() @ w2[expert].T) * weights.w2_global_scale[
                expert
            ]
            # FC2 is stored unweighted in BF16; the external reducer consumes
            # FP32 routing weights and accumulates before its final BF16 cast.
            terms[tokens, slots] = fc2.to(torch.bfloat16).float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    return terms * tensors.topk_weights.float().unsqueeze(-1)


def _inputs(rank, num_tokens, *, skewed=False, sparse=False, hidden=_HIDDEN):
    from flashinfer.moe_ep import MoEEpTensors

    generator = torch.Generator(device="cuda").manual_seed(73 + rank)
    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    rows = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    ids = torch.stack((rows % _EXPERTS, (rows + 2) % _EXPERTS), dim=-1)
    scores = (
        torch.rand(
            num_tokens, _TOP_K, dtype=torch.float32, device="cuda", generator=generator
        )
        + 0.12513
    )
    if skewed:
        ids[:] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    if sparse:
        x.zero_()
        x[:, 0] = torch.tensor([0.5, 1.0, 2.0, 4.0], device="cuda")
        ids[:] = torch.tensor([0, _EXPERTS - 1], dtype=torch.int32, device="cuda")
        # The first score is exactly halfway between BF16 1 and 1+2^-7.
        # A BF16 routing cast turns the expected nonzero difference into zero.
        scores[:, 0] = 1.00390625
        scores[:, 1] = 1.0
    return MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=scores)


def _layer(bootstrap, global_weights, *, capacity=_CAPACITY, knobs=None):
    from flashinfer.moe_ep import (
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    local_experts = _EXPERTS // bootstrap.world_size
    start = bootstrap.rank * local_experts
    local_weights = dataclasses.replace(
        global_weights,
        **{
            field.name: getattr(global_weights, field.name)[
                start : start + local_experts
            ].clone()
            for field in dataclasses.fields(global_weights)
        },
    )
    return MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=_EXPERTS,
            max_tokens_per_rank=capacity,
            token_hidden_size=global_weights.w2.shape[1],
        ),
        weights=local_weights,
        backend=MegaConfig(
            megakernel=Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=global_weights.w13.shape[1] // 2,
                top_k=_TOP_K,
                knobs=knobs,
            )
        ),
    )


def _check_numerical(
    expected_world_size,
    *,
    hidden=_HIDDEN,
    intermediate=_INTERMEDIATE,
    num_tokens=None,
    knobs=None,
    normalize_fc1=False,
):
    bootstrap = _bootstrap(expected_world_size)
    weights = _weights(hidden=hidden, intermediate=intermediate)
    if normalize_fc1:
        # Keep random FC1 variance bounded as fan-in grows. This changes only
        # the fixture's FP32 global scale, after the decoded BF16-weight MMA.
        weights = dataclasses.replace(
            weights, w13_global_scale=weights.w13_global_scale / math.sqrt(hidden)
        )
    layer = _layer(
        bootstrap, weights, capacity=max(_CAPACITY, num_tokens or 0), knobs=knobs
    )
    try:
        # Reuse one workspace across changing routing and token counts. The
        # empty source rank still owns experts needed by its peers.
        rounds = (
            ("balanced", 17, False),
            ("skewed", 17, True),
            ("single_token", 1 if bootstrap.rank == 0 else 0, False),
            ("empty_source", 0 if bootstrap.rank == 0 else 11, False),
            ("all_empty", 0, False),
            ("refill", 9, False),
        )
        if num_tokens is not None:
            rounds = (("skewed_tiles", num_tokens, True),)
        for name, count, skewed in rounds:
            tensors = _inputs(bootstrap.rank, count, skewed=skewed, hidden=hidden)
            terms = _reference_terms(
                tensors, weights, gate_up_clamp=(knobs or {}).get("gate_up_clamp")
            )
            _barrier()
            actual = layer.forward(tensors)
            _barrier()
            assert actual.dtype == torch.bfloat16
            assert actual.shape == (count, hidden)
            assert torch.isfinite(actual).all()
            if count:
                _assert_mega_oracle_term_band_close(
                    actual,
                    terms,
                    ikr=False,
                    label=(
                        f"W4A16 {name} H={hidden} I={intermediate} "
                        f"rank={bootstrap.rank}"
                    ),
                )
    finally:
        layer.destroy()
        _barrier()


def _check_scale_and_routing_contract(expected_world_size, *, knobs=None):
    bootstrap = _bootstrap(expected_world_size)
    weights = _weights(sparse=True)
    tensors = _inputs(bootstrap.rank, 4, sparse=True)
    expected = _reference_terms(tensors, weights).sum(dim=1).to(torch.bfloat16)
    assert torch.count_nonzero(expected[:, 0]) == 4
    rounded_routing = dataclasses.replace(
        tensors, topk_weights=tensors.topk_weights.bfloat16().float()
    )
    assert torch.count_nonzero(_reference_terms(rounded_routing, weights)) > 0
    assert (
        torch.count_nonzero(_reference_terms(rounded_routing, weights).sum(dim=1)) == 0
    )
    rounded_globals = dataclasses.replace(
        weights,
        w13_global_scale=weights.w13_global_scale.bfloat16().float(),
        w2_global_scale=weights.w2_global_scale.bfloat16().float(),
    )
    wrong = _reference_terms(tensors, rounded_globals).sum(dim=1).to(torch.bfloat16)
    assert not torch.equal(expected, wrong), (
        "fixture must detect early global-scale rounding"
    )

    layer = _layer(bootstrap, weights, knobs=knobs)
    try:
        actual = layer.forward(tensors)
        _barrier()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        layer.destroy()
        _barrier()


def _check_graph_replay(expected_world_size, *, knobs=None):
    bootstrap = _bootstrap(expected_world_size)
    layer = _layer(bootstrap, _weights(), knobs=knobs)
    tensors = _inputs(bootstrap.rank, 17)
    try:
        layer.warmup()
        eager = layer.forward(tensors).clone()
        _barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer.forward(tensors)
        _barrier()
        for _ in range(3):
            graph.replay()
            _barrier()
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)

        tensors.hidden_states.mul_(0.5)
        tensors.topk_ids.copy_(tensors.topk_ids.roll(1, dims=0))
        tensors.topk_weights.mul_(0.71013)
        graph.replay()
        _barrier()
        replay = captured.clone()
        eager = layer.forward(tensors)
        _barrier()
        torch.testing.assert_close(replay, eager, rtol=0, atol=0)
    finally:
        layer.destroy()
        _barrier()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "check", (_check_numerical, _check_scale_and_routing_contract, _check_graph_replay)
)
def test_w4a16_mega_single_rank(check):
    check(1)


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "check", (_check_numerical, _check_scale_and_routing_contract, _check_graph_replay)
)
def test_w4a16_mega_two_rank(check):
    check(2)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("byte_views_first", (False, True))
def test_w4a16_mega_reuses_w4a4_weights_and_dtype_aliases(byte_views_first):
    from flashinfer.moe_ep import (
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        preprocess_nvfp4_cutedsl_mega_weights,
        preprocess_w4a16_cutedsl_mega_weights,
    )

    bootstrap = _bootstrap(1)
    hidden, intermediate = 288, 448
    weights = _weights(hidden=hidden, intermediate=intermediate)
    kwargs = dict(hidden_size=hidden, intermediate_size=intermediate)
    pairs = preprocess_nvfp4_cutedsl_mega_weights(
        dataclasses.replace(weights, w13_global_scale=None, w2_global_scale=None),
        **kwargs,
    )
    prepared = preprocess_w4a16_cutedsl_mega_weights(weights, **kwargs)
    shared = tuple(
        (*pair, alpha)
        for pair, alpha in zip(
            pairs, (weights.w13_global_scale, weights.w2_global_scale), strict=True
        )
    )
    for pair, triple in zip(pairs, prepared, strict=True):
        for shared_tensor, own_tensor in zip(pair, triple[:2], strict=True):
            assert shared_tensor.dtype == own_tensor.dtype
            assert shared_tensor.shape == own_tensor.shape
            assert shared_tensor.stride() == own_tensor.stride()
            assert torch.equal(
                shared_tensor.view(torch.uint8), own_tensor.view(torch.uint8)
            )

    # A new backing allocation forces fresh launch arguments on the existing
    # compiled frontend. Same-pointer aliases alone could reuse cached args.
    byte_views = tuple(
        (
            q.transpose(1, 2).view(torch.uint8).clone().transpose(1, 2),
            sf.view(torch.uint8).clone(),
            alpha,
        )
        for q, sf, alpha in shared
    )
    initial, alternate = (
        (byte_views, shared) if byte_views_first else (shared, byte_views)
    )
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=_EXPERTS,
            max_tokens_per_rank=_CAPACITY,
            token_hidden_size=hidden,
        ),
        weights=None,
        backend=MegaConfig(
            megakernel=Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=_TOP_K,
                knobs={
                    "mma_tiler_mnk": (256, 64, 256),
                    "cluster_shape_mnk": (2, 1, 1),
                    "use_2cta_instrs": True,
                    "num_sched_stages": 2,
                    "group_hint": 512,
                    "flag_batch": 4,
                    "epi_flag_batch": (2, 4),
                    "load_balance_mode": "atomic_counter",
                    "token_back_mode": "epi_warps",
                },
            ),
            preprocess_weights=False,
            transformed_weights=initial,
        ),
    )
    tensors = _inputs(bootstrap.rank, 17, hidden=hidden)
    graph = None
    try:
        layer.warmup()
        eager = layer.forward(tensors).clone()
        _barrier()
        _assert_mega_oracle_term_band_close(
            eager, _reference_terms(tensors, weights), ikr=False, label="shared NVFP4"
        )
        frontend = layer._workspace._frontend
        compiled = frontend._mega.compiled
        layer._transformed = alternate
        actual = layer.forward(tensors)
        _barrier()
        assert frontend._mega.compiled is compiled
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer.forward(tensors)
        graph.replay()
        _barrier()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        tensors.hidden_states.mul_(0.5)
        tensors.topk_ids.copy_(tensors.topk_ids.roll(1, dims=0))
        tensors.topk_weights.mul_(0.71013)
        graph.replay()
        _barrier()
        replay = captured.clone()
        actual = layer.forward(tensors)
        _barrier()
        torch.testing.assert_close(replay, actual, rtol=0, atol=0)
        _assert_mega_oracle_term_band_close(
            actual,
            _reference_terms(tensors, weights),
            ikr=False,
            label="shared NVFP4 changed-input graph",
        )
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()
        _barrier()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_w4a16_unquantized_preparation_matches_w4a4(dtype):
    from flashinfer.moe_ep import (
        UnquantizedMoEWeights,
        preprocess_nvfp4_cutedsl_mega_weights,
        preprocess_w4a16_cutedsl_mega_weights,
    )

    _bootstrap(1)
    generator = torch.Generator(device="cuda").manual_seed(20260909)
    weights = UnquantizedMoEWeights(
        *(
            torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
            for shape in ((3, 896, 288), (3, 288, 448))
        )
    )
    kwargs = dict(hidden_size=288, intermediate_size=448)
    pairs = preprocess_nvfp4_cutedsl_mega_weights(weights, **kwargs)
    triples = preprocess_w4a16_cutedsl_mega_weights(weights, **kwargs)
    for pair, triple in zip(pairs, triples, strict=True):
        for reference, actual in zip(pair, triple[:2], strict=True):
            assert reference.dtype == actual.dtype
            assert reference.shape == actual.shape
            assert reference.stride() == actual.stride()
            assert torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
        torch.testing.assert_close(
            triple[2], torch.ones(3, device="cuda"), rtol=0, atol=0
        )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("hidden", "intermediate", "num_tokens", "normalize_fc1"),
    (
        (64, 64, 257, False),
        (192, 320, 257, False),
        (1024, 512, 257, False),
        (9472, 64, 257, True),
    ),
    ids=("h64_i64_m257", "h192_i320_m257", "h1024_i512_m257", "h9472_i64_m257"),
)
@pytest.mark.parametrize(
    "expected_world_size",
    (pytest.param(1, id="ep1"), pytest.param(2, id="ep2", marks=pytest.mark.gpu_2)),
)
def test_w4a16_mega_geometry(
    expected_world_size, hidden, intermediate, num_tokens, normalize_fc1
):
    # The skew sends every token to experts 0 and 1. 257 rows cross both the
    # 128-token tiles; EP2 also leaves one rank without local expert work.
    # The feature tails exercise FC1 and FC2 stores. H1024/I512 gives four/two
    # K256 tiles and multiple work tiles exercise the operand pipelines.
    # H9472/I64 exercises the resource-fitted raw ring at default M256/N128.
    # Its 37 FC1 K256 tiles wrap that ring alongside the one-tile FC2 tail.
    # Only that large-fan-in fixture scales FC1 globals by 1/sqrt(H).
    _check_numerical(
        expected_world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        normalize_fc1=normalize_fc1,
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("token_back_mode", ("epi_warps", "reuse_dispatch_warps"))
@pytest.mark.parametrize(
    "expected_world_size",
    (pytest.param(1, id="ep1"), pytest.param(2, id="ep2", marks=pytest.mark.gpu_2)),
)
def test_w4a16_mega_reference_schedule(expected_world_size, token_back_mode):
    # Existing swapped Mega tuning base. Batched completion must flush at
    # phase/tail boundaries, including empty experts and consecutive launches.
    knobs = {
        "group_hint": 512,
        "epi_flag_batch": (2, 4),
        "load_balance_mode": "atomic_counter",
        "flag_batch": 4,
        "token_back_mode": token_back_mode,
    }
    _check_numerical(expected_world_size, knobs=knobs)
    # Both K tails, a padded feature CTA, and three logical routed-token tiles.
    _check_numerical(
        expected_world_size, hidden=288, intermediate=448, num_tokens=257, knobs=knobs
    )
    _check_scale_and_routing_contract(expected_world_size, knobs=knobs)
    _check_graph_replay(expected_world_size, knobs=knobs)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "expected_world_size",
    (pytest.param(1, id="ep1"), pytest.param(2, id="ep2", marks=pytest.mark.gpu_2)),
)
def test_w4a16_mega_clamp(expected_world_size):
    _check_numerical(
        expected_world_size,
        hidden=288,
        intermediate=448,
        num_tokens=257,
        knobs={"gate_up_clamp": 1.5},
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("token_back_mode", ("epi_warps", "reuse_dispatch_warps"))
@pytest.mark.parametrize(
    ("tile_n", "expected_world_size"),
    (
        pytest.param(64, 1, id="n64-ep1"),
        pytest.param(64, 2, id="n64-ep2", marks=pytest.mark.gpu_2),
        pytest.param(128, 2, id="n128-ep2", marks=pytest.mark.gpu_2),
    ),
)
def test_w4a16_mega_pipeline_wrap(tile_n, expected_world_size, token_back_mode):
    # Both FC phases wrap the operand rings within one work tile: 29/9 K256
    # tiles with 32/64-element tails. N64 and N128 fit different raw depths.
    _check_numerical(
        expected_world_size,
        hidden=7200,
        intermediate=2112,
        num_tokens=257,
        normalize_fc1=True,
        knobs={
            "mma_tiler_mnk": (256, tile_n, 256),
            "cluster_shape_mnk": (2, 1, 1),
            "use_2cta_instrs": True,
            "group_hint": 512,
            "flag_batch": 4,
            "epi_flag_batch": (2, 4),
            "token_back_mode": token_back_mode,
            "load_balance_mode": "atomic_counter",
        },
    )


@pytest.mark.arch_blackwell
def test_w4a16_mega_activation_fallback():
    # At N64, this hidden size requires the two-stage activation fallback.
    # The 72 FC1 K tiles include a 96-value tail; FC2 has a 64-value tail.
    _check_numerical(
        1,
        hidden=18272,
        intermediate=64,
        num_tokens=257,
        normalize_fc1=True,
        knobs={
            "mma_tiler_mnk": (256, 64, 256),
            "cluster_shape_mnk": (2, 1, 1),
            "use_2cta_instrs": True,
            "group_hint": 512,
            "flag_batch": 4,
            "epi_flag_batch": (2, 4),
            "token_back_mode": "epi_warps",
            "load_balance_mode": "atomic_counter",
        },
    )
