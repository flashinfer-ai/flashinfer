"""Multi-rank integration tests for the Cake SM120 NVFP4 x NVFP4 (W4A4) hybrid-transport MegaMoE.

Launch on 4 or 8 sm_120 GPUs of one host (NCCL 2.30.7 with the Device API):

    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_sm120_nvfp4_cake_mega.py \
        -m "gpu_4 and arch_sm120" -v

``CAKE_SM120_NVFP4_CASES`` selects ``shape:variant[:l<L>][:int8]`` cases (default
:data:`DEFAULT_CASES`, shapes from the W4A8 test) and ``CAKE_SM120_MEGAMOE_TABLE`` the transport
table.  The routed weights are random NVFP4 (E2M1 codes, E4M3 block-16 scales including
subnormal ones), the shared expert FP8 e4m3, the activations Gaussian; every case runs with
random per-expert ``fc1_alpha`` / ``fc2_alpha`` / ``fc1_norm_const`` (one test also overrides
them per call).  The pure-torch oracle follows the kernel's numerical boundary:

* the input is quantized to NVFP4 (``scale = e4m3(amax / 6)``, ``codes = e2m1(x / scale)``);
* ``fc1 = (x_q @ w13_q) * fc1_alpha[e]`` accumulates in FP32 and rounds to BF16;
* the clamp / gated activation / route weight of the W4A8 test;
* the intermediate requantizes to NVFP4 with ``scale = e4m3((amax / 6) * fc1_norm_const[e])``;
* ``fc2 = (h_q @ w2_q) * fc2_alpha[e]`` rounds to BF16 per route; the combine sums in FP32;
* the in-kernel shared expert sees the MXFP8 requantization (UE8M0 = ceil-pow2(amax / 448)) of the
  NVFP4 input and runs the W4A8 shared path.
"""

from __future__ import annotations

import os

import pytest
import torch

from .test_moe_ep_sm120_w4a8_cake_mega import (
    DEFAULT_SWIGLU_LIMIT,
    SHAPES,
    Case,
    _bucketed,
    all_gather_cat,
    check_close,
    decode_fp8,
    expert_partial,
    init_distributed,
    int8_return_roundtrip,
    make_inputs,
    requantize_k32,
    swiglu_kernel_boundary,
    table_spec,
)

DEFAULT_CASES = (
    "v4flash:noshared,v4flash:fp8shared,dsv3:fp8shared,qwen3_235b:noshared,"
    "qwen3_30b:fp8shared:l0,v41flash:noshared,v41flash:fp8shared:int8,"
    "minimax_m3:noshared:l7,situ_nolb:noshared:l0"
)
# E4M3 weight scales 2^-8 .. ~2^-5.1 (bytes 0x02 .. 0x0f: subnormal and normal) keep W1 O(1).
WEIGHT_SCALE_BYTES = (0x02, 0x10)
SHARED_EXPONENT_RANGE = (116, 119)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _cases() -> list[Case]:
    raw = os.environ.get("CAKE_SM120_NVFP4_CASES", DEFAULT_CASES)
    cases = []
    for item in raw.split(","):
        if item.strip():
            shape, variant, *options = item.strip().split(":")
            situ = dict(SHAPES[shape].activation_fields).get("activation") == "situ"
            swiglu_limit = 0 if situ else DEFAULT_SWIGLU_LIMIT
            return_codec = "bf16"
            for option in options:
                if option == "int8":
                    return_codec = option
                else:
                    swiglu_limit = int(option.removeprefix("l"))
            cases.append(Case(SHAPES[shape], variant, swiglu_limit, return_codec))
    return cases


# --------------------------------------------------------------------------- NVFP4 helpers


def decode_nvfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Low-nibble-first E2M1 ``[..., K // 2]`` + E4M3 ``[..., K // 16]`` -> fp32 ``[..., K]``."""

    table = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    raw = packed.view(torch.uint8)
    codes = (
        torch.stack((raw & 15, raw >> 4), dim=-1)
        .reshape(*raw.shape[:-1], raw.shape[-1] * 2)
        .long()
    )
    values = table[codes & 7] * (1.0 - 2.0 * (codes >> 3).float())
    return values * scales.view(torch.float8_e4m3fn).float().repeat_interleave(
        16, dim=-1
    )


def nvfp4_roundtrip(
    values: torch.Tensor, norm: torch.Tensor | float = 1.0
) -> torch.Tensor:
    """NVFP4 quantize (the kernel / staging rule) and decode: code * scale (= value * norm)."""

    from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake import (
        nvfp4_quantize_torch,
    )

    packed, scales = nvfp4_quantize_torch(values, norm)
    return decode_nvfp4(packed, scales)


def _expert_planes(expert: int, seed: int, shape):
    generator = torch.Generator(device="cuda").manual_seed(seed * 4099 + expert)

    def plane(rows: int, k: int):
        codes = torch.randint(
            0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=generator
        )
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        scales = torch.randint(
            *WEIGHT_SCALE_BYTES,
            (rows, k // 16),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        return packed, scales

    w13, w13_scale = plane(2 * shape.intermediate, shape.hidden)
    w2, w2_scale = plane(shape.hidden, shape.intermediate)
    return w13, w13_scale, w2, w2_scale


def expert_scales(
    expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert (fc1_alpha, fc2_alpha, fc1_norm_const) of global expert ids (deterministic)."""

    e = expert.float()
    return (
        0.625 + 0.125 * torch.remainder(e, 6),
        1.5 - 0.25 * torch.remainder(e, 3),
        1.0 + torch.remainder(e, 4),
    )


def local_weights(rank: int, world_size: int, seed: int, *, shared: bool, shape):
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        CakePrequantizedMoEWeightsWithSharedExpert,
    )

    local = shape.experts // world_size
    parts = [
        _expert_planes(e, seed, shape) for e in range(rank * local, (rank + 1) * local)
    ]
    w13, w13_scale, w2, w2_scale = (
        torch.stack([part[i] for part in parts]) for i in range(4)
    )
    routed = (
        w13,
        w2,
        w13_scale.view(torch.float8_e4m3fn),
        w2_scale.view(torch.float8_e4m3fn),
    )
    if not shared:
        return MoEWeightPack(*routed)
    g = torch.Generator(device="cuda").manual_seed(seed + 7919)

    def fp8_plane(rows: int, k: int):
        codes = (torch.randn(rows, k, device="cuda", generator=g) * 4.0).to(
            torch.float8_e4m3fn
        )
        exps = torch.randint(
            *SHARED_EXPONENT_RANGE,
            (rows, k // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=g,
        )
        return codes, exps.view(torch.float8_e8m0fnu)

    s_w13, s_w13_scale = fp8_plane(2 * shape.intermediate, shape.hidden)
    s_w2, s_w2_scale = fp8_plane(shape.hidden, shape.intermediate)
    return CakePrequantizedMoEWeightsWithSharedExpert(
        *routed,
        shared_w13=s_w13,
        shared_w2=s_w2,
        shared_w13_scale=s_w13_scale,
        shared_w2_scale=s_w2_scale,
    )


class Nvfp4Oracle:
    """Dense global weights and per-expert scales; the kernel's NVFP4 boundary."""

    def __init__(self, weights, shared: bool, case: Case, shape) -> None:
        self.case, self.shape = case, shape
        self.w13 = all_gather_cat(weights.w13)
        self.w2 = all_gather_cat(weights.w2)
        self.w13_scale = all_gather_cat(weights.w13_scale)
        self.w2_scale = all_gather_cat(weights.w2_scale)
        self.shared = (
            (
                decode_fp8(weights.shared_w13, weights.shared_w13_scale),
                decode_fp8(weights.shared_w2, weights.shared_w2_scale),
            )
            if shared
            else None
        )
        self.alpha1, self.alpha2, self.norm = expert_scales(
            torch.arange(shape.experts, device="cuda")
        )

    def __call__(self, tensors, scales=None) -> torch.Tensor:
        alpha1, alpha2, norm = (
            scales if scales is not None else (self.alpha1, self.alpha2, self.norm)
        )
        x = nvfp4_roundtrip(tensors.hidden_states.float())
        rows, hidden = x.shape
        top_k = tensors.topk_ids.shape[1]
        partials = torch.zeros(
            rows, top_k, hidden, dtype=torch.float32, device=x.device
        )
        for expert in torch.unique(tensors.topk_ids[tensors.topk_ids >= 0]).tolist():
            row, slot = torch.where(tensors.topk_ids == expert)
            w13 = decode_nvfp4(self.w13[expert], self.w13_scale[expert])
            w2 = decode_nvfp4(self.w2[expert], self.w2_scale[expert])
            fc1 = ((x[row] @ w13.t()) * alpha1[expert]).to(torch.bfloat16)
            h = swiglu_kernel_boundary(
                fc1,
                tensors.topk_weights[row, slot],
                self.case.swiglu_limit,
                self.shape.activation_fields,
            )
            hq = nvfp4_roundtrip(h, norm[expert])
            partial = ((hq @ w2.t()) * alpha2[expert]).to(torch.bfloat16)
            partials[row, slot] = (
                int8_return_roundtrip(partial)
                if self.case.return_codec == "int8"
                else partial.float()
            )
        out = (
            partials.sum(dim=1)
            if top_k == 1
            else sum(partials[:, s] for s in range(top_k))
        )
        if self.shared is not None and rows:
            ones = torch.ones(rows, dtype=torch.float32, device=x.device)
            out = (
                out
                + expert_partial(
                    requantize_k32(x),
                    self.shared[0],
                    self.shared[1],
                    ones,
                    self.case.swiglu_limit,
                    self.shape.activation_fields,
                ).float()
            )
        return out.to(torch.bfloat16)


def make_layer(rank, world_size, weights, *, case: Case, shape, local_scales):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig,
    )

    alpha1, alpha2, norm = local_scales
    return MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size, rank=rank, device=torch.cuda.current_device()
        ),
        fleet_params=FleetParams(
            num_experts=shape.experts,
            max_tokens_per_rank=4096,
            token_hidden_size=shape.hidden,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(
                intermediate_size=shape.intermediate,
                top_k=shape.top_k,
                gate_up_clamp=case.swiglu_limit or None,
                **dict(shape.activation_fields),
                combine_dtype=case.return_codec,
                shared_expert="fp8_in_kernel" if case.variant == "fp8shared" else None,
                transport_table=table_spec(),
                fc1_alpha=alpha1,
                fc2_alpha=alpha2,
                fc1_norm_const=norm,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


@pytest.fixture(scope="module", params=_cases(), ids=lambda case: case.name)
def ep_layer(request):
    import torch.distributed as dist

    rank, world_size = init_distributed()
    case = request.param
    shape = case.shape.for_world(world_size)
    if shape.experts % world_size or shape.experts // world_size > 128:
        pytest.skip(f"{shape.name}: {shape.experts} experts on {world_size} ranks")
    if shape.hidden % 1024 or (
        case.variant == "fp8shared" and shape.intermediate % 256
    ):
        pytest.skip(f"{shape.name}: outside the NVFP4 region")
    shared = case.variant == "fp8shared"
    weights = local_weights(rank, world_size, seed=20261031, shared=shared, shape=shape)
    oracle = Nvfp4Oracle(weights, shared, case, shape)
    local = shape.experts // world_size
    sl = slice(rank * local, (rank + 1) * local)
    layer = make_layer(
        rank,
        world_size,
        weights,
        case=case,
        shape=shape,
        local_scales=(
            oracle.alpha1[sl].contiguous(),
            oracle.alpha2[sl].contiguous(),
            oracle.norm[sl].contiguous(),
        ),
    )
    layer.warmup(make_inputs(256, seed=1, shape=shape))
    torch.cuda.synchronize()
    yield rank, world_size, layer, oracle, shape
    layer.destroy()
    dist.barrier()


def _launch(layer) -> dict:
    return layer._workspace.last_launch


def _protocol_clean(layer) -> None:
    assert layer._kernel.protocol_error(layer._workspace) == 0


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_small_rows_use_m64_and_match_oracle(ep_layer) -> None:
    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(512, seed=100 + rank, shape=shape)
    out = layer(tensors)
    torch.cuda.synchronize()
    assert _launch(layer)["instance"].startswith("m64")
    check_close(out, oracle(tensors))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_large_rows_use_m128_and_match_oracle(ep_layer) -> None:
    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(2048, seed=150 + rank, shape=shape)
    out = layer(tensors)
    torch.cuda.synchronize()
    assert _launch(layer)["instance"].startswith("m128")
    check_close(out, oracle(tensors))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_heterogeneous_rows_including_empty_rank(ep_layer) -> None:
    rank, world_size, layer, oracle, shape = ep_layer
    pattern = (17, 1024, 0, 2048, 129, 640, 1, 777)
    rows = pattern[rank % len(pattern)]
    tensors = make_inputs(rows, seed=200 + rank, shape=shape)
    out = layer(tensors)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (rows, shape.hidden)
    check_close(out, oracle(tensors))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_schedule_switching_is_bitwise_stable(ep_layer) -> None:
    rank, world_size, layer, oracle, shape = ep_layer
    small = make_inputs(300, seed=500 + rank, shape=shape)
    large = make_inputs(1500, seed=550 + rank, shape=shape)
    first_small, first_large = layer(small).clone(), layer(large).clone()
    again_small, again_large = layer(small).clone(), layer(large).clone()
    torch.cuda.synchronize()
    assert torch.equal(first_small, again_small)
    assert torch.equal(first_large, again_large)
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_per_call_scales_override_config(ep_layer) -> None:
    """MoEEpTensors.fc1_alpha / fc2_alpha / fc1_norm_const replace the config's for one call."""

    rank, world_size, layer, oracle, shape = ep_layer
    local = shape.experts // world_size
    e = torch.arange(shape.experts, device="cuda").float()
    scales = (
        0.5 + 0.25 * torch.remainder(e, 3),
        2.0 - 0.5 * torch.remainder(e, 2),
        2.0 + torch.remainder(e, 3),
    )
    tensors = make_inputs(700, seed=650 + rank, shape=shape)
    sl = slice(rank * local, (rank + 1) * local)
    tensors.fc1_alpha, tensors.fc2_alpha, tensors.fc1_norm_const = (
        s[sl].contiguous() for s in scales
    )
    out = layer(tensors)
    torch.cuda.synchronize()
    check_close(out, oracle(tensors, scales))
    # the next call without overrides is back on the config's scales
    plain = make_inputs(700, seed=650 + rank, shape=shape)
    out2 = layer(plain)
    torch.cuda.synchronize()
    check_close(out2, oracle(plain))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_cuda_graph_replays_per_call_scales(ep_layer) -> None:
    """A bucketed call captured into a CUDA graph with per-call fc1_alpha / fc2_alpha /
    fc1_norm_const tensors (the per-expert scale tail is rewritten on the stream inside the
    graph): replays with new inputs and new scale values in the same static tensors return the
    eager bits, and a captured call without overrides replays with the config's scales."""

    import torch.distributed as dist

    rank, world_size, layer, oracle, shape = ep_layer
    local = shape.experts // world_size
    sl = slice(rank * local, (rank + 1) * local)
    e = torch.arange(shape.experts, device="cuda").float()

    def scales(k: int):
        return (
            0.5 + 0.25 * torch.remainder(e + k, 3),
            2.0 - 0.5 * torch.remainder(e + k, 2),
            2.0 + torch.remainder(e + k, 3),
        )

    bucket, rows = 512, 512 - 13 * rank
    static = make_inputs(rows, seed=1900 + rank, shape=shape)
    static.fc1_alpha, static.fc2_alpha, static.fc1_norm_const = (
        s[sl].contiguous() for s in scales(0)
    )
    plain = make_inputs(rows, seed=1950 + rank, shape=shape)
    _bucketed(layer, static, bucket)
    _bucketed(layer, plain, bucket)
    torch.cuda.synchronize()
    dist.barrier()
    graph, graph_plain = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _bucketed(layer, static, bucket)
    with torch.cuda.graph(graph_plain):
        captured_plain = _bucketed(layer, plain, bucket)
    torch.cuda.synchronize()
    dist.barrier()
    for k in (1, 2):
        fresh = make_inputs(rows, seed=2000 + 10 * k + rank, shape=shape)
        for name in ("hidden_states", "topk_ids", "topk_weights"):
            getattr(static, name).copy_(getattr(fresh, name))
        full = scales(k)
        for name, value in zip(
            ("fc1_alpha", "fc2_alpha", "fc1_norm_const"), full, strict=True
        ):
            getattr(static, name).copy_(value[sl])
        graph.replay()
        replayed = captured.clone()
        graph_plain.replay()
        replayed_plain = captured_plain.clone()
        torch.cuda.synchronize()
        eager = _bucketed(layer, static, bucket).clone()
        eager_plain = _bucketed(layer, plain, bucket).clone()
        torch.cuda.synchronize()
        assert torch.equal(replayed, eager)
        assert torch.equal(replayed_plain, eager_plain)
        check_close(replayed, oracle(static, full))
        check_close(replayed_plain, oracle(plain))
    del graph, graph_plain
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_prequantized_input_matches_quantize_input(ep_layer) -> None:
    """quantize_input=False with the staging quantizer's planes gives the identical output."""

    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake import (
        nvfp4_quantize_torch,
    )

    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(400, seed=700 + rank, shape=shape)
    out = layer(tensors).clone()
    packed, sf = nvfp4_quantize_torch(tensors.hidden_states)
    pre = MoEEpTensors(
        hidden_states=packed,
        topk_ids=tensors.topk_ids,
        topk_weights=tensors.topk_weights,
        scales=sf.view(torch.float8_e4m3fn),
    )
    backend = layer._kernel
    backend.validate_forward(pre, layer._fleet_params, quantize_input=False)
    backend.set_compile_tokens_per_rank(layer._workspace, None)
    backend.stage_inputs(pre, layer._workspace, quantize_input=False)
    out_pre = backend.compute(layer._workspace, layer._transformed, output=None).clone()
    torch.cuda.synchronize()
    assert torch.equal(out, out_pre)
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_exact_zero_outputs_stay_in_codec_capacity(ep_layer) -> None:
    """Exact zeros inside a BF16 result block (NVFP4 makes them common: a low-weight route's intermediate is mostly
    zero) must not push the block to the bounded raw-overflow area of the BF16 return codec.  W2 output channels
    0, 21 and 42 of every 64-column block are zeroed, so every returned block mixes exact zeros with O(1) values;
    single-row and small chunks must stay within capacity and match the oracle on repeated calls.  (Kept last: a
    protocol error poisons the shared workspace.)"""

    import torch.distributed as dist

    rank, world_size, layer, oracle, shape = ep_layer
    case = oracle.case
    weights = local_weights(
        rank, world_size, seed=20261031, shared=case.variant == "fp8shared", shape=shape
    )
    zero_cols = torch.cat(
        [torch.arange(b, b + 1, device="cuda") for b in range(0, shape.hidden, 64)]
    )
    zero_cols = torch.cat((zero_cols, zero_cols + 21, zero_cols + 42))
    weights.w2[:, zero_cols, :] = 0
    zoracle = Nvfp4Oracle(weights, case.variant == "fp8shared", case, shape)
    local = shape.experts // world_size
    sl = slice(rank * local, (rank + 1) * local)
    zlayer = make_layer(
        rank,
        world_size,
        weights,
        case=case,
        shape=shape,
        local_scales=(
            zoracle.alpha1[sl].contiguous(),
            zoracle.alpha2[sl].contiguous(),
            zoracle.norm[sl].contiguous(),
        ),
    )
    try:
        for step, rows in enumerate((1, 1, 3, 1, 64, 1)):
            tensors = make_inputs(rows, seed=900 + 10 * step + rank, shape=shape)
            out = zlayer(tensors)
            torch.cuda.synchronize()
            if case.variant == "noshared" and rows:
                assert (out[:, zero_cols] == 0).all()
            check_close(out, zoracle(tensors))
            assert zlayer._kernel.protocol_error(zlayer._workspace) == 0
    finally:
        zlayer.destroy()
        dist.barrier()
