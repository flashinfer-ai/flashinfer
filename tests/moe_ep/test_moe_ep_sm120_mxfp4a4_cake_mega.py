"""Multi-rank integration tests for the Cake SM120 MXFP4 x MXFP4 (W4A4) hybrid-transport MegaMoE.

Launch on 4 or 8 sm_120 GPUs of one host (NCCL 2.30.7 with the Device API):

    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_sm120_mxfp4a4_cake_mega.py \
        -m "gpu_4 and arch_sm120" -v

``CAKE_SM120_MXFP4A4_CASES`` selects ``shape:variant[:l<L>][:int8]`` cases (default
:data:`DEFAULT_CASES`, shapes from the W4A8 test) and ``CAKE_SM120_MEGAMOE_TABLE`` the transport
table.  The weights are those of the W4A8 test (random MXFP4 routed experts, FP8 e4m3 shared
expert: the two backends take the same weight pack), the activations Gaussian.  The pure-torch
oracle follows the kernel's numerical boundary:

* the input is quantized to MXFP4 (``mxfp4_quantize_torch``: UE8M0 ``2^ceil(log2(amax * fl32(1 / 6)))``
  per 32 values, ``codes = e2m1(x / scale)``);
* ``fc1 = x_q @ w13_q`` accumulates in FP32 and rounds to BF16;
* the clamp / gated activation / route weight of the W4A8 test;
* the intermediate requantizes to MXFP4 per 32 values (``scale = 2^ceil(log2(amax * (1 / 6)))``,
  RNE e2m1, saturating);
* ``fc2 = h_q @ w2_q`` rounds to BF16 per route; the combine sums in FP32;
* the in-kernel shared expert sees the exact E4M3 image of the MXFP4 input and runs the W4A8
  shared path (MXFP8 intermediate).
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
    decode_fp4,
    decode_fp8,
    expert_partial,
    init_distributed,
    int8_return_roundtrip,
    local_weights,
    make_inputs,
    swiglu_kernel_boundary,
    table_spec,
)

DEFAULT_CASES = (
    "v4flash:noshared,v4flash:fp8shared,dsv3:fp8shared,qwen3_235b:noshared,"
    "qwen3_30b:fp8shared:l0,v41flash:noshared,v41flash:fp8shared:int8,"
    "minimax_m3:noshared:l7,situ_nolb:noshared:l0"
)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_GT = (0.25, 1.25, 2.5, 5.0)
_E2M1_GE = (0.75, 1.75, 3.5)


def _cases() -> list[Case]:
    raw = os.environ.get("CAKE_SM120_MXFP4A4_CASES", DEFAULT_CASES)
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


# --------------------------------------------------------------------------- MXFP4 helpers


def mxfp4_input_roundtrip(values: torch.Tensor) -> torch.Tensor:
    """The staging quantizer, decoded (fp32)."""

    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp4_bf16_cake import (
        mxfp4_dequantize_torch,
        mxfp4_quantize_torch,
    )

    return mxfp4_dequantize_torch(*mxfp4_quantize_torch(values))


def mxfp4_requant_roundtrip(values: torch.Tensor) -> torch.Tensor:
    """The kernel's in-kernel MXFP4 requantization of the intermediate, decoded (fp32):
    UE8M0 = next power of two ``>= amax * (1 / 6)``, codes RNE e2m1 of ``v * 2^-e`` (saturating)."""

    rows, cols = values.shape
    grouped = values.float().reshape(rows, cols // 32, 32)
    bits = (grouped.abs().amax(dim=2) * (1.0 / 6.0)).contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)).clamp(
        max=254
    )
    inverse = ((254 - exponent) << 23).view(torch.float32)
    scaled = grouped * inverse.unsqueeze(2)
    magnitude = scaled.abs()
    code = torch.zeros(scaled.shape, dtype=torch.int64, device=scaled.device)
    for threshold in _E2M1_GT:
        code += (magnitude > threshold).to(torch.int64)
    for threshold in _E2M1_GE:
        code += (magnitude >= threshold).to(torch.int64)
    table = torch.tensor(E2M1, dtype=torch.float32, device=values.device)
    decoded = table[code] * torch.where(torch.signbit(scaled), -1.0, 1.0)
    scale = (exponent << 23).view(torch.float32)
    return (decoded * scale.unsqueeze(2)).reshape(rows, cols)


class Mxfp4Oracle:
    """Dense global weights; the kernel's MXFP4 activation boundary."""

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

    def __call__(self, tensors) -> torch.Tensor:
        x = mxfp4_input_roundtrip(tensors.hidden_states.float())
        rows, hidden = x.shape
        top_k = tensors.topk_ids.shape[1]
        partials = torch.zeros(
            rows, top_k, hidden, dtype=torch.float32, device=x.device
        )
        for expert in torch.unique(tensors.topk_ids[tensors.topk_ids >= 0]).tolist():
            row, slot = torch.where(tensors.topk_ids == expert)
            w13 = decode_fp4(self.w13[expert], self.w13_scale[expert])
            w2 = decode_fp4(self.w2[expert], self.w2_scale[expert])
            fc1 = (x[row] @ w13.t()).to(torch.bfloat16)
            h = swiglu_kernel_boundary(
                fc1,
                tensors.topk_weights[row, slot],
                self.case.swiglu_limit,
                self.shape.activation_fields,
            )
            partial = (mxfp4_requant_roundtrip(h) @ w2.t()).to(torch.bfloat16)
            partials[row, slot] = (
                int8_return_roundtrip(partial)
                if self.case.return_codec == "int8"
                else partial.float()
            )
        out = torch.zeros(rows, hidden, dtype=torch.float32, device=x.device)
        for s in range(top_k):
            out += partials[:, s]
        if self.shared is not None and rows:
            ones = torch.ones(rows, dtype=torch.float32, device=x.device)
            out += expert_partial(
                x,
                self.shared[0],
                self.shared[1],
                ones,
                self.case.swiglu_limit,
                self.shape.activation_fields,
            ).float()
        return out.to(torch.bfloat16)


def make_layer(rank, world_size, weights, *, case: Case, shape):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig,
    )

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
            megakernel=Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig(
                intermediate_size=shape.intermediate,
                top_k=shape.top_k,
                swiglu_limit=case.swiglu_limit,
                **dict(shape.activation_fields),
                return_codec=case.return_codec,
                shared_expert="fp8_in_kernel" if case.variant == "fp8shared" else None,
                transport_table=table_spec(),
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


def _weights(rank, world_size, case: Case, shape):
    return local_weights(
        rank,
        world_size,
        seed=20261105,
        shared=case.variant == "fp8shared",
        shape=shape,
        w13_exponent_shift=case.w13_exponent_shift,
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
        pytest.skip(f"{shape.name}: outside the FP4-activation region")
    weights = _weights(rank, world_size, case, shape)
    oracle = Mxfp4Oracle(weights, case.variant == "fp8shared", case, shape)
    layer = make_layer(rank, world_size, weights, case=case, shape=shape)
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
def test_fully_masked_rank_returns_routed_zeros(ep_layer) -> None:
    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(96, seed=300 + rank, shape=shape, masked=rank == 0)
    out = layer(tensors)
    torch.cuda.synchronize()
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
def test_cuda_graph_replays_new_inputs(ep_layer) -> None:
    """A bucketed call captured into a CUDA graph (the in-place fused quantizer runs inside the
    graph): replays with new inputs in the same static tensors return the eager bits."""

    import torch.distributed as dist

    rank, world_size, layer, oracle, shape = ep_layer
    bucket, rows = 512, 512 - 13 * rank
    static = make_inputs(rows, seed=1900 + rank, shape=shape)
    _bucketed(layer, static, bucket)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _bucketed(layer, static, bucket)
    torch.cuda.synchronize()
    dist.barrier()
    for k in (1, 2):
        fresh = make_inputs(rows, seed=2000 + 10 * k + rank, shape=shape)
        for name in ("hidden_states", "topk_ids", "topk_weights"):
            getattr(static, name).copy_(getattr(fresh, name))
        graph.replay()
        replayed = captured.clone()
        torch.cuda.synchronize()
        eager = _bucketed(layer, static, bucket).clone()
        torch.cuda.synchronize()
        assert torch.equal(replayed, eager)
        check_close(replayed, oracle(static))
    del graph
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_prequantized_input_matches_quantize_input(ep_layer) -> None:
    """quantize_input=False with the staging quantizer's planes gives the identical output."""

    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp4_bf16_cake import (
        mxfp4_quantize_torch,
    )

    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(400, seed=700 + rank, shape=shape)
    out = layer(tensors).clone()
    packed, sf = mxfp4_quantize_torch(tensors.hidden_states)
    pre = MoEEpTensors(
        hidden_states=packed.view(torch.float4_e2m1fn_x2),
        topk_ids=tensors.topk_ids,
        topk_weights=tensors.topk_weights,
        scales=sf.view(torch.float8_e8m0fnu),
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
    """Exact zeros inside a BF16 result block (FP4 intermediates make them common) must not push the
    block to the bounded raw-overflow area of the BF16 return codec: W2 output channels 0, 21 and 42 of
    every 64-column block are zeroed; single-row and small chunks must stay within capacity and match
    the oracle on repeated calls.  (Kept last: a protocol error poisons the shared workspace.)"""

    import torch.distributed as dist

    rank, world_size, layer, oracle, shape = ep_layer
    case = oracle.case
    weights = _weights(rank, world_size, case, shape)
    zero_cols = torch.arange(0, shape.hidden, 64, device="cuda")
    zero_cols = torch.cat((zero_cols, zero_cols + 21, zero_cols + 42))
    weights.w2.view(torch.uint8)[:, zero_cols, :] = 0
    zoracle = Mxfp4Oracle(weights, case.variant == "fp8shared", case, shape)
    zlayer = make_layer(rank, world_size, weights, case=case, shape=shape)
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
