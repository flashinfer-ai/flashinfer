"""Caller-owned workspace for the trtllm-gen fused-MoE APIs.

``trtllm_fused_moe_workspace_size`` sizes one buffer that a launch carves all of
its internal scratch from (``workspace_buffer=``). These tests check that the
buffered launch is bitwise identical to the per-call-allocating one, that one
buffer sized for the largest token count serves smaller ones and several layers
in turn, and that misuse is rejected.
"""

from typing import Callable

import pytest
import torch

from flashinfer import ActivationType
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import (
    Fp8QuantizationType,
    WeightLayout,
    convert_to_block_layout,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_routed_moe,
    trtllm_fused_moe_workspace_size,
)
from flashinfer.tllm_enums import RoutingInputMode, RoutingMethodType, SfLayout
from flashinfer.utils import get_compute_capability

QUANT_MODES = [
    "bf16",
    "fp8_per_tensor",
    "fp8_per_channel",
    "fp8_block_scale",
    "mxfp8",
    "nvfp4",
    "mxfp4_mxfp8",
    "mxfp4_bf16",
    "mxint4",
]

NUM_EXPERTS = 16
TOP_K = 4
HIDDEN = 1024
INTER = 1024
MAX_TOKENS = 256
TOKEN_COUNTS = [1, 37, MAX_TOKENS]

# One launch: (num_tokens, workspace_buffer, do_finalize) -> output tensor.
Launch = Callable[..., torch.Tensor]


@pytest.fixture(autouse=True)
def _require_sm100_family():
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm-gen fused MoE is only exercised on SM100 and SM103.")


def _workspace(
    quant_mode: str, max_num_tokens: int = MAX_TOKENS, **kwargs
) -> torch.Tensor:
    nbytes = trtllm_fused_moe_workspace_size(
        max_num_tokens,
        HIDDEN,
        INTER,
        NUM_EXPERTS,
        TOP_K,
        quant_mode=quant_mode,
        **kwargs,
    )
    assert nbytes > 0
    return torch.empty(nbytes, dtype=torch.uint8, device="cuda")


def _routing(num_tokens: int, generator: torch.Generator, top_k: int = TOP_K):
    logits = torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    weights, ids = torch.topk(logits.float().softmax(-1), top_k, dim=-1)
    return logits, (
        ids.to(torch.int32),
        (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16),
    )


def _bf16_weights(generator: torch.Generator):
    def prepare(w: torch.Tensor) -> torch.Tensor:
        blocks = [convert_to_block_layout(e.view(torch.uint8), 128) for e in w]
        return torch.stack(blocks).view(torch.bfloat16)

    w1 = torch.randn(
        NUM_EXPERTS, 2 * INTER, HIDDEN, device="cuda", generator=generator
    ).to(torch.bfloat16)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTER, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    return prepare(w1 * 0.05), prepare(w2 * 0.05)


def _make_bf16_case(
    seed: int,
    use_logits: bool,
    *,
    packed: bool = False,
    routing_method: RoutingMethodType = RoutingMethodType.Renormalize,
) -> Launch:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    w1, w2 = _bf16_weights(generator)
    hidden = (
        torch.randn(MAX_TOKENS, HIDDEN, device="cuda", generator=generator) * 0.1
    ).to(torch.bfloat16)
    top_k = 1 if routing_method == RoutingMethodType.Llama4 else TOP_K
    logits, routed = _routing(MAX_TOKENS, generator, top_k)
    packed_ids = (routed[0] << 16) | (
        routed[1].view(torch.int16).to(torch.int32) & 0xFFFF
    )
    grouped = routing_method == RoutingMethodType.DeepSeekV3
    bias = (
        torch.rand(NUM_EXPERTS, device="cuda", generator=generator)
        if routing_method in (RoutingMethodType.DeepSeekV3, RoutingMethodType.MiniMax2)
        else None
    )

    def launch(n: int, workspace_buffer=None, do_finalize: bool = True):
        common = dict(
            hidden_states=hidden[:n],
            gemm1_weights=w1,
            gemm2_weights=w2,
            num_experts=NUM_EXPERTS,
            top_k=top_k,
            n_group=4 if grouped else None,
            topk_group=2 if grouped else None,
            intermediate_size=INTER,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            routed_scaling_factor=1.0 if bias is not None else None,
            routing_method_type=int(routing_method),
            weight_layout=WeightLayout.BlockMajorK,
            do_finalize=do_finalize,
            # Tune only the buckets a MAX_TOKENS buffer can hold.
            tune_max_num_tokens=MAX_TOKENS,
            workspace_buffer=workspace_buffer,
        )
        if use_logits:
            out = trtllm_bf16_moe(
                routing_logits=logits[:n], routing_bias=bias, **common
            )
        else:
            out = trtllm_bf16_routed_moe(
                topk_ids=packed_ids[:n] if packed else (routed[0][:n], routed[1][:n]),
                **common,
            )
        return out if isinstance(out, torch.Tensor) else out[0]

    return launch


def _make_fp8_block_scale_case(seed: int) -> Launch:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    fp8 = torch.float8_e4m3fn

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(fp8)

    hidden = randn(MAX_TOKENS, HIDDEN, scale=0.1)
    hidden_scale = torch.ones(HIDDEN // 128, MAX_TOKENS, device="cuda")
    w1 = randn(NUM_EXPERTS, 2 * INTER, HIDDEN, scale=0.1)
    w2 = randn(NUM_EXPERTS, HIDDEN, INTER, scale=0.1)
    w1_scale = torch.ones(NUM_EXPERTS, 2 * INTER // 128, HIDDEN // 128, device="cuda")
    w2_scale = torch.ones(NUM_EXPERTS, HIDDEN // 128, INTER // 128, device="cuda")
    _, routed = _routing(MAX_TOKENS, generator)

    def launch(n: int, workspace_buffer=None, do_finalize: bool = True):
        out = trtllm_fp8_block_scale_routed_moe(
            topk_ids=(routed[0][:n], routed[1][:n]),
            routing_bias=None,
            hidden_states=hidden[:n],
            # [hidden // 128, num_tokens]: slice the token dimension contiguously.
            hidden_states_scale=hidden_scale[:, :n].contiguous(),
            gemm1_weights=w1,
            gemm1_weights_scale=w1_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            n_group=None,
            topk_group=None,
            intermediate_size=INTER,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=do_finalize,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
            workspace_buffer=workspace_buffer,
        )
        return out if isinstance(out, torch.Tensor) else out[0]

    return launch


def _make_nvfp4_case(seed: int) -> Launch:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    fp8 = torch.float8_e4m3fn

    def packed_fp4(*shape):
        # Small E2M1 magnitudes (0, 0.5, 1, 1.5) with random signs keep outputs finite.
        nibbles = torch.randint(0, 4, (*shape, 2), device="cuda", generator=generator)
        nibbles |= (
            torch.randint(0, 2, nibbles.shape, device="cuda", generator=generator) << 3
        )
        return (nibbles[..., 0] | (nibbles[..., 1] << 4)).to(torch.uint8)

    hidden = packed_fp4(MAX_TOKENS, HIDDEN // 2)
    hidden_scale = torch.ones(MAX_TOKENS, HIDDEN // 16, device="cuda").to(fp8)
    w1 = packed_fp4(NUM_EXPERTS, 2 * INTER, HIDDEN // 2)
    w2 = packed_fp4(NUM_EXPERTS, HIDDEN, INTER // 2)
    w1_scale = torch.ones(NUM_EXPERTS, 2 * INTER, HIDDEN // 16, device="cuda").to(fp8)
    w2_scale = torch.ones(NUM_EXPERTS, HIDDEN, INTER // 16, device="cuda").to(fp8)
    out_scale = torch.full((NUM_EXPERTS,), 0.05, device="cuda")
    _, routed = _routing(MAX_TOKENS, generator)

    def launch(n: int, workspace_buffer=None, do_finalize: bool = True):
        out = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(routed[0][:n], routed[1][:n]),
            routing_bias=None,
            hidden_states=hidden[:n],
            hidden_states_scale=hidden_scale[:n],
            gemm1_weights=w1,
            gemm1_weights_scale=w1_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale,
            gemm2_bias=None,
            output1_scale_scalar=out_scale,
            output1_scale_gate_scalar=out_scale,
            output2_scale_scalar=out_scale,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            n_group=None,
            topk_group=None,
            intermediate_size=INTER,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=do_finalize,
            activation_type=ActivationType.Swiglu.value,
            hidden_states_scale_layout=SfLayout.layout_linear,
            workspace_buffer=workspace_buffer,
        )
        return out if isinstance(out, torch.Tensor) else out[0]

    return launch


CASES = {
    # quant_mode, launch factory
    "bf16_logits": ("bf16", lambda seed: _make_bf16_case(seed, use_logits=True)),
    "bf16_routed": ("bf16", lambda seed: _make_bf16_case(seed, use_logits=False)),
    "fp8_block_scale": ("fp8_block_scale", _make_fp8_block_scale_case),
    "nvfp4": ("nvfp4", _make_nvfp4_case),
}


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.isfinite(expected).float().mean().item() > 0.99
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("quant_mode", QUANT_MODES)
@pytest.mark.parametrize("hidden", [HIDDEN])
@pytest.mark.parametrize("inter", [INTER])
@pytest.mark.parametrize("use_shuffled_weight", [False, True])
@pytest.mark.parametrize("num_experts", [64, 128, 256])
@pytest.mark.parametrize("top_k", [4, 8])
def test_workspace_size_is_positive_and_monotonic(
    quant_mode: str,
    hidden: int,
    inter: int,
    use_shuffled_weight: bool,
    num_experts: int,
    top_k: int,
):
    if quant_mode == "bf16" and not use_shuffled_weight:
        pytest.skip("BF16 requires shuffled BlockMajorK weights.")
    sizes = [
        trtllm_fused_moe_workspace_size(
            n,
            hidden,
            inter,
            num_experts,
            top_k,
            quant_mode=quant_mode,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=(
                WeightLayout.BlockMajorK
                if quant_mode == "bf16"
                else WeightLayout.MajorK
            ),
        )
        for n in (1, 16, 256, 4096)
    ]
    assert sizes[0] > 0
    assert sizes == sorted(sizes), sizes


def test_workspace_size_accepts_real_weights():
    """Passing the prepared weights gives the size the stand-ins give."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    w1, w2 = _bf16_weights(generator)  # BlockMajorK: 4-D, unlike the 3-D stand-ins
    fp8 = torch.float8_e4m3fn
    real = {
        "bf16": dict(gemm1_weights=w1, gemm2_weights=w2),
        "nvfp4": dict(
            gemm1_weights=torch.empty(
                NUM_EXPERTS, 2 * INTER, HIDDEN // 2, dtype=torch.uint8, device="cuda"
            ),
            gemm1_weights_scale=torch.empty(
                NUM_EXPERTS, 2 * INTER, HIDDEN // 16, dtype=fp8, device="cuda"
            ),
            gemm2_weights=torch.empty(
                NUM_EXPERTS, HIDDEN, INTER // 2, dtype=torch.uint8, device="cuda"
            ),
            gemm2_weights_scale=torch.empty(
                NUM_EXPERTS, HIDDEN, INTER // 16, dtype=fp8, device="cuda"
            ),
        ),
    }
    for quant_mode, weights in real.items():
        args = (MAX_TOKENS, HIDDEN, INTER, NUM_EXPERTS, TOP_K)
        assert trtllm_fused_moe_workspace_size(
            *args, quant_mode=quant_mode, **weights
        ) == trtllm_fused_moe_workspace_size(*args, quant_mode=quant_mode)


def _native_fp8_per_tensor_routed_query(weights_dtype: torch.dtype) -> int:
    """Size query through the exported routed FP8 entry point with Llama4 unpacked routing."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    fp8, num_tokens = torch.float8_e4m3fn, 64

    def empty(*shape, dtype=fp8):
        return torch.empty(*shape, dtype=dtype, device="cuda")

    scale = torch.ones(NUM_EXPERTS, device="cuda")
    result = get_trtllm_moe_sm100_module().moe_op.trtllm_fp8_per_tensor_scale_routed_moe(
        int(RoutingInputMode.UnpackedPrecomputed),
        torch.zeros(num_tokens, 1, dtype=torch.int32, device="cuda"),  # expert ids
        torch.ones(num_tokens, 1, dtype=weights_dtype, device="cuda"),  # weights
        None,  # routing_bias
        empty(num_tokens, HIDDEN),
        empty(NUM_EXPERTS, 2 * INTER, HIDDEN),
        scale, scale,
        empty(NUM_EXPERTS, HIDDEN, INTER),
        scale,
        empty(num_tokens, HIDDEN, dtype=torch.bfloat16),  # output
        NUM_EXPERTS, 1, None, None, INTER, 0, NUM_EXPERTS, None,
        True,  # use_routing_scales_on_input
        int(RoutingMethodType.Llama4),
        True, False, [-1, -1],  # do_finalize, enable_pdl, config_index
        ActivationType.Swiglu.value, True, None, [], [], False, None,
        None, True,  # workspace_buffer, query_workspace_size
    )  # fmt: skip
    return int(torch.from_dlpack(result[0]).item())


def test_native_size_query_skips_device_work():
    """Llama4 FP32 routing weights are cast to BF16 during launch; a size query must
    count the cast's destination without launching the cast into a null pointer."""
    fp32_bytes = _native_fp8_per_tensor_routed_query(torch.float32)
    torch.cuda.synchronize()  # an illegal address from the cast would surface here
    bf16_bytes = _native_fp8_per_tensor_routed_query(torch.bfloat16)
    # FP32 weights add the BF16 routing-scale tensor that the cast would fill.
    assert fp32_bytes > bf16_bytes > 0, (fp32_bytes, bf16_bytes)


def test_workspace_size_rejects_invalid_arguments():
    base = dict(
        max_num_tokens=8,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    )
    with pytest.raises(ValueError, match="quant_mode"):
        trtllm_fused_moe_workspace_size(**base, quant_mode="fp16")
    with pytest.raises(ValueError, match="max_num_tokens"):
        trtllm_fused_moe_workspace_size(
            **{**base, "max_num_tokens": 0}, quant_mode="bf16"
        )
    with pytest.raises(ValueError, match="local_num_experts"):
        trtllm_fused_moe_workspace_size(
            **base, quant_mode="bf16", local_num_experts=NUM_EXPERTS + 1
        )
    with pytest.raises(ValueError, match="use_per_token_scale"):
        trtllm_fused_moe_workspace_size(
            **base, quant_mode="bf16", use_per_token_scale=True
        )
    with pytest.raises(ValueError, match="num_fused_shared_experts"):
        trtllm_fused_moe_workspace_size(
            **base, quant_mode="bf16", num_fused_shared_experts=1
        )
    scale = torch.empty(1, device="cuda")
    with pytest.raises(ValueError, match="takes no weight scales"):
        trtllm_fused_moe_workspace_size(
            **base, quant_mode="bf16", gemm1_weights_scale=scale
        )
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        trtllm_fused_moe_workspace_size(
            **base, quant_mode="bf16", gemm1_weights=torch.empty(1)
        )


@pytest.mark.parametrize("case", list(CASES))
def test_workspace_size_explicit_tactic(case: str, monkeypatch):
    """Measure and execute actual configurations from each supported tile."""
    quant_mode, factory = CASES[case]
    launch = factory(seed=0)
    tactics = []
    selected = None

    def choose_one(self, name, runners, config, inputs, **kwargs):
        nonlocal tactics
        if not tactics:
            tactics = [tuple(t) for t in runners[0].get_valid_tactics(inputs, None)]
        assert tactics
        return runners[0], selected or tactics[0]

    monkeypatch.setattr(AutoTuner, "choose_one", choose_one)
    n = 32
    launch(n)  # Collect the real runner's valid configurations.
    # Cover distinct tiles and non-default configuration indices without running
    # every GEMM combination in the kernel catalog.
    by_tile = {}
    for tactic in tactics:
        by_tile.setdefault(tactic[0], []).append(tactic)
    largest = _workspace(quant_mode, max_num_tokens=n)
    sizes = []

    def allocated(workspace):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = launch(n, workspace_buffer=workspace)
        torch.cuda.synchronize()
        count = torch.cuda.memory_stats()["allocation.all.allocated"] - before
        return out, count

    for tile_tactics in by_tile.values():
        for selected in dict.fromkeys(
            ((tile_tactics[0][0], -1), tile_tactics[0], tile_tactics[-1])
        ):
            size = trtllm_fused_moe_workspace_size(
                n,
                HIDDEN,
                INTER,
                NUM_EXPERTS,
                TOP_K,
                quant_mode=quant_mode,
                tactic=selected,
            )
            assert 0 < size <= largest.numel()
            sizes.append(size)
            workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
            expected = launch(n)
            launch(n, workspace_buffer=workspace)  # Warm this configuration.
            actual, count = allocated(workspace)
            _, largest_count = allocated(largest)
            _assert_bitwise_equal(actual, expected)
            # An exact-sized buffer must not fall back to per-call scratch.
            assert count == largest_count
    assert min(sizes) < largest.numel()


@pytest.mark.parametrize("quant_mode", QUANT_MODES)
def test_workspace_size_explicit_tile_default_config(quant_mode):
    args = (32, HIDDEN, INTER, NUM_EXPERTS, TOP_K)
    maximum = trtllm_fused_moe_workspace_size(*args, quant_mode=quant_mode)
    tiles = [8, 16, 32, 64]
    if quant_mode != "mxfp4_bf16":
        tiles.append(128)
    if quant_mode in ("nvfp4", "mxfp4_mxfp8"):
        tiles.append(192)
    if quant_mode in ("mxfp8", "nvfp4", "mxfp4_mxfp8"):
        tiles.append(256)
    sizes = [
        trtllm_fused_moe_workspace_size(*args, quant_mode=quant_mode, tactic=(tile, -1))
        for tile in tiles
    ]
    # Unsupported tiles may have no valid config and return zero. The others
    # must preserve tile-dependent sizes, rather than all returning the maximum.
    valid_sizes = [size for size in sizes if size > 0]
    assert valid_sizes
    assert max(valid_sizes) == maximum
    assert min(valid_sizes) < maximum


@pytest.mark.parametrize("quant_mode", QUANT_MODES)
@pytest.mark.parametrize("config", [-1, 0])
def test_workspace_size_rejects_unsupported_tile(quant_mode, config):
    with pytest.raises((ValueError, RuntimeError)):
        trtllm_fused_moe_workspace_size(
            32,
            HIDDEN,
            INTER,
            NUM_EXPERTS,
            TOP_K,
            quant_mode=quant_mode,
            tactic=(7, config),
        )


@pytest.mark.parametrize(
    "tactic",
    [-1, (), (32,), (32, 0, 1), (-1, -1), (32, -2), (0, 0), (32, 0.5), (True, 0)],
)
def test_workspace_size_rejects_malformed_tactic(tactic):
    with pytest.raises(ValueError, match="tactic must be"):
        trtllm_fused_moe_workspace_size(
            32,
            HIDDEN,
            INTER,
            NUM_EXPERTS,
            TOP_K,
            quant_mode="bf16",
            tactic=tactic,
        )


@pytest.mark.parametrize("case", list(CASES))
def test_workspace_buffer_matches_per_call_allocation(case: str):
    """One buffer sized for MAX_TOKENS serves every smaller batch, bit for bit."""
    quant_mode, factory = CASES[case]
    launch = factory(seed=0)
    workspace = _workspace(quant_mode)
    for n in TOKEN_COUNTS:
        expected = launch(n)
        # Poison the shared buffer: the launch must not depend on its prior contents.
        workspace.fill_(0xFF)
        actual = launch(n, workspace_buffer=workspace)
        _assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize(
    "routing_input_mode", list(RoutingInputMode), ids=lambda m: m.name
)
@pytest.mark.parametrize(
    "routing_method",
    [m for m in RoutingMethodType if m != RoutingMethodType.Unspecified],
    ids=lambda m: m.name,
)
def test_workspace_buffer_covers_routing(
    routing_input_mode: RoutingInputMode, routing_method: RoutingMethodType
):
    """The logits-based size query bounds scratch for each routing path."""
    launch = _make_bf16_case(
        seed=0,
        use_logits=routing_input_mode == RoutingInputMode.FromLogits,
        packed=routing_input_mode == RoutingInputMode.PackedPrecomputed,
        routing_method=routing_method,
    )
    top_k = 1 if routing_method == RoutingMethodType.Llama4 else TOP_K
    nbytes = trtllm_fused_moe_workspace_size(
        MAX_TOKENS, HIDDEN, INTER, NUM_EXPERTS, top_k, quant_mode="bf16"
    )
    assert nbytes > 0
    workspace = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for n in TOKEN_COUNTS:
        expected = launch(n)
        # Warm the buffered path before measuring allocations.
        launch(n, workspace_buffer=workspace)
        workspace.fill_(0xFF)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        actual = launch(n, workspace_buffer=workspace)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"] - before
        # Only the returned output may allocate; scratch must fit in the buffer.
        assert allocated == 1, allocated
        _assert_bitwise_equal(actual, expected)


def test_workspace_buffer_shared_across_layers():
    """Sequential layers with different weights reuse one buffer, as an engine would."""
    layers = [_make_bf16_case(seed, use_logits=False) for seed in range(3)]
    workspace = _workspace("bf16")
    expected = [[layer(n) for n in TOKEN_COUNTS] for layer in layers]
    for i, n in enumerate(TOKEN_COUNTS):
        for layer, reference in zip(layers, expected, strict=True):
            _assert_bitwise_equal(layer(n, workspace_buffer=workspace), reference[i])


def test_workspace_buffer_avoids_per_call_allocations():
    launch = _make_bf16_case(seed=0, use_logits=False)
    workspace = _workspace("bf16")
    output = launch(MAX_TOKENS, workspace_buffer=workspace)  # warm up JIT and tactics

    def allocations(**kwargs) -> int:
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        launch(MAX_TOKENS, **kwargs)
        torch.cuda.synchronize()
        return torch.cuda.memory_stats()["allocation.all.allocated"] - before

    del output
    per_call = allocations()
    buffered = allocations(workspace_buffer=workspace)
    # The routing maps, both GEMM outputs and both batched-GEMM workspaces move into
    # the caller's buffer; only the returned output remains (12 -> 1 on B300).
    assert buffered == 1 and per_call > buffered, (buffered, per_call)


def _tuning_peak_bytes(run: Callable[[], object]) -> int:
    """Peak allocation above the starting point while ``run`` tunes from scratch."""
    AutoTuner.get().clear_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with autotune(True):
        run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def test_workspace_buffer_covers_autotuning():
    """Every profiling launch carves from the caller's buffer; tuning allocates no scratch."""
    launch = _make_bf16_case(seed=0, use_logits=False)
    workspace = _workspace("bf16")
    unbuffered = _tuning_peak_bytes(lambda: launch(MAX_TOKENS))
    buffered = _tuning_peak_bytes(
        lambda: launch(MAX_TOKENS, workspace_buffer=workspace)
    )
    # Unbuffered tuning peaks at roughly one launch's scratch, about the buffer size.
    assert buffered + workspace.numel() // 2 <= unbuffered, (
        buffered,
        unbuffered,
        workspace.numel(),
    )
    for n in TOKEN_COUNTS:
        _assert_bitwise_equal(launch(n, workspace_buffer=workspace), launch(n))


def test_workspace_buffer_autotuning_falls_back_for_larger_buckets():
    """Tuning buckets above the buffer's size allocate per call instead of failing."""
    launch = _make_bf16_case(seed=0, use_logits=False)
    workspace = _workspace("bf16", max_num_tokens=TOKEN_COUNTS[0])
    AutoTuner.get().clear_cache()
    with autotune(True):
        launch(TOKEN_COUNTS[0], workspace_buffer=workspace)
    _assert_bitwise_equal(
        launch(TOKEN_COUNTS[0], workspace_buffer=workspace), launch(TOKEN_COUNTS[0])
    )


@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_undersized_workspace_buffer_allocates_the_rest(fraction: float):
    """Scratch that does not fit is allocated per call; the carved part stays correct."""
    launch = _make_bf16_case(seed=0, use_logits=False)
    full = _workspace("bf16").numel()
    workspace = torch.full(
        (max(1, int(full * fraction)),), 0xFF, dtype=torch.uint8, device="cuda"
    )
    _assert_bitwise_equal(
        launch(MAX_TOKENS, workspace_buffer=workspace), launch(MAX_TOKENS)
    )


def test_workspace_buffer_rejects_misuse():
    launch = _make_bf16_case(seed=0, use_logits=False)
    workspace = _workspace("bf16")
    with pytest.raises(RuntimeError, match="do_finalize=True"):
        launch(MAX_TOKENS, workspace_buffer=workspace, do_finalize=False)
    with pytest.raises(RuntimeError, match="uint8 or int8"):
        float_workspace = torch.empty(
            workspace.numel(), dtype=torch.float32, device="cuda"
        )
        launch(MAX_TOKENS, workspace_buffer=float_workspace)
    with pytest.raises(RuntimeError, match="aligned"):
        launch(MAX_TOKENS, workspace_buffer=workspace[1:])
