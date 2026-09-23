"""Single-GPU NVFP4 weight checks with W4A4 and W4A16 activation modes.

NVFP4 counterpart of ``test_mxfp8_cutedsl_preprocess_vs_reference.py``: validates
that ``sm100_nvfp4_nvfp4_bf16_cutedsl.preprocess_mega_weights`` produces fp4 weights consistent
with an independent plain quant, and that a single-rank ``nvfp4_mega_moe``
launch matches a pure-torch dequant reference (fp32 GEMMs + SwiGLU + fc1-out
NVFP4 round-trip) after the in-kernel top-k reduction.

The torch oracle here is intentionally independent of the CuTeDSL-backed
``compute_megamoe_reference`` (whose GEMMs run on a reference device kernel):
W4A4 uses plain torch ops on dequantized values. W4A16 uses BF16 GEMMs
with FP32 output, a Triton helper matching the kernel's fast SwiGLU, and
ordered FP32 combine. Its deterministic path requires finite bit-exact output;
atomic in-kernel FC2 reduction is checked separately with a rounding band.

Run on one Blackwell GPU from the FlashInfer repo root (no torchrun required)::

    cd /path/to/flashinfer
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -v \\
        -m arch_blackwell --confcutdir=tests/moe_ep
"""

from __future__ import annotations

import pytest

# Verify only through the cutedsl_megamoe shim public API (plus the FI backend
# helpers); never import the src/ kernel packages directly, so a new src/ drop
# can't silently break this test.
pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")

NVFP4_BLOCK = 16
NVFP4_MODES = ("w4a4", "w4a16")


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


@pytest.fixture(scope="session")
def w4a16_single_rank_runtime(tmp_path_factory):
    """Supply the real EP1 group used by W4A16's workspace pool."""
    import torch
    import torch.distributed as dist

    _require_cuda()
    if not dist.is_initialized():
        rendezvous = tmp_path_factory.mktemp("w4a16-ep1") / "process-group"
        dist.init_process_group(
            backend="nccl",
            init_method=rendezvous.as_uri(),
            rank=0,
            world_size=1,
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )
    assert (dist.get_rank(), dist.get_world_size()) == (0, 1)
    assert dist.get_backend() == "nccl"
    # Existing moe_ep/conftest.py owns process-group teardown at session finish;
    # NCCL must not be destroyed/reinitialized between graph and precision cases.
    yield


def _single_rank_problem(
    hidden=2048,
    intermediate=1024,
    *,
    num_experts=4,
    topk=4,
    num_tokens=32,
    max_tokens=64,
    seed=7,
    make_weights=True,
):
    import torch

    num_local_experts = num_experts
    gate_up_clamp = 10.0

    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )

    w13 = w2 = None
    if make_weights:
        g = torch.Generator(device="cuda").manual_seed(13)
        w13 = torch.randn(
            num_local_experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=g,
        )
        w2 = torch.randn(
            num_local_experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device="cuda",
            generator=g,
        )
    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
        hidden_states=hidden_states,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int64),
        w13=w13,
        w2=w2,
    )


def _e2m1_decode_table(device):
    import torch

    # Standard E2M1 code points, low nibble = even element (matches the
    # kernel-side pack/unpack convention).
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _dequant_nvfp4(packed, sf_fp8, *, logical_cols):
    """Packed fp4 codes + plain per-16 e4m3 scales → fp32 (rows, logical_cols)."""
    import torch

    raw = packed.view(torch.uint8).reshape(packed.shape[0], -1)
    lut = _e2m1_decode_table(raw.device)
    lo = lut[(raw & 0x0F).to(torch.int64)]
    hi = lut[(raw >> 4).to(torch.int64)]
    vals = torch.empty(
        raw.shape[0], raw.shape[1] * 2, dtype=torch.float32, device=raw.device
    )
    vals[:, ::2] = lo
    vals[:, 1::2] = hi
    vals = vals[:, :logical_cols]
    n_blocks = logical_cols // NVFP4_BLOCK
    scales = (
        sf_fp8[:, :n_blocks].to(torch.float32).repeat_interleave(NVFP4_BLOCK, dim=-1)
    )
    return vals * scales


def _plain_nvfp4_from_bf16(problem: dict):
    """bf16 weights → kernel fp4 + plain e4m3 SF (pre-swizzle layout)."""
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        _interleave_gate_up_16,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )

    intermediate = problem["intermediate"]
    num_experts = problem["w13"].shape[0]
    norm_const = 1.0

    w13_interleaved = _interleave_gate_up_16(
        problem["w13"], intermediate_size=intermediate
    )

    fc1_weights, fc1_plain_sf = [], []
    fc2_weights, fc2_plain_sf = [], []
    for expert in range(num_experts):
        fc1_q, fc1_sf = nvfp4_quantize_per_block_16(
            w13_interleaved[expert].to(torch.float32), norm_const
        )
        fc1_weights.append(fc1_q)
        fc1_plain_sf.append(fc1_sf)
        fc2_q, fc2_sf = nvfp4_quantize_per_block_16(
            problem["w2"][expert].to(torch.float32), norm_const
        )
        fc2_weights.append(fc2_q)
        fc2_plain_sf.append(fc2_sf)

    return (
        torch.stack([w.view(torch.uint8) for w in fc1_weights], dim=0),
        torch.stack(fc1_plain_sf, dim=0),
        torch.stack([w.view(torch.uint8) for w in fc2_weights], dim=0),
        torch.stack(fc2_plain_sf, dim=0),
    )


def _torch_nvfp4_mega_reference(
    *,
    act_packed,  # (T, hidden//2) packed fp4 (uint8 view ok)
    act_sf,  # (T, >=hidden//16) e4m3 plain
    topk_idx,  # (T, topk) int64
    topk_weights,  # (T, topk) fp32
    fc1_weight,  # (E, 2I, hidden//2) packed fp4 codes (uint8)
    fc1_sf,  # (E, 2I, hidden//16) e4m3 plain
    fc2_weight,  # (E, hidden, I//2) packed fp4 codes (uint8)
    fc2_sf,  # (E, hidden, I//16) e4m3 plain
    hidden,
    intermediate,
    gate_up_clamp,
    term_transform=None,
    swiglu_alpha=None,
    swiglu_beta=None,
):
    """Pure-torch NVFP4 MegaMoE oracle (apply_topk_in_fc1=True graph).

    Mirrors the kernel's data path — dequant → fp32 fc1 GEMM → 16-interleaved
    SwiGLU fold (+clamp) → per-token topk weight folded in BEFORE the fc1-out
    NVFP4 round-trip → fp32 fc2 GEMM → bf16 terms — so disagreement is
    bounded by NVFP4 RTNE flips at fc1-out plus GEMM accumulation-order noise.

    ``term_transform``, when set, is applied to each per-(token, topk) fc2
    output term before the topk sum; the multirank oracle uses it to model the
    quantized cross-rank combine wire (``combine_roundtrip_to_fp32``).
    """
    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )

    num_tokens, topk = topk_idx.shape
    num_experts = fc1_weight.shape[0]

    act_fp32 = _dequant_nvfp4(act_packed, act_sf, logical_cols=hidden)

    out = torch.zeros(
        num_tokens, topk, hidden, dtype=torch.float32, device=act_fp32.device
    )
    for expert in range(num_experts):
        routing_mask = topk_idx == expert
        if not routing_mask.any():
            continue
        routed = routing_mask.nonzero(as_tuple=False)
        tokens, slots = routed[:, 0], routed[:, 1]

        fc1_w = _dequant_nvfp4(
            fc1_weight[expert], fc1_sf[expert], logical_cols=hidden
        )  # (2I, hidden)
        fc1_out = act_fp32[tokens] @ fc1_w.transpose(0, 1)  # (R, 2I)

        # SwiGLU over the 16-column gate/up interleave used by the NVFP4 kernel.
        m = fc1_out.shape[0]
        n_pairs = fc1_out.shape[1] // (2 * NVFP4_BLOCK)
        reshaped = fc1_out.view(m, n_pairs, 2, NVFP4_BLOCK)
        gate = reshaped[:, :, 0, :]
        up = reshaped[:, :, 1, :]
        if gate_up_clamp is not None:
            limit = abs(float(gate_up_clamp))
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        alpha = 1.0 if swiglu_alpha is None else swiglu_alpha
        beta = 0.0 if swiglu_beta is None else swiglu_beta
        swiglu = (gate * torch.sigmoid(alpha * gate) * (up + beta)).reshape(
            m, intermediate
        )

        # apply_topk_in_fc1=True: weight folded in before the fp4 round-trip
        # (post-hoc weighting would NOT match — quant changes the magnitude).
        swiglu = swiglu * topk_weights[tokens, slots].unsqueeze(-1)

        fc1_q, fc1_q_sf = nvfp4_quantize_per_block_16(swiglu, 1.0)
        swiglu_rt = _dequant_nvfp4(fc1_q, fc1_q_sf, logical_cols=intermediate)

        fc2_w = _dequant_nvfp4(
            fc2_weight[expert], fc2_sf[expert], logical_cols=intermediate
        )  # (hidden, I)
        fc2_out = swiglu_rt @ fc2_w.transpose(0, 1)
        # FC2 stores each expert term in BF16 before the top-k reduction.
        fc2_out = fc2_out.to(torch.bfloat16).float()
        if term_transform is not None:
            fc2_out = term_transform(fc2_out)
        out[tokens, slots] = fc2_out

    return out.sum(dim=1).to(torch.bfloat16)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("mode", NVFP4_MODES)
@pytest.mark.parametrize("hidden,intermediate", [(2048, 1024), (288, 448)])
@pytest.mark.parametrize("weight_dtype", ["bfloat16", "float32"])
def test_nvfp4_preprocess_fp4_weights_match_plain_quant(
    mode, hidden, intermediate, weight_dtype
):
    """``preprocess_mega_weights`` fp4 tensors match an independent plain quant."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
        to_blocked,
    )

    if mode == "w4a4":
        from flashinfer.moe_ep import (
            preprocess_nvfp4_cutedsl_mega_weights as preprocess_mega_weights,
        )
    elif mode == "w4a16":
        from flashinfer.moe_ep import (
            preprocess_bf16_nvfp4_cutedsl_mega_weights as preprocess_mega_weights,
        )
    else:
        raise AssertionError(f"unsupported precision mode: {mode}")
    problem = _single_rank_problem(hidden, intermediate)
    problem["w13"] = problem["w13"].to(getattr(torch, weight_dtype))
    problem["w2"] = problem["w2"].to(getattr(torch, weight_dtype))
    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])

    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
    )
    fc1_kernel, fc1_kernel_sf = transformed_l1
    fc2_kernel, fc2_kernel_sf = transformed_l2

    fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)

    # Kernel weights are logically (E, K//2, N) but MUST keep the packed K
    # axis stride-1 (transpose view over K-major memory) — the kernel's TMA
    # descriptors read K-major, so a materialized N-stride-1 tensor scrambles
    # every weight. Pin the stride contract explicitly.
    assert fc1_kernel.stride(1) == 1, (
        f"fc1 packed-K axis must be stride-1, got strides {fc1_kernel.stride()}"
    )
    assert fc2_kernel.stride(1) == 1, (
        f"fc2 packed-K axis must be stride-1, got strides {fc2_kernel.stride()}"
    )
    # Values: compare the K-major memory against the independent plain quant.
    torch.testing.assert_close(
        fc1_kernel.transpose(1, 2).contiguous().view(torch.uint8),
        fc1_plain,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        fc2_kernel.transpose(1, 2).contiguous().view(torch.uint8),
        fc2_plain,
        atol=0,
        rtol=0,
    )
    # Scales: preprocess swizzles the plain SF per expert.
    num_experts = fc1_plain.shape[0]
    for e in range(num_experts):
        torch.testing.assert_close(
            fc1_kernel_sf[e].view(torch.uint8).reshape(-1),
            to_blocked(fc1_sf[e]).view(torch.uint8).reshape(-1),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            fc2_kernel_sf[e].view(torch.uint8).reshape(-1),
            to_blocked(fc2_sf[e]).view(torch.uint8).reshape(-1),
            atol=0,
            rtol=0,
        )

    # Canonical prepacked inputs must produce the same weight/SF bytes and
    # K-major strides. W4A16 also accepts typed FP4 aliases.
    packed = {}
    for name in ("w13", "w2"):
        weight = problem[name]
        q, sf = nvfp4_quantize_per_block_16(
            weight.float().reshape(-1, weight.shape[-1]), 1.0
        )
        if mode == "w4a4" or weight_dtype == "float32":
            q = q.view(torch.uint8)
        packed[name] = q.reshape(*weight.shape[:-1], weight.shape[-1] // 2)
        packed[f"{name}_scale"] = sf.reshape(*weight.shape[:-1], weight.shape[-1] // 16)
    prepacked = preprocess_mega_weights(
        MoEWeightPack(**packed), intermediate_size=intermediate, hidden_size=hidden
    )
    for reference, actual in zip(
        (transformed_l1, transformed_l2), prepacked, strict=True
    ):
        for expected, result in zip(reference, actual, strict=True):
            assert (
                expected.shape == result.shape and expected.stride() == result.stride()
            )
            assert torch.equal(expected.view(torch.uint8), result.view(torch.uint8))


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "mode,tile_n,in_kernel_fc2_reduce,token_back_mode,apply_topk_in_fc1",
    [
        (mode, 128, False, token_back_mode, apply_topk_in_fc1)
        for mode in NVFP4_MODES
        for token_back_mode in ("epi_warps", "reuse_dispatch_warps")
        for apply_topk_in_fc1 in ([False, True] if mode == "w4a16" else [False])
    ]
    + [
        ("w4a16", tile_n, True, "reuse_dispatch_warps", apply_topk_in_fc1)
        for tile_n in (64, 128)
        for apply_topk_in_fc1 in (False, True)
    ],
)
@pytest.mark.parametrize(
    "hidden,intermediate,num_experts,topk",
    [
        pytest.param(2048, 1024, 4, 4, id="regular-e4"),
        # 128-misaligned (hidden % 128 == 64): exercises the ceil-div K-tail
        # and predicated epilogue paths the %64 validation relaxation opened
        # up (gpt-oss-120b geometry class).
        pytest.param(2880, 2880, 4, 4, id="tail-e4"),
        pytest.param(2048, 1024, 1, 1, id="singleton-e1"),
    ],
)
def test_nvfp4_kernel_matches_torch_reference(
    monkeypatch,
    mode,
    tile_n,
    in_kernel_fc2_reduce,
    token_back_mode,
    apply_topk_in_fc1,
    hidden,
    intermediate,
    num_experts,
    topk,
):
    """Single-rank ``nvfp4_mega_moe`` output matches the precision oracle."""
    _check_nvfp4_kernel_matches_torch_reference(
        monkeypatch,
        mode=mode,
        tile_n=tile_n,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        apply_topk_in_fc1=apply_topk_in_fc1,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("token_back_mode", ["epi_warps", "reuse_dispatch_warps"])
@pytest.mark.parametrize("activation_clamp", [0.5, None], ids=["clamp", "no-clamp"])
def test_nvfp4_w4a4_swiglu_parameters(monkeypatch, token_back_mode, activation_clamp):
    """MiniMax activation configuration and runtime overrides are W4A4-only."""
    _check_nvfp4_kernel_matches_torch_reference(
        monkeypatch,
        mode="w4a4",
        tile_n=128,
        in_kernel_fc2_reduce=False,
        token_back_mode=token_back_mode,
        apply_topk_in_fc1=False,
        hidden=2048,
        intermediate=1024,
        num_experts=4,
        topk=4,
        activation_clamp=activation_clamp,
    )


def _check_nvfp4_kernel_matches_torch_reference(
    monkeypatch,
    *,
    mode,
    tile_n,
    in_kernel_fc2_reduce,
    token_back_mode,
    apply_topk_in_fc1,
    hidden,
    intermediate,
    num_experts,
    topk,
    activation_clamp="standard",
):
    _require_cuda()

    import torch

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(
            f"nvfp4_mega_moe requires sm_100a or sm_103a; got sm_{cap[0]}{cap[1]}"
        )
    pytest.importorskip("triton")

    from flashinfer.moe_ep import MoEWeightPack

    if mode == "w4a4":
        from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
            stage_mega_moe_inputs,
        )
        from flashinfer.moe_ep import (
            preprocess_nvfp4_cutedsl_mega_weights as preprocess_mega_weights,
        )
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
            get_symm_buffer_for_mega_moe,
            nvfp4_mega_moe,
        )
    elif mode == "w4a16":
        from flashinfer.moe_ep import (
            preprocess_bf16_nvfp4_cutedsl_mega_weights as preprocess_mega_weights,
        )
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_staging import (
            stage_mega_moe_inputs,
        )
        from flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4 import (
            get_symm_buffer_for_bf16_nvfp4_mega_moe as get_symm_buffer_for_mega_moe,
            bf16_nvfp4_mega_moe as nvfp4_mega_moe,
        )
    else:
        raise AssertionError(f"unsupported precision mode: {mode}")

    # monkeypatch (not os.environ): restored after the test, so it cannot
    # silently downgrade later nvshmem-path tests in the same process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        make_weights=mode == "w4a4",
    )
    activation_pairs = [(None, None)]
    if activation_clamp != "standard":
        assert mode == "w4a4"
        problem["gate_up_clamp"] = activation_clamp
        # Keep gates near zero so ignoring alpha changes the output visibly.
        # A 0.5 clamp exercises both saturated and unsaturated gate/up values.
        problem["hidden_states"].mul_(0.1)
        problem["w13"].mul_(0.1)
        activation_pairs = [(1.702, 1.0), (1.0, 1.0), (1.0, 0.0), (1.702, 1.0)]
    rank = 0
    world_size = 1
    num_tokens = problem["num_tokens"]

    if mode == "w4a4":
        pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    elif mode == "w4a16":
        from .w4a16_reference import make_w4a16_weights

        pack = make_w4a16_weights(hidden, intermediate, num_experts)
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
    )

    if mode == "w4a4":
        fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)

    alpha1, alpha2 = None, None
    if mode == "w4a16":
        # Non-BF16-representable FP32 alphas exercise post-MMA scale use.
        alpha1 = (
            torch.linspace(0.71013, 1.23017, num_experts, device="cuda") / hidden**0.5
        )
        alpha2 = (
            torch.linspace(1.17019, 0.83023, num_experts, device="cuda")
            / intermediate**0.5
        )
    intermediate_arg = {"w4a4": 2 * intermediate, "w4a16": intermediate}[mode]

    knobs = {"token_back_mode": token_back_mode}
    if mode == "w4a16":
        knobs.update(
            mma_tiler_mnk=(256, tile_n, 256), in_kernel_fc2_reduce=in_kernel_fc2_reduce
        )

    # NOTE: the nvfp4 shim's ``intermediate`` is the fc1 output width (2*I),
    # matching the backend's ``2 * intermediate_size`` convention.
    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        intermediate_arg,
        rank,
        world_size,
        gate_up_clamp=problem["gate_up_clamp"],
        enable_in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        fc1_alpha=alpha1,
        fc2_alpha=alpha2,
        knobs=knobs,
        **(
            {"apply_topk_in_fc1": apply_topk_in_fc1}
            if mode == "w4a16"
            else dict(
                swiglu_alpha=activation_pairs[0][0],
                swiglu_beta=activation_pairs[0][1],
            )
        ),
    )
    graph = None
    try:
        scale_args = ()
        if mode == "w4a4":
            scale_args = (symm_buffer.x_sf,)
        elif mode == "w4a16":
            assert symm_buffer.x.dtype == torch.bfloat16
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            *scale_args,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )

        for step, (alpha, beta) in enumerate(activation_pairs):
            if mode == "w4a4":
                reference = _torch_nvfp4_mega_reference(
                    act_packed=symm_buffer.x[:num_tokens],
                    act_sf=scale_args[0][:num_tokens],
                    topk_idx=symm_buffer.topk_idx[:num_tokens],
                    topk_weights=symm_buffer.topk_weights[:num_tokens],
                    fc1_weight=fc1_plain,
                    fc1_sf=fc1_sf,
                    fc2_weight=fc2_plain,
                    fc2_sf=fc2_sf,
                    hidden=hidden,
                    intermediate=intermediate,
                    gate_up_clamp=problem["gate_up_clamp"],
                    swiglu_alpha=alpha,
                    swiglu_beta=beta,
                )
            elif mode == "w4a16":
                reference = _nvfp4_reference_from_weights(
                    problem,
                    pack,
                    mode=mode,
                    fc1_alpha=alpha1,
                    fc2_alpha=alpha2,
                    apply_topk_in_fc1=apply_topk_in_fc1,
                    in_kernel_fc2_reduce=in_kernel_fc2_reduce,
                )

            y_kernel = torch.empty(
                num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
            )

            def launch():
                nvfp4_mega_moe(
                    y_kernel,
                    transformed_l1,
                    transformed_l2,
                    symm_buffer,
                    **({} if step == 0 else dict(swiglu_alpha=alpha, swiglu_beta=beta)),
                    num_tokens=num_tokens,
                    gate_up_clamp=problem["gate_up_clamp"],
                )

            launch()
            torch.cuda.synchronize()
            _assert_nvfp4_reference(
                y_kernel,
                reference,
                mode=mode,
                in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            )
            if in_kernel_fc2_reduce:
                # BF16 atomic combine may reorder terms; the shared oracle band
                # covers that rounding while repeated launches catch missing zeroing.
                for _ in range(2):
                    launch()
                    _assert_nvfp4_reference(
                        y_kernel, reference, mode=mode, in_kernel_fc2_reduce=True
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                for _ in range(3):
                    graph.replay()
                    _assert_nvfp4_reference(
                        y_kernel, reference, mode=mode, in_kernel_fc2_reduce=True
                    )
    finally:
        if graph is not None:
            graph.reset()
        symm_buffer.destroy()


def _assert_nvfp4_reference(y_kernel, reference, *, mode, in_kernel_fc2_reduce=False):
    import torch

    assert mode in NVFP4_MODES, mode
    assert torch.isfinite(y_kernel).all()
    if mode == "w4a16":
        if in_kernel_fc2_reduce:
            from .mega_oracle_compare import _assert_mega_oracle_term_band_close

            assert torch.isfinite(reference).all()
            _assert_mega_oracle_term_band_close(
                y_kernel, reference, ikr=True, label="w4a16 atomic BF16 combine"
            )
        else:
            from .w4a16_reference import assert_w4a16_bits

            assert_w4a16_bits(y_kernel, reference)
        return
    assert mode == "w4a4"
    y_ref = reference
    yk = y_kernel.to(torch.float32)
    yr = y_ref.to(torch.float32)
    rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
    print(
        f"[nvfp4 oracle] rel_l2={rel_l2.item():.4g} "
        f"max|Δ|={(yk - yr).abs().max().item():.4g} "
        f"amax(ref)={yr.abs().max().item():.4g}"
    )
    # The oracle shares the kernel's quantized operands, so the residual is
    # NVFP4 RTNE flips at fc1-out + accumulation-order noise (measured
    # rel_l2≈0.0027 on GB200; kernel is bit-exact vs the CuTeDSL reference
    # launcher on the same operands). atol scales with the output range
    # (random unscaled weights put |y|~1e4 here).
    atol = 2e-3 * yr.abs().max().item()
    torch.testing.assert_close(yk, yr, atol=atol, rtol=0.05)
    assert rel_l2.item() < 0.02


def _nvfp4_reference_from_weights(
    problem,
    weights,
    *,
    mode,
    fc1_alpha=None,
    fc2_alpha=None,
    fc1_norm_const=None,
    apply_topk_in_fc1=False,
    in_kernel_fc2_reduce=False,
):
    """Shared oracle input assembly from canonical prequantized weight packs."""
    from flashinfer.moe_ep import PrequantizedMoEWeights

    assert mode in NVFP4_MODES, mode
    assert isinstance(weights, PrequantizedMoEWeights), type(weights)
    if mode == "w4a16":
        from .w4a16_reference import w4a16_reference

        return w4a16_reference(
            problem,
            weights,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            apply_topk_in_fc1=apply_topk_in_fc1,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        )
    assert mode == "w4a4"
    assert fc1_alpha is None and fc2_alpha is None and fc1_norm_const is None
    assert not apply_topk_in_fc1
    import torch
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        _interleave_gate_up_16,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )

    hidden, intermediate = problem["hidden"], problem["intermediate"]
    fc1 = _interleave_gate_up_16(
        weights.w13.view(torch.uint8), intermediate_size=intermediate
    )
    sf1 = _interleave_gate_up_16(weights.w13_scale, intermediate_size=intermediate)
    x, sf = nvfp4_quantize_per_block_16(problem["hidden_states"].float(), 1.0)
    return _torch_nvfp4_mega_reference(
        act_packed=x,
        act_sf=sf,
        topk_idx=problem["topk_ids"],
        topk_weights=problem["topk_weights"],
        fc1_weight=fc1,
        fc1_sf=sf1,
        fc2_weight=weights.w2,
        fc2_sf=weights.w2_scale,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_clamp=problem["gate_up_clamp"],
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("token_back_mode", ["epi_warps", "reuse_dispatch_warps"])
@pytest.mark.parametrize("alpha_source", ["config", "runtime"])
def test_nvfp4_w4a16_fp32_scales_and_routing(
    monkeypatch, token_back_mode, alpha_source, w4a16_single_rank_runtime
):
    """Cancellation distinguishes FP32 routing/globals from premature BF16 casts."""
    import torch
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpConfigError,
        MoEEpMegaLayer,
        MoEEpTensors,
        PrequantizedMoEWeights,
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    from .w4a16_reference import assert_w4a16_bits

    _require_cuda()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem(256, 256, topk=2, num_tokens=4, make_weights=False)
    w13 = torch.zeros((4, 512, 128), dtype=torch.uint8, device="cuda")
    w2 = torch.zeros((4, 256, 128), dtype=torch.uint8, device="cuda")
    w13[:, 0, 0] = w13[:, 256, 0] = 2  # +1 gate and up
    w2[0, 0, 0], w2[3, 0, 0] = 2, 10  # equal and opposite FC2
    weights = PrequantizedMoEWeights(
        w13=w13,
        w2=w2,
        w13_scale=torch.ones((4, 512, 16), device="cuda").to(torch.float8_e4m3fn),
        w2_scale=torch.ones((4, 256, 16), device="cuda").to(torch.float8_e4m3fn),
    )
    alphas = dict(
        fc1_alpha=torch.full((4,), 1.00390625, device="cuda"),
        fc2_alpha=torch.full((4,), 1.001953125, device="cuda"),
        fc1_norm_const=torch.full((4,), 1.1318359375, device="cuda"),
    )
    problem["hidden_states"].zero_()
    problem["hidden_states"][:, 0] = torch.tensor([0.5, 1.0, 2.0, 4.0], device="cuda")
    problem["topk_ids"][:] = torch.tensor([0, 3], device="cuda")
    problem["topk_weights"][:, 0], problem["topk_weights"][:, 1] = 1.00390625, 1.0
    # Preserve the exact, unclamped precision fixture from the W4A16 suite.
    problem["gate_up_clamp"] = None
    reference = _nvfp4_reference_from_weights(problem, weights, mode="w4a16", **alphas)
    expected = reference
    assert torch.count_nonzero(expected[:, 0]) == 4
    layer = MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=4, max_tokens_per_rank=64, token_hidden_size=256
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=256,
                top_k=2,
                knobs={"token_back_mode": token_back_mode},
                **(alphas if alpha_source == "config" else {}),
            )
        ),
    )
    graph = unnormalized_graph = None
    try:
        inputs = {
            key: problem[key] for key in ("hidden_states", "topk_ids", "topk_weights")
        }
        if alpha_source == "runtime":
            for name, alpha in alphas.items():
                for invalid in (alpha[:1], alpha.bfloat16(), alpha.cpu()):
                    with pytest.raises(MoEEpConfigError, match=f"{name} must be FP32"):
                        layer.forward(MoEEpTensors(**inputs, **{name: invalid}))
        tensors = MoEEpTensors(
            **inputs, **(alphas if alpha_source == "runtime" else {})
        )
        if alpha_source == "runtime" and token_back_mode == "epi_warps":
            # A previously captured no-norm specialization must stay live when
            # the same workspace later prepares its normalized specialization.
            plain_alphas = {name: alphas[name] for name in ("fc1_alpha", "fc2_alpha")}
            plain = MoEEpTensors(**inputs, **plain_alphas)
            plain_reference = _nvfp4_reference_from_weights(
                problem, weights, mode="w4a16", **plain_alphas
            )
            layer.forward(plain)
            unnormalized_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(unnormalized_graph):
                plain_captured = layer.forward(plain)
        actual = layer.forward(tensors)
        assert_w4a16_bits(actual, expected)
        if unnormalized_graph is not None:
            assert not torch.equal(plain_reference, expected)
            unnormalized_graph.replay()
            assert_w4a16_bits(plain_captured, plain_reference)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_graph = layer.forward(tensors)
        for step in range(3):
            if alpha_source == "runtime" and step == 1:
                alphas["fc1_alpha"].mul_(2.0)
                alphas["fc2_alpha"].mul_(0.5)
                alphas["fc1_norm_const"].mul_(1.13)
                reference = _nvfp4_reference_from_weights(
                    problem, weights, mode="w4a16", **alphas
                )
                expected = reference
            graph.replay()
            assert_w4a16_bits(actual_graph, expected)
        if alpha_source == "runtime":
            # Like W4A4, omitting an override retains the workspace's values.
            actual = layer.forward(
                MoEEpTensors(
                    **{
                        key: problem[key]
                        for key in ("hidden_states", "topk_ids", "topk_weights")
                    }
                )
            )
            assert_w4a16_bits(actual, expected)
    finally:
        if graph is not None:
            graph.reset()
        if unnormalized_graph is not None:
            unnormalized_graph.reset()
        layer.destroy()
