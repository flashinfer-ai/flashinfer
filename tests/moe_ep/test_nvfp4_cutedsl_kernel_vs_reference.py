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

Shared NVFP4/MXFP8 regressions cover the masked BF16 reducer under captured
route changes and the sparse-dispatch workspace capacity boundary.

Run on one Blackwell GPU from the FlashInfer repo root (no torchrun required)::

    cd /path/to/flashinfer
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -v \\
        -m arch_blackwell --confcutdir=tests/moe_ep
"""

from __future__ import annotations

import pytest
import torch

# Verify only through the cutedsl_megamoe shim public API (plus the FI backend
# helpers); never import the src/ kernel packages directly, so a new src/ drop
# can't silently break this test.
pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")

NVFP4_BLOCK = 16
NVFP4_MODES = ("w4a4", "w4a16")


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "kind,intermediate",
    [("nvfp4", 192), ("nvfp4", 96), ("mxfp8_e4m3", 192), ("mxfp8_e5m2", 192)],
)
@pytest.mark.parametrize("early", [True, False])
@pytest.mark.parametrize("capacity", [1, 129])
def test_blackwell_standard_provider_graph(
    kind, intermediate, early, capacity, monkeypatch
):
    """Check the exported standard kernel against staged bytes and changing routes."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    torch.manual_seed(9911)
    cfg = pkg.Sm100BlockScaledMoeConfig(
        num_total_experts=8,
        max_tokens_per_rank=capacity,
        num_topk=3,
        hidden=384,
        intermediate=intermediate,
        rank=0,
        world_size=1,
        quant_kind=kind,
        apply_routing_weights_before_fc2=early,
    )
    ws = pkg.Sm100BlockScaledSymmBuffer(cfg)
    graph = None
    try:
        weights = pkg.preprocess_block_scaled_weights(
            (
                torch.randn(8, 2 * intermediate, 384, device="cuda") / 384**0.5
            ).bfloat16(),
            (
                torch.randn(8, 384, intermediate, device="cuda") / intermediate**0.5
            ).bfloat16(),
            quant_kind=kind,
            intermediate_size=intermediate,
        )
        x = torch.randn(capacity, 384, device="cuda")
        data, scales = (
            pkg.quantize_nvfp4_block16(x)
            if kind == "nvfp4"
            else pkg.quantize_mxfp8_block32(
                x, torch.float8_e5m2 if kind == "mxfp8_e5m2" else torch.float8_e4m3fn
            )
        )
        ws.x[:capacity].view(torch.uint8).copy_(data.view(torch.uint8))
        ws.x_sf[:capacity, : scales.shape[1]].view(torch.uint8).copy_(
            scales.view(torch.uint8)
        )
        ids = (
            torch.arange(capacity * 3, device="cuda", dtype=torch.int32).reshape(-1, 3)
            % 8
        )
        scores = torch.linspace(0.1, 0.9, capacity * 3, device="cuda").reshape_as(ids)
        ws.topk_idx[:capacity].copy_(ids)
        ws.topk_weights[:capacity].copy_(scores)
        ws.note_staged_tokens(capacity)
        thunk = pkg.sm100_block_scaled_mega_launch_thunk(*weights, ws)
        thunk()
        torch.cuda.synchronize()
        indices, expected = pkg.sampled_reference(
            ws, *weights, capacity, sample_size=min(capacity, 32)
        )
        assert (
            pkg.output_error(ws.output_activation[:capacity], indices, expected) < 0.025
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            thunk()
        for state in ("holes", "empty", "last", "restore"):
            valid = torch.ones_like(ids, dtype=torch.bool)
            if state == "holes":
                valid.view(-1)[::3] = False
            if state in ("empty", "last"):
                valid.zero_()
            if state == "last":
                valid[-1, -1] = True
            ws.topk_idx[:capacity].copy_(ids).masked_fill_(~valid, -1)
            ws.topk_weights[:capacity].copy_(scores).masked_fill_(~valid, float("nan"))
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            indices, expected = pkg.sampled_reference(
                ws, *weights, capacity, sample_size=min(capacity, 32)
            )
            assert (
                pkg.output_error(ws.output_activation[:capacity], indices, expected)
                < 0.025
            ), state
    finally:
        if graph is not None:
            graph.reset()
        ws.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("scalar", ["fc1_alpha", "fc2_alpha", "fc1_norm_const"])
def test_blackwell_standard_global_scales(kind, scalar, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    import cutlass.cute as cute

    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    cfg = pkg.Sm100BlockScaledMoeConfig(
        num_total_experts=8,
        max_tokens_per_rank=1,
        num_topk=3,
        hidden=384,
        intermediate=192,
        rank=0,
        world_size=1,
        quant_kind=kind,
    )
    ws = pkg.Sm100BlockScaledSymmBuffer(cfg)
    try:
        weights = pkg.preprocess_block_scaled_weights(
            torch.zeros(8, 384, 384, device="cuda", dtype=torch.bfloat16),
            torch.zeros(8, 384, 192, device="cuda", dtype=torch.bfloat16),
            quant_kind=kind,
            intermediate_size=192,
        )
        kwargs = ws._runtime_kwargs(*weights)
        if kind == "nvfp4":
            kwargs[scalar] = None
            message = f"requires {scalar}"
        else:
            from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.block_scaled import (
                _to_cute,
            )

            scale = torch.ones(8, device="cuda", dtype=torch.float32)
            kwargs[scalar] = _to_cute(scale, assumed_align=4)
            message = f"{scalar} must be None"
        with pytest.raises(Exception, match=message):
            cute.compile(ws.kernel, **kwargs)
    finally:
        ws.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("persistent", [False, True])
def test_blackwell_standard_nvfp4_padded_aot(monkeypatch, persistent):
    """Compile a padded NVFP4 workspace with default or persistent finalization."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg
    from functools import partial
    import cutlass.torch as cutlass_torch

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setattr(
        cutlass_torch,
        "from_dlpack",
        partial(cutlass_torch.from_dlpack, enable_tvm_ffi=True),
    )
    cfg = pkg.Sm100BlockScaledMoeConfig(
        num_total_experts=8,
        max_tokens_per_rank=1,
        num_topk=3,
        hidden=384,
        intermediate=96,
        rank=0,
        world_size=1,
        quant_kind="nvfp4",
        use_persistent_finalize_kernel=persistent,
        enable_fc1_activation_per_token_scale=persistent,
    )
    ws = pkg.Sm100BlockScaledSymmBuffer(
        cfg,
        num_valid_tokens_tensor=(
            torch.ones(1, dtype=torch.int32, device="cuda") if persistent else None
        ),
    )
    try:
        loaded = ws.kernel.aot_compile()
        weights = pkg.preprocess_block_scaled_weights(
            torch.zeros(8, 192, 384, device="cuda", dtype=torch.bfloat16),
            torch.zeros(8, 384, 96, device="cuda", dtype=torch.bfloat16),
            quant_kind="nvfp4",
            intermediate_size=96,
        )
        loaded(**ws._runtime_kwargs(*weights))
        torch.cuda.synchronize()
        assert torch.count_nonzero(ws.output_activation[:1]) == 0
    finally:
        ws.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
def test_blackwell_standard_max_topk(kind, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    torch.manual_seed(7731)
    cfg = pkg.Sm100BlockScaledMoeConfig(
        num_total_experts=512,
        max_tokens_per_rank=1,
        num_topk=256,
        hidden=384,
        intermediate=192,
        rank=0,
        world_size=1,
        quant_kind=kind,
    )
    ws = pkg.Sm100BlockScaledSymmBuffer(cfg)
    graph = None
    try:
        weights = pkg.preprocess_block_scaled_weights(
            (torch.randn(512, 384, 384, device="cuda") / 384**0.5).bfloat16(),
            (torch.randn(512, 384, 192, device="cuda") / 192**0.5).bfloat16(),
            quant_kind=kind,
            intermediate_size=192,
        )
        x = torch.randn(1, 384, device="cuda")
        data, scales = (
            pkg.quantize_nvfp4_block16(x)
            if kind == "nvfp4"
            else pkg.quantize_mxfp8_block32(
                x, torch.float8_e5m2 if kind == "mxfp8_e5m2" else torch.float8_e4m3fn
            )
        )
        ws.x[0].view(torch.uint8).copy_(data[0].view(torch.uint8))
        ws.x_sf[0, : scales.shape[1]].view(torch.uint8).copy_(
            scales[0].view(torch.uint8)
        )
        ws.topk_idx[0].copy_(torch.arange(256, device="cuda", dtype=torch.int32))
        ws.topk_weights[0].fill_(1 / 256)
        ws.note_staged_tokens(1)
        thunk = pkg.sm100_block_scaled_mega_launch_thunk(*weights, ws)
        thunk()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            thunk()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        indices, expected = pkg.sampled_reference(ws, *weights, 1, sample_size=1)
        assert pkg.output_error(ws.output_activation[:1], indices, expected) < 0.025
    finally:
        if graph is not None:
            graph.reset()
        ws.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("fused", [True, False])
def test_blackwell_standard_backend_staging(kind, fused, monkeypatch):
    """Exercise registered dispatch, shared staging, and repeated batch sizes."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1" if fused else "0")
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    nvfp4 = kind == "nvfp4"
    config_type = (
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        if nvfp4
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    )
    config = config_type(
        intermediate_size=192,
        top_k=3,
        kernel_name="sm100_"
        + ("nvfp4_nvfp4" if nvfp4 else "mxfp8_mxfp8")
        + "_bf16_cutedsl_next",
        **({} if nvfp4 else {"kind": kind}),
    )
    backend = create_mega_kernel(config)
    bootstrap = moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    fleet = moe_ep.FleetParams(
        num_experts=8, max_tokens_per_rank=129, token_hidden_size=384
    )
    backend.validate_init(bootstrap, fleet)
    torch.manual_seed(8821)
    weights = backend.preprocess_weights(
        moe_ep.MoEWeightPack(
            (torch.randn(8, 384, 384, device="cuda") / 384**0.5).bfloat16(),
            (torch.randn(8, 384, 192, device="cuda") / 192**0.5).bfloat16(),
        ),
        fleet,
    )
    backend.validate_transformed_weights(weights, bootstrap, fleet)
    ws = backend.prepare_workspace(bootstrap, fleet)
    try:
        for count in (129, 1, 65, 0, 17):
            inputs = moe_ep.MoEEpTensors(
                hidden_states=torch.randn(
                    count, 384, device="cuda", dtype=torch.bfloat16
                ),
                topk_ids=torch.arange(
                    count * 3, device="cuda", dtype=torch.int64
                ).reshape(-1, 3)
                % 8,
                topk_weights=torch.full((count, 3), 1 / 3, device="cuda"),
            )
            backend.validate_forward(inputs, fleet, quantize_input=True)
            backend.stage_inputs(inputs, ws, quantize_input=True)
            output = backend.compute(ws, weights, output=None)
            torch.cuda.synchronize()
            assert output.shape == (count, 384)
            assert (ws.topk_idx[count:] == -1).all()
            if count:
                indices, expected = pkg.sampled_reference(
                    ws, *weights, count, sample_size=min(count, 32)
                )
                assert pkg.output_error(output, indices, expected) < 0.025
    finally:
        backend.destroy(ws)


def _run_blackwell_genphase_graph(
    rank, world, capacity, early, tile_n, cluster_m, kind="mxfp8_e4m3"
):
    """Native reference plus changing-route graph replay for the GenPhase prototype."""
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as genphase

    hidden, intermediate, topk, experts = 384, 192, 3, 8 * world
    intermediate = 256 if kind == "mxfp4_mxfp8" else intermediate
    torch.manual_seed(6447 + rank)
    w13 = (
        torch.randn(8, 2 * intermediate, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(8, hidden, intermediate, device="cuda") / intermediate**0.5
    ).bfloat16()
    weights = pkg.preprocess_genphase_weights(
        w13, w2, quant_kind=kind, intermediate_size=intermediate
    )
    cfg = pkg.Sm100GenphaseMoeConfig(
        num_total_experts=experts,
        max_tokens_per_rank=capacity,
        num_topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world,
        apply_routing_weights_before_fc2=early,
        quant_kind=kind,
        mma_tiler_mnk=(256, tile_n, 256 if kind == "nvfp4" else 128),
        cluster_shape_mn=(cluster_m, 1),
    )
    ws = pkg.Sm100GenphaseSymmBuffer(cfg)
    try:
        source = torch.randn(capacity, hidden, device="cuda")
        x, sf = (
            genphase.quantize_nvfp4_block16(source)
            if kind == "nvfp4"
            else pkg.quantize_genphase_mxfp8(
                source,
                torch.float8_e5m2 if kind == "mxfp8_e5m2" else torch.float8_e4m3fn,
            )
        )
        ws.x[:capacity].view(torch.uint8).copy_(x.view(torch.uint8))
        ws.x_sf[:capacity, : sf.shape[1]].view(torch.uint8).copy_(sf.view(torch.uint8))
        ids = torch.arange(capacity * topk, device="cuda", dtype=torch.int32).view(
            capacity, topk
        )
        ids = (ids + rank * 7) % experts
        scores = torch.linspace(0.1, 0.9, capacity * topk, device="cuda").view(
            capacity, topk
        )
        ws.topk_idx[:capacity].copy_(ids)
        ws.topk_weights[:capacity].copy_(scores)
        thunk = pkg.sm100_genphase_mega_launch_thunk(*weights, ws)
        thunk()
        torch.cuda.synchronize()
        indices, expected = genphase.sampled_reference(
            ws, *weights, capacity, sample_size=min(capacity, 32)
        )
        error = torch.tensor(
            genphase.output_error(ws.output_activation, indices, expected),
            device="cuda",
        )
        if world > 1:
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
        assert torch.isfinite(error) and error.item() < 0.02
        dense_terms = (
            ws._pre_reduced_activation.view(torch.bfloat16)[:capacity].float().clone()
        )
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            thunk()
        torch.cuda.current_stream().wait_stream(stream)
        for state in ("empty", "last", "holes", "asymmetric", "restore"):
            valid = torch.ones_like(ids, dtype=torch.bool)
            if state in ("empty", "last"):
                valid.zero_()
            if state == "last":
                valid[-1, -1] = True
            if state == "holes":
                valid = (
                    torch.arange(ids.numel(), device="cuda").reshape_as(ids) % 3 != 0
                )
            if state == "asymmetric":
                valid[min(rank, capacity) :].zero_()
            ws.topk_idx[:capacity].copy_(ids).masked_fill_(~valid, -1)
            ws.topk_weights[:capacity].copy_(scores).masked_fill_(~valid, float("nan"))
            control = torch.zeros((capacity, hidden), device="cuda")
            for slot in range(topk):
                term = torch.where(valid[:, slot, None], dense_terms[:, slot], 0)
                if not early:
                    # BF16 terms and FP32 scores have an exact FP64 product.
                    # Round only after adding, matching the reducer's FP32 FMA.
                    control = (
                        control.double()
                        + term.double() * scores[:, slot, None].double()
                    ).float()
                else:
                    control += term
            control = control.bfloat16()
            ws._pre_reduced_activation.fill_(255)
            if world > 1:
                dist.barrier()
            for replay in range(65):
                graph.replay()
                if replay in (0, 64):
                    torch.cuda.synchronize()
                    failed = torch.tensor(
                        not torch.equal(ws.output_activation[:capacity], control),
                        device="cuda",
                        dtype=torch.int32,
                    )
                    if world > 1:
                        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                    assert failed.item() == 0, (rank, state, replay)
        assert ws.kernel.architecture == "sm_100"
        assert ws.kernel._mainloop.tmem_plan.allocation_columns == 512
    finally:
        ws.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("capacity", [1, 65, 129, 1024])
@pytest.mark.parametrize("early", [True, False])
@pytest.mark.parametrize("tile_n,cluster_m", [(128, 4), (64, 4), (64, 2)])
def test_blackwell_genphase_graph(monkeypatch, capacity, early, tile_n, cluster_m):
    _require_cuda()
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Requires SM100/103")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run_blackwell_genphase_graph(0, 1, capacity, early, tile_n, cluster_m)


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


def _bf16_ulp(x: float) -> float:
    """bf16 spacing at magnitude ``x`` (8 significand bits incl. the hidden one)."""
    import math

    if x <= 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(x)) - 7)


def _oracle_atol(amax_ref: float) -> float:
    """Absolute tolerance for kernel-vs-oracle compares.

    ``2e-3 * amax`` covers NVFP4 RTNE flips at fc1-out plus GEMM accumulation-
    order noise (measured rel_l2 <= 3e-3).  The extra ``2 * bf16_ulp(amax)``
    covers one bf16 rounding flip of a per-(token, topk) fc2 term: both the
    kernel and the oracle store each term in bf16 before the top-k sum, so an
    fp32 accumulation-order difference at a rounding boundary shows up as one
    term ulp even where the reduced output is small (terms cancel), which the
    relative part of the compare cannot absorb.
    """
    return 2e-3 * amax_ref + 2.0 * _bf16_ulp(amax_ref)


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
    activation="swiglu",
    situ_beta=None,
    situ_linear_beta=None,
    term_transform=None,
    swiglu_alpha=None,
    swiglu_beta=None,
    fc1_activation_per_token_scale=None,  # (T,) fp32 or None
    fc1_alpha=None,  # (E,) fp32 per-expert or None (identity)
    fc2_alpha=None,  # (E,) fp32 per-expert or None (identity)
    fc1_norm_const=None,  # (E,) fp32 per-expert or None (1.0)
):
    """Pure-torch NVFP4 MegaMoE oracle (routing weights applied before FC2).

    Mirrors the kernel's data path — dequant → fp32 fc1 GEMM → 16-interleaved
    gated activation → per-token topk weight folded in BEFORE the fc1-out
    NVFP4 round-trip → fp32 fc2 GEMM → bf16 terms — so disagreement is
    bounded by NVFP4 RTNE flips at fc1-out plus GEMM accumulation-order noise.

    ``term_transform``, when set, is applied to each per-(token, topk) fc2
    output term before the topk sum; the multirank oracle uses it to model the
    quantized cross-rank combine wire (``combine_roundtrip_to_fp32``).

    ``fc1_activation_per_token_scale``, when set, multiplies token ``t``'s
    dequantized fc1 output by ``scale[t]`` BEFORE the clamp / activation --
    the kernel's ``enable_fc1_activation_per_token_scale`` fold
    (``real = fc1_alpha * scale_t * acc``).

    ``fc1_alpha`` / ``fc2_alpha`` / ``fc1_norm_const`` model the per-expert
    epilogue scalars with the kernel's semantics: ``fc1_alpha[e]`` is folded
    into the same single fp32 multiply as the per-token scale, the fc1-out
    NVFP4 round-trip uses ``fc1_norm_const[e]`` (host quantizer mirrors the
    kernel: SF = amax/6*c, data = y*c/SF, so the dequant is ``y*c`` and is NOT
    divided back), and ``fc2_alpha[e]`` scales the fp32 fc2 accumulator
    before the bf16 store.  ``None`` keeps the identity path.
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
        # Dequant fold: one fp32 multiply by ``fc1_alpha[e] * scale_t`` (the
        # kernel forms the product first, then multiplies the accumulator once),
        # applied BEFORE the clamp / activation.
        dequant = None
        if fc1_alpha is not None:
            dequant = fc1_alpha[expert].to(torch.float32)
        if fc1_activation_per_token_scale is not None:
            per_token = fc1_activation_per_token_scale[tokens].to(torch.float32)
            dequant = per_token if dequant is None else dequant * per_token
        if dequant is not None:
            if dequant.dim() == 0:
                fc1_out = fc1_out * dequant
            else:
                fc1_out = fc1_out * dequant.unsqueeze(-1)

        # Gated activation over the kernel's 16-column gate/up interleave.
        m = fc1_out.shape[0]
        n_pairs = fc1_out.shape[1] // (2 * NVFP4_BLOCK)
        reshaped = fc1_out.view(m, n_pairs, 2, NVFP4_BLOCK)
        gate = reshaped[:, :, 0, :]
        up = reshaped[:, :, 1, :]
        if activation == "situ":
            gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
            if situ_linear_beta is not None:
                up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
        else:
            if gate_up_clamp is not None:
                limit = abs(float(gate_up_clamp))
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            alpha = 1.0 if swiglu_alpha is None else swiglu_alpha
            beta = 0.0 if swiglu_beta is None else swiglu_beta
            gate = gate * torch.sigmoid(alpha * gate)
            up = up + beta
        activated = (gate * up).reshape(m, intermediate)

        # Weight folded in before the fp4 round-trip.
        # (post-hoc weighting would NOT match — quant changes the magnitude).
        activated = activated * topk_weights[tokens, slots].unsqueeze(-1)

        norm_const = 1.0 if fc1_norm_const is None else float(fc1_norm_const[expert])
        fc1_q, fc1_q_sf = nvfp4_quantize_per_block_16(activated, norm_const)
        # With a norm const the dequant is ``activated * norm_const`` (in
        # range); the kernel does not divide it back -- the caller folds 1/c
        # into fc2_alpha on the host.
        activated_rt = _dequant_nvfp4(fc1_q, fc1_q_sf, logical_cols=intermediate)

        fc2_w = _dequant_nvfp4(
            fc2_weight[expert], fc2_sf[expert], logical_cols=intermediate
        )  # (hidden, I)
        fc2_out = activated_rt @ fc2_w.transpose(0, 1)
        if fc2_alpha is not None:
            fc2_out = fc2_out * fc2_alpha[expert].to(torch.float32)
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
    "mode,tile_n,in_kernel_fc2_reduce,token_back_mode,apply_routing_weights_before_fc2",
    [
        (mode, 128, False, token_back_mode, apply_routing_weights_before_fc2)
        for mode in NVFP4_MODES
        for token_back_mode in ("epi_warps", "reuse_dispatch_warps")
        for apply_routing_weights_before_fc2 in (
            [False, True] if mode == "w4a16" else [False]
        )
    ]
    + [
        (
            "w4a16",
            tile_n,
            True,
            "reuse_dispatch_warps",
            apply_routing_weights_before_fc2,
        )
        for tile_n in (64, 128)
        for apply_routing_weights_before_fc2 in (False, True)
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
    apply_routing_weights_before_fc2,
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
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
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
        apply_routing_weights_before_fc2=False,
        hidden=2048,
        intermediate=1024,
        num_experts=4,
        topk=4,
        activation_clamp=activation_clamp,
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("token_back_mode", ["epi_warps", "reuse_dispatch_warps"])
@pytest.mark.parametrize(
    "hidden,intermediate,num_experts,topk",
    [
        pytest.param(2048, 1024, 4, 4, id="kimi-situ-e4"),
        pytest.param(3584, 3072, 56, 16, id="kimi-dep16-situ"),
    ],
)
def test_nvfp4_w4a4_situ_parameters(
    monkeypatch, token_back_mode, hidden, intermediate, num_experts, topk
):
    """Keep the upstream SiTU geometries independent of the W4A16 tactic matrix."""
    _check_nvfp4_kernel_matches_torch_reference(
        monkeypatch,
        mode="w4a4",
        tile_n=128,
        in_kernel_fc2_reduce=False,
        token_back_mode=token_back_mode,
        apply_routing_weights_before_fc2=False,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )


def _check_nvfp4_kernel_matches_torch_reference(
    monkeypatch,
    *,
    mode,
    tile_n,
    in_kernel_fc2_reduce,
    token_back_mode,
    apply_routing_weights_before_fc2,
    hidden,
    intermediate,
    num_experts,
    topk,
    activation_clamp="standard",
    activation="swiglu",
    situ_beta=None,
    situ_linear_beta=None,
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
    gate_up_clamp = problem["gate_up_clamp"] if activation == "swiglu" else None

    if mode == "w4a4":
        pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    elif mode == "w4a16":
        from tests.test_helpers.w4a16_reference import make_w4a16_weights

        pack = make_w4a16_weights(hidden, intermediate, num_experts)
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        **({"gate_up_clamp": gate_up_clamp} if mode == "w4a4" else {}),
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
        gate_up_clamp=gate_up_clamp,
        enable_in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        fc1_alpha=alpha1,
        fc2_alpha=alpha2,
        knobs=knobs,
        **(
            {"apply_routing_weights_before_fc2": (apply_routing_weights_before_fc2)}
            if mode == "w4a16"
            else dict(
                swiglu_alpha=activation_pairs[0][0],
                swiglu_beta=activation_pairs[0][1],
                activation=activation,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
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
                    gate_up_clamp=gate_up_clamp,
                    activation=activation,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
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
                    apply_routing_weights_before_fc2=(apply_routing_weights_before_fc2),
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
                    gate_up_clamp=gate_up_clamp,
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
            from tests.test_helpers.w4a16_reference import assert_w4a16_bits

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
    apply_routing_weights_before_fc2=False,
    in_kernel_fc2_reduce=False,
):
    """Shared oracle input assembly from canonical prequantized weight packs."""
    from flashinfer.moe_ep import PrequantizedMoEWeights

    assert mode in NVFP4_MODES, mode
    assert isinstance(weights, PrequantizedMoEWeights), type(weights)
    if mode == "w4a16":
        from tests.test_helpers.w4a16_reference import w4a16_reference

        return w4a16_reference(
            problem,
            weights,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        )
    assert mode == "w4a4"
    assert fc1_alpha is None and fc2_alpha is None and fc1_norm_const is None
    assert not apply_routing_weights_before_fc2
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

    from tests.test_helpers.w4a16_reference import assert_w4a16_bits

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


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "tokens,hidden,topk", [(1, 256, 4), (33, 520, 17), (33, 1024, 16)]
)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_masked_bf16_combine_graph(tokens, hidden, topk, weighted, index_dtype):
    """Check holes, poisoned scores, empty rows and dense restoration exactly."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires SM100 or newer packed FP32 arithmetic")

    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    experts = 128
    # Exercise 16-byte alignment without assuming a 32-byte-aligned allocation.
    terms = torch.empty(
        tokens * topk * hidden + 8, device="cuda", dtype=torch.bfloat16
    )[8:].reshape(tokens, topk, hidden)
    output = torch.empty(tokens * hidden + 8, device="cuda", dtype=torch.bfloat16)[
        8:
    ].reshape(tokens, hidden)
    ids = torch.zeros((tokens, topk), device="cuda", dtype=index_dtype)
    scores = torch.ones((tokens, topk), device="cuda", dtype=torch.float32)
    reducer = pkg.TopkReduce(
        hidden,
        topk,
        pkg.CombineFormat.parse("bf16"),
        sm_arch=pkg.get_cutedsl_target_arch(),
        num_experts=experts,
    )
    kwargs = dict(
        combine_quant=from_dlpack(terms, assumed_align=16),
        combine_sf=None,
        reduced_output=from_dlpack(output, assumed_align=16),
        topk_score=(
            from_dlpack(scores, assumed_align=4).mark_layout_dynamic(leading_dim=1)
            if weighted
            else None
        ),
        topk_indices=from_dlpack(ids, assumed_align=4).mark_layout_dynamic(
            leading_dim=1
        ),
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    executable = cute.compile(reducer, **kwargs)
    terms.fill_(1)
    executable(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            kwargs["stream"] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            executable(**kwargs)
        base = torch.arange(tokens * topk, device="cuda").reshape_as(ids)
        for iteration, state in enumerate(
            ("dense", "holes", "empty", "last", "restore")
        ):
            ids.copy_((base + iteration) % experts)
            if state == "holes":
                ids.view(-1)[::3] = -1
                ids.view(-1)[1::3] = experts
            elif state == "empty":
                ids.fill_(torch.iinfo(index_dtype).min)
            elif state == "last":
                ids.fill_(torch.iinfo(index_dtype).max)
                ids[-1, -1] = experts - 1
            valid = (ids >= 0) & (ids < experts)
            values = (
                (
                    torch.arange(terms.numel(), device="cuda").reshape_as(terms)
                    + iteration
                )
                % 9
                - 4
            ).to(torch.bfloat16)
            terms.copy_(values).masked_fill_(~valid.unsqueeze(-1), float("nan"))
            scores.copy_(((base % 4).float() + 1) * 0.25)
            scores.masked_fill_(~valid, float("nan"))
            expected = torch.zeros((tokens, hidden), device="cuda", dtype=torch.float32)
            for k in range(topk):
                value = torch.where(valid[:, k, None], values[:, k].float(), 0.0)
                if weighted:
                    value *= torch.where(valid[:, k, None], scores[:, k, None], 0.0)
                expected += value
            for replay in range(17):
                graph.replay()
                if replay in (0, 16):
                    torch.cuda.synchronize()
                    assert torch.isfinite(output).all(), state
                    torch.testing.assert_close(
                        output, expected.bfloat16(), rtol=0, atol=0
                    )
    finally:
        graph.reset()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("capacity", [129, 1024, 2048])
@pytest.mark.parametrize("dtype", ["nvfp4", "mxfp8"])
def test_sparse_dispatch_workspace(capacity, dtype, monkeypatch):
    """The real frontend includes publication state in the late reset prefix."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for the frontend workspace")
    import cutlass.cute as cute
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    create = (
        pkg.create_dummy_nvfp4_inputs
        if dtype == "nvfp4"
        else pkg.create_dummy_mxfp8_inputs
    )
    make_thunk = (
        pkg.nvfp4_mega_launch_thunk if dtype == "nvfp4" else pkg.mxfp8_mega_launch_thunk
    )
    _, w1, w2, buffer = create(0, 1, 4, capacity, 1, 4, 512, 256, seed=6447)
    # Construct the real layout/arguments without executing architecture-specific MMA.
    monkeypatch.setattr(cute, "compile", lambda kernel, **kwargs: lambda **args: None)
    try:
        make_thunk(w1, w2, buffer)
        kernel = buffer._frontend._mega.kernel
        if capacity <= 1024:
            region = kernel._local_region_by_name["dispatch_route_extent"]
            offset = kernel._local_offsets["dispatch_route_extent"]
            assert region.shape == (2,)
            assert offset % 16 == 0
            assert offset + region.nbytes <= kernel.local_zero_i32_count * 4
            grid = kernel._local_offsets["grid_sync_counter"]
            assert (
                offset
                >= grid + kernel._local_region_by_name["grid_sync_counter"].nbytes
            )
        else:
            assert "dispatch_route_extent" not in kernel._local_offsets
    finally:
        buffer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "hidden,intermediate,num_experts,topk,num_tokens,activation,fc1_alpha",
    [
        # 32 tokens routed to every expert: exactly the first (pre-transpose,
        # shuffle-fed) half of one 64-token epilogue subtile per expert.
        pytest.param(2048, 1024, 4, 4, 32, "standard", None, id="regular-e4"),
        # 128-misaligned hidden: ceil-div K tail + predicated epilogue.
        pytest.param(2880, 2880, 4, 4, 32, "standard", None, id="tail-e4"),
        pytest.param(2048, 1024, 4, 4, 32, "no-clamp", None, id="no-clamp"),
        # MiniMax-style (1.702, 1.0) activation with a tight clamp: mixed
        # saturated / unsaturated gate-up values, so the pre-activation
        # position of the scale is observable.
        pytest.param(2048, 1024, 4, 4, 32, "minimax", None, id="minimax-clamp"),
        # ~96 tokens per expert (192 tokens, topk 2 of 4): a full 64-token
        # subtile (both the shuffle-fed 0..31 half and the one-scale-per-lane
        # 32..63 half carry valid tokens) plus a partial second subtile.
        pytest.param(2048, 1024, 4, 2, 192, "standard", None, id="two-subtiles"),
        # Same routing with a non-identity per-expert fc1_alpha: exercises the
        # ``alpha * scale_t`` fold on both halves (the oracle folds 2*scale).
        pytest.param(2048, 1024, 4, 2, 192, "standard", 2.0, id="alpha-fold"),
        # Singleton expert (static expert extent 1) with 3 subtiles of tokens.
        pytest.param(2048, 1024, 1, 1, 192, "standard", None, id="singleton-e1"),
    ],
)
def test_nvfp4_kernel_per_token_scale_matches_torch_reference(
    monkeypatch,
    hidden,
    intermediate,
    num_experts,
    topk,
    num_tokens,
    activation,
    fc1_alpha,
):
    """``enable_fc1_activation_per_token_scale``: the kernel folds a runtime
    ``(T,)`` fp32 per-token scale into the fc1 dequant before the clamp /
    activation, matching the pure-torch oracle with the same fold.

    The epilogue applies the scale on two register schedules per 64-token
    subtile: tokens 0..31 are activated before the register transpose (scale
    fetched per register via warp shuffle) and tokens 32..63 after it (one
    scale per lane).  The 32-token cases cover the first half only; the
    192-token cases put valid tokens on both halves and across subtile
    boundaries.  Random {0.5, 1, 1.5, 2} scales differ per token, so a
    lane/token mix-up shows up as a large error.  ``fc1_alpha`` (broadcast
    per-expert scalar) checks the ``alpha * scale_t`` fold: the oracle has no
    alpha leg, so it is fed ``alpha * scale`` instead (exact in fp32).
    """
    _require_cuda()

    import torch

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(
            f"nvfp4_mega_moe requires sm_100a or sm_103a; got sm_{cap[0]}{cap[1]}"
        )
    pytest.importorskip("triton")

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        num_tokens=num_tokens,
        max_tokens=max(64, num_tokens),
    )
    swiglu_alpha, swiglu_beta = None, None
    if activation == "no-clamp":
        problem["gate_up_clamp"] = None
    elif activation == "minimax":
        problem["gate_up_clamp"] = 0.5
        problem["hidden_states"].mul_(0.1)
        problem["w13"].mul_(0.1)
        swiglu_alpha, swiglu_beta = 1.702, 1.0
    rank = 0
    world_size = 1
    num_tokens = problem["num_tokens"]

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )
    fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)

    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        rank,
        world_size,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        gate_up_clamp=problem["gate_up_clamp"],
        fc1_alpha=fc1_alpha,
        enable_fc1_activation_per_token_scale=True,
    )
    try:
        scale_buf = symm_buffer.fc1_activation_per_token_scale
        assert scale_buf is not None
        assert scale_buf.shape == (problem["max_tokens"],)
        assert scale_buf.dtype == torch.float32
        # Allocation-time default is the neutral scale.
        assert torch.equal(scale_buf, torch.ones_like(scale_buf))

        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )
        # Kernel-team tester value set {0.5, 1.0, 1.5, 2.0}: exactly
        # representable, per-token distinct, and far from the neutral 1.0.
        g = torch.Generator(device="cuda").manual_seed(23)
        scale = (
            torch.randint(1, 5, (num_tokens,), generator=g, device="cuda").to(
                torch.float32
            )
            * 0.5
        )
        scale_buf[:num_tokens].copy_(scale)

        ref_kwargs = dict(
            act_packed=symm_buffer.x[:num_tokens],
            act_sf=symm_buffer.x_sf[:num_tokens],
            topk_idx=symm_buffer.topk_idx[:num_tokens],
            topk_weights=symm_buffer.topk_weights[:num_tokens],
            fc1_weight=fc1_plain,
            fc1_sf=fc1_sf,
            fc2_weight=fc2_plain,
            fc2_sf=fc2_sf,
            hidden=problem["hidden"],
            intermediate=problem["intermediate"],
            gate_up_clamp=problem["gate_up_clamp"],
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
        # The oracle has no fc1_alpha leg: fold the (power-of-two) broadcast
        # alpha into the per-token scale it is fed -- exact in fp32.
        ref_scale = scale if fc1_alpha is None else scale * fc1_alpha
        y_ref = _torch_nvfp4_mega_reference(
            **ref_kwargs, fc1_activation_per_token_scale=ref_scale
        )
        y_ref_unscaled = _torch_nvfp4_mega_reference(**ref_kwargs)

        y_kernel = torch.empty(
            num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        nvfp4_mega_moe(
            y_kernel,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=problem["gate_up_clamp"],
        )
        torch.cuda.synchronize()

        assert torch.isfinite(y_kernel).all()
        yk = y_kernel.to(torch.float32)
        yr = y_ref.to(torch.float32)
        yu = y_ref_unscaled.to(torch.float32)
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        rel_l2_unscaled = (yk - yu).norm() / yu.norm().clamp_min(1e-6)
        print(
            f"[nvfp4 per-token-scale oracle {activation} tokens={num_tokens} "
            f"experts={num_experts} topk={topk} alpha={fc1_alpha}] "
            f"rel_l2={rel_l2.item():.4g} "
            f"rel_l2(vs unscaled ref)={rel_l2_unscaled.item():.4g} "
            f"max|d|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        # Same tolerance model as the unscaled oracle test (see _oracle_atol):
        # NVFP4 RTNE flips at fc1-out + accumulation-order noise + one bf16
        # term-rounding flip.
        atol = _oracle_atol(yr.abs().max().item())
        torch.testing.assert_close(yk, yr, atol=atol, rtol=0.05)
        assert rel_l2.item() < 0.02
        # The check has teeth: a kernel that ignored the scale (or applied it
        # to the wrong tokens) would sit far from the scaled reference.
        assert rel_l2_unscaled.item() > 0.05, (
            "per-token scale had no observable effect -- test is not discriminating"
        )

        if num_tokens == 192 and fc1_alpha is None and activation == "standard":
            # Launch-cache path: a contiguous scale compiles/launches fine, then
            # a strided re-view of the SAME storage (same data_ptr) must be
            # rejected by run() rather than hit the cached launch arguments.
            from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
                MegaMoENvfp4Inputs,
            )

            max_tokens = problem["max_tokens"]
            storage = torch.ones(2 * max_tokens, dtype=torch.float32, device="cuda")
            storage[:max_tokens].copy_(scale_buf)

            def inputs_with(scale_view):
                return MegaMoENvfp4Inputs(
                    activation=symm_buffer.x,
                    activation_sf=symm_buffer.x_sf,
                    topk_idx=symm_buffer.topk_idx,
                    topk_weights=symm_buffer.topk_weights,
                    fc1_weight=transformed_l1[0],
                    fc1_weight_sf=transformed_l1[1],
                    fc2_weight=transformed_l2[0],
                    fc2_weight_sf=transformed_l2[1],
                    fc1_alpha=symm_buffer.fc1_alpha,
                    fc2_alpha=symm_buffer.fc2_alpha,
                    fc1_norm_const=symm_buffer.fc1_norm_const,
                    output_activation=symm_buffer.output_activation,
                    fc1_activation_per_token_scale=scale_view,
                )

            frontend = symm_buffer._frontend
            y_contig = frontend.run(inputs_with(storage[:max_tokens]), sync=True)
            torch.testing.assert_close(
                y_contig[:num_tokens].to(torch.float32), yk, atol=0.0, rtol=0.0
            )
            strided = storage[::2]
            assert strided.data_ptr() == storage[:max_tokens].data_ptr()
            with pytest.raises(ValueError, match="contiguous"):
                frontend.run(inputs_with(strided), sync=True)
    finally:
        symm_buffer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "epilogue",
    [
        # Random per-expert fc1_alpha / fc2_alpha / fc1_norm_const (the same
        # value sets as make_dummy_epilogue_params) on top of the per-token
        # scale: checks the alpha*scale_t fold, the norm-const SF shift and
        # the fc2 alpha against an independent reference.
        pytest.param("random", id="random-scalars"),
        # Deployment recipe from the kernel-team design: quantize the fc1
        # output with norm const 16 (SF shifted up 4 binades) and fold the
        # 1/16 into fc2_alpha on the host.  Small-magnitude MiniMax-style
        # values keep every SF inside the e4m3 normal range, where this must
        # be indistinguishable from norm const 1 with the unfolded fc2_alpha.
        pytest.param("deploy-norm16", id="deploy-norm16"),
    ],
)
def test_nvfp4_kernel_per_token_scale_with_epilogue_scalars(monkeypatch, epilogue):
    """Per-token scale combined with non-identity per-expert epilogue scalars,
    vs the pure-torch oracle carrying the same scalars."""
    _require_cuda()

    import torch

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(
            f"nvfp4_mega_moe requires sm_100a or sm_103a; got sm_{cap[0]}{cap[1]}"
        )
    pytest.importorskip("triton")

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        make_dummy_epilogue_params,
        nvfp4_mega_moe,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    num_experts, topk, num_tokens = 4, 2, 192
    problem = _single_rank_problem(
        num_experts=num_experts, topk=topk, num_tokens=num_tokens, max_tokens=192
    )
    swiglu_alpha, swiglu_beta = None, None
    g = torch.Generator(device="cuda").manual_seed(29)
    fc1_alpha, fc2_alpha, fc1_norm_const = make_dummy_epilogue_params(
        num_experts, generator=g
    )
    if epilogue == "deploy-norm16":
        problem["gate_up_clamp"] = 0.5
        problem["hidden_states"].mul_(0.1)
        problem["w13"].mul_(0.1)
        swiglu_alpha, swiglu_beta = 1.702, 1.0
        fc2_alpha_unfolded = fc2_alpha
        fc1_norm_const = torch.full_like(fc1_norm_const, 16.0)
        fc2_alpha = fc2_alpha_unfolded / 16.0  # exact (power of two)

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )
    fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)

    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        0,
        1,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        gate_up_clamp=problem["gate_up_clamp"],
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
        enable_fc1_activation_per_token_scale=True,
    )
    try:
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )
        scale = (
            torch.randint(1, 5, (num_tokens,), generator=g, device="cuda").to(
                torch.float32
            )
            * 0.5
        )
        symm_buffer.fc1_activation_per_token_scale[:num_tokens].copy_(scale)

        ref_kwargs = dict(
            act_packed=symm_buffer.x[:num_tokens],
            act_sf=symm_buffer.x_sf[:num_tokens],
            topk_idx=symm_buffer.topk_idx[:num_tokens],
            topk_weights=symm_buffer.topk_weights[:num_tokens],
            fc1_weight=fc1_plain,
            fc1_sf=fc1_sf,
            fc2_weight=fc2_plain,
            fc2_sf=fc2_sf,
            hidden=problem["hidden"],
            intermediate=problem["intermediate"],
            gate_up_clamp=problem["gate_up_clamp"],
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            fc1_activation_per_token_scale=scale,
            fc1_alpha=fc1_alpha,
        )
        # Exact kernel math: norm const in the SF, fc2_alpha applied as-is.
        y_ref = _torch_nvfp4_mega_reference(
            **ref_kwargs, fc2_alpha=fc2_alpha, fc1_norm_const=fc1_norm_const
        )

        y_kernel = torch.empty(
            num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        nvfp4_mega_moe(
            y_kernel,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=problem["gate_up_clamp"],
        )
        torch.cuda.synchronize()

        assert torch.isfinite(y_kernel).all()
        yk = y_kernel.to(torch.float32)
        yr = y_ref.to(torch.float32)
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        msg = (
            f"[nvfp4 per-token-scale + epilogue scalars {epilogue}] "
            f"rel_l2={rel_l2.item():.4g} max|d|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        atol = _oracle_atol(yr.abs().max().item())
        torch.testing.assert_close(yk, yr, atol=atol, rtol=0.05)
        assert rel_l2.item() < 0.02

        if epilogue == "deploy-norm16":
            # The recipe is a pure SF-exponent shift (+4 binades): for every
            # block whose SF stays inside the e4m3 normal range under both
            # norm consts it reproduces the norm-const-1 / unfolded-fc2_alpha
            # result bit for bit, so the two oracles differ only through the
            # blocks the shift is meant to help (SF subnormal / flushed at c=1)
            # or hurt (SF saturated at 448 for c=16; excluded here by the small
            # magnitudes).  Bound that residual instead of demanding equality.
            y_plain = _torch_nvfp4_mega_reference(
                **ref_kwargs, fc2_alpha=fc2_alpha_unfolded, fc1_norm_const=None
            )
            yp = y_plain.to(torch.float32)
            rel_l2_recipe = (yr - yp).norm() / yp.norm().clamp_min(1e-6)
            msg += f" recipe-vs-plain rel_l2={rel_l2_recipe.item():.3g}"
            assert rel_l2_recipe.item() < 0.02, (
                "norm16 + fc2_alpha/16 deviates from norm 1 + fc2_alpha by "
                f"rel_l2={rel_l2_recipe.item():.3g}: more than a few blocks left "
                "the e4m3 normal range"
            )
        print(msg)
    finally:
        symm_buffer.destroy()


def test_nvfp4_per_token_scale_forward_validation():
    """``MoEEpTensors.fc1_activation_per_token_scale`` must be present iff the
    kernel config enables it, and be ``(num_tokens,)`` fp32 CUDA."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        validate_fc1_activation_per_token_scale,
    )

    ok = torch.ones(8, dtype=torch.float32, device="cuda")
    validate_fc1_activation_per_token_scale(ok, num_tokens=8, enabled=True)
    validate_fc1_activation_per_token_scale(None, num_tokens=8, enabled=False)
    with pytest.raises(MoEEpConfigError, match="required"):
        validate_fc1_activation_per_token_scale(None, num_tokens=8, enabled=True)
    with pytest.raises(
        MoEEpConfigError, match="enable_fc1_activation_per_token_scale=False"
    ):
        validate_fc1_activation_per_token_scale(ok, num_tokens=8, enabled=False)
    with pytest.raises(MoEEpConfigError, match="shape"):
        validate_fc1_activation_per_token_scale(ok[:4], num_tokens=8, enabled=True)
    with pytest.raises(MoEEpConfigError, match="float32"):
        validate_fc1_activation_per_token_scale(
            ok.to(torch.bfloat16), num_tokens=8, enabled=True
        )
    with pytest.raises(MoEEpConfigError, match="CUDA"):
        validate_fc1_activation_per_token_scale(ok.cpu(), num_tokens=8, enabled=True)


def test_nvfp4_shim_per_token_scale_input_validation():
    """The shim rejects a per-token scale that disagrees with the session flag
    (present-but-disabled, missing-but-enabled, wrong shape / dtype) before
    any compile or launch."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        MegaMoENvfp4Config,
        MegaMoENvfp4Inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Frontend,
    )

    tokens, topk, experts, hidden, intermediate = 64, 2, 4, 256, 256
    base = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=tokens,
        num_topk=topk,
        num_total_experts=experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    dev = "cuda"
    e = experts

    def make_inputs(scale):
        return MegaMoENvfp4Inputs(
            activation=torch.zeros(
                tokens, hidden // 2, dtype=torch.uint8, device=dev
            ).view(torch.float4_e2m1fn_x2),
            activation_sf=torch.zeros(
                tokens, hidden // 16, dtype=torch.float8_e4m3fn, device=dev
            ),
            topk_idx=torch.zeros(tokens, topk, dtype=torch.int64, device=dev),
            topk_weights=torch.zeros(tokens, topk, dtype=torch.float32, device=dev),
            fc1_weight=torch.zeros(
                e, hidden // 2, intermediate, dtype=torch.uint8, device=dev
            ).view(torch.float4_e2m1fn_x2),
            fc1_weight_sf=torch.zeros(e, 8, dtype=torch.float8_e4m3fn, device=dev),
            fc2_weight=torch.zeros(
                e, intermediate // 4, hidden, dtype=torch.uint8, device=dev
            ).view(torch.float4_e2m1fn_x2),
            fc2_weight_sf=torch.zeros(e, 8, dtype=torch.float8_e4m3fn, device=dev),
            fc1_alpha=torch.ones(e, dtype=torch.float32, device=dev),
            fc2_alpha=torch.ones(e, dtype=torch.float32, device=dev),
            fc1_norm_const=torch.ones(e, dtype=torch.float32, device=dev),
            output_activation=torch.zeros(
                tokens, hidden, dtype=torch.bfloat16, device=dev
            ),
            fc1_activation_per_token_scale=scale,
        )

    scale = torch.ones(tokens, dtype=torch.float32, device=dev)
    on = MegaMoENvfp4Frontend(
        MegaMoENvfp4Config(**base, enable_fc1_activation_per_token_scale=True)
    )
    off = MegaMoENvfp4Frontend(MegaMoENvfp4Config(**base))

    on._validate_inputs(make_inputs(scale), num_tokens=tokens)
    off._validate_inputs(make_inputs(None), num_tokens=tokens)
    with pytest.raises(ValueError, match="required"):
        on._validate_inputs(make_inputs(None), num_tokens=tokens)
    with pytest.raises(ValueError, match="enable_fc1_activation_per_token_scale=False"):
        off._validate_inputs(make_inputs(scale), num_tokens=tokens)
    with pytest.raises(ValueError, match="shape"):
        on._validate_inputs(make_inputs(scale[: tokens // 2]), num_tokens=tokens)
    with pytest.raises(ValueError, match="float32"):
        on._validate_inputs(make_inputs(scale.to(torch.bfloat16)), num_tokens=tokens)
    # The dispatch warps read the scale as a flat fp32 plane (base + token*4),
    # so a strided view of the right shape/dtype must be rejected, not misread.
    strided = torch.ones(2 * tokens, dtype=torch.float32, device=dev)[::2]
    assert strided.shape == (tokens,)
    with pytest.raises(ValueError, match="contiguous"):
        on._validate_inputs(make_inputs(strided), num_tokens=tokens)
    # run() revalidates only on a launch-cache miss, so a re-view of the same
    # storage (same data_ptr, different stride / shape / dtype) must change the
    # launch key -- otherwise the strided view would reuse the cached launch
    # arguments and be read as if it were contiguous.
    storage = torch.ones(2 * tokens, dtype=torch.float32, device=dev)
    key_contig = on._launch_cache_key(make_inputs(storage[:tokens]), tokens)
    assert storage[::2].data_ptr() == storage[:tokens].data_ptr()
    assert on._launch_cache_key(make_inputs(storage[::2]), tokens) != key_contig
    assert (
        on._launch_cache_key(make_inputs(storage[: tokens // 2]), tokens) != key_contig
    )
    assert (
        on._launch_cache_key(make_inputs(storage[:tokens].view(torch.int32)), tokens)
        != key_contig
    )
