"""Fused DataPreprocess staging vs the torch-composed staging path.

The cutedsl mega backends stage activations via a single fused quant+repack
launch (``fused_quant_stage``); the original torch staging survives behind
``FLASHINFER_MEGA_FUSED_STAGE=0``. The repo's torch quantizers are bit-matched
to the kernel (see ``src/src/inputs_process.py``'s harness), so the two paths
must agree bit-exactly on data, scales, and routing.

Run on one Blackwell GPU from the FlashInfer repo root::

    cd /path/to/flashinfer
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    CUDA_VISIBLE_DEVICES=0 pytest tests/moe_ep/test_fused_quant_stage.py -v \\
        -m arch_blackwell --confcutdir=tests/moe_ep
"""

from __future__ import annotations

import pytest

pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")


def _require_blackwell():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(f"staging kernel needs sm_100/sm_103; got sm_{cap[0]}{cap[1]}")


def _make_buffers(quant_type: str, capacity: int, hidden: int, topk: int):
    import torch

    if quant_type == "nvfp4":
        x = (
            torch.zeros(capacity * hidden // 2, dtype=torch.uint8, device="cuda")
            .view(torch.float4_e2m1fn_x2)
            .reshape(capacity, hidden // 2)
        )
        sf = torch.zeros(
            capacity, hidden // 16, dtype=torch.float8_e4m3fn, device="cuda"
        )
    else:
        data_dtype = (
            torch.float8_e4m3fn if quant_type == "mxfp8_e4m3" else torch.float8_e5m2
        )
        x = torch.zeros(capacity, hidden, dtype=data_dtype, device="cuda")
        sf = (
            torch.zeros(capacity * (hidden // 32), dtype=torch.uint8, device="cuda")
            .view(torch.float8_e8m0fnu)
            .reshape(capacity, hidden // 32)
        )
    # Dirty the routing tail so the capacity re-mask is actually exercised;
    # reset the tail-fill memo for this (possibly reused) address so the
    # fused path treats the dirty buffer as fully live.
    topk_idx = torch.full((capacity, topk), 7, dtype=torch.int64, device="cuda")
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import note_staged_tokens

    note_staged_tokens(topk_idx, capacity)
    topk_weights = torch.zeros(capacity, topk, dtype=torch.float32, device="cuda")
    return x, sf, topk_idx, topk_weights


def _make_batch(num_tokens: int, hidden: int, topk: int, num_experts: int, seed: int):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1, sorted=False)
    return hidden_states, topk_ids.to(torch.int64), topk_weights.to(torch.float32)


def _stage(quant_type: str, monkeypatch, fused: bool, batch, buffers, norm_const):
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1" if fused else "0")
    hidden_states, topk_ids, topk_weights = batch
    x, sf, idx_out, w_out = buffers
    if quant_type == "nvfp4":
        from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
            stage_mega_moe_inputs,
        )

        stage_mega_moe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            x,
            sf,
            idx_out,
            w_out,
            norm_const=norm_const,
        )
    else:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.staging import (
            stage_mega_moe_inputs,
        )

        stage_mega_moe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            x,
            sf,
            idx_out,
            w_out,
            kind=quant_type,
        )


@pytest.mark.arch_blackwell
def test_mxfp8_staging_alternates_legacy_and_genphase_routing(monkeypatch):
    """A compiled int64 stager cannot be reused for an int32 output ABI."""
    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import note_staged_tokens

    _require_blackwell()
    hidden, topk, capacity = 384, 3, 68
    batch = _make_batch(65, hidden, topk, 16, seed=6447)
    reference = _make_buffers("mxfp8_e4m3", capacity, hidden, topk)
    _stage("mxfp8_e4m3", monkeypatch, False, batch, reference, None)
    for input_dtype, output_dtype in (
        (torch.int64, torch.int64),
        (torch.int64, torch.int32),
        (torch.int32, torch.int32),
        (torch.int64, torch.int64),
    ):
        buffers = list(_make_buffers("mxfp8_e4m3", capacity, hidden, topk))
        buffers[2] = buffers[2].to(output_dtype)
        note_staged_tokens(buffers[2], capacity)
        current = (batch[0], batch[1].to(input_dtype), batch[2])
        _stage("mxfp8_e4m3", monkeypatch, True, current, buffers, None)
        torch.cuda.synchronize()
        for actual, expected in zip(buffers, reference, strict=True):
            if actual.element_size() == 1:
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            else:
                assert torch.equal(actual, expected.to(actual.dtype))


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.arch_blackwell)]
)
@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
def test_genphase_staging_mixed_input_formats(monkeypatch, device, kind):
    """A prequantized batch must update the fused stager's shared tail memo."""
    from types import SimpleNamespace
    from unittest import mock

    import torch

    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.staging import (
        stage_genphase_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        forget_staged_tokens,
        staged_tokens,
    )
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    if device == "cuda":
        _require_blackwell()
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1" if device == "cuda" else "0")
    capacity, hidden, topk = 68, 384, 3
    generator = torch.Generator(device=device).manual_seed(6447)
    x = torch.randn(
        capacity, hidden, dtype=torch.bfloat16, device=device, generator=generator
    )
    qx, sx = pkg.quantize_genphase_mxfp8(
        x, torch.float8_e5m2 if kind == "mxfp8_e5m2" else torch.float8_e4m3fn
    )
    ids = torch.arange(topk, device=device).expand(capacity, topk).contiguous()
    scores = torch.full((capacity, topk), 1 / topk, device=device)
    workspace = SimpleNamespace(
        config=SimpleNamespace(hidden=hidden, sf_vec_size=32, quant_kind=kind),
        x=torch.zeros_like(qx),
        x_sf=torch.zeros(capacity, 128, dtype=torch.uint8, device=device).view(
            torch.float8_e8m0fnu
        ),
        topk_idx=torch.full((capacity, topk), -1, dtype=torch.int32, device=device),
        topk_weights=torch.zeros_like(scores),
        note_staged_tokens=mock.Mock(),
    )
    try:
        # Exercise the memo before capture makes tail masking unconditional.
        for count, quantize_input in (
            (1, True),
            (65, False),
            (1, True),
            (0, False),
            (1, True),
            (capacity, False),
            (1, True),
        ):
            tensors = MoEEpTensors(
                hidden_states=x[:count] if quantize_input else qx[:count],
                scales=None if quantize_input else sx[:count],
                topk_ids=ids[:count],
                topk_weights=scores[:count],
            )
            stage_genphase_inputs(tensors, workspace, quantize_input=quantize_input)
            assert staged_tokens(workspace.topk_idx) == count
            workspace.note_staged_tokens.assert_called_with(count)
            assert torch.equal(workspace.topk_idx[:count], ids[:count].int())
            assert (workspace.topk_idx[count:] == -1).all()
            assert torch.equal(workspace.topk_weights[:count], scores[:count])
            assert torch.equal(
                workspace.x[:count].view(torch.uint8), qx[:count].view(torch.uint8)
            )
            assert torch.equal(
                workspace.x_sf[:count, : hidden // 32].view(torch.uint8),
                sx[:count].view(torch.uint8),
            )
            assert (workspace.x_sf[:, hidden // 32 :].view(torch.uint8) == 0).all()
        if device == "cuda":
            prequantized = MoEEpTensors(
                hidden_states=qx[:65],
                scales=sx[:65],
                topk_ids=ids[:65],
                topk_weights=scores[:65],
            )
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                stage_genphase_inputs(prequantized, workspace, quantize_input=False)
            torch.cuda.current_stream().wait_stream(stream)
            # Replay grows the live routes without updating the host memo.
            for _ in range(2):
                stage_genphase_inputs(tensors, workspace, quantize_input=True)
                graph.replay()
                stage_genphase_inputs(tensors, workspace, quantize_input=True)
                assert staged_tokens(workspace.topk_idx) == 1
                assert (workspace.topk_idx[1:] == -1).all()
    finally:
        forget_staged_tokens(workspace.topk_idx)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("quant_type", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("num_tokens", [32, 64])  # partial + full capacity
def test_fused_stage_bit_matches_torch_stage(monkeypatch, quant_type, num_tokens):
    import torch

    _require_blackwell()

    hidden, topk, num_experts, capacity = 2048, 4, 16, 64
    norm_const = 2.0 if quant_type == "nvfp4" else 1.0
    batch = _make_batch(num_tokens, hidden, topk, num_experts, seed=17)

    ref = _make_buffers(quant_type, capacity, hidden, topk)
    got = _make_buffers(quant_type, capacity, hidden, topk)
    _stage(quant_type, monkeypatch, False, batch, ref, norm_const)
    _stage(quant_type, monkeypatch, True, batch, got, norm_const)
    torch.cuda.synchronize()

    for name, r, g in (
        ("x", ref[0], got[0]),
        ("x_sf", ref[1], got[1]),
    ):
        assert torch.equal(r.view(torch.uint8), g.view(torch.uint8)), (
            f"{quant_type} {name} mismatch (tokens={num_tokens})"
        )
    assert torch.equal(ref[2], got[2]), "topk_idx mismatch (incl. -1 tail mask)"
    assert torch.equal(ref[3], got[3]), "topk_weights mismatch"
    if num_tokens < capacity:
        assert (got[2][num_tokens:] == -1).all()


@pytest.mark.arch_blackwell
def test_fused_stage_launch_cache_tracks_new_data_and_token_count(monkeypatch):
    """Cache-hit relaunch (new data, same ptrs) and cache-rebuild (new n)."""
    import torch

    _require_blackwell()

    hidden, topk, num_experts, capacity = 2048, 4, 16, 64
    buffers = _make_buffers("nvfp4", capacity, hidden, topk)

    for seed, num_tokens in ((3, 32), (5, 32), (7, 48)):
        batch = _make_batch(num_tokens, hidden, topk, num_experts, seed=seed)
        ref = _make_buffers("nvfp4", capacity, hidden, topk)
        _stage("nvfp4", monkeypatch, False, batch, ref, 1.0)
        _stage("nvfp4", monkeypatch, True, batch, buffers, 1.0)
        torch.cuda.synchronize()
        assert torch.equal(ref[0].view(torch.uint8), buffers[0].view(torch.uint8)), (
            f"x mismatch at seed={seed} tokens={num_tokens}"
        )
        assert torch.equal(ref[2], buffers[2])


@pytest.mark.arch_blackwell
def test_fused_stage_bit_matches_deep_gemm_torch_stage(monkeypatch):
    """dg staging: fused DataPreprocess(mxfp8_e4m3) == per_token_cast_to_fp8.

    deep_gemm's packed-ue8m0 recipe is byte-identical to the cutedsl mxfp8
    recipe (scales are the same e8m0 bytes packed 4-per-int32), so the fused
    path must reproduce the torch path bit-exactly through the byte views.
    """
    import torch

    _require_blackwell()
    pytest.importorskip("deep_gemm")

    from flashinfer.moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.staging import (
        stage_mega_moe_inputs,
    )

    hidden, topk, num_experts, num_tokens = 2048, 4, 16, 48
    batch = _make_batch(num_tokens, hidden, topk, num_experts, seed=23)
    hidden_states, topk_ids, topk_weights = batch

    def _dg_buffers():
        # deep_gemm layout: sliced-to-n views, scales packed 4-per-int32.
        x = torch.zeros(num_tokens, hidden, dtype=torch.float8_e4m3fn, device="cuda")
        sf = torch.zeros(num_tokens, hidden // 128, dtype=torch.int32, device="cuda")
        idx = torch.full((num_tokens, topk), 7, dtype=torch.int64, device="cuda")
        w = torch.zeros(num_tokens, topk, dtype=torch.float32, device="cuda")
        return x, sf, idx, w

    ref = _dg_buffers()
    got = _dg_buffers()
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "0")
    stage_mega_moe_inputs(hidden_states, topk_weights, topk_ids, *ref)
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1")
    stage_mega_moe_inputs(hidden_states, topk_weights, topk_ids, *got)
    torch.cuda.synchronize()

    assert torch.equal(ref[0].view(torch.uint8), got[0].view(torch.uint8)), "x"
    assert torch.equal(ref[1].view(torch.uint8), got[1].view(torch.uint8)), "x_sf"
    assert torch.equal(ref[2], got[2]), "topk_idx"
    assert torch.equal(ref[3], got[3]), "topk_weights"


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("quant_type", ["nvfp4", "mxfp8_e4m3"])
def test_zero_token_stage_masks_stale_rows(monkeypatch, quant_type):
    """A zero-token staging must re-mask rows the previous batch left live.

    Regression: both the staging wrapper and ``fused_quant_stage`` used to
    early-return on ``num_tokens == 0`` before tail masking and the live-count
    memo update, leaving the previous batch's rows routed and
    ``staged_tokens()`` reporting the stale count.
    """
    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        fused_quant_stage,
        staged_tokens,
    )

    _require_blackwell()

    hidden, topk, num_experts, capacity = 2048, 4, 16, 64
    norm_const = 1.0 if quant_type == "nvfp4" else None

    # Wrapper path: live batch then a zero-token batch.
    buffers = _make_buffers(quant_type, capacity, hidden, topk)
    _stage(
        quant_type,
        monkeypatch,
        True,
        _make_batch(32, hidden, topk, num_experts, seed=1),
        buffers,
        norm_const,
    )
    _stage(
        quant_type,
        monkeypatch,
        True,
        _make_batch(0, hidden, topk, num_experts, seed=2),
        buffers,
        norm_const,
    )
    torch.cuda.synchronize()
    assert (buffers[2] == -1).all(), "zero-token staging left stale live rows"
    assert staged_tokens(buffers[2]) == 0

    # Direct fused_quant_stage path (bypasses the wrapper's early-out).
    buffers = _make_buffers(quant_type, capacity, hidden, topk)
    x, sf, idx_out, w_out = buffers
    batch = _make_batch(0, hidden, topk, num_experts, seed=3)
    fused_quant_stage(
        batch[0],
        batch[1],
        batch[2],
        x,
        sf,
        idx_out,
        w_out,
        quant_type=quant_type,
        norm_const=norm_const,
    )
    torch.cuda.synchronize()
    assert (idx_out == -1).all(), "zero-token fused_quant_stage left live rows"
    assert staged_tokens(idx_out) == 0


@pytest.mark.arch_blackwell
def test_fused_stage_tail_memo_shrinking_batches(monkeypatch):
    """Tail re-mask memo: shrink/grow sequences must keep [n:] masked."""
    import torch

    _require_blackwell()

    hidden, topk, num_experts, capacity = 2048, 4, 16, 64
    buffers = _make_buffers("nvfp4", capacity, hidden, topk)
    for seed, num_tokens in ((1, 64), (2, 32), (3, 48), (4, 16), (5, 16)):
        batch = _make_batch(num_tokens, hidden, topk, num_experts, seed=seed)
        _stage("nvfp4", monkeypatch, True, batch, buffers, 1.0)
        torch.cuda.synchronize()
        assert (buffers[2][num_tokens:] == -1).all(), (
            f"tail not masked after n={num_tokens}"
        )
        assert (buffers[2][:num_tokens] >= 0).all()

    # Mixed fused/torch sequence: the torch fallback must update the memo so
    # a later fused shrink cannot skip a needed fill.
    _stage(
        "nvfp4",
        monkeypatch,
        True,
        _make_batch(16, hidden, topk, num_experts, seed=6),
        buffers,
        1.0,
    )
    _stage(
        "nvfp4",
        monkeypatch,
        False,
        _make_batch(64, hidden, topk, num_experts, seed=7),
        buffers,
        1.0,
    )
    _stage(
        "nvfp4",
        monkeypatch,
        True,
        _make_batch(32, hidden, topk, num_experts, seed=8),
        buffers,
        1.0,
    )
    torch.cuda.synchronize()
    assert (buffers[2][32:] == -1).all(), "fused-after-torch shrink left live tail"
