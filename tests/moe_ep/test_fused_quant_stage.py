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

pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")


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
    # Dirty the routing tail so the capacity re-mask is actually exercised.
    topk_idx = torch.full((capacity, topk), 7, dtype=torch.int64, device="cuda")
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
        from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging import (
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
        from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.staging import (
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

    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.staging import (
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
