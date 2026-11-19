"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Silicon tests for the specialized SM120 fp8 [128,128] block-scale MoE
kernels (agent-generated, contract E=256, top_k=8, hidden=2048,
intermediate=512, silu-gated, bf16 activations), whose sole public home is
the unified MoE API (``SpecializedFp8BlockConfig`` /
``SpecializedFp8BlockRunner``; the unified-surface tests live in
``test_specialized_fp8_blockscale_sm120_layer.py``).

These tests exercise the kernels' own module directly
(``maybe_run_specialized``: the guard's token allowlist, kill switch,
fail-closed refusals), numeric correctness against an fp32 block-dequant
reference MoE, a CUDA-graph capture/replay smoke, and the NEGATIVE CONTROL
for the flat vendor entry: ``cutlass_fused_moe`` keeps its upstream surface
(stock SM120 dsfp8 refusal, no "specialized"/"auto" backend) and never
reaches these kernels.
"""

import pytest
import torch


def _is_validated_sm120_device():
    # Gate on the CURRENT device: DEVICE = "cuda" resolves to it, so checking
    # device 0 would mis-gate on multi-GPU hosts.
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    # The backend's dispatch surface requires the validated device profile
    # (CC 12.0 and the exact SM count its cooperative grids are sized for),
    # so other SM120 parts skip rather than exercise the stock refusal.
    return (props.major, props.minor) == (12, 0) and props.multi_processor_count == 188


def _has_min_cuda_toolkit():
    # Shared probe with the runner's support check: a cached module artifact
    # needs no compiler; otherwise the JIT-selected nvcc (FLASHINFER_NVCC
    # override, else $CUDA_HOME/bin/nvcc) must be CUDA 13.2+ (ptxas accepts
    # cvt.rn.bf16x2.e4m3x2 only from PTX ISA 9.2). torch.version.cuda is
    # never substituted: it says nothing about the ability to build.
    try:
        from flashinfer.fused_moe.runners import _specialized_jit_nvcc_release
        from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
            gen_specialized_fused_moe_fp8_blockscale_sm120_module,
        )

        if gen_specialized_fused_moe_fp8_blockscale_sm120_module().is_compiled:
            return True
        v = _specialized_jit_nvcc_release()
        return v is not None and v >= (13, 2)
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _is_validated_sm120_device(),
        reason="the specialized fp8 block-scale MoE backend serves only the "
        "validated SM120 device profile (CC 12.0, 188 SMs)",
    ),
    pytest.mark.skipif(
        not _has_min_cuda_toolkit(),
        reason="the specialized backend JIT-builds an sm_120a module "
        "(CUDA 13.2+ toolkit required for PTX ISA 9.2)",
    ),
]

DEVICE = "cuda"
E, TOPK, H, I = 256, 8, 2048, 512
REL_TOL = 2e-2


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cpu").manual_seed(20260102)
    w13 = torch.randn(E, 2 * I, H, generator=gen).to(torch.float8_e4m3fn)
    w2 = torch.randn(E, H, I, generator=gen).to(torch.float8_e4m3fn)
    w13_scale = torch.rand(E, (2 * I) // 128, H // 128, generator=gen)
    w2_scale = torch.rand(E, H // 128, I // 128, generator=gen)
    w13_scale = (w13_scale * 0.02 + 0.01).float()
    w2_scale = (w2_scale * 0.02 + 0.01).float()
    return (
        w13.to(DEVICE),
        w2.to(DEVICE),
        w13_scale.contiguous().to(DEVICE),
        w2_scale.contiguous().to(DEVICE),
    )


def _routing(t, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(t, E, generator=gen)
    ids = torch.topk(scores, TOPK, dim=-1).indices.to(torch.int32)
    tw = torch.softmax(torch.rand(t, TOPK, generator=gen), dim=-1)
    x = torch.randn(t, H, generator=gen).to(torch.bfloat16)
    return (
        x.to(DEVICE),
        ids.contiguous().to(DEVICE),
        tw.float().contiguous().to(DEVICE),
    )


def _dequant_block(w_fp8, scale):
    rows, cols = w_fp8.shape
    s = scale.to(torch.float32)
    s = s.repeat_interleave(128, dim=0)[:rows]
    s = s.repeat_interleave(128, dim=1)[:, :cols]
    return w_fp8.to(torch.float32) * s


def _reference(x, ids, tw, weights):
    w13, w2, w13_scale, w2_scale = weights
    t = x.shape[0]
    x32 = x.to(torch.float32)
    out = torch.zeros(t, H, dtype=torch.float32, device=x.device)
    ids64 = ids.to(torch.int64)
    for e in torch.unique(ids64).tolist():
        tok, slot = (ids64 == e).nonzero(as_tuple=True)
        w13_e = _dequant_block(w13[e], w13_scale[e])
        w2_e = _dequant_block(w2[e], w2_scale[e])
        gate_up = x32[tok] @ w13_e.T
        gate, up = gate_up[:, :I], gate_up[:, I:]
        inter = (gate * torch.sigmoid(gate)) * up
        out.index_add_(0, tok, (inter @ w2_e.T) * tw[tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


def _flat_call(x, ids, tw, weights, output=None, **overrides):
    """The flat vendor entry (negative-control arm): upstream surface only."""
    from flashinfer.fused_moe import cutlass_fused_moe

    w13, w2, w13_scale, w2_scale = weights
    kwargs = dict(
        input=x,
        token_selected_experts=ids,
        token_final_scales=tw,
        fc1_expert_weights=w13,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=[w13_scale, w2_scale],
        use_deepseek_fp8_block_scale=True,
    )
    if output is not None:
        kwargs["output"] = output
    kwargs.update(overrides)
    out = cutlass_fused_moe(**kwargs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _dispatch(x, ids, tw, weights, output=None):
    """Call the kernels' own module directly (what the unified
    SpecializedFp8BlockRunner does); returns (output_or_None, reason)."""
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        maybe_run_specialized,
    )

    w13, w2, w13_scale, w2_scale = weights
    return maybe_run_specialized(
        x,
        ids,
        tw,
        w13,
        w2,
        [w13_scale, w2_scale],
        output,
        context_ok=True,
    )


def _assert_close_to_reference(out, ref):
    scale = ref.float().abs().max().clamp(min=1e-3).item()
    rel_l2 = (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    ).item()
    max_abs = (out.float() - ref.float()).abs().max().item()
    assert rel_l2 < REL_TOL, f"rel_l2 {rel_l2}"
    assert max_abs < 0.05 * scale, f"max_abs {max_abs}"


@pytest.mark.parametrize("t", [4, 16])
def test_module_dispatch_matches_reference(weights, t):
    """The kernels' own module serves allowlisted token counts; the output
    must match the fp32 block-dequant reference."""
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        specialized_moe_fp8_blockscale_stats,
    )

    x, ids, tw = _routing(t, seed=100 + t)
    ref = _reference(x, ids, tw, weights)
    before = specialized_moe_fp8_blockscale_stats()["specialized_dispatches"]
    out, reason = _dispatch(x, ids, tw, weights)
    torch.cuda.synchronize()
    assert reason is None and out is not None
    after = specialized_moe_fp8_blockscale_stats()["specialized_dispatches"]
    assert after == before + 1
    _assert_close_to_reference(out, ref)


def test_off_allowlist_token_count_declined(weights):
    """A token count off the shipped allowlist is declined with the reason
    recorded (the unified layer then routes the call elsewhere or refuses —
    the kernels never serve it)."""
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        specialized_moe_fp8_blockscale_stats,
    )

    before = specialized_moe_fp8_blockscale_stats()["fallback_reasons"].get(
        "num_tokens_off_allowlist", 0
    )
    x, ids, tw = _routing(7, seed=1)
    out, reason = _dispatch(x, ids, tw, weights)
    assert out is None and reason == "num_tokens_off_allowlist"
    reasons = specialized_moe_fp8_blockscale_stats()["fallback_reasons"]
    # Exactly this call's refusal must be counted: an earlier off-allowlist
    # call in the session must not satisfy the assertion for this one.
    assert reasons.get("num_tokens_off_allowlist", 0) == before + 1


def test_flat_entry_negative_control(weights):
    """The flat vendor entry keeps its upstream surface: the stock SM120
    dsfp8 refusal on an allowlisted shape (the specialized kernels never
    serve it), a "cutlass" default, and no "specialized"/"auto" backend."""
    import inspect

    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        specialized_moe_fp8_blockscale_stats,
    )

    assert (
        inspect.signature(cutlass_fused_moe).parameters["backend"].default == "cutlass"
    )
    x, ids, tw = _routing(16, seed=2)
    before = specialized_moe_fp8_blockscale_stats()["specialized_dispatches"]
    with pytest.raises(NotImplementedError):
        _flat_call(x, ids, tw, weights)
    for name in ("specialized", "auto"):
        with pytest.raises(ValueError, match="unsupported fused MoE backend"):
            _flat_call(x, ids, tw, weights, backend=name)
    assert specialized_moe_fp8_blockscale_stats()["specialized_dispatches"] == before


def test_kill_switch_declines(weights, monkeypatch):
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    x, ids, tw = _routing(16, seed=5)
    out, reason = _dispatch(x, ids, tw, weights)
    assert out is None and reason == "kill_switch"


@pytest.mark.parametrize(
    "case", ["weight_offset", "scale_offset", "short", "long", "cpu"]
)
def test_bad_warmup_never_enters_kernel(weights, monkeypatch, case):
    from flashinfer.fused_moe.specialized import fp8_blockscale_moe_sm120 as mod

    monkeypatch.setattr(mod, "_kernel_entry", lambda: pytest.fail("entered kernel"))
    monkeypatch.delenv(mod._DISABLE_ENV, raising=False)
    receipts = set(mod._warmed)
    w13, w2, s13, s2 = weights
    scales = [s13, s2]
    device = w13.device
    if case in ("weight_offset", "scale_offset"):
        src = w2 if case == "weight_offset" else s2
        offset = torch.empty(src.numel() + 1, dtype=src.dtype, device=device)[
            1:
        ].view_as(src)
        assert offset.is_contiguous() and offset.data_ptr() % 16
        if case == "weight_offset":
            w2 = offset
        else:
            scales[1] = offset
        reason = "operand_alignment"
    elif case == "cpu":
        device = "cpu"
        reason = "CUDA device"
    else:
        scales = scales[:1] if case == "short" else scales + [s2]
        reason = "quant_scales_count"
    with pytest.raises(ValueError, match=reason):
        mod.warmup(w13, w2, scales, device=device, t_values=(256,))
    assert mod._warmed == receipts


def test_cuda_graph_replay_matches_eager(weights):
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        warmup,
    )

    w13, w2, w13_scale, w2_scale = weights
    warmup(w13, w2, [w13_scale, w2_scale], t_values=(16,))
    t = 16
    x, ids, tw = _routing(t, seed=6)
    out = torch.zeros(t, H, dtype=torch.bfloat16, device=DEVICE)
    assert _dispatch(x, ids, tw, weights, output=out)[1] is None
    torch.cuda.synchronize()
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert _dispatch(x, ids, tw, weights, output=out)[1] is None
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        assert _dispatch(x, ids, tw, weights, output=out)[1] is None
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    # a few bf16 ulp at the output magnitude: accumulation order may
    # differ between eager and replay (bf16-atomics class)
    tol = 2.0**-5 * eager.float().abs().max().clamp(min=1e-3).item()
    max_abs = (out.float() - eager.float()).abs().max().item()
    assert max_abs <= tol, f"graphed vs eager max_abs {max_abs} > {tol}"


# ---------------------------------------------------------------------------
# Concurrency: independently executing calls and independently captured
# graphs must never share live scratch, and the software-grid-barrier
# persistent kernels must never run without a residency guarantee.
# ---------------------------------------------------------------------------
#
# The analytic construction makes every intermediate an exact power of two,
# so a correct execution is EXACT and any cross-execution scratch bleed is a
# large, deterministic error: all-ones fp8 weights, W13 scale 2^-4 and W2
# scale 2^-12, constant bf16 activations a, valid cycling expert ids,
# routing weights 1/8. Then gate = up = 128a, inter = 16384a^2, and
# out = 512 * inter * 2^-12 = 2048 a^2 in every element (a=1 -> 2048,
# a=2 -> 8192), with silu's sigmoid factor within one bf16 ulp of 1.


@pytest.fixture(scope="module")
def analytic_weights():
    w13 = torch.ones(E, 2 * I, H).to(torch.float8_e4m3fn).to(DEVICE)
    w2 = torch.ones(E, H, I).to(torch.float8_e4m3fn).to(DEVICE)
    w13_scale = torch.full((E, (2 * I) // 128, H // 128), 2.0**-4, device=DEVICE)
    w2_scale = torch.full((E, H // 128, I // 128), 2.0**-12, device=DEVICE)
    return w13, w2, w13_scale, w2_scale


def _analytic_inputs(t, a):
    x = torch.full((t, H), float(a), dtype=torch.bfloat16, device=DEVICE)
    ids = (
        (torch.arange(t * TOPK, dtype=torch.int32, device=DEVICE) % E)
        .reshape(t, TOPK)
        .contiguous()
    )
    tw = torch.full((t, TOPK), 1.0 / TOPK, dtype=torch.float32, device=DEVICE)
    return x, ids, tw


def _assert_exact(out, a, label):
    expected = 2048.0 * a * a
    bad = (out.float() != expected).sum().item()
    mx = (out.float() - expected).abs().max().item()
    assert bad == 0, (
        f"{label}: {bad} wrong elements, max abs err {mx} (expected {expected})"
    )


def _sync_with_deadline(streams, seconds=180.0):
    """Wait for the given streams' queued work, failing (instead of hanging
    the suite) if the device never completes — the pre-fix deadlock
    signature of a spin-barrier grid launched without a residency guarantee
    or with shared barrier words. Events are recorded on the streams under
    test: an event on the default stream would carry no dependency on them
    and the asserts would read outputs still being written."""
    import time

    events = []
    for s in streams:
        ev = torch.cuda.Event()
        ev.record(s)
        events.append(ev)
    t0 = time.monotonic()
    while not all(ev.query() for ev in events):
        if time.monotonic() - t0 > seconds:
            raise AssertionError(
                f"device did not complete within {seconds}s: persistent-kernel deadlock"
            )
        time.sleep(0.05)


@pytest.mark.parametrize("t", [4, 16, 256])
def test_concurrent_streams_isolated_scratch(analytic_weights, t):
    """Two independent eager calls on two streams, same device, different
    activations: each call owns its own scratch (per device AND stream), so
    both outputs are exact. Before the fix the per-device module scratch
    (and, for T<=16, the module-global barrier counters and intermediate)
    let one call consume the other's FC1 results."""
    x1, ids, tw = _analytic_inputs(t, 1)
    x2, _, _ = _analytic_inputs(t, 2)
    out1 = torch.empty(t, H, dtype=torch.bfloat16, device=DEVICE)
    out2 = torch.empty(t, H, dtype=torch.bfloat16, device=DEVICE)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for trial in range(10):
        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            for _ in range(4):
                assert _dispatch(x1, ids, tw, analytic_weights, output=out1)[1] is None
        with torch.cuda.stream(s2):
            for _ in range(4):
                assert _dispatch(x2, ids, tw, analytic_weights, output=out2)[1] is None
        _sync_with_deadline([s1, s2])
        _assert_exact(out1, 1, f"t={t} trial={trial} stream1")
        _assert_exact(out2, 2, f"t={t} trial={trial} stream2")


def test_independent_graph_captures_replay_concurrently(analytic_weights):
    """Two separately captured CUDA graphs (no shared memory pool), replayed
    concurrently on two streams: scratch is allocated from independent
    graph pools, even on the same capture stream, so both outputs
    stay exact under concurrent replay — the reviewer's T=256 wrong-result
    reproduction."""
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import warmup

    t = 256
    w13, w2, w13_scale, w2_scale = analytic_weights
    warmup(w13, w2, [w13_scale, w2_scale], t_values=(t,))
    x1, ids, tw = _analytic_inputs(t, 1)
    x2, _, _ = _analytic_inputs(t, 2)
    out1 = torch.empty(t, H, dtype=torch.bfloat16, device=DEVICE)
    out2 = torch.empty(t, H, dtype=torch.bfloat16, device=DEVICE)

    # eager warm runs for each layer (the torch.cuda.graph pattern)
    assert _dispatch(x1, ids, tw, analytic_weights, output=out1)[1] is None
    assert _dispatch(x2, ids, tw, analytic_weights, output=out2)[1] is None
    torch.cuda.synchronize()

    g1, g2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        assert _dispatch(x1, ids, tw, analytic_weights, output=out1)[1] is None
    with torch.cuda.graph(g2):
        assert _dispatch(x2, ids, tw, analytic_weights, output=out2)[1] is None

    # serial replays are correct
    out1.zero_(), out2.zero_()
    torch.cuda.synchronize()
    g1.replay()
    g2.replay()
    torch.cuda.synchronize()
    _assert_exact(out1, 1, "serial replay g1")
    _assert_exact(out2, 2, "serial replay g2")

    # concurrent replays on independent streams must stay correct
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    failures = 0
    worst = 0.0
    for _trial in range(30):
        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            g1.replay()
        with torch.cuda.stream(s2):
            g2.replay()
        _sync_with_deadline([s1, s2])
        for out, a in ((out1, 1), (out2, 2)):
            err = (out.float() - 2048.0 * a * a).abs().max().item()
            if err != 0.0:
                failures += 1
                worst = max(worst, err)
    assert failures == 0, f"{failures}/60 concurrent replays wrong, max abs err {worst}"


def test_recapture_releases_live_workspace(analytic_weights):
    import gc

    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import warmup

    w13, w2, s13, s2 = analytic_weights
    warmup(w13, w2, [s13, s2], t_values=(16, 256))
    small = _analytic_inputs(16, 1)
    large = _analytic_inputs(256, 2)
    small_out = torch.empty_like(small[0])
    large_out = torch.empty_like(large[0])

    def capture(inputs, out):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, reason = _dispatch(*inputs, analytic_weights, output=out)
            assert reason is None
        return graph

    keep = capture(small, small_out)
    keep.replay()
    torch.cuda.synchronize()
    gc.collect()

    def live():
        stats = torch.cuda.memory_stats(small_out.device)
        return stats["allocated_bytes.all.current"], stats["allocation.all.current"]

    baseline_bytes, baseline_entries = live()
    for _ in range(20):
        graph = capture(large, large_out)
        graph.replay()
        torch.cuda.synchronize()
        del graph
        gc.collect()
        nbytes, entries = live()
        assert nbytes <= baseline_bytes and entries <= baseline_entries, (
            nbytes - baseline_bytes,
            entries - baseline_entries,
        )
    small_out.zero_()
    keep.replay()
    torch.cuda.synchronize()
    _assert_exact(small_out, 1, "surviving small graph")


def test_persistent_decode_residency_fails_closed(analytic_weights):
    """The T<=16 persistent kernels spin in software grid barriers sized for
    a fully resident grid. On an execution context that cannot co-schedule
    that grid (here: a green-context partition with a fraction of the SMs),
    the cooperative launch must be REFUSED (fail closed) or complete
    correctly — never spin forever the way an unguaranteed plain launch
    can."""
    pytest.importorskip("cuda")
    from flashinfer.green_ctx import split_device_green_ctx_by_sm_count

    dev = torch.device("cuda", torch.cuda.current_device())
    try:
        streams, resources = split_device_green_ctx_by_sm_count(dev, [16])
    except Exception as exc:  # green contexts unavailable on this stack
        pytest.skip(f"green context partition unavailable: {exc}")
    if int(resources[0].sm.smCount) >= 188:
        pytest.skip("partition does not constrain the persistent grid")
    t = 16
    x, ids, tw = _analytic_inputs(t, 1)
    out = torch.empty(t, H, dtype=torch.bfloat16, device=DEVICE)
    stream = streams[0]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        try:
            res, reason = _dispatch(x, ids, tw, analytic_weights, output=out)
        except RuntimeError as exc:
            message = str(exc).lower()
            if not (
                "persistent decode t=16 cooperative launch failed" in message
                and any(
                    text in message
                    for text in (
                        "too large",
                        "too many blocks in cooperative launch",
                        "cooperativelaunchtoolarge",
                    )
                )
            ):
                raise
        else:
            assert reason is None and res is out
            _sync_with_deadline([stream])
            _assert_exact(out, 1, "constrained-context t=16")
    probe = torch.ones(1, device=dev).add_(1)
    torch.cuda.synchronize(dev)
    assert probe.item() == 2
