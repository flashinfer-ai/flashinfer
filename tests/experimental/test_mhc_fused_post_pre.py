"""Tests for the experimental flashinfer.mhc.mhc_fused_post_pre (fused mHC post -> pre +
input RMSNorm).

The first group runs without a GPU (support checks, warmup policy and the composable
fallback against the reference). The second group needs an SM103 GPU: specialized
kernel vs reference over every regime boundary, the fallback outside the supported
surface, CUDA graph capture with and without the pre-capture warmup, and the operand
checks.
"""

import pytest
import torch

import flashinfer.mhc as mhc
from flashinfer.experimental.mhc_fused_post_pre import backend
from flashinfer.experimental.mhc_fused_post_pre.reference import (
    mhc_fused_post_pre_reference,
)

_CONSTANTS = dict(
    rms_eps=1e-5,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
    norm_eps=1e-5,
)
_OUTPUTS = ("residual_cur", "post_mix_cur", "comb_mix_cur", "layer_input_cur")
# (atol, rtol, max abs error) per output; >= 99.9% of the elements within atol + rtol*|ref|.
_TOLERANCES = {
    "residual_cur": (0.015625, 0.02, 0.125),
    "post_mix_cur": (0.004, 0.004, 0.02),
    "comb_mix_cur": (0.004, 0.004, 0.02),
    "layer_input_cur": (0.015625, 0.02, 0.125),
}


def _inputs(
    num_tokens: int,
    device: str,
    seed: int = 0,
    hidden_size: int = 4096,
    hc_mult: int = 4,
) -> dict:
    """Inputs as the mHC blocks produce them: ``post_layer_mix`` is a sigmoid gate
    scaled by 2, ``comb_res_mix`` a Sinkhorn-normalized (doubly stochastic) mix."""
    generator = torch.Generator(device=device).manual_seed(seed)
    mix = hc_mult * (2 + hc_mult)

    def randn(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=device, generator=generator
        )

    comb = randn(num_tokens, hc_mult, hc_mult).softmax(dim=-1)
    for _ in range(20):
        comb = comb / comb.sum(dim=-2, keepdim=True)
        comb = comb / comb.sum(dim=-1, keepdim=True)
    return {
        "x": (0.5 * randn(num_tokens, hidden_size)).to(torch.bfloat16),
        "residual": (0.5 * randn(num_tokens, hc_mult, hidden_size)).to(torch.bfloat16),
        "post_layer_mix": (2.0 * torch.sigmoid(randn(num_tokens, hc_mult))),
        "comb_res_mix": comb.contiguous(),
        "fn": (randn(mix, hc_mult * hidden_size) / (hc_mult * hidden_size) ** 0.5),
        "hc_scale": torch.tensor([0.75, -1.25, 1.5], device=device),
        "hc_base": 0.5 * randn(mix),
        "norm_weight": (1.0 + 0.1 * randn(hidden_size)).to(torch.bfloat16),
    }


def _assert_matches(got, ref) -> None:
    for name, g, r in zip(_OUTPUTS, got, ref, strict=True):
        atol, rtol, cap = _TOLERANCES[name]
        g = g.float().reshape(r.shape)
        r = r.float()
        assert torch.isfinite(g).all(), name
        err = (g - r).abs()
        matched = (err <= atol + rtol * r.abs()).float().mean().item()
        assert matched >= 0.999, (name, matched)
        assert err.max().item() <= cap, (name, err.max().item())


# ----------------------------------------------------------------------------------
# GPU-free
# ----------------------------------------------------------------------------------


def test_entry_points_are_experimental():
    for api in (
        mhc.mhc_fused_post_pre,
        mhc.mhc_fused_post_pre_supported,
        mhc.mhc_fused_post_pre_warmup,
    ):
        assert api.is_experimental is True


def test_supported_surface():
    surface = backend.SUPPORTED_SURFACE
    assert surface["archs"] == ("sm_103",)
    assert (surface["hidden_size"], surface["hc_mult"], surface["mix_width"]) == (
        4096,
        4,
        24,
    )
    assert surface["num_tokens_range"] == (33, 16384)
    assert tuple(surface["constants"].values()) == tuple(_CONSTANTS.values())


@pytest.mark.parametrize(
    "num_tokens, changes, reason",
    [
        (48, {"norm_weight": None}, "norm_weight"),
        (48, {"rms_eps": 1e-6}, "constants"),
        (48, {"norm_eps": 1e-6}, "constants"),
        (48, {"hc_pre_eps": 1e-5}, "constants"),
        (48, {"sinkhorn_repeat": 19}, "constants"),
        (48, {"hc_post_mult_value": 1.0}, "constants"),
        (32, {}, "num_tokens"),
        (16385, {}, "num_tokens"),
        (33, {}, "device"),
        (16384, {}, "device"),
    ],
)
def test_guard_reasons(num_tokens, changes, reason):
    args = dict(_CONSTANTS)
    norm_weight = torch.ones(4096, dtype=torch.bfloat16)
    if "norm_weight" in changes:
        norm_weight = changes.pop("norm_weight")
    args.update(changes)
    residual = torch.empty(num_tokens, 4, 4096, dtype=torch.bfloat16)
    assert backend._reject_reason(residual, norm_weight, **args) == reason


@pytest.mark.parametrize("shape", [(48, 8, 4096), (48, 4, 7168)])
def test_guard_rejects_other_geometry(shape):
    residual = torch.empty(shape, dtype=torch.bfloat16)
    norm_weight = torch.ones(shape[-1], dtype=torch.bfloat16)
    reason = backend._reject_reason(residual, norm_weight, *_CONSTANTS.values())
    assert reason == "shape"


@pytest.mark.parametrize("num_tokens", [33, 48, 129])
def test_fallback_matches_reference(num_tokens):
    inputs = _inputs(num_tokens, "cpu")
    got = mhc.mhc_fused_post_pre(**inputs, **_CONSTANTS)
    ref = mhc_fused_post_pre_reference(**inputs, **_CONSTANTS)
    for g, r in zip(got, ref, strict=True):
        assert torch.equal(g, r.view(g.shape))
    assert got[1].shape == (num_tokens, 4, 1)


def test_fallback_false_declines_and_out_is_untouched():
    inputs = _inputs(48, "cpu")
    out = (
        torch.full((48, 4, 4096), 7.0, dtype=torch.bfloat16),
        torch.full((48, 4), 7.0),
        torch.full((48, 16), 7.0),
        torch.full((48, 4096), 7.0, dtype=torch.bfloat16),
    )
    before = backend._stats["declined_calls"]
    assert mhc.mhc_fused_post_pre(**inputs, out=out, fallback=False) is None
    assert backend._stats["declined_calls"] == before + 1
    assert all(bool((t == 7.0).all()) for t in out)


def test_supported_is_false_off_device():
    assert not mhc.mhc_fused_post_pre_supported(**_inputs(48, "cpu"))


def test_supported_without_out_never_allocates(monkeypatch):
    inputs = _inputs(48, "cpu")

    def no_allocation(*args, **kwargs):
        raise AssertionError("mhc_fused_post_pre_supported must not allocate outputs")

    monkeypatch.setattr(torch, "empty", no_allocation)
    assert not mhc.mhc_fused_post_pre_supported(**inputs)


def test_warmup_declines_other_geometry_without_building():
    state = backend._state
    loaded = state["module"]
    assert mhc.mhc_fused_post_pre_warmup(hidden_size=7168) is False
    assert mhc.mhc_fused_post_pre_warmup(sinkhorn_repeat=19) is False
    assert state["module"] is loaded


# ----------------------------------------------------------------------------------
# SM103
# ----------------------------------------------------------------------------------


def _is_sm103() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (10, 3)


sm103 = pytest.mark.skipif(not _is_sm103(), reason="specialized kernel is SM103-only")

# Every regime boundary of the kernel dispatch for decode / stream cluster capacities
# 8..10 (decode narrow/wide, stream default/long-register split, short-stream,
# crossover, long-stream), plus representative token counts of each regime.
_SWEEP = [33, 40, 47, 48, 56, 64, 65, 72, 73, 80, 81, 96, 128, 129, 144, 145, 160, 161]
_SWEEP += [256, 512, 1023, 1024, 1284, 1536, 1799, 1800, 1920, 2048, 2049, 2544, 2545]
_SWEEP += [
    2864,
    2865,
    3184,
    3185,
    3230,
    4096,
    4099,
    6244,
    6245,
    8192,
    11476,
    16128,
    16384,
]


@sm103
@pytest.mark.parametrize("num_tokens", _SWEEP)
def test_specialized_matches_reference(num_tokens):
    assert mhc.mhc_fused_post_pre_warmup()
    inputs = _inputs(num_tokens, "cuda", seed=num_tokens)
    before = backend._stats["specialized_dispatches"]
    got = mhc.mhc_fused_post_pre(**inputs, **_CONSTANTS, fallback=False)
    assert got is not None
    assert backend._stats["specialized_dispatches"] == before + 1
    ref = mhc_fused_post_pre_reference(**inputs, **_CONSTANTS)
    torch.cuda.synchronize()
    _assert_matches(got, ref)


@sm103
def test_every_variant_dispatches():
    assert mhc.mhc_fused_post_pre_warmup()
    stats = backend.stats()
    spans = stats["variants_by_num_tokens"][torch.cuda.current_device()]
    assert "no_kernel_for_shape" not in spans
    assert stats["distinct_kernels_for_surface"] == 7
    before = stats["dispatch_by_variant"]
    for lo, _ in spans.values():
        # The first token count of each span always selects that variant.
        mhc.mhc_fused_post_pre(**_inputs(lo, "cuda"), fallback=False)
    after = backend.stats()["dispatch_by_variant"]
    assert all(after[name] == before[name] + 1 for name in spans)


@sm103
def test_off_surface_takes_fallback():
    assert mhc.mhc_fused_post_pre_warmup()
    inputs = _inputs(32, "cuda")
    before = backend._stats["specialized_dispatches"]
    assert not mhc.mhc_fused_post_pre_supported(**inputs)
    assert mhc.mhc_fused_post_pre(**inputs, fallback=False) is None
    got = mhc.mhc_fused_post_pre(**inputs)
    assert backend._stats["specialized_dispatches"] == before
    _assert_matches(got, mhc_fused_post_pre_reference(**inputs, **_CONSTANTS))


@sm103
@pytest.mark.parametrize("num_tokens", [48, 64, 2048])
def test_cuda_graph_capture_after_warmup(num_tokens):
    assert mhc.mhc_fused_post_pre_warmup()
    inputs = _inputs(num_tokens, "cuda", seed=1)
    out = (
        torch.empty(num_tokens, 4, 4096, dtype=torch.bfloat16, device="cuda"),
        torch.empty(num_tokens, 4, device="cuda"),
        torch.empty(num_tokens, 16, device="cuda"),
        torch.empty(num_tokens, 4096, dtype=torch.bfloat16, device="cuda"),
    )
    before = backend._stats["specialized_dispatches"]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        served = mhc.mhc_fused_post_pre(**inputs, out=out, fallback=False)
    assert served is not None
    assert backend._stats["specialized_dispatches"] == before + 1
    fresh = _inputs(num_tokens, "cuda", seed=2)
    for key in ("x", "residual", "post_layer_mix", "comb_res_mix"):
        inputs[key].copy_(fresh[key])
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches(out, mhc_fused_post_pre_reference(**inputs, **_CONSTANTS))


@sm103
def test_capture_without_warmup_declines(monkeypatch):
    # A device that was never prepared: under capture nothing is built or prepared.
    monkeypatch.setattr(backend, "_state", dict(backend._state))
    backend._state["devices"] = {}
    inputs = _inputs(48, "cuda")
    before = backend._stats["capture_unready"]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        served = mhc.mhc_fused_post_pre(**inputs, fallback=False)
    assert served is None
    assert backend._stats["capture_unready"] == before + 1


@sm103
def test_operand_checks_decline():
    assert mhc.mhc_fused_post_pre_warmup()
    inputs = _inputs(48, "cuda")
    assert mhc.mhc_fused_post_pre_supported(**inputs)
    misaligned = torch.empty(48 * 4096 + 8, dtype=torch.bfloat16, device="cuda")
    misaligned = misaligned[4 : 4 + 48 * 4096].view(48, 4096)
    misaligned.copy_(inputs["x"])
    assert not mhc.mhc_fused_post_pre_supported(**{**inputs, "x": misaligned})
    strided = inputs["residual"].transpose(0, 1).contiguous().transpose(0, 1)
    assert not mhc.mhc_fused_post_pre_supported(**{**inputs, "residual": strided})
    # Declined calls still compute through the fallback.
    got = mhc.mhc_fused_post_pre(**{**inputs, "x": misaligned})
    _assert_matches(got, mhc_fused_post_pre_reference(**inputs, **_CONSTANTS))


@sm103
@pytest.mark.parametrize("num_tokens", [48, 16384])
def test_supported_checks_inputs_only_without_out(num_tokens):
    assert mhc.mhc_fused_post_pre_warmup()
    inputs = _inputs(num_tokens, "cuda")
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    assert mhc.mhc_fused_post_pre_supported(**inputs)
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.max_memory_allocated() == allocated
    # Caller-provided outputs are still checked.
    out = (
        torch.empty(num_tokens, 4, 4096, dtype=torch.bfloat16, device="cuda"),
        torch.empty(num_tokens, 4, device="cuda"),
        torch.empty(num_tokens, 4, 4, device="cuda"),
        torch.empty(num_tokens, 4096, dtype=torch.bfloat16, device="cuda"),
    )
    assert mhc.mhc_fused_post_pre_supported(**inputs, out=out)
    aliased = (out[0], out[1], out[2], inputs["x"])
    assert not mhc.mhc_fused_post_pre_supported(**inputs, out=aliased)


@sm103
def test_tensor_map_cache_hits_on_reused_buffers():
    assert mhc.mhc_fused_post_pre_warmup()
    inputs = _inputs(1284, "cuda")
    out = tuple(t.contiguous() for t in mhc.mhc_fused_post_pre(**inputs))
    out = (out[0], out[1].view(1284, 4), out[2], out[3])
    before = backend.stats()["tensor_map_cache"]
    for _ in range(3):
        mhc.mhc_fused_post_pre(**inputs, out=out)
    after = backend.stats()["tensor_map_cache"]
    hits = after["hits"] - before["hits"]
    misses = after["misses"] - before["misses"]
    # Three maps per call; at most two of them can share a cache slot.
    assert hits + misses == 9 and hits >= 3


@sm103
def test_outer_shape_views():
    assert mhc.mhc_fused_post_pre_warmup()
    flat = _inputs(64, "cuda")
    nested = dict(flat)
    for key in ("x", "residual", "post_layer_mix", "comb_res_mix"):
        nested[key] = flat[key].view(2, 32, *flat[key].shape[1:])
    got = mhc.mhc_fused_post_pre(**nested)
    assert got[0].shape == (2, 32, 4, 4096) and got[1].shape == (2, 32, 4, 1)
    ref = mhc_fused_post_pre_reference(**flat, **_CONSTANTS)
    _assert_matches([g.reshape(r.shape) for g, r in zip(got, ref, strict=True)], ref)
