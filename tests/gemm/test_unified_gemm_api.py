"""GPU tests for the unified GEMM API (GemmLayer, packs, Linear, linear, gemm).

Per quant pair and per backend that ``supported_backends`` reports usable on
this GPU: accuracy against an fp32 reference built from the *same* quantized
operands, bit-level parity with the legacy function the adapter delegates to
(``mm_fp4`` / ``bmm_fp8``), the bias epilogue and its report, deterministic
selection outside ``autotune(True)`` and measured selection inside it, CUDA
graph capture/replay, the functional doors, and the ``torch.nn.Linear``
contract. CPU-only config tests live in ``tests/cpu/gemm/test_unified_gemm_api.py``.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest
import torch
import torch.nn.functional as F

import flashinfer.gemm.nn as nn_mod
from flashinfer import bmm_fp8, mm_fp4
from flashinfer.autotuner import autotune
from flashinfer.gemm import (
    BackendUnsupported,
    CublasGemmConfig,
    CudnnGemmConfig,
    CuteDslGemmConfig,
    CutlassGemmConfig,
    EpilogueConfig,
    GemmActivationPack,
    GemmConfig,
    GemmLayer,
    GemmProblem,
    Linear,
    TrtllmGemmConfig,
    gemm,
    linear,
    prepare_activations,
    prepare_weight_pack,
    supported_backends,
)
from flashinfer.gemm import runners as R
from flashinfer.gemm.api import BackendOptions, QuantConfig, QuantFormat
from flashinfer.quantization.fp4_quantization import e2m1_and_ufp8sf_scale_to_float
from flashinfer.utils import get_compute_capability

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = torch.device("cuda")
_CC = get_compute_capability(DEVICE)
ARCH = _CC[0] * 10 + _CC[1]

NVFP4 = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
FP8 = QuantConfig(weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor)
PAIRS = {"nvfp4": NVFP4, "fp8": FP8}
BACKENDS = {
    "cutlass": CutlassGemmConfig(),
    "cute-dsl": CuteDslGemmConfig(),
    "trtllm": TrtllmGemmConfig(),
    "cudnn": CudnnGemmConfig(),
    "cublas": CublasGemmConfig(),
}
M, N, K = 48, 256, 512


def _single(pair_name: str, backend_key: str, **kw) -> GemmConfig:
    return GemmConfig(
        quant=PAIRS[pair_name],
        backend=BackendOptions((BACKENDS[backend_key],)),
        **kw,
    )


def _usable(pair_name: str, backend_key: str, n: int = N, k: int = K) -> str | None:
    """Reason the backend cannot serve the pair here, or None when it can."""
    (report,) = supported_backends(
        _single(pair_name, backend_key), GemmProblem(n=n, k=k), DEVICE
    )
    return None if report.ok else report.reason


def _usable_backends(pair_name: str) -> list[str]:
    return [b for b in BACKENDS if _usable(pair_name, b) is None]


def _skip_unless_usable(pair_name: str, backend_key: str) -> None:
    reason = _usable(pair_name, backend_key)
    if reason is not None:
        pytest.skip(f"{backend_key} cannot serve {pair_name} on sm{ARCH}: {reason}")


def _dequant_reference(
    act: GemmActivationPack, view: dict, pair_name: str
) -> torch.Tensor:
    """fp32 ``A @ W^T`` from the quantized operands the kernel actually consumes."""
    if pair_name == "fp8":
        a = act.a.float() * act.a_scale.float()
        w = view["b"].float() * view["b_scale"].float()
        return a @ w.T

    # Host-side dequantizer (as the fuzzer's _snap_nvfp4 uses it): CPU tensors,
    # flat uint8 scales, and the *inverse* of the quantization global scale.
    def deq(q, sf, gs):
        return e2m1_and_ufp8sf_scale_to_float(
            q.cpu(),
            sf.cpu().view(torch.uint8).reshape(-1),
            (1.0 / gs).cpu(),
            16,
            1,
            True,
        ).to(DEVICE, torch.float32)

    a = deq(act.a, act.a_scale, act.a_global_scale)
    w = deq(view["b"], view["b_descale"], view["b_global_scale"])
    return a @ w.T


def _inputs(pair_name: str, m: int = M, n: int = N, k: int = K, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16)
    return x, w


# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_prepare_weight_pack_shares_views_across_same_recipe(pair_name):
    usable = _usable_backends(pair_name)
    if not usable:
        pytest.skip(f"no backend serves {pair_name} on sm{ARCH}")
    cfg = GemmConfig(
        quant=PAIRS[pair_name],
        backend=BackendOptions(tuple(BACKENDS[b] for b in usable)),
    )
    _, w = _inputs(pair_name)
    pack = prepare_weight_pack(w, cfg)
    assert set(pack.native_views) == set(usable)
    recipes = {k: v["recipe"] for k, v in pack.native_views.items()}
    for key in usable:
        if key == "trtllm":
            assert recipes[key] == "nvfp4_128x4_shuffled"
        else:
            # Same recipe => the very same view object (no second weight copy).
            first = next(k for k in usable if k != "trtllm")
            assert pack.get_view(key) is pack.get_view(first)
    assert pack.logical_shape == (N, K)


def test_prepare_weight_pack_rejects_unservable_config():
    with pytest.raises(BackendUnsupported):
        _, w = _inputs("fp8")
        # cuBLAS has no NVFP4 GEMM: nothing to prepare.
        prepare_weight_pack(
            w, GemmConfig(quant=NVFP4, backend=BackendOptions((CublasGemmConfig(),)))
        )


# ---------------------------------------------------------------------------
# GemmLayer per backend: accuracy and legacy parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_key", list(BACKENDS))
@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
@pytest.mark.parametrize("m", [1, M, 300])
def test_layer_matches_reference_and_legacy(pair_name, backend_key, m):
    _skip_unless_usable(pair_name, backend_key)
    cfg = _single(pair_name, backend_key)
    x, w = _inputs(pair_name, m=m)
    pack = prepare_weight_pack(w, cfg)
    act = prepare_activations(x, quant=cfg.quant)
    layer = GemmLayer(cfg, device=DEVICE)
    layer.prepare(pack)

    out = layer(act)
    assert out.shape == (m, N) and out.dtype == torch.bfloat16
    assert layer.winner_backend == backend_key

    ref = _dequant_reference(act, pack.get_view(backend_key), pair_name)
    cos = F.cosine_similarity(ref.reshape(-1), out.float().reshape(-1), dim=0)
    assert cos > 0.99, f"cosine {cos} vs dequantized reference"

    # Parity with the legacy entry point the adapter delegates to: same kernel,
    # same input list, same tuner key, so the outputs agree to rounding.
    view = pack.get_view(backend_key)
    if pair_name == "fp8":
        legacy = bmm_fp8(
            act.a.unsqueeze(0),
            view["b3"],
            act.a_scale,
            view["b_scale"],
            torch.bfloat16,
            backend=backend_key,
        )[0]
    else:
        alpha = (1.0 / (act.a_global_scale * view["b_global_scale"])).reshape(1)
        legacy = mm_fp4(
            act.a,
            view["b_t"],
            act.a_scale,
            view["b_descale_t"],
            alpha,
            torch.bfloat16,
            backend=backend_key,
        )
    torch.testing.assert_close(out, legacy, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_bias_is_applied_and_reported(pair_name):
    usable = _usable_backends(pair_name)
    if not usable:
        pytest.skip(f"no backend serves {pair_name} on sm{ARCH}")
    cfg = _single(pair_name, usable[0], epilogue=EpilogueConfig(bias=True))
    x, w = _inputs(pair_name)
    bias = torch.randn(N, device=DEVICE, dtype=torch.bfloat16)
    pack = prepare_weight_pack(w, cfg, bias=bias)
    act = prepare_activations(x, quant=cfg.quant)
    layer = GemmLayer(cfg, device=DEVICE)
    out = layer(act, pack)
    ref = _dequant_reference(act, pack.get_view(usable[0]), pair_name) + bias.float()
    cos = F.cosine_similarity(ref.reshape(-1), out.float().reshape(-1), dim=0)
    assert cos > 0.99
    report = layer.winner_report()
    assert report is not None and report.backend_key == usable[0]
    assert report.fused_bias is False  # MVP: the layer adds the bias after the GEMM


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_auto_is_deterministic_outside_tuning_and_measured_inside(pair_name):
    usable = _usable_backends(pair_name)
    if len(usable) < 2:
        pytest.skip(
            f"need two usable backends for {pair_name} on sm{ARCH}, have {usable}"
        )
    cands = tuple(BACKENDS[b] for b in usable)
    cfg = GemmConfig(quant=PAIRS[pair_name], backend=BackendOptions(cands))
    x, w = _inputs(pair_name)
    pack = prepare_weight_pack(w, cfg)
    act = prepare_activations(x, quant=cfg.quant)
    layer = GemmLayer(cfg, device=DEVICE)
    layer.prepare(pack)

    # Outside autotune(True): first candidate in BackendOptions order, no measurement.
    layer(act)
    assert layer.winner_backend == usable[0]
    assert len(layer._winners) == 1

    # Inside: every admitted candidate is measured; the winner is one of them
    # and is cached for the bucket afterwards.
    with autotune(True):
        out = layer(act)
    assert layer.winner_backend in usable
    tuned = layer.winner_backend
    layer(act)
    assert layer.winner_backend == tuned
    ref = _dequant_reference(act, pack.get_view(tuned), pair_name)
    assert F.cosine_similarity(ref.reshape(-1), out.float().reshape(-1), dim=0) > 0.99

    layer.reset_winner()
    assert layer.winner_backend is None and not layer._winners


def test_explicit_backend_never_falls_back():
    reason = _usable("nvfp4", "cute-dsl")
    if reason is not None:
        pytest.skip(f"cute-dsl unavailable: {reason}")
    # N % 8 != 0 is a cute-dsl constraint; an explicit single candidate must
    # raise with its name, not silently run another backend.
    cfg = _single("nvfp4", "cute-dsl")
    x, w = _inputs("nvfp4", n=N + 4)
    pack = prepare_weight_pack(w, cfg)
    act = prepare_activations(x, quant=cfg.quant)
    layer = GemmLayer(cfg, device=DEVICE)
    with pytest.raises(BackendUnsupported, match="cute-dsl"):
        layer(act, pack)


def test_supported_backends_reports_reasons():
    reports = supported_backends(
        GemmConfig(
            quant=NVFP4,
            backend=BackendOptions((CublasGemmConfig(), CutlassGemmConfig())),
        ),
        GemmProblem(n=N, k=K),
        DEVICE,
    )
    by_key = {r.backend_key: r for r in reports}
    assert by_key["cublas"].ok is False and "quant pair" in by_key["cublas"].reason
    assert by_key["cublas"].auto_rank == -1
    cutlass_ok = _usable("nvfp4", "cutlass") is None
    assert by_key["cutlass"].ok is cutlass_ok
    if cutlass_ok:
        assert by_key["cutlass"].auto_rank == 0
    # trtllm: bf16 output only (checked after the arch gate, so only where it exists).
    if ARCH not in (100, 103, 107):
        return
    (r,) = supported_backends(
        GemmConfig(
            quant=dataclasses.replace(NVFP4, output=QuantFormat.FP16),
            backend=BackendOptions((TrtllmGemmConfig(),)),
        ),
        GemmProblem(n=N, k=K),
        DEVICE,
    )
    assert r.ok is False and "bf16" in r.reason


# ---------------------------------------------------------------------------
# CUDA graphs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_cuda_graph_replay(pair_name):
    usable = _usable_backends(pair_name)
    if not usable:
        pytest.skip(f"no backend serves {pair_name} on sm{ARCH}")
    cfg = _single(pair_name, usable[0])
    x, w = _inputs(pair_name)
    pack = prepare_weight_pack(w, cfg)
    act = prepare_activations(x, quant=cfg.quant)
    layer = GemmLayer(cfg, device=DEVICE)
    layer.prepare(pack)
    out = torch.empty(M, N, device=DEVICE, dtype=torch.bfloat16)
    # Warm up (selection, JIT) outside capture, then capture the lean path.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            layer(act, out=out)
    torch.cuda.current_stream().wait_stream(s)
    eager = out.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        layer(act, out=out)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# PyTorch doors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_linear_module_matches_nn_linear(pair_name):
    usable = _usable_backends(pair_name)
    if not usable:
        pytest.skip(f"no backend serves {pair_name} on sm{ARCH}")
    cfg = GemmConfig(
        quant=PAIRS[pair_name],
        backend=BackendOptions(tuple(BACKENDS[b] for b in usable)),
    )
    torch.manual_seed(0)
    ref = torch.nn.Linear(K, N, bias=True, device=DEVICE, dtype=torch.bfloat16)
    lin = Linear(K, N, bias=True, device=DEVICE, dtype=torch.bfloat16, config=cfg)
    lin.load_state_dict(ref.state_dict())
    x = torch.randn(2, M, K, device=DEVICE, dtype=torch.bfloat16)

    # Before prepare_weights: plain F.linear on the parameters.
    torch.testing.assert_close(lin(x), ref(x))

    lin.prepare_weights()
    assert lin.prepared and lin.winner_backend is None
    out = lin(x)
    assert out.shape == (2, M, N) and out.dtype == torch.bfloat16
    assert lin.winner_backend == usable[0]
    assert lin.winner_report().fused_bias is False
    cos = F.cosine_similarity(
        ref(x).float().reshape(-1), out.float().reshape(-1), dim=0
    )
    assert cos > 0.98, f"cosine {cos} vs torch.nn.Linear in bf16"

    # A caller-quantized pack goes straight through.
    act = prepare_activations(x.reshape(-1, K), quant=cfg.quant)
    out2 = lin(act)
    torch.testing.assert_close(out2, out.reshape(-1, N), rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_functional_linear_and_gemm(pair_name):
    usable = _usable_backends(pair_name)
    if not usable:
        pytest.skip(f"no backend serves {pair_name} on sm{ARCH}")
    cfg = _single(pair_name, usable[0])
    x, w = _inputs(pair_name)
    bias = torch.randn(N, device=DEVICE, dtype=torch.bfloat16)
    ref = F.linear(x.float(), w.float(), bias.float())

    # No config: F.linear.
    torch.testing.assert_close(linear(x, w, bias), F.linear(x, w, bias))

    out = linear(x, w, bias, config=cfg)
    assert F.cosine_similarity(ref.reshape(-1), out.float().reshape(-1), dim=0) > 0.98
    out_again = linear(x, w, bias, config=cfg)  # pack cached by weight identity
    torch.testing.assert_close(out, out_again)
    # The bias is applied even though cfg.epilogue.bias is False (F.linear semantics).
    out_no_bias = linear(x, w, None, config=cfg)
    torch.testing.assert_close(
        (out.float() - out_no_bias.float()).mean(0), bias.float(), rtol=5e-2, atol=5e-2
    )

    pack = prepare_weight_pack(w, cfg, bias=bias)
    act = prepare_activations(x, quant=cfg.quant)
    out_pack = gemm(
        act, pack, config=dataclasses.replace(cfg, epilogue=EpilogueConfig(bias=True))
    )
    assert (
        F.cosine_similarity(ref.reshape(-1), out_pack.float().reshape(-1), dim=0) > 0.98
    )


# ---------------------------------------------------------------------------
# Caches: winner per weight shape, layer per normalized device, pack per weight
# ---------------------------------------------------------------------------


def _first_usable(pair_name: str) -> str:
    usable = _usable_backends(pair_name)
    if not usable:
        pytest.skip(f"no backend serves {pair_name} on sm{ARCH}")
    return usable[0]


def _spy_validate(monkeypatch) -> list:
    """Record every runner.validate call as the weight view's shape."""
    calls: list = []
    orig = R.GemmRunner.validate

    def spy(self, act, view, out):
        calls.append(tuple(view["b"].shape))
        return orig(self, act, view, out)

    monkeypatch.setattr(R.GemmRunner, "validate", spy)
    return calls


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_winner_is_cached_per_weight_shape(pair_name, monkeypatch):
    """One layer serving two weights of different (N, K): each shape is admitted
    (validate) and selected on its own, a repeat is a cache hit, and a K
    mismatch is rejected on every call, not only on the first one of a bucket."""
    cfg = _single(pair_name, _first_usable(pair_name))
    calls = _spy_validate(monkeypatch)
    x_a, w_a = _inputs(pair_name)
    x_b, w_b = _inputs(pair_name, n=2 * N, k=2 * K, seed=1)
    pack_a, pack_b = prepare_weight_pack(w_a, cfg), prepare_weight_pack(w_b, cfg)
    act_a = prepare_activations(x_a, quant=cfg.quant)
    act_b = prepare_activations(x_b, quant=cfg.quant)
    layer = GemmLayer(cfg, device=DEVICE)

    out_a = layer(act_a, pack_a)
    per_shape = len(calls)
    assert per_shape >= 1
    out_b = layer(act_b, pack_b)
    assert len(calls) == 2 * per_shape, "the second shape skipped admission"
    assert len(layer._winners) == 2
    for out, act, pack in ((out_a, act_a, pack_a), (out_b, act_b, pack_b)):
        ref = _dequant_reference(act, pack.get_view(layer.winner_backend), pair_name)
        assert (
            F.cosine_similarity(ref.reshape(-1), out.float().reshape(-1), dim=0) > 0.99
        )

    layer(act_a, pack_a)
    assert len(calls) == 2 * per_shape, "a repeat of a cached shape re-validated"

    # act K=K against weight K=2K: rejected by validate, never launched.
    with pytest.raises(BackendUnsupported):
        layer(act_a, pack_b)
    layer(act_a, pack_a)  # the cached winner of the matching pair is intact
    with pytest.raises(BackendUnsupported):
        layer(act_a, pack_b)


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_functional_doors_share_one_layer_per_device_and_key_per_shape(pair_name):
    cfg = _single(pair_name, _first_usable(pair_name))
    nn_mod._LAYER_CACHE.clear()
    bare = nn_mod._layer_for(cfg, torch.device("cuda"))
    indexed = nn_mod._layer_for(cfg, torch.device("cuda", torch.cuda.current_device()))
    assert bare is indexed and len(nn_mod._LAYER_CACHE) == 1

    x_a, w_a = _inputs(pair_name)
    x_b, w_b = _inputs(pair_name, n=2 * N, k=2 * K, seed=1)
    gemm(
        prepare_activations(x_a, quant=cfg.quant),
        prepare_weight_pack(w_a, cfg),
        config=cfg,
    )
    gemm(
        prepare_activations(x_b, quant=cfg.quant),
        prepare_weight_pack(w_b, cfg),
        config=cfg,
    )
    assert len(nn_mod._LAYER_CACHE) == 1
    assert len(bare._winners) == 2


@pytest.mark.parametrize("pair_name", ["nvfp4", "fp8"])
def test_linear_pack_cache_holds_live_weights_and_warns_on_eviction(
    pair_name, monkeypatch
):
    cfg = _single(pair_name, _first_usable(pair_name))
    prepared = {"n": 0}
    orig = nn_mod.prepare_weight_pack

    def counting(*a, **k):
        prepared["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(nn_mod, "prepare_weight_pack", counting)
    x, _ = _inputs(pair_name)
    weights = [torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)]

    # Within the limit: every weight is prepared once, however many forwards.
    monkeypatch.setattr(nn_mod, "_PACK_CACHE_LIMIT", 3)
    nn_mod._PACK_CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(3):
            for w in weights:
                linear(x, w, config=cfg)
    assert prepared["n"] == 3

    # One over the limit: the eviction is loud.
    monkeypatch.setattr(nn_mod, "_PACK_CACHE_LIMIT", 2)
    nn_mod._PACK_CACHE.clear()
    prepared["n"] = 0
    with pytest.warns(RuntimeWarning, match="prepared-weight cache"):
        for w in weights:
            linear(x, w, config=cfg)
    assert prepared["n"] == 3
    nn_mod._PACK_CACHE.clear()


def test_nvfp4_alpha_is_formed_once_per_call_and_never_with_a_static_scale():
    backend = _first_usable("nvfp4")
    cfg = _single("nvfp4", backend)
    x, w = _inputs("nvfp4")
    pack = prepare_weight_pack(w, cfg)
    view = pack.get_view(backend)
    torch.testing.assert_close(
        view["b_global_scale_inv"], (1.0 / view["b_global_scale"]).reshape(1)
    )

    # Dynamic activation scale: one alpha per (activation pack, weight view),
    # equal to the legacy 1 / (a_gs * b_gs).
    act = prepare_activations(x, quant=NVFP4)
    layer = GemmLayer(cfg, device=DEVICE)
    layer.prepare(pack)
    layer(act)
    ((memo_view, alpha),) = act._alpha_by_view.values()
    assert memo_view is view
    expected = (1.0 / (act.a_global_scale * view["b_global_scale"])).reshape(1)
    torch.testing.assert_close(alpha, expected.to(torch.float32))
    layer(act)
    assert len(act._alpha_by_view) == 1

    # Static activation scale folded in at load: nothing is formed per call.
    static = prepare_weight_pack(w, cfg, activation_global_scale=act.a_global_scale)
    act_s = prepare_activations(x, quant=NVFP4, global_scale=act.a_global_scale)
    layer_s = GemmLayer(cfg, device=DEVICE)
    layer_s.prepare(static)
    out_s = layer_s(act_s)
    assert not act_s._alpha_by_view
    torch.testing.assert_close(out_s, layer(act), rtol=2e-2, atol=2e-2)
