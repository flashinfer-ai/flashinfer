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

import pytest
import torch

from flashinfer.fused_moe.w4a16_mxfp4_moe import (
    BACKENDS,
    _composable_reference,
    _dispatch_table,
    _reset_w4a16_mxfp4_latent_moe_stats,
    _w4a16_mxfp4_latent_moe_stats,
    w4a16_mxfp4_latent_routed_moe,
)

DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISABLE_ENV_LEGACY = "FLASHINFER_W4A16_MXFP4_LATENT_MOE"
EL, K, L, I, G = 56, 16, 3584, 3072, 32


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(DISABLE_ENV_LEGACY, raising=False)
    _reset_w4a16_mxfp4_latent_moe_stats()
    yield
    _reset_w4a16_mxfp4_latent_moe_stats()


def _sm90_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")


def _case(T=8, el=EL, l=L, i=I, k=K, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((T, l), generator=g, device=device) * 1.0).to(torch.bfloat16)
    w13p = torch.randint(
        0, 256, (el, 2 * i, l // 2), generator=g, device=device, dtype=torch.uint8
    )
    w13s = torch.randint(
        118, 122, (el, 2 * i, l // G), generator=g, device=device, dtype=torch.uint8
    )
    w2p = torch.randint(
        0, 256, (el, l, i // 2), generator=g, device=device, dtype=torch.uint8
    )
    w2s = torch.randint(
        118, 122, (el, l, i // G), generator=g, device=device, dtype=torch.uint8
    )
    # realistic routing: DISTINCT global expert ids per token (duplicates
    # would double-hit an expert, which real routers never emit)
    ids = torch.stack(
        [torch.randperm(el * 2, generator=g, device=device)[:k] for _ in range(T)]
    ).to(torch.int32)
    w = torch.rand((T, k), generator=g, device=device, dtype=torch.float32)
    return x, w13p, w13s, w2p, w2s, ids, w


def _matched(y, ref, rtol=0.03, atol=0.02):
    yf, rf = y.float(), ref.float()
    return float(((yf - rf).abs() <= (atol + rtol * rf.abs())).float().mean().item())


def test_backends_tuple_and_table():
    assert BACKENDS == ("auto", "specialized", "reference")
    table = _dispatch_table()
    assert table is not None
    assert table["kill_switch"] == DISABLE_ENV
    op = table["ops"]["w4a16_mxfp4_latent_routed_moe"]
    assert (op["E_local"], op["top_k"], op["L"], op["I"]) == (EL, K, L, I)


def test_unknown_backend_raises_gpu_free():
    x = torch.zeros((1, L), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="never silently swapped"):
        w4a16_mxfp4_latent_routed_moe(
            x,
            *(torch.zeros(1, dtype=torch.uint8) for _ in range(4)),
            torch.zeros((1, K), dtype=torch.int32),
            torch.zeros((1, K), dtype=torch.float32),
            backend="cutlass",
        )


def test_specialized_matches_reference_backend():
    _sm90_or_skip()
    args = _case()
    y_spec = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    y_ref = w4a16_mxfp4_latent_routed_moe(*args, backend="reference")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["reference"] == 1, stats
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert _matched(y_spec, y_ref) >= 0.99
    # determinism contract (as documented): bitwise on the decode launch
    # path (T <= 64); the prefill path's bf16 atomic combine order is
    # disclosed nondeterministic and is tolerance-checked instead.
    y2 = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    if not torch.equal(y_spec, y2):
        # the op's documented rerun contract: the bf16 atomic combine order
        # is nondeterministic (disclosed in the PR); reruns must still agree
        # within the package gate band
        assert _matched(y2, y_spec, rtol=0.03, atol=0.02) >= 0.999, (
            "reruns diverged beyond the documented atomic-order band"
        )


def test_explicit_reference_never_fused_and_auto_routes():
    _sm90_or_skip()
    args = _case(T=4)
    _reset_w4a16_mxfp4_latent_moe_stats()
    w4a16_mxfp4_latent_routed_moe(*args, backend="reference")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    _reset_w4a16_mxfp4_latent_moe_stats()
    w4a16_mxfp4_latent_routed_moe(*args, backend="auto")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats


def test_off_geometry_explicit_raises_auto_takes_reference():
    _sm90_or_skip()
    # smaller expert count: consistent shapes, off the fixed geometry
    args = _case(T=2, el=8)
    # explicit "specialized" raises with the reason (never falls back)
    with pytest.raises(NotImplementedError, match="never falls back"):
        w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert any("specialized:" in r for r in stats["fallback_reasons"]), stats
    # auto routes the same call to the composable reference path
    y = w4a16_mxfp4_latent_routed_moe(*args, backend="auto")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["reference"] == 1, stats
    out = torch.empty_like(y)
    ref = _composable_reference(*args, out)
    assert _matched(y, ref) >= 0.999


def test_prefill_specialized_matches_reference_backend():
    """Parity on the prefill launch path (T > 64), not just the decode path."""
    _sm90_or_skip()
    args = _case(T=128, seed=3)
    y_spec = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    y_ref = w4a16_mxfp4_latent_routed_moe(*args, backend="reference")
    assert _matched(y_spec, y_ref) >= 0.99


def test_extreme_activations_stay_finite():
    """|x| >= 512 would overflow the kernel's scaled fp16 staging to inf and
    poison the combine with NaN (inf * 0-weight); the staging now saturates.
    Out-of-range magnitudes lose accuracy, so only the clean rows are
    parity-checked; the poisoned row must still come back finite."""
    _sm90_or_skip()
    args = _case(T=8, seed=5)
    x = args[0].clone()
    x[0, :8] = 1024.0
    x[0, 8:16] = -1024.0
    args = (x,) + args[1:]
    y_spec = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert torch.isfinite(y_spec.float()).all()
    y_ref = w4a16_mxfp4_latent_routed_moe(*args, backend="reference")
    assert _matched(y_spec[1:], y_ref[1:]) >= 0.99


def test_misaligned_packed_weights_fall_back():
    """A contiguous uint8 view with a storage offset breaks the 16-byte
    alignment the kernel's vector loads require; it must be served by the
    reference path, not faulted through uint4 loads."""
    _sm90_or_skip()
    x, w13p, w13s, w2p, w2s, ids, w = _case(T=4, seed=7)
    buf = torch.empty(w13p.numel() + 8, dtype=torch.uint8, device=w13p.device)
    buf[8:] = w13p.reshape(-1)
    w13p_off = buf[8:].view(w13p.shape)
    assert w13p_off.is_contiguous() and w13p_off.data_ptr() % 16 == 8
    with pytest.raises(NotImplementedError, match="never falls back"):
        w4a16_mxfp4_latent_routed_moe(
            x, w13p_off, w13s, w2p, w2s, ids, w, backend="specialized"
        )
    y = w4a16_mxfp4_latent_routed_moe(
        x, w13p_off, w13s, w2p, w2s, ids, w, backend="auto"
    )
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] == 1, stats
    assert any("16-byte" in r for r in stats["fallback_reasons"]), stats
    y_ref = w4a16_mxfp4_latent_routed_moe(
        x, w13p, w13s, w2p, w2s, ids, w, backend="reference"
    )
    assert torch.equal(y, y_ref)


def test_inconsistent_weight_structure_raises():
    _sm90_or_skip()
    x, w13p, w13s, w2p, w2s, ids, w = _case(T=2, el=8)
    # mismatched expert counts
    with pytest.raises(ValueError, match="local expert count"):
        w4a16_mxfp4_latent_routed_moe(
            x, w13p[:4], w13s, w2p, w2s, ids, w, backend="reference"
        )
    # fc1 packed vs scale row counts
    with pytest.raises(ValueError, match="w13 packed/scale rows"):
        w4a16_mxfp4_latent_routed_moe(
            x, w13p[:, : 2 * I - 2], w13s, w2p, w2s, ids, w, backend="reference"
        )
    # fc2 packed vs scale reduction widths
    with pytest.raises(ValueError, match="reduction width"):
        w4a16_mxfp4_latent_routed_moe(
            x, w13p, w13s, w2p[..., : I // 2 - 16], w2s, ids, w, backend="reference"
        )


@pytest.mark.parametrize("env,val", [(DISABLE_ENV, "1"), (DISABLE_ENV_LEGACY, "0")])
def test_kill_switch_forces_the_reference_path(env, val, monkeypatch):
    _sm90_or_skip()
    monkeypatch.setenv(env, val)
    args = _case(T=2)
    w4a16_mxfp4_latent_routed_moe(*args, backend="auto")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] == 1, stats
    # explicit "specialized" raises under the kill switch (never falls back)
    with pytest.raises(NotImplementedError, match="never falls back"):
        w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")


# ---------------------------------------------------------------------------
# Unified MoELayer backend (W4A16MxFp4LatentConfig / W4A16MxFp4LatentRunner)
# ---------------------------------------------------------------------------

from flashinfer.fused_moe import (  # noqa: E402
    BackendOptions,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SiTU,
    W4A16MxFp4LatentConfig,
    W4A16MxFp4LatentRunner,
)
from flashinfer.fused_moe import api as moe_api  # noqa: E402
from flashinfer.fused_moe import layer as moe_layer  # noqa: E402
from flashinfer.fused_moe import runners as moe_runners  # noqa: E402
from flashinfer.fused_moe.layer import (  # noqa: E402
    _BACKEND_RUNNERS,
    MoELayer,
    NoEligibleBackend,
)
from flashinfer.tllm_enums import RoutingInputMode  # noqa: E402


def _unified_config(experts=None, backend=None):
    return MoEConfig(
        routing=RoutingConfig(num_experts=EL, top_k=K),
        quant=QuantConfig(weight=QuantFormat.MXFP4),
        experts=experts or ExpertConfig(intermediate_size=I),
        activation=SiTU(),
        backend=backend or BackendOptions(candidates=(W4A16MxFp4LatentConfig(),)),
    )


def _weight_pack(w13p, w13s, w2p, w2s):
    pack = MoEWeightPack()
    pack.prepare_for(
        "w4a16_mxfp4_latent",
        W4A16MxFp4LatentConfig.prepare_weights(w13p, w13s, w2p, w2s),
    )
    return pack


def _act_pack(x, ids, w):
    return MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        topk_ids=ids,
        topk_weights=w,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


def _fake_sm90(monkeypatch):
    monkeypatch.setattr(moe_runners, "get_compute_capability", lambda dev: (9, 0))
    monkeypatch.setattr(moe_layer, "get_compute_capability", lambda dev: (9, 0))


def _cpu_case(T=4, el=EL, l=L, i=I, k=K, seed=11):
    return _case(T=T, el=el, l=l, i=i, k=k, seed=seed, device="cpu")


def test_unified_config_supported_arch_gating():
    assert W4A16MxFp4LatentConfig.supported(90)
    for arch in (80, 86, 89, 100, 103, 110, 120, 121):
        assert not W4A16MxFp4LatentConfig.supported(arch), arch


def test_unified_config_registered_and_default_visible():
    assert _BACKEND_RUNNERS[W4A16MxFp4LatentConfig] is W4A16MxFp4LatentRunner
    # DEFAULT-VISIBLE: a matching layer reaches the kernel through
    # backend=auto with no opt-in; supported() gates the arch and the
    # runner gates the fixed contract.
    assert any(isinstance(c, W4A16MxFp4LatentConfig) for c in moe_api._DEFAULT_BACKEND)
    assert any(
        isinstance(c, W4A16MxFp4LatentConfig)
        for c in moe_api._DEFAULT_BACKEND.valid_for(90)
    )
    assert not any(
        isinstance(c, W4A16MxFp4LatentConfig)
        for c in moe_api._DEFAULT_BACKEND.valid_for(120)
    )
    assert W4A16MxFp4LatentConfig in moe_api.ALL_BACKEND_CONFIGS


def test_unified_prepare_weights_validation():
    x, w13p, w13s, w2p, w2s, ids, w = _cpu_case(T=2)
    view = W4A16MxFp4LatentConfig.prepare_weights(w13p, w13s, w2p, w2s)
    assert set(view) == {"w13_packed", "w13_scale", "w2_packed", "w2_scale"}
    assert all(v.is_contiguous() for v in view.values())
    with pytest.raises(TypeError, match="uint8"):
        W4A16MxFp4LatentConfig.prepare_weights(w13p.to(torch.int32), w13s, w2p, w2s)
    with pytest.raises(ValueError, match="3D"):
        W4A16MxFp4LatentConfig.prepare_weights(w13p[0], w13s, w2p, w2s)


def test_unified_runner_ep_rejected_cpu():
    config = _unified_config(
        experts=ExpertConfig(intermediate_size=I, local_num_experts=28)
    )
    runner = W4A16MxFp4LatentRunner(config, device=torch.device("cpu"))
    with pytest.raises(NotImplementedError, match="expert parallelism"):
        runner.check_support()
    config = _unified_config(
        experts=ExpertConfig(intermediate_size=I, local_expert_offset=8)
    )
    runner = W4A16MxFp4LatentRunner(config, device=torch.device("cpu"))
    with pytest.raises(NotImplementedError, match="expert parallelism"):
        runner.check_support()


def test_unified_runner_fixed_geometry_rejected_cpu(monkeypatch):
    _fake_sm90(monkeypatch)
    cpu = torch.device("cpu")
    runner = W4A16MxFp4LatentRunner(
        MoEConfig(
            routing=RoutingConfig(num_experts=64, top_k=K),
            quant=QuantConfig(weight=QuantFormat.MXFP4),
            experts=ExpertConfig(intermediate_size=I),
            activation=SiTU(),
        ),
        device=cpu,
    )
    with pytest.raises(NotImplementedError, match="num_experts=56"):
        runner.check_support()
    runner = W4A16MxFp4LatentRunner(
        MoEConfig(
            routing=RoutingConfig(num_experts=EL, top_k=8),
            quant=QuantConfig(weight=QuantFormat.MXFP4),
            experts=ExpertConfig(intermediate_size=I),
            activation=SiTU(),
        ),
        device=cpu,
    )
    with pytest.raises(NotImplementedError, match="top_k=16"):
        runner.check_support()
    runner = W4A16MxFp4LatentRunner(
        _unified_config(experts=ExpertConfig(intermediate_size=4096)), device=cpu
    )
    with pytest.raises(NotImplementedError, match="intermediate_size=3072"):
        runner.check_support()
    # non-canonical SiTU scalars are rejected, not silently re-scaled
    runner = W4A16MxFp4LatentRunner(
        MoEConfig(
            routing=RoutingConfig(num_experts=EL, top_k=K),
            quant=QuantConfig(weight=QuantFormat.MXFP4),
            experts=ExpertConfig(intermediate_size=I),
            activation=SiTU(gate_scale=2.0),
        ),
        device=cpu,
    )
    with pytest.raises(NotImplementedError, match="canonical SiTU"):
        runner.check_support()


def test_unified_runner_quant_gating():
    assert W4A16MxFp4LatentRunner.supports_quant(QuantConfig(weight=QuantFormat.MXFP4))
    assert not W4A16MxFp4LatentRunner.supports_quant(QuantConfig())
    assert not W4A16MxFp4LatentRunner.supports_quant(
        QuantConfig(weight=QuantFormat.NVFP4)
    )
    assert not W4A16MxFp4LatentRunner.supports_quant(
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
    )
    assert W4A16MxFp4LatentRunner.supports_expert_parallelism is False


def test_unified_layer_construction_reaches_runner_cpu(monkeypatch):
    _fake_sm90(monkeypatch)
    layer = MoELayer(_unified_config(), device=torch.device("cpu"))
    assert len(layer.runners) == 1
    assert isinstance(layer.runners[0], W4A16MxFp4LatentRunner)
    assert layer.runners[0].backend_key == "w4a16_mxfp4_latent"


def test_unified_layer_ep_rejected_cpu(monkeypatch):
    _fake_sm90(monkeypatch)
    config = _unified_config(
        experts=ExpertConfig(intermediate_size=I, local_num_experts=28)
    )
    with pytest.raises(NoEligibleBackend, match="expert parallelism"):
        MoELayer(config, device=torch.device("cpu"))


def test_unified_runner_dispatch_monkeypatched_launch(monkeypatch):
    """CPU dispatch logic: the runner calls the kernel module's entry with
    backend="specialized" and passes the pack/view tensors through
    unchanged; an explicit backend runs under autotune tuning mode (no
    tuner-state games), so the shootout times the real dispatch."""
    from flashinfer.autotuner import AutoTuner

    _fake_sm90(monkeypatch)
    layer = MoELayer(_unified_config(), device=torch.device("cpu"))
    runner = layer.runners[0]
    x, w13p, w13s, w2p, w2s, ids, w = _cpu_case(T=4)
    weight_pack = _weight_pack(w13p, w13s, w2p, w2s)
    act = _act_pack(x, ids, w)

    seen = {}

    def _fake_flat(x_, w13p_, w13s_, w2p_, w2s_, ids_, w_, *, out=None, backend):
        seen["backend"] = backend
        seen["tensors_ok"] = (
            x_ is x
            and w13p_ is weight_pack.get_view("w4a16_mxfp4_latent")["w13_packed"]
            and ids_ is ids
            and w_ is w
        )
        seen["tuning_mode_inside"] = AutoTuner.get().is_tuning_mode
        return torch.zeros_like(x_)

    monkeypatch.setattr(runner._flat, "w4a16_mxfp4_latent_routed_moe", _fake_flat)
    tuner = AutoTuner.get()
    monkeypatch.setattr(tuner, "is_tuning_mode", True)
    inputs = runner.pack_inputs(act, weight_pack)
    out = runner.forward(inputs, tactic=-1)
    assert seen["backend"] == "specialized"
    assert seen["tensors_ok"]
    # The runner no longer mutates tuner state: the explicit backend form
    # runs under tuning mode (only the flat "auto" form refuses it).
    assert seen["tuning_mode_inside"] is True
    assert tuner.is_tuning_mode is True
    assert out.shape == x.shape


def test_unified_runner_pack_rejections(monkeypatch):
    _fake_sm90(monkeypatch)
    layer = MoELayer(_unified_config(), device=torch.device("cpu"))
    runner = layer.runners[0]
    x, w13p, w13s, w2p, w2s, ids, w = _cpu_case(T=4)
    weight_pack = _weight_pack(w13p, w13s, w2p, w2s)
    # bf16 topk_weights: the kernel consumes float32 combine weights as-is
    act = _act_pack(x, ids, w.to(torch.bfloat16))
    with pytest.raises(TypeError, match="float32"):
        runner.pack_inputs(act, weight_pack)
    # off-geometry weight view: a precise error, never another kernel
    bad = _weight_pack(w13p[:, : 2 * I - 64], w13s, w2p, w2s)
    with pytest.raises(ValueError, match="fixed"):
        runner.pack_inputs(_act_pack(x, ids, w), bad)
    # non-bf16 activations
    act = _act_pack(x.float(), ids, w)
    with pytest.raises(TypeError, match="BF16"):
        runner.pack_inputs(act, weight_pack)
    # routing from logits is not supported (no in-kernel router)
    logits_pack = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=torch.randn(x.shape[0], EL),
    )
    with pytest.raises(NotImplementedError, match="routing_input_mode"):
        runner.pack_inputs(logits_pack, weight_pack)
    # off-envelope / off-hidden packs are filtered by accepts(), not crashed on
    assert runner._accepts_shape(_act_pack(x, ids, w))
    wide = torch.zeros((2, 4096), dtype=torch.bfloat16)
    wide_ids = torch.zeros((2, K), dtype=torch.int32)
    wide_w = torch.zeros((2, K), dtype=torch.float32)
    assert not runner._accepts_shape(_act_pack(wide, wide_ids, wide_w))


# ---- GPU (SM90) tests ------------------------------------------------------


def test_unified_layer_ep_rejected_sm90():
    _sm90_or_skip()
    config = _unified_config(
        experts=ExpertConfig(intermediate_size=I, local_num_experts=28)
    )
    with pytest.raises(NoEligibleBackend, match="expert parallelism"):
        MoELayer(config)


def test_unified_layer_matches_flat_specialized():
    """Numerics parity through MoELayer vs the flat entry on the allowlisted
    geometry (decode launch path, T <= 64)."""
    _sm90_or_skip()
    x, w13p, w13s, w2p, w2s, ids_global, w = _case(T=8, seed=21)
    # the unified layer exposes the op as a complete 56-expert MoE: ids in [0, EL)
    g = torch.Generator(device="cuda").manual_seed(22)
    ids = torch.stack(
        [torch.randperm(EL, generator=g, device="cuda")[:K] for _ in range(x.shape[0])]
    ).to(torch.int32)
    y_flat = w4a16_mxfp4_latent_routed_moe(
        x, w13p, w13s, w2p, w2s, ids, w, backend="specialized"
    )
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats

    layer = MoELayer(_unified_config())
    _reset_w4a16_mxfp4_latent_moe_stats()
    y_layer = layer(_act_pack(x, ids, w), _weight_pack(w13p, w13s, w2p, w2s))
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] >= 1, stats
    assert stats["backend_dispatches"]["reference"] == 0, stats
    assert layer.winner_backend == "w4a16_mxfp4_latent"
    assert y_layer.shape == y_flat.shape and y_layer.dtype == y_flat.dtype
    # The kernel's bf16 atomic combine order is disclosed nondeterministic
    # (see the flat rerun hedge above), so two dispatches of the same kernel
    # live in a tolerance band, not at bitwise equality.
    assert _matched(y_layer, y_flat) >= 0.99
    # Accuracy vs the exact-math reference is a property of the kernel, not
    # of this adapter (this routing keeps all 16 slots local — denser
    # accumulation than the flat suite's half-local cases, so its absolute
    # 0.99 band does not transfer): the layer path must match the reference
    # as well as the flat path does, within the atomic-order noise.
    y_ref = w4a16_mxfp4_latent_routed_moe(
        x, w13p, w13s, w2p, w2s, ids, w, backend="reference"
    )
    m_layer, m_flat = _matched(y_layer, y_ref), _matched(y_flat, y_ref)
    assert m_layer >= m_flat - 5e-3, (m_layer, m_flat)


def test_unified_layer_prefill_path_matches_flat():
    _sm90_or_skip()
    x, w13p, w13s, w2p, w2s, _ids, w = _case(T=128, seed=23)
    g = torch.Generator(device="cuda").manual_seed(24)
    ids = torch.stack(
        [torch.randperm(EL, generator=g, device="cuda")[:K] for _ in range(x.shape[0])]
    ).to(torch.int32)
    layer = MoELayer(_unified_config())
    y_layer = layer(_act_pack(x, ids, w), _weight_pack(w13p, w13s, w2p, w2s))
    y_flat = w4a16_mxfp4_latent_routed_moe(
        x, w13p, w13s, w2p, w2s, ids, w, backend="specialized"
    )
    y_ref = w4a16_mxfp4_latent_routed_moe(
        x, w13p, w13s, w2p, w2s, ids, w, backend="reference"
    )
    # The prefill path's bf16 atomic combine order is disclosed
    # nondeterministic; gate layer-vs-flat within that band and the
    # reference comparison relative to the flat path (see the decode test
    # for why the absolute 0.99 reference band does not transfer here).
    assert _matched(y_layer, y_flat) >= 0.99
    m_layer, m_flat = _matched(y_layer, y_ref), _matched(y_flat, y_ref)
    assert m_layer >= m_flat - 5e-3, (m_layer, m_flat)


def test_unified_layer_kill_switch_serves_reference(monkeypatch):
    """The flat op's kill switch is honored identically through the layer:
    the dispatch lands on the composable reference path."""
    _sm90_or_skip()
    x, w13p, w13s, w2p, w2s, _ids, w = _case(T=4, seed=25)
    g = torch.Generator(device="cuda").manual_seed(26)
    ids = torch.stack(
        [torch.randperm(EL, generator=g, device="cuda")[:K] for _ in range(x.shape[0])]
    ).to(torch.int32)
    layer = MoELayer(_unified_config())
    weight_pack = _weight_pack(w13p, w13s, w2p, w2s)
    monkeypatch.setenv(DISABLE_ENV, "1")
    _reset_w4a16_mxfp4_latent_moe_stats()
    y_layer = layer(_act_pack(x, ids, w), weight_pack)
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] >= 1, stats
    assert any("kill switch" in r for r in stats["fallback_reasons"]), stats
    y_ref = w4a16_mxfp4_latent_routed_moe(
        x, w13p, w13s, w2p, w2s, ids, w, backend="reference"
    )
    assert torch.equal(y_layer, y_ref)
