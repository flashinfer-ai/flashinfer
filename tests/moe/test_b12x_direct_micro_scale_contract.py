"""Direct scale contract against precise static and decoded-weight FP32 MoE.

This contract checks scale meaning across different quantization implementations.
It is separate from the default fast-MMA numerical comparison.
"""

import pytest
import torch

from flashinfer import B12xMoEWrapper
from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.cute_dsl.utils import convert_sf_from_mma_layout
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
    unswizzle_block_scale,
)

from .utils import create_moe_tensors

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_cute_dsl_available()),
    reason="CUDA + CuTe-DSL required",
)

E, H, I, TOPK = 512, 2560, 320, 10


def _decode_weight(packed, scales):
    # The checkpoint bytes store the low nibble first. These are logical
    # weights before either activation quantizer, with no kernel arithmetic.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
        dtype=torch.float32,
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).long()
    values = lut[codes].reshape(packed.shape[0], packed.shape[1] * 2)
    return values * scales.repeat_interleave(16, dim=1)


@pytest.fixture(scope="module")
def scale_inputs():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 required")
    torch.backends.cuda.matmul.allow_tf32 = False
    t = create_moe_tensors(
        num_tokens=6,
        hidden_size=H,
        intermediate_size=I,
        num_experts=E,
        num_local_experts=E,
        top_k=TOPK,
        seed=20260916,
        interleave_gated_weights=False,
        use_nontrivial_alphas=True,
    )
    decoded = {}
    for key, rows, cols in (("w1", 2 * I, H), ("w2", H, I)):
        sf = t[key + "_weight_sf"]
        assert torch.unique(sf.view(torch.uint8).flatten()[:4096]).numel() > 1
        sw = convert_sf_from_mma_layout(sf, rows, cols, E, 16)
        sw = sw.reshape(E, ((rows + 127) // 128) * 128, -1)
        decoded[key] = {
            e: _decode_weight(
                t[key + "_weight"][e],
                unswizzle_block_scale(sw[e], rows, cols // 16),
            )
            for e in (*range(9), E - 1)
        }
    return t, decoded


def _wrapper():
    return B12xMoEWrapper(
        num_experts=E,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=I,
        use_cuda_graph=True,
        max_num_tokens=8192,
    )


def _kwargs(t, m, scale_kind, duplicate=False):
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, E - 1], device="cuda")
    if duplicate:
        ids.zero_()
    kwargs = dict(
        x=t["x_bf16"][:m],
        w1_weight=t["w1_weight"],
        w1_weight_sf=t["w1_weight_sf"],
        w1_alpha=t["w1_alpha"],
        w2_weight=t["w2_weight"],
        w2_weight_sf=t["w2_weight_sf"],
        w2_alpha=t["w2_alpha"],
        token_selected_experts=ids.to(torch.int32).repeat(m, 1),
        token_final_scales=t["token_final_scales"][:m],
        fc2_input_scale=torch.linspace(0.7, 1.3, E, device="cuda"),
    )
    if scale_kind == "scalar":
        kwargs["input_global_scale"] = torch.tensor(0.8, device="cuda")
        kwargs["fc2_input_scale"] = torch.tensor(0.7, device="cuda")
    elif scale_kind == "vector":
        kwargs["input_global_scale"] = torch.linspace(0.8, 1.2, E, device="cuda")
    return kwargs


def _expert_value(t, e):
    return t.flatten()[0 if t.numel() == 1 else e]


def _fp32_reference(kwargs, decoded):
    """FP32 logical MoE before activation rounding, using exact packed weights.

    MMA quantization divides activations by the ordinary API global scale.
    Explicit FC1 globals are folded into alpha by the wrapper, cancelling
    that divisor; legacy alpha serves both purposes. FC2 alpha stays separate.
    """
    x = kwargs["x"].float()
    ids, routing = kwargs["token_selected_experts"], kwargs["token_final_scales"]
    out = torch.zeros_like(x)
    input_gs = kwargs.get("input_global_scale", kwargs["w1_alpha"])
    for e in torch.unique(ids).tolist():
        d1 = _expert_value(input_gs, e)
        d2 = _expert_value(kwargs["fc2_input_scale"], e)
        if d1.item() == 0 or d2.item() == 0:
            continue
        alpha1 = kwargs["w1_alpha"][e]
        if "input_global_scale" in kwargs:
            alpha1 = alpha1 * d1
        uv = (x @ decoded["w1"][e].T) * (alpha1 / d1)
        activated = torch.nn.functional.silu(uv[:, I:]) * uv[:, :I]
        projected = (activated @ decoded["w2"][e].T) * (kwargs["w2_alpha"][e] / d2)
        weight = (routing * (ids == e)).sum(-1, keepdim=True)
        out += projected * weight
    return out


def _run(moe, kwargs, backend):
    old = md._FORCED_BACKEND
    md._FORCED_BACKEND = backend
    try:
        return moe.run(**kwargs)
    finally:
        md._FORCED_BACKEND = old


def _static_precise(moe, kwargs):
    # The wrapper has no fast_math argument. Keep its ordinary public scale
    # folding and prepared views, overriding only the internal reference's
    # compile option. Every case uses this same reference method.
    original = md.launch_sm120_static_moe

    def precise_launch(**launch_kwargs):
        launch_kwargs["fast_math"] = False
        return original(**launch_kwargs)

    md.launch_sm120_static_moe = precise_launch
    try:
        return _run(moe, kwargs, "static")
    finally:
        md.launch_sm120_static_moe = original


def _inverse(t):
    return torch.where(t != 0, 1.0 / t, t)


def _reciprocal_call(moe, kwargs, input_recip, fc2_recip):
    # Exercise the lower-level reciprocal flag with a prepared weight view:
    # the FC1 alpha fold must remain the same as for the ordinary wrapper.
    old = md._FORCED_BACKEND
    md._FORCED_BACKEND = "direct_micro"
    try:
        m = kwargs["x"].shape[0]
        return md.launch_sm120_static_moe(
            workspace=moe._static_workspace,
            weights=moe._weight_views,
            a=kwargs["x"],
            topk_ids=kwargs["token_selected_experts"],
            topk_weights=kwargs["token_final_scales"],
            input_gs=input_recip,
            down_input_scale=fc2_recip,
            scatter_output=moe._moe_output[:m],
            num_experts=E,
            num_tokens=m,
            k=H,
            n=moe._static_workspace.n,
            top_k=TOPK,
            input_scales_are_reciprocal=True,
        )
    finally:
        md._FORCED_BACKEND = old


def _error(actual, expected, tolerance):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    error = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    assert error < tolerance, error


def _poison(moe):
    moe._static_workspace.dm_intermediate.fill_(float("nan"))


def _trace_getters(monkeypatch):
    calls = []
    for name, backend in (
        ("_get_direct_micro_kernel", "direct_micro"),
        ("_get_micro_kernel", "micro"),
        ("_get_static_kernel", "static"),
    ):
        original = getattr(md, name)

        def traced(*args, _original=original, _backend=backend, **kwargs):
            calls.append(_backend)
            return _original(*args, **kwargs)

        monkeypatch.setattr(md, name, traced)
    return calls


def _check(actual, reference, oracle):
    _error(actual, reference, 0.03)
    _error(actual, oracle, 0.35)
    _error(reference, oracle, 0.35)


@pytest.mark.parametrize(
    "m,scale_kind,duplicate",
    [(m, kind, False) for m in (1, 2, 3, 4) for kind in ("scalar", "vector", "legacy")]
    + [(4, "vector", True)],
)
def test_direct_micro_scale_contract(
    scale_inputs, monkeypatch, m, scale_kind, duplicate
):
    monkeypatch.setattr(md, "_FORCED_BACKEND", None)
    monkeypatch.delenv(md._STATIC_SOURCE_SCALES_ENV, raising=False)
    calls = _trace_getters(monkeypatch)
    backend = None if m == 4 else "direct_micro"
    t, decoded = scale_inputs
    kwargs = _kwargs(t, m, scale_kind, duplicate)
    actual_moe, reference_moe = _wrapper(), _wrapper()
    _poison(actual_moe)
    actual = _run(actual_moe, kwargs, backend).clone()
    assert calls[-1] == "direct_micro"
    assert actual_moe._weight_views.source_scales
    reference = _static_precise(reference_moe, kwargs).clone()
    oracle = _fp32_reference(kwargs, decoded)
    _error(actual, reference, 0.03)
    _error(actual, oracle, 0.35)
    _error(reference, oracle, 0.35)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(actual_moe, kwargs, backend)
    assert calls[-1] == "direct_micro"
    for _ in range(3):
        _poison(actual_moe)
        graph.replay()
        _error(captured, actual, 0.03)
        _error(captured, oracle, 0.35)

    input_recip = _inverse(kwargs.get("input_global_scale", kwargs["w1_alpha"]))
    fc2_recip = _inverse(kwargs["fc2_input_scale"])
    reciprocal = _reciprocal_call(actual_moe, kwargs, input_recip, fc2_recip).clone()
    _error(reciprocal, actual, 0.03)
    reciprocal_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(reciprocal_graph):
        reciprocal_buffer = _reciprocal_call(actual_moe, kwargs, input_recip, fc2_recip)
    for _ in range(3):
        _poison(actual_moe)
        reciprocal_graph.replay()
        _error(reciprocal_buffer, actual, 0.03)
        _error(reciprocal_buffer, oracle, 0.35)

    # Change the caller tensor in place, including an active zero-scale expert.
    # The already captured graph must execute the conversion on the new values.
    fc2_scale = kwargs["fc2_input_scale"]
    original_ptr = fc2_scale.data_ptr()
    if fc2_scale.numel() == 1:
        fc2_scale.zero_()
    else:
        fc2_scale.copy_(torch.linspace(1.3, 0.7, E, device="cuda"))
        fc2_scale[0] = 0
    assert fc2_scale.data_ptr() == original_ptr
    fc2_recip.copy_(_inverse(fc2_scale))
    updated = _run(actual_moe, kwargs, backend).clone()
    assert calls[-1] == "direct_micro"
    updated_reference = _static_precise(reference_moe, kwargs).clone()
    updated_oracle = _fp32_reference(kwargs, decoded)
    assert not torch.equal(actual, updated), "live scale must affect the output"
    _error(updated, updated_reference, 0.03)
    _error(updated, updated_oracle, 0.35)
    _error(updated_reference, updated_oracle, 0.35)
    for _ in range(3):
        for replay, buffer in (
            (graph, captured),
            (reciprocal_graph, reciprocal_buffer),
        ):
            _poison(actual_moe)
            replay.replay()
            _error(buffer, updated, 0.03)
            _error(buffer, updated_oracle, 0.35)
    if fc2_scale.numel() == 1 or duplicate:
        assert torch.count_nonzero(updated) == 0


@pytest.mark.parametrize("fallback", ["source_off", "w1_strided", "w2_strided"])
def test_m4_source_layout_fallback(scale_inputs, monkeypatch, fallback):
    monkeypatch.setattr(md, "_FORCED_BACKEND", None)
    monkeypatch.delenv(md._STATIC_SOURCE_SCALES_ENV, raising=False)
    calls = _trace_getters(monkeypatch)
    t, decoded = scale_inputs
    kwargs = _kwargs(t, 4, "vector")
    if fallback == "source_off":
        monkeypatch.setenv(md._STATIC_SOURCE_SCALES_ENV, "0")
    else:
        key = "w1_weight" if fallback == "w1_strided" else "w2_weight"
        packed = kwargs[key]
        kwargs[key] = torch.stack((packed, torch.zeros_like(packed)), -1)[..., 0]
        assert not kwargs[key].is_contiguous()
    candidate, reference_moe = _wrapper(), _wrapper()
    _poison(candidate)
    actual = _run(candidate, kwargs, None).clone()
    assert calls[-1] == "micro"
    # Static's compiled TMA descriptor expects contiguous packed weights.
    # Keep the candidate strided, and give the oracle the same weight values
    # in the original contiguous layout rather than testing static strides.
    reference_kwargs = dict(kwargs)
    for key in ("w1_weight", "w2_weight"):
        assert torch.equal(kwargs[key], t[key])
        reference_kwargs[key] = t[key]
    reference = _static_precise(reference_moe, reference_kwargs).clone()
    oracle = _fp32_reference(kwargs, decoded)
    _check(actual, reference, oracle)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(candidate, kwargs, None)
    assert calls[-1] == "micro"
    for _ in range(3):
        _poison(candidate)
        graph.replay()
        _check(captured, reference, oracle)
        _error(captured, actual, 0.03)


def test_m3_m4_m6_m4_shared_wrapper_graph(scale_inputs, monkeypatch):
    monkeypatch.setattr(md, "_FORCED_BACKEND", None)
    monkeypatch.delenv(md._STATIC_SOURCE_SCALES_ENV, raising=False)
    calls = _trace_getters(monkeypatch)
    t, decoded = scale_inputs
    kwargs = {m: _kwargs(t, m, "vector") for m in (3, 4, 6)}
    # Hold the same FC1/FC2 scale tensors through every warm-up and replay.
    # Distinct FC1 tensor pointers would test an unrelated folded-alpha cache.
    for m in (4, 6):
        for key in ("input_global_scale", "fc2_input_scale"):
            kwargs[m][key] = kwargs[3][key]
    candidate, reference_moe = _wrapper(), _wrapper()
    selected = {3: "direct_micro", 4: "direct_micro", 6: "static"}
    expected, oracles, eager, graphs, buffers = {}, {}, {}, {}, {}
    for m in (3, 4, 6):
        _poison(candidate)
        eager[m] = _run(candidate, kwargs[m], None).clone()
        assert calls[-1] == selected[m]
        expected[m] = _static_precise(reference_moe, kwargs[m]).clone()
        oracles[m] = _fp32_reference(kwargs[m], decoded)
        _check(eager[m], expected[m], oracles[m])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            buffers[m] = _run(candidate, kwargs[m], None)
        graphs[m] = graph
        assert calls[-1] == selected[m]
    for _ in range(3):
        for m in (3, 4, 6, 4):
            _poison(candidate)
            actual = _run(candidate, kwargs[m], None).clone()
            assert calls[-1] == selected[m]
            _check(actual, expected[m], oracles[m])
            _poison(candidate)
            graphs[m].replay()
            _check(buffers[m], expected[m], oracles[m])
            _error(buffers[m], eager[m], 0.03)
