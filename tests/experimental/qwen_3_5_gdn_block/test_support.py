# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the geometry, the support probe (never raises; typed declines), the plan-time errors, the backend registry
and plan cache with a fake backend, and import isolation (no kernel or DSL load until a FROST plan is built)."""

import os
import pathlib
import subprocess
import sys

import pytest
import torch

from flashinfer.experimental.qwen_3_5_gdn_block import api, support
from flashinfer.experimental.qwen_3_5_gdn_block.backends.frost import admission
from flashinfer.qwen_3_5_gdn_block import (
    Qwen35GdnDecline,
    Qwen35GdnGeometry,
    prepare_qwen_3_5_gdn_block,
    qwen_3_5_gdn_block_decode,
    qwen_3_5_gdn_block_support,
    qwen_3_5_gdn_block_verify,
)
from tests.experimental.qwen_3_5_gdn_block.reference import QWEN38_27B

pytestmark = pytest.mark.cpu_only

FROST = Qwen35GdnGeometry(**QWEN38_27B)
CPU = torch.device("cpu")


def test_geometry_form_checks_and_derived_widths():
    assert (FROST.key_dim, FROST.value_dim, FROST.conv_dim) == (2048, 6144, 10240)
    assert FROST.conv_state_len(1) == 3 and FROST.conv_state_len(6) == 8
    for bad, match in (
        (dict(QWEN38_27B, d_model=0), "positive int"),
        (dict(QWEN38_27B, num_k_heads=True), "positive int"),
        (dict(QWEN38_27B, head_k_dim=128.0), "positive int"),
        (dict(QWEN38_27B, num_v_heads=40), "multiple of"),
        (dict(QWEN38_27B, conv_kernel=1), "at least 2"),
        (dict(QWEN38_27B, norm_eps=0.0), "norm_eps"),
    ):
        with pytest.raises(ValueError, match=match):
            Qwen35GdnGeometry(**bad)


@pytest.mark.parametrize(
    "args, kind, needle",
    [
        ((CPU, "not a geometry", "decode"), "argument", "Qwen35GdnGeometry"),
        ((CPU, FROST, "prefill"), "form", "prefill"),
        ((CPU, FROST, "decode", "nope"), "backend", "nope"),
        ((CPU, FROST, "decode", 7), "backend", "7"),
        (
            (CPU, Qwen35GdnGeometry(**dict(QWEN38_27B, d_model=4096)), "decode"),
            "geometry",
            "d_model",
        ),
        (
            (CPU, Qwen35GdnGeometry(**dict(QWEN38_27B, num_v_heads=32)), "verify"),
            "geometry",
            "num_v_heads",
        ),
        ((CPU, FROST, "decode"), "device", "no CUDA device"),
        (("not a device", FROST, "verify"), "device", "no CUDA device"),
    ],
)
def test_support_declines_are_typed_and_never_raise(args, kind, needle):
    sup = qwen_3_5_gdn_block_support(*args)
    assert not sup.served and sup.kind == kind, sup
    assert needle in sup.reason, sup.reason
    assert support.unsupported_reason(*args) == sup.reason


@pytest.mark.parametrize(
    "kw, kind",
    [
        (dict(requests=0), "argument"),
        (dict(requests=2.0), "argument"),
        (dict(tokens_per_request=-1), "argument"),
        (dict(tokens_per_request=True), "argument"),
    ],
)
def test_malformed_step_shapes_are_argument_declines(kw, kind):
    sup = qwen_3_5_gdn_block_support(CPU, FROST, "verify", **kw)
    assert not sup.served and sup.kind == kind, sup


@pytest.fixture
def as_sm100(monkeypatch):
    """The FROST admission as it reads on a 148-SM SM100 part with CuTe DSL installed (device facts and DSL probe
    patched)."""
    monkeypatch.setattr(admission, "device_cc", lambda device: (10, 0))
    monkeypatch.setattr(admission, "_sm_count", lambda device: 148)
    monkeypatch.setattr(
        "flashinfer.cute_dsl.availability.is_cute_dsl_available", lambda: True
    )


@pytest.mark.parametrize(
    "form, requests, tokens, served",
    [
        ("decode", None, None, True),
        ("decode", 1, None, True),
        ("decode", 512, 1, True),
        ("decode", 4, 2, False),
        ("verify", None, None, True),
        ("verify", 1, 10, True),
        ("verify", 1, 11, False),
        ("verify", 2, 8, True),
        ("verify", 2, 9, False),
        ("verify", 4, 6, True),
        ("verify", 5, 4, False),
        ("verify", 3, 1, False),
        ("verify", None, 11, False),
    ],
)
def test_frost_step_shape_envelope(as_sm100, form, requests, tokens, served):
    sup = qwen_3_5_gdn_block_support(
        "cuda:0", FROST, form, requests=requests, tokens_per_request=tokens
    )
    assert sup.served == served, sup
    if not served:
        assert sup.kind == "shape", sup


def test_a_raising_support_function_is_a_backend_decline():
    def support_fn(*args):
        raise RuntimeError("boom")

    api.register_backend("raises", support_fn, lambda spec: None)
    try:
        sup = qwen_3_5_gdn_block_support(CPU, FROST, "decode", "raises")
        assert not sup.served and sup.kind == "backend" and "boom" in sup.reason
    finally:
        api.unregister_backend("raises")


def test_plan_time_errors_are_typed():
    with pytest.raises(NotImplementedError, match="cc 10.0"):
        prepare_qwen_3_5_gdn_block("decode", geometry=FROST, device=CPU, requests=1)
    with pytest.raises(NotImplementedError, match="d_model"):
        prepare_qwen_3_5_gdn_block(
            "decode",
            geometry=Qwen35GdnGeometry(**dict(QWEN38_27B, d_model=2048)),
            device=CPU,
            requests=1,
        )
    with pytest.raises(ValueError, match="requests"):
        prepare_qwen_3_5_gdn_block("decode", geometry=FROST, device=CPU, requests=0)
    with pytest.raises(ValueError, match="param_dtypes"):
        prepare_qwen_3_5_gdn_block(
            "decode",
            geometry=FROST,
            device=CPU,
            requests=1,
            param_dtypes=(torch.float16,) * 3,
        )


# --------------------------------------------------------------------------- the registry with a fake backend

SMALL = Qwen35GdnGeometry(
    d_model=64,
    num_v_heads=4,
    num_k_heads=2,
    head_k_dim=16,
    head_v_dim=16,
    conv_kernel=4,
)


class _Recorder:
    def __init__(self, spec):
        self.spec, self.prepared, self.ran = spec, 0, []

    def prepare(self, inputs):
        self.prepared += 1

    def run(self, inputs):
        inputs.out.view(torch.uint8).fill_(1)
        if inputs.out_bf16 is not None:
            inputs.out_bf16.fill_(2.0)
        self.ran.append(inputs)


@pytest.fixture
def fake_backend():
    built = []

    def support_fn(device, geometry, form, requests, tokens_per_request):
        if torch.device(device).type != "cpu":
            return Qwen35GdnDecline("device", "the fake backend runs on the CPU")
        return None

    def build_fn(spec):
        built.append(_Recorder(spec))
        return built[-1]

    api.register_backend("fake", support_fn, build_fn)
    yield built
    api.unregister_backend("fake")
    api.clear_plan_cache()


def _tensors(g, requests, tokens, form, slots=6, param_dtype=torch.bfloat16):
    rows = requests * tokens
    hv, dv, dk = g.num_v_heads, g.head_v_dim, g.head_k_dim
    qkvz = torch.zeros(rows, g.conv_dim + hv * dv, dtype=torch.bfloat16)
    t = dict(
        qkv=qkvz[:, : g.conv_dim],
        z=qkvz[:, g.conv_dim :].view(rows, hv, dv),
        hidden=torch.zeros(rows, g.d_model, dtype=torch.bfloat16),
        ba_weight=torch.zeros(2 * hv, g.d_model, dtype=torch.bfloat16),
        A_log=torch.zeros(hv),
        dt_bias=torch.zeros(hv, dtype=param_dtype),
        ssm_state=torch.zeros(slots, hv, dv, dk),
        norm_weight=torch.ones(dv, dtype=param_dtype),
        out_scale=torch.ones(1),
        conv_state=torch.zeros(
            slots, g.conv_state_len(tokens), g.conv_dim, dtype=torch.bfloat16
        ).transpose(1, 2),
        conv_weight=torch.zeros(g.conv_dim, g.conv_kernel, dtype=torch.bfloat16),
    )
    if form == "decode":
        t["state_indices"] = torch.arange(1, rows + 1, dtype=torch.int32)
    else:
        t["state_indices"] = torch.arange(1, rows + 1, dtype=torch.int32).view(
            requests, tokens
        )
        t["cu_seqlens"] = torch.arange(0, rows + 1, tokens, dtype=torch.int32)
        t["num_accepted_tokens"] = torch.ones(requests, dtype=torch.int32)
    return t


def test_registry_and_plan_cache_with_a_fake_backend(fake_backend):
    assert "frost" in api.registered_backends() and "fake" in api.registered_backends()
    with pytest.raises(ValueError, match="already registered"):
        api.register_backend("fake", lambda *a: None, lambda s: None)
    assert qwen_3_5_gdn_block_support(CPU, SMALL, "verify", "fake").served

    t = _tensors(SMALL, 3, 1, "decode")
    q = qwen_3_5_gdn_block_decode(**t, geometry=SMALL, backend="fake")
    assert q.shape == (3, 4, 16) and q.dtype == torch.float8_e4m3fn
    assert bool((q.view(torch.uint8) == 1).all())
    q2 = qwen_3_5_gdn_block_decode(**t, geometry=SMALL, backend="fake")
    assert q2 is q  # the cached plan's own output
    assert len(fake_backend) == 1 and fake_backend[0].prepared == 2
    spec = fake_backend[0].spec
    assert (spec.backend, spec.form, spec.requests, spec.tokens_per_request) == (
        "fake",
        "decode",
        3,
        1,
    )
    assert spec.param_dtypes == (torch.float32, torch.bfloat16, torch.bfloat16)

    # another dtype of the parameters, another row count, another form: other plans
    qwen_3_5_gdn_block_decode(
        **_tensors(SMALL, 3, 1, "decode", param_dtype=torch.float32),
        geometry=SMALL,
        backend="fake",
    )
    qwen_3_5_gdn_block_decode(
        **_tensors(SMALL, 4, 1, "decode"), geometry=SMALL, backend="fake"
    )
    tv = _tensors(SMALL, 2, 4, "verify")
    out = torch.empty(8, 4, 16, dtype=torch.float8_e4m3fn)
    out_bf16 = torch.empty(8, 4, 16, dtype=torch.bfloat16)
    ret = qwen_3_5_gdn_block_verify(
        **tv, geometry=SMALL, backend="fake", out=out, out_bf16=out_bf16
    )
    assert ret is out and bool((out_bf16 == 2.0).all())
    assert len(fake_backend) == 4
    inputs = fake_backend[-1].ran[-1]
    assert inputs.num_accepted_tokens is tv["num_accepted_tokens"]

    # an explicit plan; a plan of another form is refused
    plan = prepare_qwen_3_5_gdn_block(
        "verify",
        geometry=SMALL,
        device=CPU,
        requests=2,
        tokens_per_request=4,
        backend="fake",
    )
    plan.run(**tv)
    with pytest.raises(ValueError, match="form='verify'"):
        qwen_3_5_gdn_block_decode(**t, geometry=SMALL, plan=plan)

    api.clear_plan_cache()
    qwen_3_5_gdn_block_decode(**t, geometry=SMALL, backend="fake")
    assert len(fake_backend) == 6  # rebuilt after the clear


@pytest.mark.parametrize(
    "name, value, match",
    [
        ("qkv", torch.zeros(3, 99, dtype=torch.bfloat16), "qkv must have shape"),
        ("hidden", torch.zeros(3, 64, dtype=torch.float16), "hidden must be"),
        ("A_log", torch.zeros(5), "A_log must have shape"),
        (
            "ssm_state",
            torch.zeros(6, 4, 16, 16).transpose(2, 3),
            "each slot contiguous",
        ),
        (
            "conv_state",
            torch.zeros(6, 10, 128, dtype=torch.bfloat16).transpose(1, 2),
            "conv_state must have shape",
        ),
        ("state_indices", torch.zeros(3, dtype=torch.int64), "state_indices must be"),
        ("out_scale", torch.ones(2), "one-element"),
    ],
)
def test_bind_inputs_names_the_operand(fake_backend, name, value, match):
    t = _tensors(SMALL, 3, 1, "decode")
    t[name] = value
    with pytest.raises(ValueError, match=match):
        qwen_3_5_gdn_block_decode(**t, geometry=SMALL, backend="fake")


# --------------------------------------------------------------------------- import isolation

_REPO_ROOT = pathlib.Path(__file__).parents[3]

_ISOLATION = """
import sys
import flashinfer
before = set(sys.modules)
import flashinfer.qwen_3_5_gdn_block as q
added = set(sys.modules) - before
assert not any(k.startswith("flashinfer.experimental.qwen_3_5_gdn_block") for k in added), sorted(added)
for name in ("qwen_3_5_gdn_block_support", "qwen_3_5_gdn_block_decode", "qwen_3_5_gdn_block_verify",
             "prepare_qwen_3_5_gdn_block"):
    assert getattr(q, name).is_experimental is True, name
import torch
g = q.Qwen35GdnGeometry(d_model=5120, num_v_heads=48, num_k_heads=16, head_k_dim=128, head_v_dim=128, conv_kernel=4)
sup = q.qwen_3_5_gdn_block_support(torch.device("cpu"), g, "verify")
assert not sup.served and sup.kind == "device", sup
loaded = {k for k in sys.modules if k.startswith("flashinfer.experimental.qwen_3_5_gdn_block")}
for heavy in ("runner", "gdn", "gdn_verify"):
    assert f"flashinfer.experimental.qwen_3_5_gdn_block.backends.frost.{heavy}" not in loaded, sorted(loaded)
print("ok")
"""


def test_import_and_probe_load_no_kernel():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_REPO_ROOT), env.get("PYTHONPATH")))
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _ISOLATION],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[
        -4000:
    ]
