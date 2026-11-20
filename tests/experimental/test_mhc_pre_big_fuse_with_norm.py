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

# Tests for the experimental mhc_pre_big_fuse_with_norm API: the support check,
# the composable implementation against a float64 reference (any host), and the
# SM100 / SM103 cluster kernel against the composable implementation, a float64
# Sinkhorn on extreme logits, and CUDA-graph capture.

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.mhc as mhc
from flashinfer.experimental.mhc_pre_big_fuse_with_norm import backend
from flashinfer.utils import get_compute_capability

H = 4096
HC = 4
MIX = HC * (2 + HC)
CONSTS = dict(
    rms_eps=1e-5,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
    norm_eps=1e-5,
)
KILL_SWITCH = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Comb logits that stress the Sinkhorn: inputs that drive the scale vectors' power-of-two gauge
# the furthest, peaked / same-column / near-permutation / Hall-violating patterns with large
# gaps, exact ties, |x| up to 1e30, and one-hot rows that all peak in one column.
# fmt: off
EXTREME_COMB_LOGITS = {
    "gauge_drift_cycle": [
        -0.001066722092218697, -40.849876403808594, -68.89411926269531, -21.326635360717773,
        -9.155299949270557e-07, -17.12651252746582, -180.71405029296875, -133.67942810058594,
        -3.600817706228554e-07, -40.78141784667969, -206.076904296875, -18.190526962280273,
        -0.00023028221039567143, -23.177854537963867, -27.096738815307617, -36.789642333984375
    ],
    "gauge_drift_side": [
        -2.438597732634662e-07, -15.393172264099121, -33.856449127197266, -26.9609317779541,
        -6.436353601202427e-07, -18.90549659729004, -54.867042541503906, -25.174983978271484,
        -6.910326919751242e-05, -30.612531661987305, -23.293027877807617, -59.79174041748047,
        -6.739954551449046e-05, -14.53339958190918, -22.531600952148438, -29.887998580932617
    ],
    "exponent_reach": [
        -29.140092849731445, -37.4259033203125, -155.30215454101562, 0.12972554564476013,
        -29.431982040405273, -73.03096771240234, -16.719484329223633, 0.0, 0.0,
        -3.8528892993927, -15.567123413085938, -22.057586669921875, -15.968350410461426,
        -80.7394027709961, -98.5098876953125, 8.106473501356959e-07
    ],
    "iid_wide": [
        -6.956757545471191, -2.8573360443115234, -4.691375732421875, -9.76917552947998,
        -5.8167595863342285, 20.65135955810547, -8.403948783874512, -8.17123794555664,
        4.197597503662109, -2.217984199523926, 9.423771858215332, 0.7752690315246582,
        -9.630749702453613, 12.668396949768066, -1.534620761871338, -1.9574013948440552
    ],
    "three_rows_one_column_noisy": [
        -0.04553842917084694, -5.968676567077637, -6.008527755737305, -6.036728382110596,
        0.052443988621234894, -5.9925127029418945, -5.935831069946289, -5.921547889709473,
        0.02344232238829136, -5.93055534362793, -5.970357418060303, -5.924376964569092,
        -5.983889102935791, -6.004086494445801, -6.015609264373779, -0.08347638696432114
    ],
    "peaked": [
        -92.28055572509766, -1.897629737854004, 0.0, -76.70032501220703, -17.874927520751953,
        -26.17359161376953, -98.40567016601562, 0.0, -113.32694244384766, -148.49537658691406,
        0.0, -77.13370513916016, -79.97402954101562, -82.64522552490234, -73.06037902832031, 0.0
    ],
    "same_column": [
        0.0, -17.61452293395996, -30.93341636657715, -47.217613220214844, 0.0,
        -96.44166564941406, -124.4942626953125, -162.72108459472656, 0.0, -145.14852905273438,
        -146.66363525390625, -62.89583969116211, 0.0, -26.72414207458496, -33.25218200683594,
        -29.900049209594727
    ],
    "near_permutation": [
        -173.6573944091797, 0.0, -31.658599853515625, -37.261940002441406,
        -0.010502124205231667, -0.5176069140434265, 0.0, -102.33224487304688,
        -51.40129470825195, -145.13021850585938, -165.3909149169922, 0.0, 0.0,
        -140.92306518554688, -106.04235076904297, -86.0247573852539
    ],
    "hall_violating": [
        -0.7919648289680481, -1.7849817276000977, -87.25366973876953, -87.25366973876953,
        -2.0112876892089844, -6.858701229095459, -87.25366973876953, -87.25366973876953,
        -10.774415969848633, -14.573726654052734, -87.25366973876953, -87.25366973876953,
        -44.290863037109375, -51.5079231262207, -58.8403434753418, -36.99163818359375
    ],
    "exact_ties": [
        -22.773212432861328, -22.773212432861328, 0.0, 0.0, -22.773212432861328,
        -22.773212432861328, 0.0, -22.773212432861328, -22.773212432861328, -22.773212432861328,
        -22.773212432861328, 0.0, -22.773212432861328, 0.0, -22.773212432861328,
        -22.773212432861328
    ],
    "huge_magnitudes": [
        -257.8782043457031, 0.33135437965393066, -1.5496666974465625e+27, -1.3206441402435303,
        -104573206528.0, 4.0163316101559825e+20, -87907008512.0, 132733.03125, 115300.8515625,
        -4.878467442907394e+25, -1.3332345396144374e+18, 3.4476771072855245e+17,
        -5.964796044908062e+20, -2.1706987664155787e+22, -2751713.5, -4.135654456950416e+23
    ],
    "three_rows_one_column": [
        0.0, -6.0, -6.0, -6.0, 0.0, -6.0, -6.0, -6.0, 0.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0,
        0.0
    ],
    "all_equal": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ],
    "one_hot_same_column": [
        0.0, -1.0000000150474662e+30, -1.0000000150474662e+30, -1.0000000150474662e+30, 0.0,
        -1.0000000150474662e+30, -1.0000000150474662e+30, -1.0000000150474662e+30, 0.0,
        -1.0000000150474662e+30, -1.0000000150474662e+30, -1.0000000150474662e+30, 0.0,
        -1.0000000150474662e+30, -1.0000000150474662e+30, -1.0000000150474662e+30
    ],
}
# fmt: on


@pytest.fixture(autouse=True)
def _kill_switch_unset(monkeypatch):
    monkeypatch.delenv(KILL_SWITCH, raising=False)


def _require_sm10x() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if get_compute_capability(torch.device("cuda")) not in backend.COMPUTE_CAPABILITIES:
        pytest.skip(
            "the mHC pre big-fuse + RMSNorm cluster kernel targets SM100 / SM103"
        )


def _inputs(t: int, ns: int, seed: int = 0, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(seed * 7919 + t * 131 + ns)
    amp = torch.exp(
        torch.empty(t, 1, 1, device=device).uniform_(-3.9, 0.7, generator=g)
    )
    new_r = torch.randn(t, HC, H, device=device, generator=g) * amp
    residual = new_r.to(torch.bfloat16)
    fn = torch.randn(MIX, HC * H, device=device, generator=g) * 0.03
    xs = new_r.reshape(t, HC, ns, H // ns)
    fs = fn.view(MIX, HC, ns, H // ns)
    gemm_out_mul = torch.einsum("tjsk,njsk->stn", xs, fs).contiguous()
    gemm_out_sqrsum = xs.square().sum(dim=(1, 3)).transpose(0, 1).contiguous()
    hc_scale = torch.tensor([0.09, 0.16, 0.21], device=device)
    hc_base = torch.randn(MIX, device=device, generator=g)
    norm_weight = (0.2 + 0.01 * torch.randn(H, device=device, generator=g)).to(
        torch.bfloat16
    )
    return dict(
        gemm_out_mul=gemm_out_mul,
        gemm_out_sqrsum=gemm_out_sqrsum,
        hc_scale=hc_scale,
        hc_base=hc_base,
        residual=residual,
        norm_weight=norm_weight,
    )


def _sinkhorn_fp64(logits: torch.Tensor, eps: float = 1e-6, repeat: int = 20):
    x = logits.double().reshape(-1, HC, HC)
    c = torch.softmax(x, dim=-1) + eps
    c = c / (c.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        c = c / (c.sum(dim=-1, keepdim=True) + eps)
        c = c / (c.sum(dim=-2, keepdim=True) + eps)
    return c.reshape(-1, HC * HC)


def _reference_fp64(inp: dict, consts: dict):
    """float64 evaluation of the op contract (bf16 rounding of ol kept, as in the contract)."""
    f64 = torch.float64
    residual = inp["residual"].to(f64)
    t_tokens = residual.shape[0]
    sq = inp["gemm_out_sqrsum"].to(f64).sum(dim=0)
    rms = torch.rsqrt(sq / (HC * H) + consts["rms_eps"])
    mixes = inp["gemm_out_mul"].to(f64).sum(dim=0) * rms.unsqueeze(-1)
    scale = inp["hc_scale"].to(f64)
    base = inp["hc_base"].to(f64)
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + consts["hc_pre_eps"]
    post = (
        torch.sigmoid(mixes[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
        * consts["hc_post_mult_value"]
    )
    comb = _sinkhorn_fp64(
        mixes[:, 2 * HC :] * scale[2] + base[2 * HC :],
        consts["hc_sinkhorn_eps"],
        consts["sinkhorn_repeat"],
    )
    ol = (pre.unsqueeze(-1) * residual).sum(dim=1)
    r = torch.rsqrt(ol.square().mean(dim=-1, keepdim=True) + consts["norm_eps"])
    li = ol.to(torch.bfloat16).to(f64) * r * inp["norm_weight"].to(f64)
    return post, comb.reshape(t_tokens, HC * HC), li


def _check(out, ref):
    post, comb, li = out
    rpost, rcomb, rli = ref
    torch.testing.assert_close(post, rpost, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(comb, rcomb, atol=1e-7, rtol=1e-5)
    err = (li.float() - rli.float()).abs()
    assert bool((err <= 1e-6 + 2.0**-5 * rli.float().abs()).all())
    exact = (li.view(torch.int16) == rli.view(torch.int16)).float().mean().item()
    assert exact >= 0.995, exact


def _reject_count(reason: str) -> int:
    return backend.STATS["rejects"].get(reason, 0)


# ---------------------------------------------------------------- any host (no GPU needed) ------


def test_api_is_experimental():
    for fn in (
        mhc.mhc_pre_big_fuse_with_norm,
        mhc.mhc_pre_big_fuse_with_norm_supported,
        mhc.mhc_pre_big_fuse_with_norm_warmup,
    ):
        assert getattr(fn, "is_experimental", False), fn.__name__


def test_core_import_does_not_load_the_backend():
    code = (
        "import sys, flashinfer, flashinfer.mhc; "
        "assert 'flashinfer.experimental.mhc_pre_big_fuse_with_norm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_support_envelope_matches_the_kernel_constants():
    src = (
        Path(backend.__file__).parent / "csrc" / "mhc_pre_big_fuse_with_norm_cluster.cu"
    ).read_text()

    def constexpr(name: str) -> float:
        m = re.search(rf"constexpr (?:int|float) {name} = ([0-9.e+-]+)f?;", src)
        assert m is not None, name
        return float(m.group(1))

    assert constexpr("H") == backend.HIDDEN_SIZE == H
    assert constexpr("HC") == backend.HC_MULT == HC
    assert constexpr("RMS_EPS") == backend.CONSTANTS["rms_eps"] == CONSTS["rms_eps"]
    assert constexpr("HC_PRE_EPS") == backend.CONSTANTS["hc_pre_eps"]
    assert constexpr("SK_EPS") == backend.CONSTANTS["hc_sinkhorn_eps"]
    assert constexpr("POST_MULT") == backend.CONSTANTS["hc_post_mult_value"]
    assert constexpr("NORM_EPS") == backend.CONSTANTS["norm_eps"] == CONSTS["norm_eps"]
    assert backend.CONSTANTS == CONSTS
    assert sorted(backend.NUM_SPLITS) == [8, 64]
    assert (backend.MIN_TOKENS, backend.MAX_TOKENS) == (1, 32)


def test_support_rejects_before_touching_the_device():
    # CPU tensors: every semantic / shape / layout check runs before the device check.
    inp = _inputs(2, 8, device="cpu")
    cases = [
        ("num_tokens", dict(inp, **_inputs(33, 8, device="cpu")), CONSTS),
        ("num_splits", dict(inp, **_inputs(2, 16, device="cpu")), CONSTS),
        ("constants", inp, dict(CONSTS, hc_post_mult_value=1.0)),
        ("constants", inp, dict(CONSTS, sinkhorn_repeat=19)),
        ("no_norm_weight", dict(inp, norm_weight=None), CONSTS),
        (
            "input_layout",
            dict(
                inp,
                residual=inp["residual"].transpose(0, 1).contiguous().transpose(0, 1),
            ),
            CONSTS,
        ),  # fmt: skip
        ("device", inp, CONSTS),
    ]
    for reason, args, consts in cases:
        before = _reject_count(reason)
        assert not mhc.mhc_pre_big_fuse_with_norm_supported(**args, **consts)
        assert _reject_count(reason) == before + 1, reason
    misaligned = torch.empty(2 * HC * HC + 1, dtype=torch.float32)[1:].view(2, HC * HC)
    before = _reject_count("alignment")
    assert not mhc.mhc_pre_big_fuse_with_norm_supported(
        **inp, **CONSTS, comb_mix=misaligned
    )
    assert _reject_count("alignment") == before + 1


def test_kill_switch_is_read_at_call_time(monkeypatch):
    inp = _inputs(1, 8, device="cpu")
    monkeypatch.setenv(KILL_SWITCH, "1")
    before = _reject_count("kill_switch")
    assert not mhc.mhc_pre_big_fuse_with_norm_supported(**inp, **CONSTS)
    assert _reject_count("kill_switch") == before + 1
    monkeypatch.delenv(KILL_SWITCH)
    mhc.mhc_pre_big_fuse_with_norm_supported(**inp, **CONSTS)
    assert _reject_count("kill_switch") == before + 1


@pytest.mark.parametrize("t,ns", [(1, 8), (3, 8), (5, 64), (40, 4)])
def test_composable_path_matches_fp64_reference(t, ns):
    inp = _inputs(t, ns, seed=t, device="cpu")
    before = backend.STATS["composable_calls"]
    out = mhc.mhc_pre_big_fuse_with_norm(**inp, **CONSTS)
    assert backend.STATS["composable_calls"] == before + 1
    post, comb, li = out
    assert post.shape == (t, HC) and post.dtype == torch.float32
    assert comb.shape == (t, HC * HC) and comb.dtype == torch.float32
    assert li.shape == (t, H) and li.dtype == torch.bfloat16
    rpost, rcomb, rli = _reference_fp64(inp, CONSTS)
    torch.testing.assert_close(post.double(), rpost, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(comb.double(), rcomb, atol=1e-6, rtol=1e-5)
    # layer_input: bf16(ol) may round one ulp apart, plus the final bf16 rounding
    err = (li.double() - rli).abs()
    assert bool((err <= 1e-6 + 2.0**-6 * rli.abs()).all())


# ---------------------------------------------------------------- SM100 / SM103 GPUs ------------


@pytest.mark.parametrize("t", [1, 2, 3, 4, 8, 13, 16, 24, 25, 31, 32])
@pytest.mark.parametrize("ns", [8, 64])
def test_kernel_matches_composable(t, ns, monkeypatch):
    _require_sm10x()
    inp = _inputs(t, ns)
    assert mhc.mhc_pre_big_fuse_with_norm_supported(**inp, **CONSTS)
    before = backend.STATS["kernel_calls"]
    out = mhc.mhc_pre_big_fuse_with_norm(**inp, **CONSTS)
    assert backend.STATS["kernel_calls"] == before + 1
    monkeypatch.setenv(KILL_SWITCH, "1")
    assert not mhc.mhc_pre_big_fuse_with_norm_supported(**inp, **CONSTS)
    ref = mhc.mhc_pre_big_fuse_with_norm(**inp, **CONSTS)
    _check(out, ref)


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_sinkhorn_extreme_comb_logits(enable_pdl):
    # The comb logits are injected through hc_base with hc_scale[2] = 0, so they are exactly
    # the rows above; comb_mix must be finite and match a float64 Sinkhorn.
    _require_sm10x()
    for ns in (8, 64):
        for name, row in EXTREME_COMB_LOGITS.items():
            logits = torch.tensor(row, dtype=torch.float32)
            inp = _inputs(2, ns)
            inp["hc_scale"][2] = 0.0
            inp["hc_base"][2 * HC :] = logits.cuda()
            assert mhc.mhc_pre_big_fuse_with_norm_supported(**inp, **CONSTS)
            _, comb, _ = mhc.mhc_pre_big_fuse_with_norm(
                **inp, **CONSTS, enable_pdl=enable_pdl
            )
            ref = _sinkhorn_fp64(logits)
            got = comb.double().cpu()
            assert torch.isfinite(got).all(), name
            bound = 1e-7 + 1e-5 * ref.abs()
            assert bool(((got - ref).abs() <= bound).all()), name


def test_unsupported_calls_take_composable_path():
    _require_sm10x()
    inp = _inputs(33, 8)
    assert not mhc.mhc_pre_big_fuse_with_norm_supported(**inp, **CONSTS)
    before = backend.STATS["composable_calls"]
    mhc.mhc_pre_big_fuse_with_norm(**inp, **CONSTS)
    assert backend.STATS["composable_calls"] == before + 1
    consts = dict(CONSTS, hc_post_mult_value=1.0)
    assert not mhc.mhc_pre_big_fuse_with_norm_supported(**_inputs(2, 8), **consts)


def _graph_outputs(t):
    return tuple(
        torch.empty(s, dtype=d, device="cuda")
        for s, d in (
            ((t, HC), torch.float32),
            ((t, HC * HC), torch.float32),
            ((t, H), torch.bfloat16),
        )
    )


def test_cuda_graph_capture_after_warmup(monkeypatch):
    _require_sm10x()
    assert mhc.mhc_pre_big_fuse_with_norm_warmup(torch.device("cuda"))
    inp = _inputs(2, 8)
    outs = _graph_outputs(2)
    before = backend.STATS["kernel_calls"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mhc.mhc_pre_big_fuse_with_norm(
            **inp, **CONSTS, post_mix=outs[0], comb_mix=outs[1], layer_input=outs[2]
        )
    assert backend.STATS["kernel_calls"] == before + 1
    graph.replay()
    torch.cuda.synchronize()
    monkeypatch.setenv(KILL_SWITCH, "1")
    ref = mhc.mhc_pre_big_fuse_with_norm(**inp, **CONSTS)
    _check(outs, ref)


def test_cuda_graph_capture_without_loaded_module_falls_back(monkeypatch):
    # A capturing call never builds or loads the module: it captures the composable path.
    _require_sm10x()
    assert mhc.mhc_pre_big_fuse_with_norm_warmup(torch.device("cuda"))
    monkeypatch.setitem(backend._state, "module", None)
    inp = _inputs(2, 8)
    outs = _graph_outputs(2)
    unready = _reject_count("capture_unready")
    kernel_calls = backend.STATS["kernel_calls"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mhc.mhc_pre_big_fuse_with_norm(
            **inp, **CONSTS, post_mix=outs[0], comb_mix=outs[1], layer_input=outs[2]
        )
    assert _reject_count("capture_unready") == unready + 1
    assert backend.STATS["kernel_calls"] == kernel_calls
    assert backend._state["module"] is None
    graph.replay()
    torch.cuda.synchronize()
    monkeypatch.setenv(KILL_SWITCH, "1")
    _check(outs, mhc.mhc_pre_big_fuse_with_norm(**inp, **CONSTS))
