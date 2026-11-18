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

The cake backends of ``mm_fp4`` (per-token alpha) and ``bmm_bf16`` as the last
candidate of ``backend="auto"``: listed last only where they serve every M
of the call's tuning bucket (so a cached cake winner is never a cache miss for
another M of that bucket), never the untuned default, measured under
``flashinfer.autotune()``, and removed by
``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES=1``. Outside tuning they are listed
only while the tuner holds an entry of their runner class, so an untuned call
does no cake work and keeps the other backends' runner list.
"""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from flashinfer import SfLayout, autotune, bmm_bf16, mm_bf16, mm_fp4, nvfp4_quantize
from flashinfer.autotune_cache import autotune_v2
from flashinfer.autotuner import AutoTuner, ProfilingCacheKey
from flashinfer.cute_dsl.availability import is_cute_dsl_available
from flashinfer.experimental.cake_nvfp4_per_token import (
    auto_candidate as cake_candidate,
)
from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cb
from flashinfer.fused_moe.utils import map_to_hybrid_bucket_uncapped
from flashinfer.gemm import gemm_base
from flashinfer.utils import get_compute_capability

DISABLE_ENV = "FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES"
EXPERIMENTAL_ENV = "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS"
SWAP_AB_DISABLE_ENV = "FLASHINFER_DISABLE_CAKE_BF16_BMM_SWAP_AB"
CAKE_FP4_RUNNER = "CakeFp4GemmRunner"
CAKE_BMM_RUNNER = "CakeBf16BmmRunner"
FP4_N, FP4_K = 2112, 7168


def _cake_arch() -> bool:
    return get_compute_capability(torch.device("cuda")) in ((10, 0), (10, 3))


def _skip_unless_cake_bf16_bmm_supported():
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if compute_capability not in {
        (10, 0),
        (10, 3),
    } or not bmm_bf16.is_backend_supported("cake", compute_capability_number):
        pytest.skip("CAKE BF16 BMM requires SM100 or SM103.")


def _skip_unless_cake_fp4_supported():
    if not _cake_arch():
        pytest.skip("cake per-token NVFP4 GEMM requires SM100 or SM103.")
    if not cb.generated_program_available(torch.device("cuda")):
        pytest.skip("no generated per-token NVFP4 program registered for this GPU")
    if not is_cute_dsl_available():
        pytest.skip("per-token alpha auto selection needs the cute-dsl backend.")


@pytest.fixture
def fresh_tuner(monkeypatch):
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    return tuner


def _spy_choose_one(monkeypatch, force_last: bool = False) -> list:
    """Record every AutoTuner.choose_one call; ``force_last`` returns the last
    runner with its default tactic instead of the tuner's choice."""
    calls = []
    original = AutoTuner.choose_one

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if force_last:
            runner, tactic = runners[-1], -1
        else:
            runner, tactic = original(
                self, custom_op, runners, tuning_config, inputs, **kwargs
            )
        calls.append(
            SimpleNamespace(
                op=custom_op,
                runners=[type(r).__name__ for r in runners],
                runner=type(runner).__name__,
                is_first=runner is runners[0],
                tactic=tactic,
            )
        )
        return runner, tactic

    monkeypatch.setattr(AutoTuner, "choose_one", spy)
    return calls


def _record_runners(monkeypatch) -> list:
    """Record the runner names of every AutoTuner.choose_one call before the
    tuner runs, so a call that raises while tuning is still recorded."""
    offered = []
    original = AutoTuner.choose_one

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        offered.append([type(r).__name__ for r in runners])
        return original(self, custom_op, runners, tuning_config, inputs, **kwargs)

    monkeypatch.setattr(AutoTuner, "choose_one", spy)
    return offered


class _GroupPlanChecked(Exception):
    """Raised by the stubbed autotune-group plan check once it has recorded."""


def _record_group_plans(monkeypatch, tuner) -> tuple[list, list]:
    """Put ``tuner`` in tuning mode without timing anything. Record the runner
    names of every AutoTuner.choose_one call (``offered``) and the runner names
    that its autotune-group plan check compares across ranks (``planned``).
    The check raises _GroupPlanChecked instead of running a collective; a call
    without a group returns runners[0] with its default tactic."""
    from flashinfer.autotuner import autotuner as autotuner_module

    offered, planned = [], []
    choose_one = AutoTuner.choose_one

    def check(phase, selection=(), error=None):
        assert phase == "operation" and selection[0] == "choose_one", phase
        planned.append(list(selection[2]))
        raise _GroupPlanChecked

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        offered.append([type(r).__name__ for r in runners])
        if autotuner_module._tune_process_group is None:
            return runners[0], -1
        return choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs)

    monkeypatch.setattr(autotuner_module, "_check_tuning_group_phase", check)
    monkeypatch.setattr(AutoTuner, "choose_one", spy)
    monkeypatch.setattr(tuner, "is_tuning_mode", True)
    return offered, planned


def _count_calls(monkeypatch, owner, name: str) -> list:
    """Wrap ``owner.name`` so every call is recorded and still runs."""
    calls = []
    original = getattr(owner, name)

    def counting(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counting)
    return calls


def _hold_cake_entry(tuner, runner_name: str) -> None:
    """Give ``tuner`` a winner of the cake runner class for an op no call
    uses, as after a tuning run that cake won elsewhere: cake is then listed
    outside tuning, while every call of the test still misses the cache."""
    key = ProfilingCacheKey(
        custom_op="cake_candidate_gate_test",
        runner_class_name=runner_name,
        runner_hash=0,
        nearest_profile=((1,),),
        extras=(),
    )
    tuner.profiling_cache[key] = (-1, None)


def _raise_if_called(monkeypatch, owner, *names: str) -> None:
    """Replace each ``owner.name`` by a function that fails the test."""
    for name in names:

        def fail(*args, _name=name, **kwargs):
            raise AssertionError(f"{_name} was called")

        monkeypatch.setattr(owner, name, fail)


def _cake_wins_timing(monkeypatch, runner_name: str) -> list:
    """Make tuning time the runner class ``runner_name`` at 1 and every other
    runner at 2 without launching, so that runner wins every bucket where it
    is offered. Returns the names of the runners timed."""
    timed = []

    def timing(self, runner, inputs, tactic, tuning_config, *args, **kwargs):
        timed.append(type(runner).__name__)
        return 1.0 if type(runner).__name__ == runner_name else 2.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", timing)
    return timed


def _set_switches(monkeypatch, cake_enabled: bool, experimental: bool):
    if cake_enabled:
        monkeypatch.delenv(DISABLE_ENV, raising=False)
    else:
        monkeypatch.setenv(DISABLE_ENV, "1")
    if experimental:
        monkeypatch.setenv(EXPERIMENTAL_ENV, "1")
    else:
        monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)


# ---------------------------------------------------------------------------
# mm_fp4, per-token alpha
# ---------------------------------------------------------------------------


def _dequantize(fp4, sf):
    """FP32 dequantisation of packed E2M1 + swizzled 128x4 E4M3 scales (rows x K)."""
    fp4 = fp4.view(torch.uint8)
    rows, kh = fp4.shape
    k = 2 * kh
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=fp4.device,
    )
    nib = torch.stack([fp4 & 0xF, fp4 >> 4], dim=-1).reshape(rows, k).long()
    sf_logical = cb.unswizzle_sf_128x4(sf.view(torch.uint8), rows, k)
    scales = sf_logical.view(torch.float8_e4m3fn).float()
    return (e2m1[nib].view(rows, k // 16, 16) * scales[:, :, None]).view(rows, k)


def _assert_matches(out, ref):
    got = out.float()
    err = (got - ref).abs()
    ulp = ref.abs() * (2.0**-8 if out.dtype == torch.bfloat16 else 2.0**-11)
    bound = 1e-2 + 1e-2 * ref.abs() + ulp
    assert torch.isfinite(got).all()
    bad = int((err > bound).sum())
    assert bad == 0, f"{bad} elements exceed |ref| * 1e-2 + 1e-2 + half ulp"


def _fp4_case(m, n=FP4_N, k=FP4_K, seed=0):
    """NVFP4 operands with a per-token alpha that differs on every row, and the
    FP32 product of the dequantized operands scaled by it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
    g_in = (448 * 6) / a.float().abs().max()
    g_w = (448 * 6) / b.float().abs().max()
    a_fp4, a_sf = nvfp4_quantize(
        a, g_in, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b, g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    row = 0.25 + torch.arange(m, device="cuda", dtype=torch.float32) / m
    alpha = (row / (g_in * g_w)).float().contiguous()
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(b_fp4, b_sf).T) * alpha[:, None]
    return (a_fp4, b_fp4, a_sf, b_sf, alpha), ref


def _mm_fp4(operands, out_dtype, backend="auto", alpha=None, **kwargs):
    a_fp4, b_fp4, a_sf, b_sf, per_token_alpha = operands
    return mm_fp4(
        a_fp4,
        b_fp4.T,
        a_sf,
        b_sf.T,
        per_token_alpha if alpha is None else alpha,
        out_dtype,
        None,
        block_size=16,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=True,
        **kwargs,
    )


def _fp4_runner_inputs(operands, out):
    a_fp4, b_fp4, a_sf, b_sf, alpha = operands
    workspace = torch.empty(1, device="cuda", dtype=torch.uint8)
    return [a_fp4, b_fp4.T, a_sf, b_sf.T, alpha, out.dtype, out, 16, True, workspace]


@pytest.mark.parametrize("m", [16, 128])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4_cake_last_and_untuned_default_unchanged(
    monkeypatch, fresh_tuner, m, out_dtype
):
    """(a) cake follows cute-dsl; (b) the untuned call still runs runners[0]."""
    _skip_unless_cake_fp4_supported()
    operands, ref = _fp4_case(m)
    calls = _spy_choose_one(monkeypatch)

    _set_switches(monkeypatch, cake_enabled=False, experimental=True)
    baseline = _mm_fp4(operands, out_dtype)
    assert mm_fp4.suitable_auto_backends == ["cute-dsl"]
    before = calls[-1]

    # No cake entry in the tuner: the switched-on call lists what the
    # switched-off call lists.
    _set_switches(monkeypatch, cake_enabled=True, experimental=True)
    _mm_fp4(operands, out_dtype)
    assert mm_fp4.suitable_auto_backends == ["cute-dsl"]
    assert calls[-1].runners == before.runners

    _hold_cake_entry(fresh_tuner, CAKE_FP4_RUNNER)
    out = _mm_fp4(operands, out_dtype)
    assert mm_fp4.suitable_auto_backends == ["cute-dsl", "cake"]
    after = calls[-1]

    assert after.op == before.op == "fp4_gemm"
    assert after.runners == [*before.runners, CAKE_FP4_RUNNER]
    assert after.runners[0] != CAKE_FP4_RUNNER
    assert before.is_first and after.is_first and after.tactic == before.tactic
    assert after.runner == before.runner
    _assert_matches(baseline, ref)
    _assert_matches(out, ref)


@pytest.mark.parametrize(
    "case", ["switch_off", "experimental_gate_closed", "pdl_off", "scalar_alpha"]
)
def test_mm_fp4_cake_absent(monkeypatch, fresh_tuner, case):
    """(a) No cake candidate when switched off, without the experimental opt-in,
    for inputs cake rejects, or for a scalar alpha; the call still runs."""
    _skip_unless_cake_fp4_supported()
    operands, ref = _fp4_case(16)
    # A held cake entry lists cake outside tuning, so only the case leaves it out.
    _hold_cake_entry(fresh_tuner, CAKE_FP4_RUNNER)
    calls = _spy_choose_one(monkeypatch)
    _set_switches(
        monkeypatch,
        cake_enabled=case != "switch_off",
        experimental=case != "experimental_gate_closed",
    )
    kwargs = {}
    if case == "pdl_off":
        kwargs["enable_pdl"] = False
    if case == "scalar_alpha":
        alpha = operands[4][:1].clone()
        ref = ref * (alpha / operands[4])[:, None]
        kwargs["alpha"] = alpha

    out = _mm_fp4(operands, torch.bfloat16, **kwargs)

    assert "cake" not in mm_fp4.suitable_auto_backends
    assert calls and CAKE_FP4_RUNNER not in calls[-1].runners
    if case == "scalar_alpha":
        torch.testing.assert_close(
            out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
        )
    else:
        _assert_matches(out, ref)


@pytest.mark.parametrize("m", [16, 128])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4_cake_measured_under_autotune(monkeypatch, fresh_tuner, m, out_dtype):
    """(c) Tuning measures cake with cute-dsl; any winner matches the reference,
    also when the tuned choice is replayed without tuning."""
    _skip_unless_cake_fp4_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=True)
    operands, ref = _fp4_case(m, seed=1)
    calls = _spy_choose_one(monkeypatch)
    # The cake runner resolves mm_fp4_per_token at call time.
    cake_launches = _count_calls(monkeypatch, cb, "mm_fp4_per_token")

    with autotune(True, tuning_buckets=(m,)):
        tuned = _mm_fp4(operands, out_dtype)
    tuning_launches = len(cake_launches)
    replayed = _mm_fp4(operands, out_dtype)

    assert calls[0].runners[-1] == CAKE_FP4_RUNNER
    assert tuning_launches > 0, "tuning never ran the cake runner"
    assert f"fp4_gemm::{CAKE_FP4_RUNNER}" not in fresh_tuner.stats.failed_tactics
    assert calls[-1].runner == calls[0].runner
    _assert_matches(tuned, ref)
    _assert_matches(replayed, ref)


@pytest.mark.parametrize("m", [16, 128])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4_cake_runner_forced(monkeypatch, fresh_tuner, m, out_dtype):
    """(d) The cake runner, chosen through the tuner path or called directly,
    reproduces backend="cake" bit for bit."""
    _skip_unless_cake_fp4_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=True)
    operands, ref = _fp4_case(m, seed=2)
    explicit = _mm_fp4(operands, out_dtype, backend="cake")

    _hold_cake_entry(fresh_tuner, CAKE_FP4_RUNNER)
    calls = _spy_choose_one(monkeypatch, force_last=True)
    forced = _mm_fp4(operands, out_dtype)
    assert calls[-1].runner == CAKE_FP4_RUNNER

    runner = cake_candidate.cake_gemm_fp4_runner(False, True)
    direct = torch.empty_like(explicit)
    inputs = _fp4_runner_inputs(operands, direct)
    assert runner.get_valid_tactics(inputs, None) == [-1]
    runner(inputs=inputs, tactic=-1)
    torch.cuda.synchronize()

    _assert_matches(explicit, ref)
    torch.testing.assert_close(forced, explicit, rtol=0, atol=0)
    torch.testing.assert_close(direct, explicit, rtol=0, atol=0)


def test_mm_fp4_cake_runner_rejects_unsupported_inputs():
    """(a) The runner offers no tactic for operands cake does not serve."""
    _skip_unless_cake_fp4_supported()
    operands, _ = _fp4_case(16)
    out = torch.empty((16, FP4_N), device="cuda", dtype=torch.bfloat16)
    inputs = _fp4_runner_inputs(operands, out)
    runner = cake_candidate.cake_gemm_fp4_runner
    assert runner(False, True).get_valid_tactics(inputs, None) == [-1]
    assert runner(True, True).get_valid_tactics(inputs, None) == []  # 8x4 scales
    assert runner(False, False).get_valid_tactics(inputs, None) == []  # no PDL
    for index, value in ((4, inputs[4][:1]), (7, 32), (8, False)):
        bad = list(inputs)
        bad[index] = value
        assert runner(False, True).get_valid_tactics(bad, None) == []
    bad = list(inputs)
    bad[1] = inputs[1].contiguous()  # row-major b
    assert runner(False, True).get_valid_tactics(bad, None) == []
    bad = list(inputs)
    buffer = torch.empty(out.numel() + 8, device="cuda", dtype=out.dtype)
    bad[6] = buffer[8:].view_as(out)  # 16-byte offset: still aligned
    assert runner(False, True).get_valid_tactics(bad, None) == [-1]
    bad[6] = buffer[1 : 1 + out.numel()].view_as(out)  # 2-byte offset
    assert runner(False, True).get_valid_tactics(bad, None) == []


@pytest.mark.parametrize("n", [8, 16, 24, 32])
def test_mm_fp4_cake_runner_needs_n_32(monkeypatch, fresh_tuner, n):
    """(a) Every cake GEMM program stores the output through a TMA box at least
    32 columns wide, so N in {8, 16, 24} (N % 8 == 0, which the checker
    accepts) gets no cake tactic and no cake runner; N = 32 does."""
    _skip_unless_cake_fp4_supported()
    m = 16
    operands, _ = _fp4_case(m, n=n)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    inputs = _fp4_runner_inputs(operands, out)
    runner = cake_candidate.cake_gemm_fp4_runner(False, True)
    expected = [-1] if n >= 32 else []
    assert runner.get_valid_tactics(inputs, None) == expected
    assert cake_candidate.cake_fp4_gemm_serves(inputs, False, True) == (n >= 32)
    if n >= 32:
        return
    # The auto call never offers cake, so tuning cannot time it and raise.
    _set_switches(monkeypatch, cake_enabled=True, experimental=True)
    offered = _record_runners(monkeypatch)
    try:
        with autotune(True, tuning_buckets=(m,)):
            _mm_fp4(operands, torch.bfloat16)
    except Exception:
        if offered:
            raise
        pytest.skip(f"no other per-token NVFP4 backend serves N={n}")
    assert offered and all(CAKE_FP4_RUNNER not in runners for runners in offered)


# ---------------------------------------------------------------------------
# bmm_bf16
# ---------------------------------------------------------------------------


def _bmm_case(b, m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((b, m, k), device="cuda", dtype=torch.bfloat16, generator=g)
    mat2 = torch.randn(
        (b, n, k), device="cuda", dtype=torch.bfloat16, generator=g
    ).transpose(-2, -1)
    return a, mat2, torch.bmm(a.float(), mat2.float())


def test_bmm_bf16_cake_last_and_untuned_default_unchanged(monkeypatch, fresh_tuner):
    """(a) cake is listed last; (b) the untuned call still runs runners[0].
    The bmm cake backend is stable: the experimental gate does not apply."""
    _skip_unless_cake_bf16_bmm_supported()
    a, mat2, ref = _bmm_case(4, 64, 256, 256)
    calls = _spy_choose_one(monkeypatch)
    assert "cake" not in bmm_bf16.experimental_backends

    _set_switches(monkeypatch, cake_enabled=False, experimental=False)
    baseline = bmm_bf16(a, mat2, backend="auto")
    assert "cake" not in bmm_bf16.suitable_auto_backends
    before = calls[-1]

    # No cake entry in the tuner: the switched-on call lists what the
    # switched-off call lists.
    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    bmm_bf16(a, mat2, backend="auto")
    assert "cake" not in bmm_bf16.suitable_auto_backends
    assert calls[-1].runners == before.runners

    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    out = bmm_bf16(a, mat2, backend="auto")
    auto_backends = bmm_bf16.suitable_auto_backends
    after = calls[-1]

    assert auto_backends[-1] == "cake" and len(auto_backends) > 1
    assert after.op == before.op == "bf16_gemm"
    assert after.runners == [*before.runners, CAKE_BMM_RUNNER]
    assert after.runners[0] != CAKE_BMM_RUNNER
    assert before.is_first and after.is_first and after.tactic == before.tactic
    assert after.runner == before.runner
    expected = ref.to(torch.bfloat16)
    torch.testing.assert_close(baseline, expected, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize(
    "case", ["switch_off", "unsupported_k", "mm_bf16_2d", "bucket_runs_swap_ab"]
)
def test_bmm_bf16_cake_absent(monkeypatch, fresh_tuner, case):
    """(a) No cake candidate when switched off, for a K cake rejects, for the 2D
    mm_bf16 calls that share the bf16_gemm runner list, or where the tuning
    bucket of M (8, a swap-AB row) runs another cake program than M=5."""
    _skip_unless_cake_bf16_bmm_supported()
    _set_switches(monkeypatch, cake_enabled=case != "switch_off", experimental=True)
    # A held cake entry lists cake outside tuning, so only the case leaves it out.
    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    calls = _spy_choose_one(monkeypatch)
    if case == "mm_bf16_2d":
        a, mat2, ref = _bmm_case(1, 64, 256, 256)
        out = mm_bf16(a[0], mat2[0], backend="auto")
        ref = ref[0]
    elif case == "bucket_runs_swap_ab":
        monkeypatch.delenv("FLASHINFER_DISABLE_CAKE_BF16_BMM_SWAP_AB", raising=False)
        a, mat2, ref = _bmm_case(2, 5, 1024, 1024)
        out = bmm_bf16(a, mat2, backend="auto")
        # The checker accepts the call; only the runner list leaves cake out.
        assert bmm_bf16.suitable_auto_backends[-1] == "cake"
    else:
        k = 128 if case == "unsupported_k" else 256
        a, mat2, ref = _bmm_case(4, 64, 256, k)
        out = bmm_bf16(a, mat2, backend="auto")
        assert "cake" not in bmm_bf16.suitable_auto_backends

    assert calls and all(CAKE_BMM_RUNNER not in call.runners for call in calls)
    torch.testing.assert_close(out, ref.to(out.dtype), rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_bmm_bf16_cake_measured_under_autotune(monkeypatch, fresh_tuner, res_dtype):
    """(c) Tuning measures cake with the other backends; any winner matches the
    reference, also when the tuned choice is replayed without tuning."""
    _skip_unless_cake_bf16_bmm_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    a, mat2, ref = _bmm_case(4, 64, 256, 256, seed=1)
    calls = _spy_choose_one(monkeypatch)
    cake_launches = _count_calls(monkeypatch, gemm_base, "_run_cake_bf16_bmm")
    out = torch.empty((4, 64, 256), device="cuda", dtype=res_dtype)
    replayed = torch.empty_like(out)

    with autotune(True, tuning_buckets=(64,)):
        bmm_bf16(a, mat2, out=out, out_dtype=res_dtype, backend="auto")
    tuning_launches = len(cake_launches)
    bmm_bf16(a, mat2, out=replayed, out_dtype=res_dtype, backend="auto")

    assert calls[0].runners[-1] == CAKE_BMM_RUNNER
    assert tuning_launches > 0, "tuning never ran the cake runner"
    assert f"bf16_gemm::{CAKE_BMM_RUNNER}" not in fresh_tuner.stats.failed_tactics
    assert calls[-1].runner == calls[0].runner
    expected = ref.to(res_dtype)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(replayed, expected, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize(
    "shape", [(4, 64, 256, 256), (2, 8, 1024, 1024), (16, 48, 80, 64)]
)
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_bmm_bf16_cake_runner_forced(monkeypatch, fresh_tuner, shape, res_dtype):
    """(d) The cake runner, chosen through the tuner path or called directly,
    reproduces backend="cake" bit for bit (dispatcher and swap-AB rows)."""
    _skip_unless_cake_bf16_bmm_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    monkeypatch.delenv(SWAP_AB_DISABLE_ENV, raising=False)
    b, m, n, k = shape
    a, mat2, ref = _bmm_case(b, m, n, k, seed=2)
    explicit = bmm_bf16(a, mat2, out_dtype=res_dtype, backend="cake")
    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    runner = gemm_base._cake_bf16_bmm_runner()  # pyright: ignore[reportPrivateUsage]
    direct = torch.empty_like(explicit)
    workspace = torch.empty(1, device="cuda", dtype=torch.uint8)
    inputs = [a, mat2, None, False, direct, workspace]

    # The default bucket of the swap-AB row M=8 also holds M=5..7, which run
    # the dispatcher, so cake is not offered there; round_up maps every M to
    # its own bucket.
    swap_ab_row = (b, m, n, k) == (2, 8, 1024, 1024)
    if swap_ab_row:
        assert runner.get_valid_tactics(inputs, None) == []
    buckets = autotune(False, round_up=True) if swap_ab_row else nullcontext()
    calls = _spy_choose_one(monkeypatch, force_last=True)
    with buckets:
        forced = bmm_bf16(a, mat2, out_dtype=res_dtype, backend="auto")
        assert calls[-1].runner == CAKE_BMM_RUNNER
        assert runner.get_valid_tactics(inputs, None) == [-1]
    runner(inputs=inputs, tactic=-1)
    torch.cuda.synchronize()

    torch.testing.assert_close(explicit, ref.to(res_dtype), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(forced, explicit, rtol=0, atol=0)
    torch.testing.assert_close(direct, explicit, rtol=0, atol=0)


def test_cake_bf16_bmm_bucket_verdict():
    """(a) The bucket-wide check: a bucket that holds a swap-AB row and an M
    running the dispatcher, or an M past the int32 limits, has no cake."""
    verdict = gemm_base._cake_bf16_bmm_bucket_verdict  # pyright: ignore[reportPrivateUsage]
    hybrid = map_to_hybrid_bucket_uncapped

    def one_m_per_bucket(m):
        return m

    for target in ("sm_100a", "sm_103a"):
        # Swap-AB rows M=8, 16, 33 share buckets (4, 8], (8, 16], (32, 64].
        assert not verdict(hybrid, 8, target, 2, 1024, 1024, torch.bfloat16, 2)
        assert not verdict(hybrid, 16, target, 4, 1024, 1024, torch.float32, 4)
        assert not verdict(hybrid, 64, target, 2, 1032, 1024, torch.bfloat16, 2)
        # No row of these (batch, N, K, dtype) in the bucket: one program.
        assert verdict(hybrid, 4, target, 2, 1024, 1024, torch.bfloat16, 2)
        assert verdict(hybrid, 32, target, 2, 1024, 1024, torch.bfloat16, 2)
        assert verdict(hybrid, 8, target, 3, 1024, 1024, torch.bfloat16, 2)
        assert verdict(hybrid, 8, target, 2, 1024, 256, torch.bfloat16, 2)
        # A bucket of one M runs one program, a row or not.
        assert verdict(one_m_per_bucket, 8, target, 2, 1024, 1024, torch.bfloat16, 2)
        assert verdict(one_m_per_bucket, 5, target, 2, 1024, 1024, torch.bfloat16, 2)
    # Swap-AB routing off: every M runs the dispatcher.
    assert verdict(hybrid, 8, None, 2, 1024, 1024, torch.bfloat16, 2)
    # Batch 512, K 1024: A has at most 2**31 - 1 elements for M <= 4095, so
    # bucket (3584, 4096] holds an M cake rejects and bucket (3072, 3584] none.
    assert not verdict(hybrid, 4096, None, 512, 64, 1024, torch.bfloat16, 2)
    assert verdict(hybrid, 3584, None, 512, 64, 1024, torch.bfloat16, 2)


# (batch, N, K, output dtype, every M of the default bucket of one swap-AB row)
_SWAP_AB_ROW_BUCKETS = [
    pytest.param(2, 1024, 1024, torch.bfloat16, 8, (5, 6, 7, 8), id="row_m8"),
    pytest.param(4, 1024, 1024, torch.float16, 16, tuple(range(9, 17)), id="row_m16"),
    pytest.param(2, 1032, 1024, torch.bfloat16, 33, (33, 34, 40, 64), id="row_m33"),
]


@pytest.mark.parametrize(
    "batch, n, k, res_dtype, row_m, bucket_ms", _SWAP_AB_ROW_BUCKETS
)
def test_bmm_bf16_cake_absent_for_every_m_of_swap_ab_row_bucket(
    monkeypatch, fresh_tuner, batch, n, k, res_dtype, row_m, bucket_ms
):
    """(a) A default bucket that holds a swap-AB row and M that run the
    dispatcher offers cake for none of its M, the row included, and the cake
    runner offers no tactic at any of them, so a cake winner of the bucket
    can never be a cache miss for another M of it."""
    _skip_unless_cake_bf16_bmm_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    monkeypatch.delenv(SWAP_AB_DISABLE_ENV, raising=False)
    device = torch.device("cuda")
    row = (batch, row_m, n, k, res_dtype)
    assert gemm_base._cake_bf16_bmm_swap_ab_row_route(device, row) is not None  # pyright: ignore[reportPrivateUsage]
    assert row_m in bucket_ms
    assert len({map_to_hybrid_bucket_uncapped(m) for m in bucket_ms}) == 1
    runner = gemm_base._cake_bf16_bmm_runner()  # pyright: ignore[reportPrivateUsage]
    workspace = torch.empty(1, device="cuda", dtype=torch.uint8)
    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    calls = _spy_choose_one(monkeypatch)

    for m in bucket_ms:
        a, mat2, ref = _bmm_case(batch, m, n, k)
        out = torch.empty((batch, m, n), device="cuda", dtype=res_dtype)
        bmm_bf16(a, mat2, out=out, out_dtype=res_dtype, backend="auto")
        # The checker accepts the call; only the runner list leaves cake out.
        assert bmm_bf16.suitable_auto_backends[-1] == "cake"
        assert CAKE_BMM_RUNNER not in calls[-1].runners, f"M={m}"
        inputs = [a, mat2, None, False, out, workspace]
        assert runner.get_valid_tactics(inputs, None) == [], f"M={m}"
        torch.testing.assert_close(out, ref.to(res_dtype), rtol=2e-2, atol=5e-2)


def test_bmm_bf16_cake_not_timed_on_swap_ab_row_bucket(monkeypatch, fresh_tuner):
    """(c) Tuning a call whose bucket offers cake times cake on the other
    profiled buckets only where it is offered for every M: never on the bucket
    (8, 16] of the swap-AB row M=16, so the replayed calls of that bucket find
    the tuned non-cake winner."""
    _skip_unless_cake_bf16_bmm_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    monkeypatch.delenv(SWAP_AB_DISABLE_ENV, raising=False)
    batch, n, k, res_dtype = 4, 1024, 1024, torch.float16
    device = torch.device("cuda")
    row = (batch, 16, n, k, res_dtype)
    assert gemm_base._cake_bf16_bmm_swap_ab_row_route(device, row) is not None  # pyright: ignore[reportPrivateUsage]
    launched_m = []
    run_cake = gemm_base._run_cake_bf16_bmm  # pyright: ignore[reportPrivateUsage]

    def recording(A, B, out):
        launched_m.append(A.shape[1])
        return run_cake(A, B, out)

    monkeypatch.setattr(gemm_base, "_run_cake_bf16_bmm", recording)
    calls = _spy_choose_one(monkeypatch)

    a, mat2, ref = _bmm_case(batch, 64, n, k, seed=3)
    out = torch.empty((batch, 64, n), device="cuda", dtype=res_dtype)
    with autotune(True, tuning_buckets=(8, 16, 64)):
        bmm_bf16(a, mat2, out=out, out_dtype=res_dtype, backend="auto")
    assert calls[0].runners[-1] == CAKE_BMM_RUNNER  # bucket (32, 64]: no row
    assert launched_m, "tuning never ran the cake runner"
    assert all(map_to_hybrid_bucket_uncapped(m) != 16 for m in launched_m)
    torch.testing.assert_close(out, ref.to(res_dtype), rtol=2e-2, atol=5e-2)

    del launched_m[:]
    # Whichever runner won, cake is listed outside tuning from here on, so only
    # the bucket check leaves it out of the replayed calls.
    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    for m in (9, 16):
        a, mat2, ref = _bmm_case(batch, m, n, k, seed=4)
        out = torch.empty((batch, m, n), device="cuda", dtype=res_dtype)
        bmm_bf16(a, mat2, out=out, out_dtype=res_dtype, backend="auto")
        assert CAKE_BMM_RUNNER not in calls[-1].runners, f"M={m}"
        torch.testing.assert_close(out, ref.to(res_dtype), rtol=2e-2, atol=5e-2)
    assert not launched_m


def test_bmm_bf16_cake_runner_rejects_unsupported_inputs():
    """(a) The runner offers no tactic for operands cake does not serve."""
    _skip_unless_cake_bf16_bmm_supported()
    a, mat2, _ = _bmm_case(1, 16, 32, 64)
    out = torch.empty((1, 16, 32), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(1, device="cuda", dtype=torch.uint8)
    runner = gemm_base._cake_bf16_bmm_runner()  # pyright: ignore[reportPrivateUsage]

    def tactics(a_, b_, bias_, out_):
        return runner.get_valid_tactics([a_, b_, bias_, False, out_, workspace], None)

    assert tactics(a, mat2, None, out) == [-1]
    misaligned_a = torch.empty(a.numel() + 1, device="cuda", dtype=a.dtype)[1:]
    misaligned_a = misaligned_a.view_as(a).copy_(a)
    assert tactics(misaligned_a, mat2, None, out) == []
    assert tactics(a, mat2, torch.zeros(32, device="cuda"), out) == []  # bias
    assert tactics(a[0], mat2[0], None, out[0]) == []  # 2D (mm_bf16)
    assert tactics(a, mat2.contiguous(), None, out) == []  # row-major B
    out_over_a = a.view(-1)[: out.numel()].view_as(out)
    assert tactics(a, mat2, None, out_over_a) == []  # out overlaps A


def test_cake_candidates_absent_on_unsupported_arch(monkeypatch, fresh_tuner):
    """(a) Off SM100/SM103 neither API lists cake, and its runners offer no
    tactic for operands they would serve there."""
    if _cake_arch():
        pytest.skip("covers architectures without the cake GEMMs")
    compute_capability = get_compute_capability(torch.device("cuda"))
    cc = compute_capability[0] * 10 + compute_capability[1]
    assert not mm_fp4.is_backend_supported("cake", cc)
    assert not bmm_bf16.is_backend_supported("cake", cc)
    _set_switches(monkeypatch, cake_enabled=True, experimental=True)
    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    _hold_cake_entry(fresh_tuner, CAKE_FP4_RUNNER)

    a, mat2, ref = _bmm_case(4, 64, 256, 256)
    out = torch.empty((4, 64, 256), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(1, device="cuda", dtype=torch.uint8)
    bmm_runner = gemm_base._cake_bf16_bmm_runner()  # pyright: ignore[reportPrivateUsage]
    assert (
        bmm_runner.get_valid_tactics([a, mat2, None, False, out, workspace], None) == []
    )
    if bmm_bf16.is_compute_capability_supported(cc):
        calls = _spy_choose_one(monkeypatch)
        bmm_bf16(a, mat2, out=out, backend="auto")
        assert "cake" not in bmm_bf16.suitable_auto_backends
        assert all(CAKE_BMM_RUNNER not in call.runners for call in calls)
        torch.testing.assert_close(out, ref.to(torch.bfloat16), rtol=2e-2, atol=5e-2)

    if compute_capability[0] >= 10:  # nvfp4_quantize needs SM100+
        operands, _ = _fp4_case(16)
        fp4_out = torch.empty((16, FP4_N), device="cuda", dtype=torch.bfloat16)
        fp4_runner = cake_candidate.cake_gemm_fp4_runner(False, True)
        inputs = _fp4_runner_inputs(operands, fp4_out)
        assert fp4_runner.get_valid_tactics(inputs, None) == []


# ---------------------------------------------------------------------------
# Listing gate: no cake work outside tuning until the tuner holds a cake entry
# ---------------------------------------------------------------------------


def _saved_runner_names(path) -> set:
    """Runner class names of the tactic entries of an autotune cache file."""
    configs = json.loads(path.read_text())
    return {
        value[0]
        for key, value in configs.items()
        if not key.startswith("_") and isinstance(value, list) and value
    }


def test_runner_class_gate_follows_tuner_sources(monkeypatch, fresh_tuner):
    """The gate is closed for a fresh tuner, open in tuning mode, for a winner
    of the class in any in-memory partition, for a loaded config of the class,
    while a managed store is active and for a bundled config of the class, and
    closed again once those entries are gone."""
    from flashinfer.autotuner import autotuner as autotuner_module

    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    gate = fresh_tuner.may_select_runner_class
    assert not gate(CAKE_BMM_RUNNER) and not gate(CAKE_FP4_RUNNER)

    with autotune(True):
        assert gate(CAKE_BMM_RUNNER)
    assert not gate(CAKE_BMM_RUNNER)

    # An in-memory winner, added and removed after the answer was memoized.
    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    assert gate(CAKE_BMM_RUNNER) and not gate(CAKE_FP4_RUNNER)
    (key,) = list(fresh_tuner.profiling_cache)
    del fresh_tuner.profiling_cache[key]
    assert not gate(CAKE_BMM_RUNNER)
    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    assert gate(CAKE_BMM_RUNNER)
    fresh_tuner.clear_cache()
    assert not gate(CAKE_BMM_RUNNER)

    # A store or policy partition of the in-memory winners.
    fresh_tuner._winner_partitions[("__policy__", "gate")] = {key: (-1, None)}  # pyright: ignore[reportPrivateUsage]
    assert gate(CAKE_BMM_RUNNER)
    fresh_tuner.clear_cache()
    assert not gate(CAKE_BMM_RUNNER)

    # A loaded config (load_configs / autotune(cache=...)).
    fresh_tuner._file_configs[key.file_key] = (CAKE_BMM_RUNNER, -1)  # pyright: ignore[reportPrivateUsage]
    assert gate(CAKE_BMM_RUNNER) and not gate(CAKE_FP4_RUNNER)
    fresh_tuner.clear_cache()
    assert not gate(CAKE_BMM_RUNNER)

    # An active managed store: its entries are read lazily, so always open.
    monkeypatch.setattr(fresh_tuner, "_managed_cache", SimpleNamespace())
    assert gate(CAKE_BMM_RUNNER)
    monkeypatch.setattr(fresh_tuner, "_managed_cache", None)
    assert not gate(CAKE_BMM_RUNNER)

    # Bundled configs, read only with FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1.
    monkeypatch.setattr(
        autotuner_module,
        "_bundled_configs_hold_runner_class",
        lambda name: name == CAKE_BMM_RUNNER,
    )
    assert not gate(CAKE_BMM_RUNNER)
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    assert gate(CAKE_BMM_RUNNER) and not gate(CAKE_FP4_RUNNER)


def test_runner_class_gate_scans_bundled_configs_only_when_loading(
    monkeypatch, fresh_tuner
):
    """Without ``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1`` the gate never scans
    (and so never imports) the bundled config modules. No GPU needed."""
    from flashinfer.autotuner import autotuner as autotuner_module

    scanned = []

    def scan(name):
        scanned.append(name)
        return False

    monkeypatch.setattr(autotuner_module, "_bundled_configs_hold_runner_class", scan)
    gate = fresh_tuner.may_select_runner_class
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)
    assert not gate(CAKE_BMM_RUNNER)
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    assert not gate(CAKE_BMM_RUNNER)
    assert scanned == []
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    assert not gate(CAKE_BMM_RUNNER)
    assert scanned == [CAKE_BMM_RUNNER]


def test_disable_env_closes_cake_candidates_before_the_gate(monkeypatch, fresh_tuner):
    """``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES=1`` leaves every automatic
    Cake candidate out without consulting the tuner's gate. No GPU needed."""
    from flashinfer.gemm import gemm_svdquant

    _set_switches(monkeypatch, cake_enabled=False, experimental=False)
    _raise_if_called(monkeypatch, fresh_tuner, "may_select_runner_class")
    assert not gemm_base._cake_auto_candidate_open(CAKE_BMM_RUNNER)  # pyright: ignore[reportPrivateUsage]
    assert not gemm_base._cake_auto_candidate_open(CAKE_FP4_RUNNER)  # pyright: ignore[reportPrivateUsage]
    heuristic = gemm_svdquant._heuristic_func_nvfp4_svdquant  # pyright: ignore[reportPrivateUsage]
    assert heuristic(["cutlass", "cake"]) == ["cutlass"]


def test_group_tuning_closes_cake_candidates(monkeypatch, fresh_tuner):
    """Tuning under an autotune process group lists no automatic Cake
    candidate, since ranks with M in different tuning buckets would list
    different runners. Without tuning the group changes nothing. No GPU needed."""
    from flashinfer.autotuner import autotuner as autotuner_module

    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    is_open = gemm_base._cake_auto_candidate_open  # pyright: ignore[reportPrivateUsage]
    bmm_heuristic = gemm_base._heuristic_func_bmm_bf16  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(fresh_tuner, "is_tuning_mode", True)
    assert is_open(CAKE_BMM_RUNNER) and is_open(CAKE_FP4_RUNNER)
    assert bmm_heuristic(["cudnn", "cake"], None, None) == ["cudnn", "cake"]

    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())
    assert fresh_tuner.may_select_runner_class(CAKE_BMM_RUNNER)
    assert not is_open(CAKE_BMM_RUNNER) and not is_open(CAKE_FP4_RUNNER)
    assert bmm_heuristic(["cudnn", "cake"], None, None) == ["cudnn"]

    # Serving with the group still set: listed as without a group.
    monkeypatch.setattr(fresh_tuner, "is_tuning_mode", False)
    assert not is_open(CAKE_BMM_RUNNER)
    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    _hold_cake_entry(fresh_tuner, CAKE_FP4_RUNNER)
    assert is_open(CAKE_BMM_RUNNER) and is_open(CAKE_FP4_RUNNER)


def test_bmm_bf16_group_tuning_plans_the_same_runners_for_every_m(
    monkeypatch, fresh_tuner
):
    """(c) Under an autotune process group every rank must list the same
    runners: AutoTuner.choose_one compares their class names across ranks
    before timing. A rank at M=64 (bucket (32, 64], which offers cake while
    tuning) and a rank at M=16 (bucket (8, 16] of a swap-AB row, which does
    not) plan the same runners, without cake. Without a group M=64 still
    lists cake."""
    from flashinfer.autotuner import autotuner as autotuner_module

    _skip_unless_cake_bf16_bmm_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    monkeypatch.delenv(SWAP_AB_DISABLE_ENV, raising=False)
    batch, n, k, res_dtype = 4, 1024, 1024, torch.float16
    row = (batch, 16, n, k, res_dtype)
    device = torch.device("cuda")
    assert gemm_base._cake_bf16_bmm_swap_ab_row_route(device, row) is not None  # pyright: ignore[reportPrivateUsage]
    cases = [_bmm_case(batch, m, n, k, seed=5) for m in (64, 16)]
    offered, planned = _record_group_plans(monkeypatch, fresh_tuner)

    for a, mat2, ref in cases:
        out = bmm_bf16(a, mat2, out_dtype=res_dtype, backend="auto")
        torch.testing.assert_close(out, ref.to(res_dtype), rtol=2e-2, atol=5e-2)
    assert not planned
    with_cake, without_cake = offered
    assert CAKE_BMM_RUNNER not in without_cake
    assert with_cake == [*without_cake, CAKE_BMM_RUNNER]

    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())
    for a, mat2, _ in cases:
        with pytest.raises(_GroupPlanChecked):
            bmm_bf16(a, mat2, out_dtype=res_dtype, backend="auto")
    assert offered[2:] == planned == [without_cake, without_cake]


def test_cake_runner_class_names_match_the_gate():
    """The gate keys on the class names the runners are cached under."""
    assert type(gemm_base._cake_bf16_bmm_runner()).__name__ == CAKE_BMM_RUNNER  # pyright: ignore[reportPrivateUsage]
    assert gemm_base._CAKE_BF16_BMM_RUNNER == CAKE_BMM_RUNNER  # pyright: ignore[reportPrivateUsage]
    for layout in (False, True):
        for pdl in (False, True):
            runner = cake_candidate.cake_gemm_fp4_runner(layout, pdl)
            assert type(runner).__name__ == CAKE_FP4_RUNNER
    assert gemm_base._CAKE_FP4_GEMM_RUNNER == CAKE_FP4_RUNNER  # pyright: ignore[reportPrivateUsage]


def test_bmm_bf16_untuned_call_does_no_cake_work(monkeypatch, fresh_tuner):
    """Without tuning and without a cake entry the call runs no cake check and
    builds no cake runner, and lists exactly the switched-off runners; a held
    cake entry makes the same call reach the cake checks."""
    _skip_unless_cake_bf16_bmm_supported()
    a, mat2, ref = _bmm_case(4, 64, 256, 256)
    calls = _spy_choose_one(monkeypatch)
    _set_switches(monkeypatch, cake_enabled=False, experimental=False)
    bmm_bf16(a, mat2, backend="auto")
    base = calls[-1]

    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    _raise_if_called(
        monkeypatch,
        gemm_base,
        "_cake_bf16_bmm_offered",
        "_cake_bf16_bmm_serves",
        "_cake_bf16_bmm_serves_bucket",
        "_cake_bf16_bmm_runner",
    )
    out = bmm_bf16(a, mat2, backend="auto")
    assert "cake" not in bmm_bf16.suitable_auto_backends
    assert calls[-1].runners == base.runners
    assert calls[-1].is_first and calls[-1].tactic == base.tactic
    torch.testing.assert_close(out, ref.to(torch.bfloat16), rtol=2e-2, atol=5e-2)

    _hold_cake_entry(fresh_tuner, CAKE_BMM_RUNNER)
    with pytest.raises(AssertionError, match="was called"):
        bmm_bf16(a, mat2, backend="auto")


def test_bmm_bf16_cake_winner_served_after_tuning_and_from_cache_file(
    monkeypatch, fresh_tuner, tmp_path
):
    """A bucket that cake won is served by cake outside tuning, in the tuning
    process and in a later tuner that loads the saved cache file; clearing the
    caches closes the gate again."""
    _skip_unless_cake_bf16_bmm_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    a, mat2, ref = _bmm_case(4, 64, 256, 256, seed=5)
    expected = ref.to(torch.bfloat16)
    cache = tmp_path / "bmm_bf16_cake.json"
    calls = _spy_choose_one(monkeypatch)
    timed = _cake_wins_timing(monkeypatch, CAKE_BMM_RUNNER)

    with autotune(True, tuning_buckets=(64,), cache=str(cache)):
        bmm_bf16(a, mat2, backend="auto")
    assert CAKE_BMM_RUNNER in timed and len(set(timed)) > 1
    assert calls[-1].runner == CAKE_BMM_RUNNER
    assert CAKE_BMM_RUNNER in _saved_runner_names(cache)

    # The tuning process, outside tuning.
    out = bmm_bf16(a, mat2, backend="auto")
    assert calls[-1].runners[-1] == CAKE_BMM_RUNNER
    assert calls[-1].runner == CAKE_BMM_RUNNER
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)

    # A later tuner: closed until the cache file is loaded.
    later = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", later)
    bmm_bf16(a, mat2, backend="auto")
    assert CAKE_BMM_RUNNER not in calls[-1].runners and calls[-1].is_first
    with autotune(False, cache=str(cache)):
        out = bmm_bf16(a, mat2, backend="auto")
        assert calls[-1].runner == CAKE_BMM_RUNNER
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)
    # Loaded configs stay loaded after the context.
    bmm_bf16(a, mat2, backend="auto")
    assert calls[-1].runner == CAKE_BMM_RUNNER

    later.clear_cache()
    out = bmm_bf16(a, mat2, backend="auto")
    assert CAKE_BMM_RUNNER not in calls[-1].runners and calls[-1].is_first
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)


def test_bmm_bf16_cake_winner_served_from_managed_store(
    monkeypatch, fresh_tuner, tmp_path
):
    """A bucket that cake won under ``autotune_v2`` is served by cake from the
    managed on-disk store: after the context in the tuning process, and in a
    later tuner only once it replays that store."""
    _skip_unless_cake_bf16_bmm_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=False)
    a, mat2, ref = _bmm_case(4, 64, 256, 256, seed=8)
    expected = ref.to(torch.bfloat16)
    calls = _spy_choose_one(monkeypatch)
    timed = _cake_wins_timing(monkeypatch, CAKE_BMM_RUNNER)

    with autotune_v2(cache_root=str(tmp_path), tuning_buckets=(64,)):
        bmm_bf16(a, mat2, backend="auto")
    assert CAKE_BMM_RUNNER in timed and len(set(timed)) > 1
    assert calls[-1].runner == CAKE_BMM_RUNNER
    entries = [
        json.loads(path.read_text()) for path in tmp_path.glob("v2/*/entries/*.json")
    ]
    assert CAKE_BMM_RUNNER in {entry["runner"] for entry in entries}

    # The tuning process, outside the context: the store stays attached.
    out = bmm_bf16(a, mat2, backend="auto")
    assert calls[-1].runner == CAKE_BMM_RUNNER
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)

    # A later tuner: closed until it replays the store, which it reads lazily.
    later = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", later)
    bmm_bf16(a, mat2, backend="auto")
    assert CAKE_BMM_RUNNER not in calls[-1].runners and calls[-1].is_first
    timed.clear()
    with autotune_v2(mode="replay", cache_root=str(tmp_path)):
        out = bmm_bf16(a, mat2, backend="auto")
        assert calls[-1].runners[-1] == CAKE_BMM_RUNNER
        assert calls[-1].runner == CAKE_BMM_RUNNER
    assert timed == []
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-2)


def test_is_contiguous_layout_matches_torch():
    """The memoized fp4 layout check reads contiguity from shape and stride
    the way ``torch.Tensor.is_contiguous`` does."""
    base = torch.empty(6, 8, 4)
    views = [
        base,
        base[:, :1],
        base[:1],
        base.transpose(0, 1),
        base[:, ::2],
        base[:, :0],
        base.permute(2, 0, 1),
        torch.empty(1, 8).expand(5, 8),
        torch.empty(8, 1).expand(8, 1),
        torch.empty(0, 3).t(),
    ]
    for view in views:
        layout = cake_candidate._is_contiguous_layout(view.shape, view.stride())  # pyright: ignore[reportPrivateUsage]
        assert layout == view.is_contiguous(), (view.shape, view.stride())


def test_mm_fp4_untuned_call_does_no_cake_work(monkeypatch, fresh_tuner):
    """Without tuning and without a cake entry the per-token call runs no cake
    check and builds no cake runner, and lists cute-dsl only; a held cake entry
    makes the same call reach the cake checks."""
    _skip_unless_cake_fp4_supported()
    operands, ref = _fp4_case(16, seed=6)
    calls = _spy_choose_one(monkeypatch)
    _set_switches(monkeypatch, cake_enabled=False, experimental=True)
    _mm_fp4(operands, torch.bfloat16)
    base = calls[-1]

    _set_switches(monkeypatch, cake_enabled=True, experimental=True)
    _raise_if_called(
        monkeypatch, cake_candidate, "cake_fp4_gemm_serves", "cake_gemm_fp4_runner"
    )
    out = _mm_fp4(operands, torch.bfloat16)
    assert mm_fp4.suitable_auto_backends == ["cute-dsl"]
    assert calls[-1].runners == base.runners
    assert calls[-1].is_first and calls[-1].tactic == base.tactic
    _assert_matches(out, ref)

    _hold_cake_entry(fresh_tuner, CAKE_FP4_RUNNER)
    with pytest.raises(AssertionError, match="was called"):
        _mm_fp4(operands, torch.bfloat16)


def test_mm_fp4_cake_winner_served_after_tuning(monkeypatch, fresh_tuner):
    """A bucket that cake won is served by cake outside tuning; clearing the
    caches closes the gate again."""
    _skip_unless_cake_fp4_supported()
    _set_switches(monkeypatch, cake_enabled=True, experimental=True)
    operands, ref = _fp4_case(16, seed=7)
    calls = _spy_choose_one(monkeypatch)
    timed = _cake_wins_timing(monkeypatch, CAKE_FP4_RUNNER)

    with autotune(True, tuning_buckets=(16,)):
        _mm_fp4(operands, torch.bfloat16)
    assert CAKE_FP4_RUNNER in timed and len(set(timed)) > 1
    assert calls[-1].runner == CAKE_FP4_RUNNER

    out = _mm_fp4(operands, torch.bfloat16)
    assert mm_fp4.suitable_auto_backends == ["cute-dsl", "cake"]
    assert calls[-1].runner == CAKE_FP4_RUNNER
    _assert_matches(out, ref)

    fresh_tuner.clear_cache()
    out = _mm_fp4(operands, torch.bfloat16)
    assert mm_fp4.suitable_auto_backends == ["cute-dsl"]
    assert CAKE_FP4_RUNNER not in calls[-1].runners and calls[-1].is_first
    _assert_matches(out, ref)
