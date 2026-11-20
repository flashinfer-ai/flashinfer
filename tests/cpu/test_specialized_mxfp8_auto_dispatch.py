"""Exercise mm_mxfp8 auto dispatch, not GPU kernels or autotuner profiling."""

from concurrent.futures import ThreadPoolExecutor
from inspect import getclosurevars
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

from flashinfer import mm_mxfp8, utils
from flashinfer.gemm import gemm_base as gemm


class Runner:
    def __init__(self):
        self.launches = []
        self.supported = True
        self.compatible = True

    def supports_inputs(self, inputs):
        return self.supported

    def is_tactic_compatible(self, inputs, tactic):
        return self.compatible

    def __call__(self, *, inputs, tactic):
        self.launches.append((inputs[5].dtype, tactic))


@pytest.fixture
def dispatch(monkeypatch):
    # This fixture exercises dispatch with CPU buffers; placement has its own tests.
    monkeypatch.setattr(gemm, "_check_mm_mxfp8_tensor_devices", lambda *args: None)
    # backend_requirement captures the checker table at decoration time. Limit
    # this test to two backends without rebinding or redecorating the public API.
    checks = getclosurevars(mm_mxfp8.has_backend).nonlocals["backend_checks"]
    for name in set(checks) - {"specialized", "cutlass"}:
        monkeypatch.delitem(checks, name)
    for module in (utils, gemm):
        monkeypatch.setattr(module, "get_compute_capability", lambda device: (10, 7))
    monkeypatch.setattr(gemm, "is_sm12x_supported", lambda device: False)
    monkeypatch.setattr(
        gemm,
        "specialized_mm_mxfp8_eligible",
        lambda a, b, sa, sb, out, dtype, layout: None
        if dtype == torch.bfloat16
        else "out_dtype",
    )
    specialized, stock = Runner(), Runner()
    monkeypatch.setattr(
        gemm, "SpecializedMxfp8SmallMRunner", lambda layout: specialized
    )
    module = SimpleNamespace(cutlass_mxfp8_gemm_runner=lambda: stock)
    monkeypatch.setattr(gemm, "get_cutlass_mxfp8_gemm_module", lambda major: module)
    workspace = torch.empty(1, dtype=torch.uint8, device="cpu")
    monkeypatch.setattr(gemm, "_get_stream_cache_buf", lambda *args: workspace)

    def choose_one(*, runners, inputs, **kwargs):
        expected = (
            [specialized, stock] if inputs[5].dtype == torch.bfloat16 else [stock]
        )
        assert runners == expected
        return runners[0], 7

    tuner = SimpleNamespace(choose_one=choose_one)
    monkeypatch.setattr(gemm.AutoTuner, "get", staticmethod(lambda: tuner))
    a = torch.empty((4, 128), dtype=torch.float8_e4m3fn, device="cpu")
    b = torch.empty((128, 128), dtype=a.dtype, device="cpu").T
    scale = torch.empty(512, dtype=torch.uint8, device="cpu")  # one 128x4 atom
    return SimpleNamespace(
        args=(a, b, scale, scale),
        specialized=specialized,
        stock=stock,
        tuner=tuner,
        workspace=workspace,
    )


def test_interleaved_auto(dispatch, monkeypatch):
    barrier = Barrier(2, timeout=10)

    def workspace(*args):
        # Both heuristics have returned; neither body has read its choice yet.
        barrier.wait()
        return dispatch.workspace

    monkeypatch.setattr(gemm, "_get_stream_cache_buf", workspace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(mm_mxfp8, *dispatch.args, out_dtype=dtype)
            for dtype in (torch.bfloat16, torch.float16)
        ]
        for future in futures:
            future.result()
    assert dispatch.specialized.launches == [(torch.bfloat16, 7)]
    assert dispatch.stock.launches == [(torch.float16, 7)]


@pytest.mark.parametrize("inner_fails", [False, True])
def test_nested_auto_restores_outer_choice(dispatch, monkeypatch, inner_fails):
    entered = False

    def workspace(*args):
        nonlocal entered
        if not entered:
            entered = True
            if inner_fails:
                with pytest.raises(RuntimeError, match="inner workspace"):
                    mm_mxfp8(*dispatch.args, out_dtype=torch.float16)
            else:
                mm_mxfp8(*dispatch.args, out_dtype=torch.float16)
        elif inner_fails:
            # Fail after the inner FP16 choice has replaced the outer one.
            raise RuntimeError("inner workspace")
        return dispatch.workspace

    monkeypatch.setattr(gemm, "_get_stream_cache_buf", workspace)
    mm_mxfp8(*dispatch.args, out_dtype=torch.bfloat16)
    assert dispatch.specialized.launches == [(torch.bfloat16, 7)]
    assert dispatch.stock.launches == ([] if inner_fails else [(torch.float16, 7)])


@pytest.mark.parametrize("form", ["keyword", "positional", "default"])
def test_skip_check_resolves_auto(dispatch, form):
    if form == "positional":
        mm_mxfp8(*dispatch.args, None, torch.float16, False, "auto", skip_check=True)
    else:
        kwargs = {"backend": "auto"} if form == "keyword" else {}
        mm_mxfp8(*dispatch.args, out_dtype=torch.float16, skip_check=True, **kwargs)
    assert dispatch.stock.launches == [(torch.float16, 7)]
    assert not dispatch.specialized.launches


@pytest.mark.parametrize("refusal", ["inputs", "tactic", "not_a_candidate"])
def test_revalidate_tuner_pick(dispatch, refusal):
    first, stock = dispatch.specialized, dispatch.stock
    foreign = Runner()  # Accepts everything but was not offered on this call.
    first.supported = refusal != "inputs"
    first.compatible = refusal == "inputs"  # otherwise rejects even tactic -1
    picked = foreign if refusal == "not_a_candidate" else first
    dispatch.tuner.choose_one = lambda **kwargs: (picked, 17)

    mm_mxfp8(*dispatch.args)
    assert stock.launches == [(torch.bfloat16, -1)]
    assert not first.launches and not foreign.launches

    stock.launches.clear()
    stock.supported = False
    with pytest.raises(ValueError, match="No mm_mxfp8 runner accepts these inputs"):
        mm_mxfp8(*dispatch.args)
    assert not stock.launches and not first.launches and not foreign.launches
