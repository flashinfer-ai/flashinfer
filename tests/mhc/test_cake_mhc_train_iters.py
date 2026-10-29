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

Tests of the run-time Sinkhorn iteration count of the Cake four-stream mHC training package (SM100 / SM103 / SM107).

``sinkhorn_iters`` is an integer from 1 to ``SINKHORN_ITERS_MAX`` (64).  The default 20 runs the compiled programs,
whose chain is unrolled for it; every other count selects the run-time-count programs of the coefficient pass and the
token-owner pass, which take the count as a kernel argument.  This module checks:

* the selection (pure host rules: at 20 no run-time-count record launches, at any other count the coefficient
  and token-owner passes launch exactly their run-time-count record) is checked on CPU in
  ``tests/cpu/mhc/test_cake_mhc_train_rules.py``;
* the numerics at counts 1, 2, 5, 10, 19, 21, 30, 50 and 64 on T = 1, 7, 1024 and 16263: every output against an FP64
  reference evaluated at the same count, under the absolute bars of ``test_cake_mhc_train.py`` (BF16 outputs at atol =
  rtol = 1e-2 with no mismatch; the FP32 statistics, gradient statistics and parameter gradients at rel-L2 <= 1e-5;
  ``dscale`` / ``dbase`` per element <= 1e-4 of max(|ref|, RMS(ref))), and the comparator-relative clause of the
  parameter gradients over 25 draws at three counts;
* the paths: at each count the checkpoint replay (``forward_stats``) is bitwise the ``forward_prepare`` record, and the
  autograd functions (``save_stats`` True and False) give bitwise the outputs and gradients of the direct entry points;
* that changing the count in one process loads no module after the first run-time-count call, and the argument checks.

The reference is the same arithmetic as ``test_cake_mhc_train.py``'s with the count as a parameter.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_mhc_train")
loader = cake_package("cake_mhc_train", "cake_jit")
forward_prepare = pkg.cake_mhc_train_forward_prepare
forward_stats = pkg.cake_mhc_train_forward_stats
forward_finish = pkg.cake_mhc_train_forward_finish
backward_prepare = pkg.cake_mhc_train_backward_prepare
backward_finish = pkg.cake_mhc_train_backward_finish

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, N, MAPPING_ROWS = 4, 4096, 16384, 24
RMS_EPS, HC_EPS, SINKHORN_ITERS, SINKHORN_ITERS_MAX = 1e-5, 1e-6, 20, 64
COUNTS = (1, 2, 5, 10, 19, 21, 30, 50, 64)
T_COUNTS = (1, 7, 1024, 16263)
BF16_ATOL = BF16_RTOL = 1e-2
FP32_REL_L2_MAX = 1e-5
FP32_ELEM_REL_MAX = 1e-4
FP32_COMPARATOR_FACTOR = 2.0
PARAM_CLAUSE_DRAWS = 25
BLOCK_TOKENS = 8192
SEED = 2032
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(_device()) in SUPPORTED_CAPABILITIES
        and pkg.is_cake_mhc_train_supported(_device())
    )


needs_device = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_train needs an SM100 / SM103 / SM107 device with exported routes",
)


# ----------------------------------------------------------------------------------------------------------------
# Reference (the arithmetic of test_cake_mhc_train.py with the Sinkhorn count as a parameter)


def _set_ieee_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _sinkhorn(logits: torch.Tensor, iters: int, hc_eps: float = HC_EPS) -> torch.Tensor:
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _block(X, fn, scale, base, B, dY, dU, *, dtype, iters: int) -> dict:
    T = X.shape[0]
    with torch.enable_grad():
        Xd = X.detach().to(dtype).requires_grad_(True)
        fnd = fn.detach().to(dtype).requires_grad_(True)
        scaled = scale.detach().to(dtype).requires_grad_(True)
        based = base.detach().to(dtype).requires_grad_(True)
        Bd = B.detach().to(dtype).requires_grad_(True)
        u = Xd.reshape(T, N)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + RMS_EPS)
        z = (u * r) @ fnd.T
        pre = torch.sigmoid(z[:, 0:4] * scaled[0] + based[0:4]) + HC_EPS
        post = 2 * torch.sigmoid(z[:, 4:8] * scaled[1] + based[4:8])
        C = _sinkhorn(
            (z[:, 8:] * scaled[2] + based[8:]).reshape(T, STREAMS, STREAMS), iters
        )
        U_pre = (pre.unsqueeze(-1) * Xd).sum(dim=1)
        Y_pre = post.unsqueeze(-1) * Bd.unsqueeze(1) + torch.einsum(
            "tij,tid->tjd", C, Xd
        )
        dUd, dYd = dU.detach().to(dtype), dY.detach().to(dtype)
        dX32, dfn, dscale, dbase, dB32 = torch.autograd.grad(
            [U_pre, Y_pre], [Xd, fnd, scaled, based, Bd], grad_outputs=[dUd, dYd]
        )
    with torch.no_grad():
        Bd = Bd.detach()
        return {
            "r": r.detach().reshape(T), "z": z.detach(), "pre": pre.detach(), "post": post.detach(), "C": C.detach(),
            "U32": U_pre.detach(), "Y32": Y_pre.detach(), "dB32": dB32, "dX32": dX32,
            "dpost": torch.einsum("tjd,td->tj", dYd, Bd), "dC": torch.einsum("tid,tjd->tij", Xd.detach(), dYd),
            "dfn": dfn, "dscale": dscale, "dbase": dbase,
        }  # fmt: skip


def _oracle(i: dict, iters: int, *, dtype=F64) -> dict:
    """FP64 in row blocks of BLOCK_TOKENS (the oracle), or IEEE FP32 in one block (the comparator)."""
    T = i["X"].shape[0]
    if dtype == F32:
        _set_ieee_fp32()
    bt = T if dtype == F32 else max(1, min(BLOCK_TOKENS, T))
    blocks = [
        _block(
            i["X"][lo : lo + bt], i["fn"], i["scale"], i["base"], i["B"][lo : lo + bt], i["dY"][lo : lo + bt],
            i["dU"][lo : lo + bt], dtype=dtype, iters=iters,
        )
        for lo in range(0, T, bt)
    ]  # fmt: skip
    out = {
        k: torch.cat([b[k] for b in blocks], dim=0)
        for k in blocks[0]
        if k not in ("dfn", "dscale", "dbase")
    }
    for k in ("dfn", "dscale", "dbase"):
        acc = blocks[0][k].clone()
        for b in blocks[1:]:
            acc += b[k]
        out[k] = acc
    return out


def _make_inputs(T: int, *, seed: int = SEED) -> dict:
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    return {
        "X": randn(T, STREAMS, HIDDEN, dtype=BF16), "fn": (randn(MAPPING_ROWS, N) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3), "base": 0.1 * randn(MAPPING_ROWS), "B": randn(T, HIDDEN, dtype=BF16),
        "dY": randn(T, STREAMS, HIDDEN, dtype=BF16), "dU": randn(T, HIDDEN, dtype=BF16),
    }  # fmt: skip


def _run(i: dict, iters: int) -> dict:
    U, stats = forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], sinkhorn_iters=iters
    )
    Y = forward_finish(i["X"], i["B"], stats)
    dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
    dX, dfn, dscale, dbase = backward_finish(
        i["X"],
        i["dY"],
        i["dU"],
        i["B"],
        i["fn"],
        i["scale"],
        i["base"],
        stats,
        gstats,
        sinkhorn_iters=iters,
    )
    return {"U": U, "stats": stats, "Y": Y, "dB": dB, "gstats": gstats, "dX": dX, "dfn": dfn, "dscale": dscale,
            "dbase": dbase}  # fmt: skip


def _stats_columns(stats: torch.Tensor) -> dict:
    T = stats.shape[0]
    return {
        "r": stats[:, 0], "z": stats[:, 1:25], "pre": stats[:, 25:29], "post": stats[:, 29:33],
        "C": stats[:, 33:49].reshape(T, STREAMS, STREAMS),
    }  # fmt: skip


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.double() - b.double()).norm().item()
    den = b.double().norm().item()
    return num / den if den > 0 else (0.0 if num == 0 else math.inf)


def _max_rel_elem(a: torch.Tensor, b: torch.Tensor) -> float:
    ref = b.double()
    err = (a.double() - ref).abs()
    rms = ref.square().mean().sqrt().item()
    if rms == 0:
        return 0.0 if float(err.max()) == 0 else math.inf
    return float((err / ref.abs().clamp_min(rms)).max())


def _assert_bf16(out: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    assert out.dtype == BF16 and tuple(out.shape) == tuple(ref.shape), label
    a, b = out.double(), ref.double()
    assert torch.isfinite(a).all(), f"{label}: non-finite values"
    bad = int(((a - b).abs() > BF16_ATOL + BF16_RTOL * b.abs()).sum())
    assert bad == 0, (
        f"{label}: {bad} of {a.numel()} elements outside atol = rtol = 1e-2"
    )


def _assert_fp32(
    out: torch.Tensor, ref: torch.Tensor, label: str, *, per_element: bool = False
) -> None:
    assert out.dtype == F32 and tuple(out.shape) == tuple(ref.shape), label
    assert torch.isfinite(out).all(), f"{label}: non-finite values"
    error = _rel_l2(out, ref)
    assert error <= FP32_REL_L2_MAX, (
        f"{label}: rel-L2 {error:.3e} > {FP32_REL_L2_MAX:.0e}"
    )
    if per_element:
        worst = _max_rel_elem(out, ref)
        assert worst <= FP32_ELEM_REL_MAX, (
            f"{label}: per-element relative error {worst:.3e} > {FP32_ELEM_REL_MAX:.0e}"
        )


def _check(T: int, iters: int, *, seed: int = SEED) -> dict:
    i = _make_inputs(T, seed=seed)
    res = _run(i, iters)
    ref = _oracle(i, iters)
    label = f"T={T} sinkhorn_iters={iters}"
    for out, key in (("U", "U32"), ("Y", "Y32"), ("dB", "dB32"), ("dX", "dX32")):
        _assert_bf16(res[out], ref[key], f"{out} {label}")
    for name, out in _stats_columns(res["stats"]).items():
        _assert_fp32(out.contiguous(), ref[name], f"{name} {label}")
    _assert_fp32(res["gstats"][:, 0:4].contiguous(), ref["dpost"], f"dpost {label}")
    _assert_fp32(
        res["gstats"][:, 4:20].reshape(T, STREAMS, STREAMS).contiguous(),
        ref["dC"],
        f"dC {label}",
    )
    assert torch.equal(
        res["gstats"][:, 20:], torch.zeros_like(res["gstats"][:, 20:])
    ), f"gstats pad {label}"
    _assert_fp32(res["dfn"], ref["dfn"], f"dfn {label}")
    _assert_fp32(res["dscale"], ref["dscale"], f"dscale {label}", per_element=True)
    _assert_fp32(res["dbase"], ref["dbase"], f"dbase {label}", per_element=True)
    return {"inputs": i, "result": res, "ref": ref}


# ----------------------------------------------------------------------------------------------------------------
# Tests


@needs_device
@pytest.mark.parametrize("iters", COUNTS)
@pytest.mark.parametrize("T", T_COUNTS)
def test_counts_match_fp64(T: int, iters: int) -> None:
    out = _check(T, iters)
    # The count reaches the kernels: where the FP64 Sinkhorn matrices of this count and of 20 lie apart at the test's
    # tolerance, the record's matrix is nearer this count's. A chain at its fixed point (T=1 from about 19 steps on)
    # leaves the two FP64 matrices closer than that and nothing to tell apart; the values alone are checked there.
    C = _stats_columns(out["result"]["stats"])["C"]
    ref_n, ref_20 = out["ref"]["C"], _oracle(out["inputs"], SINKHORN_ITERS)["C"]
    if _rel_l2(ref_n, ref_20) > 4 * FP32_REL_L2_MAX:
        assert _rel_l2(C, ref_n) < _rel_l2(C, ref_20), (
            f"stats C at sinkhorn_iters={iters} is nearer the count of 20's"
        )


@needs_device
@pytest.mark.parametrize("iters", (1, 5, 64))
@pytest.mark.parametrize("T", (7, 257))
def test_parameter_gradients_within_2x_comparator_over_draws(
    T: int, iters: int
) -> None:
    """The comparator-relative clause of record over 25 draws at the count: rel-L2 <= 1e-5 on every draw (dscale /
    dbase per element <= 1e-4), and median and maximum of the rel-L2 <= 2x the IEEE-FP32 comparator's at the same
    count; the statistics r / z / pre / post / C hold the same form."""
    rows: dict[str, list[tuple[float, float]]] = {}
    for seed in range(SEED, SEED + PARAM_CLAUSE_DRAWS):
        i = _make_inputs(T, seed=seed)
        res = _run(i, iters)
        ref, comp = _oracle(i, iters), _oracle(i, iters, dtype=F32)
        outs = {
            **{k: res[k] for k in ("dfn", "dscale", "dbase")},
            **_stats_columns(res["stats"]),
        }
        for k, out in outs.items():
            out = out.contiguous()
            error = _rel_l2(out, ref[k])
            assert error <= FP32_REL_L2_MAX, (
                f"{k} T={T} sinkhorn_iters={iters} seed={seed}: rel-L2 {error:.3e}"
            )
            if k in ("dscale", "dbase"):
                assert _max_rel_elem(out, ref[k]) <= FP32_ELEM_REL_MAX, (
                    k,
                    T,
                    iters,
                    seed,
                )
            rows.setdefault(k, []).append((error, _rel_l2(comp[k], ref[k])))
    for k, pairs in rows.items():
        kernel, comparator = [p[0] for p in pairs], [p[1] for p in pairs]
        print(
            f"{k} T={T} sinkhorn_iters={iters}: kernel rel-L2 median {statistics.median(kernel):.3e} max "
            f"{max(kernel):.3e}; comparator median {statistics.median(comparator):.3e} max {max(comparator):.3e}",
            flush=True,
        )
        assert statistics.median(kernel) <= FP32_COMPARATOR_FACTOR * statistics.median(
            comparator
        ), (k, T, iters)
        assert max(kernel) <= FP32_COMPARATOR_FACTOR * max(comparator), (k, T, iters)


@needs_device
@pytest.mark.parametrize("iters", COUNTS)
@pytest.mark.parametrize("T", (129, 2049))
def test_paths_agree_bitwise(T: int, iters: int) -> None:
    """At each count: the checkpoint replay is bitwise the forward record, and the autograd functions (statistics saved
    or replayed) give bitwise the outputs and gradients of the direct entry points on the same branch."""
    i = _make_inputs(T, seed=SEED + 41)
    U, stats = forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], sinkhorn_iters=iters
    )
    assert torch.equal(
        stats,
        forward_stats(i["X"], i["fn"], i["scale"], i["base"], sinkhorn_iters=iters),
    )
    B = (U.float() + i["B"].float()).to(
        BF16
    )  # the branch: dB/dU = 1, so its backward hands dB on as dU
    Y = forward_finish(i["X"], B, stats)
    dB, gstats = backward_prepare(i["dY"], B, i["X"], stats)
    direct = backward_finish(
        i["X"],
        i["dY"],
        dB,
        B,
        i["fn"],
        i["scale"],
        i["base"],
        stats,
        gstats,
        sinkhorn_iters=iters,
    )
    for save_stats in (True, False):
        X = i["X"].clone().requires_grad_(True)
        fn = i["fn"].clone().requires_grad_(True)
        scale = i["scale"].clone().requires_grad_(True)
        base = i["base"].clone().requires_grad_(True)
        Ua, handle = pkg.cake_mhc_train_prepare(
            X, fn, scale, base, sinkhorn_iters=iters, save_stats=save_stats
        )
        Ua.retain_grad()
        Ba = (Ua.float() + i["B"].float()).to(BF16)
        Ya = pkg.cake_mhc_train_finish(X, Ba, handle)
        (Ya.float() * i["dY"].float()).sum().backward()
        label = f"T={T} sinkhorn_iters={iters} save_stats={save_stats}"
        assert torch.equal(Ua.detach(), U) and torch.equal(Ya.detach(), Y), label
        assert torch.equal(Ua.grad, dB), label
        for got, want, name in zip(
            (X.grad, fn.grad, scale.grad, base.grad),
            direct,
            ("dX", "dfn", "dscale", "dbase"),
            strict=True,
        ):
            assert torch.equal(got, want), (
                f"autograd {name} differs from the direct entry point, {label}"
            )


@needs_device
def test_changing_count_does_not_rebuild() -> None:
    """The count is a launch argument of the run-time-count programs: after one call at a non-default count, calls at
    other counts (and back at the default) load no module, and each follows the FP64 reference at its count."""
    _check(64, SINKHORN_ITERS)
    _check(64, 3)
    loaded = loader.load.cache_info().misses
    for iters in (1, 2, 7, 33, 64, SINKHORN_ITERS, 11):
        i = _make_inputs(255, seed=SEED + iters)
        res = _run(i, iters)
        ref = _oracle(i, iters)
        _assert_bf16(res["dX"], ref["dX32"], f"dX sinkhorn_iters={iters}")
        _assert_fp32(res["dfn"], ref["dfn"], f"dfn sinkhorn_iters={iters}")
    assert loader.load.cache_info().misses == loaded, (
        "a changed sinkhorn_iters loaded a module"
    )


@needs_device
def test_count_validation() -> None:
    assert pkg.SHIPPED_SINKHORN_ITERS == (SINKHORN_ITERS,)
    assert pkg.SINKHORN_ITERS_MAX == loader.SINKHORN_ITERS_MAX == SINKHORN_ITERS_MAX
    i = _make_inputs(8, seed=SEED + 9)
    X, fn, scale, base = i["X"], i["fn"], i["scale"], i["base"]
    for iters in (0, -1, 2.5, 20.5, True, float("inf"), float("nan"), "20"):
        with pytest.raises(ValueError):
            forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        with pytest.raises(ValueError):
            forward_stats(X, fn, scale, base, sinkhorn_iters=iters)
    for iters in (SINKHORN_ITERS_MAX + 1, 1000):
        with pytest.raises(NotImplementedError):
            forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        with pytest.raises(NotImplementedError):
            pkg.cake_mhc_train_prepare(X, fn, scale, base, sinkhorn_iters=iters)
    for iters in (1, 20.0, SINKHORN_ITERS_MAX):
        forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
