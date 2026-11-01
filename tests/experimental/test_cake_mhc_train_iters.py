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

Tests of the run-time Sinkhorn iteration count and the epsilons of the Cake
four-stream mHC training package with the opt-in non-deterministic backward
(``flashinfer.experimental.cake.cake_mhc_train_backend``, every entry point
with ``deterministic=False``; SM100 / SM103 / SM107; FlashInfer issue #8068), in
the form of ``test_cake_mhc_det_train_iters.py``.

``sinkhorn_iters`` is an integer from 1 to ``SINKHORN_ITERS_MAX`` (64).  The
non-deterministic backward launches the package's copy of the deterministic
token-owner pass, whose run-time-count record takes every count but the
compiled 20; the atomic fn split and column-slab pass do not depend on the
count.  The forward and the backward prepare are the package's copies of the
deterministic programs (the same bits at every count, checked here against the
deterministic package).  This module checks:

* the numerics at counts 1, 2, 5, 10, 19, 21, 30, 50 and 64 on T = 1, 7, 1024
  and 16263, on every one of 5 launches: ``dX``, ``dscale``, ``dbase`` bitwise
  the deterministic backward's at the same count and ``dfn`` within ``2 P u
  sum_t |dz'_t| |x_t|``; against an FP64 reference at the same count the BF16
  ``dX`` gate (atol = rtol = 1e-2, no mismatch) and rel-L2 <= 1e-5
  (``dscale`` / ``dbase`` per element <= 1e-4); where the FP64 ``dfn`` of the
  count and of 20 lie apart, the kernel's ``dfn`` is nearer the count's; the
  comparator-relative clause of the parameter gradients over 25 draws at three
  counts;
* the paths: at each count the package's forward prepare, statistics replay,
  residual mixing and backward prepare are bitwise the deterministic
  package's, the checkpoint replay is bitwise the forward record, and the
  autograd functions (statistics saved: the backward path;
  replayed: the checkpoint path) agree with the direct entry point and with
  each other (``dX`` / ``dscale`` / ``dbase`` bitwise, ``dfn`` within the
  bound);
* that changing the count in one process loads no module after the first
  run-time-count call, the count checks, and non-default epsilons with a
  run-time count.

The cases above run at ``H = 4096``;
``test_counts_match_fp64_at_every_hidden_size`` and
``test_paths_agree_at_every_hidden_size`` repeat the numerics and the paths at
the counts 1, 7, 20 and 64 at hidden 5120 and 7168 where the device's
architecture routes them (SM100 / SM103; SM107 routes 4096 only).
"""

from __future__ import annotations

import functools
import math
import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

KEY = "cake.cake_mhc_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
# The deterministic package: the same-input comparison point.
det = cake_package("cake_mhc_det_train")
det_loader = cake_package("cake_mhc_det_train", "cake_jit")
# The package's entry points with the mode they require.
forward_prepare = functools.partial(
    pkg.cake_mhc_train_forward_prepare, deterministic=False
)
forward_stats = functools.partial(pkg.cake_mhc_train_forward_stats, deterministic=False)
forward_finish = functools.partial(
    pkg.cake_mhc_train_forward_finish, deterministic=False
)
backward_prepare = functools.partial(
    pkg.cake_mhc_train_backward_prepare, deterministic=False
)
backward_finish = pkg.cake_mhc_train_backward_finish
det_backward_finish = det.cake_mhc_det_train_backward_finish

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, MAPPING_ROWS = 4, 4096, 24
HIDDEN_SIZES = (4096, 5120, 7168)
RMS_EPS, HC_EPS, SINKHORN_ITERS, SINKHORN_ITERS_MAX = 1e-5, 1e-6, 20, 64
COUNTS = (1, 2, 5, 10, 19, 21, 30, 50, 64)
T_COUNTS = (1, 7, 1024, 16263)
HIDDEN_COUNTS = (1, 7, 20, 64)  # the counts of the hidden sizes other than 4096
BF16_ATOL = BF16_RTOL = 1e-2
FP32_REL_L2_MAX = 1e-5
FP32_ELEM_REL_MAX = 1e-4
FP32_COMPARATOR_FACTOR = 2.0
PARAM_CLAUSE_DRAWS = 25
BLOCK_TOKENS = 8192
SEED = 2032
# Launches of the non-deterministic backward per draw: every gate holds on each (the worst is reported).
REPEATS = 5
FP32_UNIT_ROUNDOFF = 2.0**-24
ABS_SLACK = 1.02
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
_PARAMETER = ("dfn", "dscale", "dbase")
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")
_EXACT = ("dX", "dscale", "dbase")  # bitwise the deterministic package's


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_mhc_train_supported(_device())
        and loader.route_applies(
            device_capability=_capability(), hidden=HIDDEN, kind="bwd_finish"
        )
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Both packages' libraries of the local architecture (this package's, and the deterministic package's for the
    # same-input comparisons); a prebuilt or cached library loads without compiling.
    if _supported():
        arch = _arch()
        for spec in [*det_loader.specs_for_arch(arch), *loader.specs_for_arch(arch)]:
            spec.build_and_load()
    yield


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _require_hidden(hidden: int) -> int:
    """``hidden`` if the device's architecture routes every kind of it, else the case is skipped."""
    arch = _arch()
    sizes = tuple(loader.supported_hidden_sizes(arch)) if arch else ()
    if hidden not in sizes:
        pytest.skip(
            f"hidden {hidden} has no exported route on {arch} (exported: {sizes})"
        )
    return hidden


def _parts(T: int, hidden: int = HIDDEN) -> int:
    """``P``: the chunk count of the column-slab pass, the number of partials each ``dfn`` element sums."""
    parts = loader.host_plan(T, _sm_count(), _arch(), hidden)["parts"]
    assert 1 <= parts <= 32, (T, hidden, parts)
    return parts


def _lpt_runs(T: int, hidden: int = HIDDEN) -> bool:
    """Whether the column slab runs its two-size ``lpt`` record at ``T`` on this device (H 4096 only)."""
    return loader.bwd_slab_atomic_program(T, _sm_count(), _arch(), hidden) == "lpt"


def _slab_parts(T: int, hidden: int = HIDDEN) -> int:
    """The chunk count the column slab launches: ``slab_plan``'s ``lpt_parts`` under ``lpt``, else :func:`_parts`."""
    if not _lpt_runs(T, hidden):
        return _parts(T, hidden)
    parts = loader.slab_plan(T, _sm_count(), _arch(), hidden)["lpt_parts"]
    assert 1 <= parts <= 32, (T, hidden, parts)
    return parts


def _dfn_factor(T: int, hidden: int = HIDDEN) -> float:
    """F of the dfn bound F u (|carried| + ABS_SLACK sum_t |dz'_t| |x_t|) at T (``test_cake_mhc_train.py``'s
    ``_dfn_factor``, which derives it): 2 P where the column slab runs dynr (both sides sum the same per-chunk
    partials), 2 max(P, P_k7) + 10 where it runs lpt (the partials are another partition's)."""
    if not _lpt_runs(T, hidden):
        return 2.0 * _parts(T, hidden)
    return 2.0 * max(_slab_parts(T, hidden), _parts(T, hidden)) + 10.0


# ----------------------------------------------------------------------------------------------------------------
# Reference (test_cake_mhc_train.py's arithmetic: the epsilons and the Sinkhorn count a case carries)


def _set_ieee_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _sinkhorn(logits: torch.Tensor, hc_eps: float, iters: int) -> torch.Tensor:
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _coefficients(z, scale, base, hc_eps: float, iters: int):
    T = z.shape[0]
    pre = torch.sigmoid(z[:, 0:4] * scale[0] + base[0:4]) + hc_eps
    post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = _sinkhorn(
        (z[:, 8:] * scale[2] + base[8:]).reshape(T, STREAMS, STREAMS), hc_eps, iters
    )
    return pre, post, C


def _constants(i: dict) -> dict:
    """The launch constants of a case: the epsilons and the Sinkhorn count it carries (default: the model's)."""
    return {k: i[k] for k in ("rms_eps", "hc_eps", "sinkhorn_iters") if k in i}


def _reference_constants(i: dict) -> tuple[float, float, int]:
    return (
        i.get("rms_eps", RMS_EPS),
        i.get("hc_eps", HC_EPS),
        i.get("sinkhorn_iters", SINKHORN_ITERS),
    )


def _block(X, fn, scale, base, B, dY, dU, *, dtype, rms_eps, hc_eps, iters) -> dict:
    T = X.shape[0]
    with torch.enable_grad():
        Xd = X.detach().to(dtype).requires_grad_(True)
        fnd = fn.detach().to(dtype).requires_grad_(True)
        scaled = scale.detach().to(dtype).requires_grad_(True)
        based = base.detach().to(dtype).requires_grad_(True)
        u = Xd.reshape(T, -1)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)
        z = (u * r) @ fnd.T
        pre, post, C = _coefficients(z, scaled, based, hc_eps, iters)
        U_pre = (pre.unsqueeze(-1) * Xd).sum(dim=1)
        Y_pre = post.unsqueeze(-1) * B.detach().to(dtype).unsqueeze(1) + torch.einsum(
            "tij,tid->tjd", C, Xd
        )
        dX32, dfn, dscale, dbase = torch.autograd.grad(
            [U_pre, Y_pre],
            [Xd, fnd, scaled, based],
            grad_outputs=[dU.detach().to(dtype), dY.detach().to(dtype)],
        )
    return {"dX32": dX32, "dfn": dfn, "dscale": dscale, "dbase": dbase}


def _oracle(i: dict, *, dtype=F64) -> dict:
    """The reference gradients at the case's constants: FP64 in row blocks of BLOCK_TOKENS (the oracle), or IEEE FP32
    in one block (the comparator)."""
    T = i["X"].shape[0]
    if dtype == F32:
        _set_ieee_fp32()
    rms_eps, hc_eps, iters = _reference_constants(i)
    bt = T if dtype == F32 else max(1, min(BLOCK_TOKENS, T))
    blocks = [
        _block(
            i["X"][lo : lo + bt], i["fn"], i["scale"], i["base"], i["B"][lo : lo + bt], i["dY"][lo : lo + bt],
            i["dU"][lo : lo + bt], dtype=dtype, rms_eps=rms_eps, hc_eps=hc_eps, iters=iters,
        )
        for lo in range(0, T, bt)
    ]  # fmt: skip
    out = {"dX32": torch.cat([b["dX32"] for b in blocks], dim=0)}
    for k in _PARAMETER:
        acc = blocks[0][k].clone()
        for b in blocks[1:]:
            acc += b[k]
        out[k] = acc
    return out


def _dzp(i: dict) -> torch.Tensor:
    """The FP64 ``dz' = r dL/dz`` of the reference at the case's constants, ``[T, 24]`` (the backward of the
    coefficients alone, so no ``[T, 4H]`` autograd graph is held)."""
    T, n = i["X"].shape[0], STREAMS * i["X"].shape[2]
    rms_eps, hc_eps, iters = _reference_constants(i)
    fn, scale, base = (i[k].double() for k in ("fn", "scale", "base"))
    blocks = [torch.zeros((0, MAPPING_ROWS), dtype=F64, device=_device())]
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        X = i["X"][lo:hi].double()
        dY = i["dY"][lo:hi].double()
        dU = i["dU"][lo:hi].double()
        B = i["B"][lo:hi].double()
        u = X.reshape(hi - lo, n)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)
        with torch.enable_grad():
            z = ((u * r) @ fn.T).requires_grad_(True)
            pre, post, C = _coefficients(z, scale, base, hc_eps, iters)
            (dz,) = torch.autograd.grad(
                [pre, post, C],
                [z],
                grad_outputs=[
                    torch.einsum("tid,td->ti", X, dU),
                    torch.einsum("tjd,td->tj", dY, B),
                    torch.einsum("tid,tjd->tij", X, dY),
                ],
            )
        blocks.append(dz * r)
    return torch.cat(blocks, dim=0)


def _dfn_bound(i: dict) -> torch.Tensor:
    """Per-element bound of ``|dfn_a - dfn_b|`` for two sums of per-chunk partials in two orders:
    ``F u * 1.02 sum_t |dz'_t| |x_t|`` with ``F`` = :func:`_dfn_factor` (``test_cake_mhc_train.py``'s
    ``_dfn_bound``)."""
    T, hidden = i["X"].shape[0], i["X"].shape[2]
    dzp = _dzp(i).abs()
    A = torch.zeros((MAPPING_ROWS, STREAMS * hidden), dtype=F64, device=_device())
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        A += dzp[lo:hi].T @ i["X"][lo:hi].reshape(hi - lo, -1).double().abs()
    return _dfn_factor(T, hidden) * FP32_UNIT_ROUNDOFF * ABS_SLACK * A


# ----------------------------------------------------------------------------------------------------------------
# Inputs, launches and gates


def _make_inputs(T: int, *, seed: int = SEED, hidden: int = HIDDEN) -> dict:
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    return {
        "X": randn(T, STREAMS, hidden, dtype=BF16), "fn": (randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3), "base": 0.1 * randn(MAPPING_ROWS), "B": randn(T, hidden, dtype=BF16),
        "dY": randn(T, STREAMS, hidden, dtype=BF16), "dU": randn(T, hidden, dtype=BF16),
    }  # fmt: skip


def _forward(i: dict) -> dict:
    """The forward prepare at the case's constants and the backward prepare: the records ``backward_finish`` takes."""
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"], **_constants(i))
    _dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
    return {"stats": stats, "gstats": gstats}


def _backward(entry, i: dict, prep: dict, **kwargs) -> dict:
    out = entry(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"],
        **{**_constants(i), **kwargs},
    )  # fmt: skip
    return dict(zip(_OUTPUTS, out, strict=True))


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    return _backward(backward_finish, i, prep, deterministic=False, **kwargs)


def _det_backward(i: dict, prep: dict, **kwargs) -> dict:
    return _backward(det_backward_finish, i, prep, **kwargs)


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _nd_repeats(i: dict, prep: dict, n: int = REPEATS) -> list[dict]:
    """``n`` launches of the non-deterministic backward, each output finite."""
    outs = []
    for repeat in range(n):
        out = _snapshot(_nd_backward(i, prep))
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{k} repeat {repeat}: non-finite values"
        outs.append(out)
    return outs


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


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        a.detach().reshape(-1).view(torch.uint8),
        b.detach().reshape(-1).view(torch.uint8),
    )


def _assert_exact(got, want, label: str) -> None:
    """``(dX, dfn, dscale, dbase)`` sequences or dicts: ``dX``, ``dscale``, ``dbase`` bitwise."""
    for n, k in enumerate(_OUTPUTS):
        if k in _EXACT:
            a = got[k] if isinstance(got, dict) else got[n]
            b = want[k] if isinstance(want, dict) else want[n]
            assert _same_bits(a, b), f"{k} {label}: bits differ"


def _assert_dfn_within(
    got: torch.Tensor, want: torch.Tensor, bound: torch.Tensor, label: str
) -> None:
    assert got.dtype == F32 and tuple(got.shape) == tuple(want.shape), label
    assert tuple(got.shape) == tuple(bound.shape) and got.shape[0] == MAPPING_ROWS, (
        label
    )
    assert torch.isfinite(got).all(), f"{label}: non-finite dfn"
    diff = (got.double() - want.double()).abs()
    over = int((diff > bound).sum())
    ratio = float((diff / bound.clamp_min(torch.finfo(F64).tiny)).max())
    assert over == 0, (
        f"{label}: {over} dfn elements beyond 2 P u sum_t |dz'_t| |x_t| (largest ratio {ratio:.3e})"
    )


def _check(T: int, iters: int, *, seed: int = SEED, hidden: int = HIDDEN) -> dict:
    """One draw at the count: every launch against the deterministic backward and the FP64 reference."""
    i = _make_inputs(T, seed=seed, hidden=hidden)
    i["sinkhorn_iters"] = iters
    prep = _forward(i)
    want = _snapshot(_det_backward(i, prep))
    bound = _dfn_bound(i)
    outs = _nd_repeats(i, prep)
    ref = _oracle(i)
    label = f"T={T} sinkhorn_iters={iters} H={hidden}"
    for repeat, out in enumerate(outs):
        what = f"{label} launch {repeat}"
        _assert_exact(out, want, f"{what} vs deterministic")
        _assert_dfn_within(
            out["dfn"], want["dfn"], bound, f"dfn {what} vs deterministic"
        )
        _assert_bf16(out["dX"], ref["dX32"], f"dX {what}")
        _assert_fp32(out["dfn"], ref["dfn"], f"dfn {what}")
        _assert_fp32(out["dscale"], ref["dscale"], f"dscale {what}", per_element=True)
        _assert_fp32(out["dbase"], ref["dbase"], f"dbase {what}", per_element=True)
    return {"inputs": i, "outs": outs, "ref": ref}


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("iters", COUNTS)
@pytest.mark.parametrize("T", T_COUNTS)
def test_counts_match_fp64(T: int, iters: int) -> None:
    _check_counts(T, iters, HIDDEN)


@pytest.mark.parametrize("iters", HIDDEN_COUNTS)
@pytest.mark.parametrize("T", T_COUNTS)
@pytest.mark.parametrize("hidden", HIDDEN_SIZES[1:], ids=lambda h: f"H{h}")
def test_counts_match_fp64_at_every_hidden_size(
    hidden: int, T: int, iters: int
) -> None:
    """The counts 1, 7, 20 and 64 at hidden 5120 and 7168 against the deterministic backward and the FP64 reference
    of the size (H 4096 is ``test_counts_match_fp64``)."""
    _check_counts(T, iters, _require_hidden(hidden))


def _check_counts(T: int, iters: int, hidden: int) -> None:
    out = _check(T, iters, hidden=hidden)
    # The count reaches the kernels: where the FP64 dfn of this count and of 20 lie apart at the test's tolerance,
    # every launch's dfn is nearer this count's (test_cake_mhc_det_train_iters.py's check of the record's C, on the
    # output of the non-deterministic route). A chain at its fixed point leaves the two FP64 gradients closer than
    # that and nothing to tell apart; the values alone are checked there.
    ref_n = out["ref"]["dfn"]
    ref_20 = _oracle({**out["inputs"], "sinkhorn_iters": SINKHORN_ITERS})["dfn"]
    if _rel_l2(ref_n, ref_20) > 4 * FP32_REL_L2_MAX:
        for repeat, o in enumerate(out["outs"]):
            assert _rel_l2(o["dfn"], ref_n) < _rel_l2(o["dfn"], ref_20), (
                f"dfn at sinkhorn_iters={iters} H={hidden} launch {repeat} is nearer the count of 20's"
            )


@pytest.mark.parametrize("iters", (1, 5, 64))
@pytest.mark.parametrize("T", (7, 257))
def test_parameter_gradients_within_2x_comparator_over_draws(
    T: int, iters: int
) -> None:
    """The comparator-relative clause of record over 25 draws at the count, on the worst of 5 launches of every
    draw: rel-L2 <= 1e-5 on every draw (dscale / dbase per element <= 1e-4), and median and maximum of the rel-L2
    <= 2x the IEEE-FP32 comparator's at the same count."""
    rows: dict[str, list[tuple[float, float]]] = {}
    for seed in range(SEED, SEED + PARAM_CLAUSE_DRAWS):
        i = _make_inputs(T, seed=seed)
        i["sinkhorn_iters"] = iters
        outs = _nd_repeats(i, _forward(i))
        ref, comp = _oracle(i), _oracle(i, dtype=F32)
        for k in _PARAMETER:
            error = max(_rel_l2(o[k], ref[k]) for o in outs)
            assert error <= FP32_REL_L2_MAX, (
                f"{k} T={T} sinkhorn_iters={iters} seed={seed}: rel-L2 {error:.3e}"
            )
            if k in ("dscale", "dbase"):
                worst = max(_max_rel_elem(o[k], ref[k]) for o in outs)
                assert worst <= FP32_ELEM_REL_MAX, (k, T, iters, seed, worst)
            rows.setdefault(k, []).append((error, _rel_l2(comp[k], ref[k])))
    for k, pairs in rows.items():
        kernel, comparator = [p[0] for p in pairs], [p[1] for p in pairs]
        print(
            f"{k} T={T} sinkhorn_iters={iters} deterministic=False: worst-launch rel-L2 median "
            f"{statistics.median(kernel):.3e} max {max(kernel):.3e}; comparator median "
            f"{statistics.median(comparator):.3e} max {max(comparator):.3e}",
            flush=True,
        )
        assert statistics.median(kernel) <= FP32_COMPARATOR_FACTOR * statistics.median(
            comparator
        ), (k, T, iters)
        assert max(kernel) <= FP32_COMPARATOR_FACTOR * max(comparator), (k, T, iters)


@pytest.mark.parametrize("iters", COUNTS)
@pytest.mark.parametrize("T", (129, 2049))
def test_paths_agree(T: int, iters: int) -> None:
    """At each count: the package's forward prepare, statistics replay, residual mixing and backward prepare are
    bitwise the deterministic package's; the checkpoint replay is bitwise the forward record; the direct entry
    point, the autograd functions with the statistics saved (the backward path) and replayed (the checkpoint path)
    and the deterministic backward on the same branch give ``dX`` / ``dscale`` / ``dbase`` bitwise and ``dfn`` within
    the bound of one another; the autograd ``U`` / ``Y`` / ``dU`` are bitwise the direct entry points'."""
    _check_paths(T, iters, HIDDEN)


@pytest.mark.parametrize("iters", HIDDEN_COUNTS)
@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("hidden", HIDDEN_SIZES[1:], ids=lambda h: f"H{h}")
def test_paths_agree_at_every_hidden_size(hidden: int, T: int, iters: int) -> None:
    """``test_paths_agree`` at hidden 5120 and 7168 and the counts 1, 7, 20 and 64."""
    _check_paths(T, iters, _require_hidden(hidden))


def _check_paths(T: int, iters: int, hidden: int) -> None:
    i = _make_inputs(T, seed=SEED + 41, hidden=hidden)
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
    U_det, stats_det = det.cake_mhc_det_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], sinkhorn_iters=iters
    )
    dB_det, gstats_det = det.cake_mhc_det_train_backward_prepare(
        i["dY"], B, i["X"], stats_det
    )
    for name, got, want in (
        ("U", U, U_det),
        ("stats", stats, stats_det),
        ("Y", Y, det.cake_mhc_det_train_forward_finish(i["X"], B, stats_det)),
        ("dB", dB, dB_det),
        ("grad_stats", gstats, gstats_det),
    ):
        assert _same_bits(got, want), (
            f"{name} T={T} sinkhorn_iters={iters} H={hidden}: differs from the deterministic package"
        )
    operands = (i["X"], i["dY"], dB, B, i["fn"], i["scale"], i["base"], stats, gstats)
    direct = backward_finish(*operands, sinkhorn_iters=iters, deterministic=False)
    direct_det = det_backward_finish(*operands, sinkhorn_iters=iters)
    bound = _dfn_bound({**i, "B": B, "dU": dB, "sinkhorn_iters": iters})
    label = f"T={T} sinkhorn_iters={iters} H={hidden}"
    _assert_exact(direct, direct_det, f"direct {label} vs deterministic")
    _assert_dfn_within(
        direct[1], direct_det[1], bound, f"dfn direct {label} vs deterministic"
    )
    paths = {}
    for save_stats in (True, False):
        X = i["X"].clone().requires_grad_(True)
        fn = i["fn"].clone().requires_grad_(True)
        scale = i["scale"].clone().requires_grad_(True)
        base = i["base"].clone().requires_grad_(True)
        Ua, handle = pkg.cake_mhc_train_prepare(
            X, fn, scale, base, sinkhorn_iters=iters, save_stats=save_stats,
            deterministic=False,
        )  # fmt: skip
        Ua.retain_grad()
        Ba = (Ua.float() + i["B"].float()).to(BF16)
        Ya = pkg.cake_mhc_train_finish(X, Ba, handle, deterministic=False)
        (Ya.float() * i["dY"].float()).sum().backward()
        what = f"{label} save_stats={save_stats}"
        assert torch.equal(Ua.detach(), U) and torch.equal(Ya.detach(), Y), what
        assert torch.equal(Ua.grad, dB), what
        got = (X.grad, fn.grad, scale.grad, base.grad)
        for want, name in ((direct, "direct"), (direct_det, "deterministic")):
            _assert_exact(got, want, f"autograd {what} vs {name}")
            _assert_dfn_within(got[1], want[1], bound, f"dfn autograd {what} vs {name}")
        paths[save_stats] = got
    _assert_exact(paths[False], paths[True], f"{label}: checkpoint vs backward path")
    _assert_dfn_within(
        paths[False][1],
        paths[True][1],
        bound,
        f"dfn {label}: checkpoint vs backward path",
    )


def test_changing_count_does_not_rebuild() -> None:
    """The count is a launch argument of the run-time-count programs: after one call at a non-default count, calls at
    other counts (and back at the default) load no module of either package, and each follows the FP64 reference at
    its count."""
    _check(64, SINKHORN_ITERS)
    _check(64, 3)
    loaded = (loader.load.cache_info().misses, det_loader.load.cache_info().misses)
    for iters in (1, 2, 7, 33, 64, SINKHORN_ITERS, 11):
        i = _make_inputs(255, seed=SEED + iters)
        i["sinkhorn_iters"] = iters
        res = _nd_backward(i, _forward(i))
        ref = _oracle(i)
        _assert_bf16(res["dX"], ref["dX32"], f"dX sinkhorn_iters={iters}")
        _assert_fp32(res["dfn"], ref["dfn"], f"dfn sinkhorn_iters={iters}")
    assert (
        loader.load.cache_info().misses,
        det_loader.load.cache_info().misses,
    ) == loaded, "a changed sinkhorn_iters loaded a module"


def test_count_validation() -> None:
    """The package checks the count like the deterministic package: a non-integer, non-positive or bool count is a
    ``ValueError``, one above ``SINKHORN_ITERS_MAX`` a ``NotImplementedError``, on ``backward_finish``,
    ``forward_prepare`` and ``cake_mhc_train_prepare``; the counts 1, 20.0 and ``SINKHORN_ITERS_MAX`` run and give the deterministic
    backward's ``dX`` / ``dscale`` / ``dbase`` bits at the same count."""
    assert det.SHIPPED_SINKHORN_ITERS == pkg.SHIPPED_SINKHORN_ITERS == (SINKHORN_ITERS,)
    assert det.SINKHORN_ITERS_MAX == loader.SINKHORN_ITERS_MAX == SINKHORN_ITERS_MAX
    i = _make_inputs(8, seed=SEED + 9)
    X, fn, scale, base = i["X"], i["fn"], i["scale"], i["base"]
    prep = _forward(i)
    for iters in (0, -1, 2.5, 20.5, True, float("inf"), float("nan"), "20"):
        with pytest.raises(ValueError):
            _nd_backward(i, prep, sinkhorn_iters=iters)
        with pytest.raises(ValueError):
            forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        with pytest.raises(ValueError):
            pkg.cake_mhc_train_prepare(
                X, fn, scale, base, sinkhorn_iters=iters, deterministic=False
            )
    for iters in (SINKHORN_ITERS_MAX + 1, 1000):
        with pytest.raises(NotImplementedError):
            _nd_backward(i, prep, sinkhorn_iters=iters)
        with pytest.raises(NotImplementedError):
            forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        with pytest.raises(NotImplementedError):
            pkg.cake_mhc_train_prepare(
                X, fn, scale, base, sinkhorn_iters=iters, deterministic=False
            )
    for iters in (1, 20.0, SINKHORN_ITERS_MAX):
        case = {**i, "sinkhorn_iters": iters}
        cprep = _forward(case)
        got = _snapshot(_nd_backward(case, cprep))
        want = _det_backward(case, cprep)
        _assert_exact(got, want, f"sinkhorn_iters={iters!r} vs deterministic")
        _assert_dfn_within(
            got["dfn"], want["dfn"], _dfn_bound({**i, "sinkhorn_iters": int(iters)}),
            f"dfn sinkhorn_iters={iters!r} vs deterministic",
        )  # fmt: skip


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize(
    ("iters", "rms_eps", "hc_eps"),
    (
        (7, RMS_EPS, HC_EPS),
        (64, RMS_EPS, HC_EPS),
        (SINKHORN_ITERS, 2e-5, 2e-6),
        (7, 1e-6, 1e-5),
    ),
)
def test_run_time_count_and_epsilons(
    T: int, iters: int, rms_eps: float, hc_eps: float
) -> None:
    """A run-time Sinkhorn count (the deterministic token-owner pass's run-time-count record) and non-default
    epsilons with ``deterministic=False``: ``dX``, ``dscale``, ``dbase`` bitwise the deterministic backward's at the
    same constants on every launch, ``dfn`` within the bound; against the FP64 reference at the same constants the
    BF16 ``dX`` gate and rel-L2 <= 1e-5 (``dscale`` / ``dbase`` per element <= 1e-4) on the worst launch."""
    i = _make_inputs(T, seed=SEED + 41)
    i.update(sinkhorn_iters=iters, rms_eps=rms_eps, hc_eps=hc_eps)
    prep = _forward(i)
    want = _snapshot(_det_backward(i, prep))
    bound = _dfn_bound(i)
    outs = _nd_repeats(i, prep)
    label = f"T={T} sinkhorn_iters={iters} rms_eps={rms_eps:g} hc_eps={hc_eps:g}"
    for repeat, out in enumerate(outs):
        _assert_exact(out, want, f"{label} launch {repeat}")
        _assert_dfn_within(
            out["dfn"], want["dfn"], bound, f"dfn {label} launch {repeat}"
        )
    ref = _oracle(i)
    _assert_bf16(outs[0]["dX"], ref["dX32"], f"dX {label}")
    for k in _PARAMETER:
        error = max(_rel_l2(o[k], ref[k]) for o in outs)
        assert error <= FP32_REL_L2_MAX, (
            f"{k} {label}: rel-L2 {error:.3e} > {FP32_REL_L2_MAX:.0e}"
        )
        if k != "dfn":
            worst = max(_max_rel_elem(o[k], ref[k]) for o in outs)
            assert worst <= FP32_ELEM_REL_MAX, (
                f"{k} {label}: per-element relative error {worst:.3e} > {FP32_ELEM_REL_MAX:.0e}"
            )
