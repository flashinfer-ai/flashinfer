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

Tests for the Cake four-stream mHC training forward / backward (hidden 4096,
4 streams; SM100 / SM103 / SM107).

Every output of the package is compared with a pure-torch FP64 reference of
the function (autograd through the FP64 forward, BF16 rounding only at the
``U`` / ``Y`` / ``dB`` / ``dX`` boundaries) and, for the FP32 outputs, with
the IEEE-FP32 evaluation of the same reference (TF32 off):

* BF16 ``U``, ``Y``, ``dB``, ``dX``: ``|out - ref| <= 1e-2 + 1e-2 |ref|`` on
  every element (unit-scale inputs), no mismatch allowed;
* the exposed per-token FP32 statistics ``r``, ``z``, ``pre``, ``post``, ``C``:
  relative L2 error <= 1e-5 on every draw; the "<= 2x the IEEE-FP32
  comparator" clause in the same >= 25-draw distribution form as the parameter
  gradients below (the single-draw ratio printed as the literal record);
* FP32 ``dfn``, ``dscale``, ``dbase``: on every draw relative L2 error <= 1e-5
  and (``dscale`` / ``dbase``) per element <= 1e-4 relative (absolute floor
  1e-4 x the vector's RMS); the "<= 2x the IEEE-FP32 comparator" clause is a
  distribution over >= 25 draws of a case (the contract owner's decision of
  2026-10-19: the 3-element ``dscale`` sits at the FP32 resolution floor, where
  the comparator's own error is a fraction of an ulp and one draw's ratio is
  sampling luck): the median and the maximum of the kernel's rel-L2 over the
  draws are <= 2x the comparator's median and maximum.  Each fixed-draw test
  prints its single-draw ratio as the literal record;
* token counts 0, 1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 255, 1000, 2049,
  16263, 20001 and the 13 performance rows (1024 ... 65536); the distribution
  clause at 1, 2, 3, 7, 63, 129, 255, 1000, 2049 (the kernel test suite
  carries it at every correctness token count);
* saturated sigmoid tails (``scale z + base >= 17`` on every token): on every
  one of >= 25 draws the tail gradients of ``dbase`` / ``dscale`` / ``dfn`` are
  non-zero and within 1e-3 relative of FP64, and the parameter gradients are
  judged under that tail clause (rel-L2 and per element <= 1e-3; the 1e-5 /
  1e-4 / 2x figures are printed as the literal record: the comparator's FP32
  sigmoid rounds to 1 and loses the tail);
* single-cast cancellation (inputs whose three ``dX`` contributions cancel on
  one stream): on the elements with ``|dX| <= 2^-6 max |contribution|`` the
  kernel's BF16 ``dX`` is measured in BF16 ulps of the reference,
  ``|dX - dX_fp64| / ulp_bf16(dX_fp64)`` (the contract owner's decision of
  2026-10-20: the error in units of the reference's BF16 spacing, finite at
  zero, where a bit distance counts every representable value in between).
  The gate is the kernel test suite's refined band gate aggregated over >= 64
  draws of the case (the contract owner's decision of 2026-10-19): on the
  resolvable band (``|dX| >= 2^-14 max |contribution|``) the error is <= 2 ulp
  on every draw, or the pooled ``n(> 2 ulp)`` and the maximum over the draws
  are within 2x the IEEE-FP32 single-cast comparator's.  Every draw prints its band and
  spec-literal mask figures (maximum ulp, ``n(> 2 ulp)``, strict band verdict,
  the single-draw refined verdict) as the literal record; the IEEE-FP32
  comparator itself exceeds 2 ulp on that band;
* checkpoint replay (``forward_stats``, ``cake_mhc_train_prepare(save_stats=False)``)
  gives bitwise the same statistics and gradients as the saved path;
* the autograd wrappers keep nothing of a connection alive once its backward
  has run (plain graphs, non-reentrant and reentrant checkpoints, a backward
  without the prepare node), and repeated backward passes over a retained graph
  and interleaved connections give bitwise the gradients of a fresh graph and of
  the connections run one after the other;
* four consecutive ``accumulate=True`` backward passes equal the FP64 sum of
  the four gradients plus the carried-in values within the FP32 gates, in the
  same >= 25-draw form;
* three launches on the same inputs (outputs and workspaces re-poisoned) give
  bitwise identical outputs; no non-deterministic reduction exists;
* ``fn x 1e3`` (``|z| ~ 1e3``; a regime of this suite beyond the contract's
  distributions): the BF16 gates hold unchanged and the FP32 outputs stay
  within 2x the IEEE-FP32 comparator in the same distribution form (rel-L2
  median and maximum; per element ``max(1e-4, 2x the comparator's maximum)``),
  the absolute bars being below the FP32 representation of the pre-activations
  there; literal verdicts printed.

The largest cases hold the FP64 reference in row blocks of 8192 tokens and
the FP32 comparator in one block (about 40 GB of device memory at T = 65536).
This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_mhc_train`` facade and the ``flashinfer`` top-level
exposure are covered by ``test_cake_mhc_train_api.py``; the host rules by
``test_cake_mhc_train_rules.py``.
"""

from __future__ import annotations

import contextlib
import math
import re
import statistics
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_mhc_train")
cake_build = cake_package("cake_mhc_train", "cake_build")
loader = cake_package("cake_mhc_train", "cake_jit")
forward_prepare = pkg.cake_mhc_train_forward_prepare
forward_stats = pkg.cake_mhc_train_forward_stats
forward_finish = pkg.cake_mhc_train_forward_finish
backward_prepare = pkg.cake_mhc_train_backward_prepare
backward_finish = pkg.cake_mhc_train_backward_finish
forward_workspace = pkg.cake_mhc_train_forward_workspace
backward_workspace = pkg.cake_mhc_train_backward_workspace
forward_workspace_bytes = pkg.cake_mhc_train_forward_workspace_bytes
backward_workspace_bytes = pkg.cake_mhc_train_backward_workspace_bytes
is_supported = pkg.is_cake_mhc_train_supported

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, N, MAPPING_ROWS = 4, 4096, 16384, 24
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
T_CORRECTNESS = (
    0,
    1,
    2,
    3,
    7,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    1000,
    2049,
    16263,
    20001,
)
T_PERFORMANCE = (
    16263,
    14928,
    15288,
    15566,
    15590,
    16114,
    16132,
    16164,
    1024,
    4096,
    8192,
    32768,
    65536,
)
T_CHECKED = tuple(sorted((set(T_CORRECTNESS) | set(T_PERFORMANCE)) - {0}))
BF16_ATOL = BF16_RTOL = 1e-2
FP32_REL_L2_MAX = 1e-5
FP32_COMPARATOR_FACTOR = 2.0
FP32_ELEM_REL_MAX = 1e-4
CANCEL_FACTOR = 2.0**-6
CANCEL_ULP_MAX = 2
CANCEL_FLOOR = 2.0**-14
TAIL_REL_MAX = 1e-3
BLOCK_TOKENS = 8192
SEED = 2032
# The comparator-relative clauses are seed distributions (contract owner decisions of 2026-10-19): the parameter
# gradients over PARAM_CLAUSE_DRAWS draws, the cancellation band over CANCEL_CLAUSE_DRAWS draws (seeds SEED, SEED + 1, ...).
PARAM_CLAUSE_DRAWS = 25
CANCEL_CLAUSE_DRAWS = 64
T_DISTRIBUTION = (1, 2, 3, 7, 63, 129, 255, 1000, 2049)
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before the first test, through the spec
    # objects the package's loader itself builds on first use; a prebuilt or cached library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


# ----------------------------------------------------------------------------------------------------------------
# Reference arithmetic (FP64 oracle; the same code in FP32 after _set_ieee_fp32 is the comparator)


def _set_ieee_fp32() -> None:
    """TF32 off everywhere so that FP32 means IEEE FP32 FMA."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for path in (
        "fp32_precision",
        "cuda.matmul.fp32_precision",
        "cudnn.conv.fp32_precision",
    ):
        obj = torch.backends
        parts = path.split(".")
        for name in parts[:-1]:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, parts[-1]):
            with contextlib.suppress(Exception):  # a read-only control on this build
                setattr(obj, parts[-1], "ieee")


def _round_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """One round-to-nearest-even step to BF16, also from FP64 (``Tensor.to`` rounds FP64 twice, through FP32)."""
    if x.dtype != F64:
        return x.to(BF16)
    x32 = x.float()
    b = x32.to(BF16)
    bits = x32.contiguous().view(torch.int32)
    tie = (bits & 0xFFFF) == 0x8000
    exact = x32.double() == x
    fix = tie & ~exact
    up = x.abs() > x32.abs().double()
    hi = (bits + 0x8000) & -65536
    lo = bits & -65536
    fixed = torch.where(up, hi, lo).view(F32).to(BF16)
    return torch.where(fix, fixed, b)


def _bf16_ulp_spacing(x: torch.Tensor) -> torch.Tensor:
    """The BF16 spacing (one ulp) at ``|x|``, FP64 in and out: ``2^(floor(log2 |x|) - 7)`` for normal magnitudes and
    the subnormal spacing ``2^-133`` below ``2^-126`` (and at zero)."""
    e = torch.floor(torch.log2(x.to(F64).abs().clamp_min(2.0**-126)))
    return torch.exp2(e - 7)


def _bf16_ulp_error(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """The error of a BF16 output in units of the FP64 reference's BF16 spacing: ``|out - ref| / ulp_bf16(ref)`` (FP64).
    A correctly rounded output is within 0.5; the measure is continuous across binades and finite where the reference
    is tiny or zero, unlike a bit distance between two BF16 values."""
    assert out.dtype == BF16, out.dtype
    ref64 = ref.to(F64)
    return (out.to(F64) - ref64).abs() / _bf16_ulp_spacing(ref64)


def _sinkhorn(logits: torch.Tensor, hc_eps: float = HC_EPS) -> torch.Tensor:
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(SINKHORN_ITERS - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _coefficients(z, scale, base, hc_eps: float = HC_EPS):
    T = z.shape[0]
    pre = torch.sigmoid(z[:, 0:4] * scale[0] + base[0:4]) + hc_eps
    post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = _sinkhorn((z[:, 8:] * scale[2] + base[8:]).reshape(T, STREAMS, STREAMS), hc_eps)
    return pre, post, C


def _block(
    X,
    fn,
    scale,
    base,
    B,
    dY,
    dU,
    *,
    dtype,
    want_contrib: bool,
    rms_eps: float = RMS_EPS,
    hc_eps: float = HC_EPS,
) -> dict:
    """One row block of the reference: the forward in ``dtype``, the backward by autograd of the FP arithmetic."""
    T = X.shape[0]
    with torch.enable_grad():
        Xd = X.detach().to(dtype).requires_grad_(True)
        fnd = fn.detach().to(dtype).requires_grad_(True)
        scaled = scale.detach().to(dtype).requires_grad_(True)
        based = base.detach().to(dtype).requires_grad_(True)
        Bd = B.detach().to(dtype).requires_grad_(True)
        u = Xd.reshape(T, N)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)
        z = (u * r) @ fnd.T
        pre, post, C = _coefficients(z, scaled, based, hc_eps)
        U_pre = (pre.unsqueeze(-1) * Xd).sum(dim=1)
        Y_pre = post.unsqueeze(-1) * Bd.unsqueeze(1) + torch.einsum(
            "tij,tid->tjd", C, Xd
        )
        dUd = dU.detach().to(dtype)
        dYd = dY.detach().to(dtype)
        dX32, dfn, dscale, dbase, dB32 = torch.autograd.grad(
            [U_pre, Y_pre], [Xd, fnd, scaled, based, Bd], grad_outputs=[dUd, dYd]
        )
    with torch.no_grad():
        Xd, Bd = Xd.detach(), Bd.detach()
        out = {
            "r": r.detach().reshape(T),
            "z": z.detach(),
            "pre": pre.detach(),
            "post": post.detach(),
            "C": C.detach(),
            "U32": U_pre.detach(),
            "Y32": Y_pre.detach(),
            "dB32": dB32,
            "dpost": torch.einsum("tjd,td->tj", dYd, Bd),
            "dC": torch.einsum("tid,tjd->tij", Xd, dYd),
            "dX32": dX32,
            "dfn": dfn,
            "dscale": dscale,
            "dbase": dbase,
        }
        if want_contrib:
            out["collapse"] = pre.detach().unsqueeze(-1) * dUd.unsqueeze(1)
            out["mix"] = torch.einsum("tij,tjd->tid", C.detach(), dYd)
    return out


_PER_TOKEN = (
    "r",
    "z",
    "pre",
    "post",
    "C",
    "U32",
    "Y32",
    "dB32",
    "dpost",
    "dC",
    "dX32",
    "collapse",
    "mix",
)
_PARAMETER = ("dfn", "dscale", "dbase")
_STATS = ("r", "z", "pre", "post", "C")


def _oracle(
    i: dict,
    *,
    dtype=F64,
    block_tokens: int | None = BLOCK_TOKENS,
    want_contrib: bool = False,
) -> dict:
    """The reference of one input set: FP64 in row blocks (the oracle) or IEEE FP32 in one block (the comparator).
    The epsilons of the case (``i["rms_eps"]`` / ``i["hc_eps"]``, default the model constants) enter the arithmetic."""
    T = i["X"].shape[0]
    if dtype == F32:
        _set_ieee_fp32()
    bt = T if not block_tokens else max(1, min(block_tokens, T))
    eps = {"rms_eps": i.get("rms_eps", RMS_EPS), "hc_eps": i.get("hc_eps", HC_EPS)}
    blocks = [
        _block(
            i["X"][lo : lo + bt], i["fn"], i["scale"], i["base"], i["B"][lo : lo + bt], i["dY"][lo : lo + bt],
            i["dU"][lo : lo + bt], dtype=dtype, want_contrib=want_contrib, **eps,
        )
        for lo in range(0, T, bt)
    ]  # fmt: skip
    if len(blocks) == 1:
        return blocks[0]
    out = {
        k: torch.cat([b[k] for b in blocks], dim=0)
        for k in _PER_TOKEN
        if k in blocks[0]
    }
    for k in _PARAMETER:
        acc = blocks[0][k].clone()
        for b in blocks[1:]:
            acc += b[k]
        out[k] = acc
    return out


def _make_inputs(
    T: int, *, mode: str = "unit", seed: int = SEED, fn_scale: float = 1.0, zero_rows=()
) -> dict:
    """The inputs of one case: ``X, B, dY, dU ~ N(0, 1)`` BF16; ``fn ~ 1e-2 N(0, 1)`` (``1e-4`` in the ``tail`` and
    ``cancel`` regimes); ``scale ~ 1 + 0.1 N``; ``base ~ 0.1 N`` (``base[:8] = 20`` in ``tail``, so every pre / post
    logit is >= 17); in ``cancel`` ``dU := BF16(-(sum_j C_0j dY_j) / pre_0)`` from the FP64 statistics, so the
    collapse and residual-mixing contributions of stream 0 to ``dX`` cancel."""
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    fs = {"unit": 1e-2, "tail": 1e-4, "cancel": 1e-4}[mode]
    X = randn(T, STREAMS, HIDDEN, dtype=BF16)
    fn = randn(MAPPING_ROWS, N) * fs
    scale = 1.0 + 0.1 * randn(3)
    base = 0.1 * randn(MAPPING_ROWS)
    if mode == "tail":
        base[:8] = 20.0
    B = randn(T, HIDDEN, dtype=BF16)
    dY = randn(T, STREAMS, HIDDEN, dtype=BF16)
    dU = randn(T, HIDDEN, dtype=BF16)
    if fn_scale != 1.0:
        fn = fn * float(fn_scale)
    if zero_rows:
        X[list(zero_rows)] = 0
    fn = fn.contiguous()
    if mode == "cancel" and T > 0:
        with torch.no_grad():
            dU_new = torch.empty_like(dU)
            for lo in range(0, T, BLOCK_TOKENS):
                hi = min(T, lo + BLOCK_TOKENS)
                u = X[lo:hi].double().reshape(hi - lo, N)
                r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + RMS_EPS)
                z = (u * r) @ fn.double().T
                pre, _post, C = _coefficients(z, scale.double(), base.double())
                mix_0 = torch.einsum("tj,tjd->td", C[:, 0, :], dY[lo:hi].double())
                dU_new[lo:hi] = _round_to_bf16(-mix_0 / pre[:, 0:1])
            dU = dU_new
    return {
        "X": X,
        "fn": fn,
        "scale": scale,
        "base": base,
        "B": B,
        "dY": dY,
        "dU": dU,
        "mode": mode,
        "T": T,
    }


def _run(i: dict, *, fwd_ws=None, bwd_ws=None) -> dict:
    """The package end to end: forward prepare / finish, backward prepare / finish on the inputs of a case (with the
    case's epsilons when it carries non-default ones)."""
    eps = {k: i[k] for k in ("rms_eps", "hc_eps") if k in i}
    U, stats = forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], workspace=fwd_ws, **eps
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
        workspace=bwd_ws,
        **eps,
    )
    return {
        "U": U,
        "stats": stats,
        "Y": Y,
        "dB": dB,
        "gstats": gstats,
        "dX": dX,
        "dfn": dfn,
        "dscale": dscale,
        "dbase": dbase,
    }


def _stats_columns(stats: torch.Tensor) -> dict:
    T = stats.shape[0]
    return {
        "r": stats[:, 0],
        "z": stats[:, 1:25],
        "pre": stats[:, 25:29],
        "post": stats[:, 29:33],
        "C": stats[:, 33:49].reshape(T, STREAMS, STREAMS),
    }


# ----------------------------------------------------------------------------------------------------------------
# Gates


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
    assert out.dtype == BF16 and out.is_contiguous(), label
    assert tuple(out.shape) == tuple(ref.shape), (
        label,
        tuple(out.shape),
        tuple(ref.shape),
    )
    a, b = out.double(), ref.double()
    assert torch.isfinite(a).all(), f"{label}: non-finite values"
    if a.numel() == 0:
        return
    err = (a - b).abs()
    bad = err > BF16_ATOL + BF16_RTOL * b.abs()
    n_bad = int(bad.sum())
    assert n_bad == 0, (
        f"{label}: {n_bad} of {a.numel()} elements outside atol = rtol = 1e-2 (max abs err {float(err.max()):.3e})"
    )


def _assert_fp32(
    out: torch.Tensor,
    ref: torch.Tensor,
    comp: torch.Tensor | None,
    label: str,
    *,
    per_element: bool = False,
    regime: str = "unit",
    parameter: bool = False,
) -> dict:
    """The contract's FP32 gates on one draw, and the literal record of the comparator-relative clause.

    Statistics (``parameter=False``): rel-L2 <= 1e-5 and <= 2x the IEEE-FP32 comparator's rel-L2 on the same inputs,
    asserted on this draw. ``regime="large_logits"`` marks the ``fn x 1e3`` regime of ``test_large_logits`` (|z| ~ 1e3;
    an addition of this suite, outside the contract's input distributions): there the pre-activations ``scale z + base``
    sit beyond what FP32 arithmetic represents to the resolution of the absolute bars, so the comparator-relative clause
    alone governs and the literal verdicts of the absolute bars are printed beside it.

    Distribution-clause outputs (``parameter=True``: ``dfn`` / ``dscale`` / ``dbase`` and the per-token statistics
    ``r`` / ``z`` / ``pre`` / ``post`` / ``C``): the absolute bars are asserted on this draw (``unit``: rel-L2 <= 1e-5, per element <= 1e-4; ``tail``: the tail clause, rel-L2 and per element <= 1e-3, the
    1e-5 / 1e-4 figures printed; ``large_logits``: none); the "<= 2x the comparator" clause is a distribution over
    >= 25 draws (``_assert_parameter_distribution``), so on one draw it is printed as the literal record and never
    asserted. Returns the draw's figures."""
    assert out.dtype == F32 and out.is_contiguous(), label
    assert tuple(out.shape) == tuple(ref.shape), (
        label,
        tuple(out.shape),
        tuple(ref.shape),
    )
    assert torch.isfinite(out).all(), f"{label}: non-finite values"
    if out.numel() == 0:
        return {}
    error = _rel_l2(out, ref)
    comparator = _rel_l2(comp, ref) if comp is not None else None
    ratio = None if comparator is None else _ratio(error, comparator)
    worst = _max_rel_elem(out, ref) if per_element else None
    comparator_worst = (
        _max_rel_elem(comp, ref) if per_element and comp is not None else None
    )
    row = {
        "kernel": error, "comparator": comparator, "ratio": ratio,
        "kernel_elem": worst, "comparator_elem": comparator_worst,
    }  # fmt: skip
    met = {True: "met", False: "NOT met"}
    if parameter:
        clause = "n/a" if ratio is None else met[ratio <= FP32_COMPARATOR_FACTOR]
        print(
            f"{label}: rel-L2 {error:.3e} (literal 1e-5 bar {met[error <= FP32_REL_L2_MAX]}), IEEE-FP32 comparator "
            f"{'n/a' if comparator is None else f'{comparator:.3e}'}; single-draw 2x clause {clause} (literal record: the "
            f"clause of record is the >= {PARAM_CLAUSE_DRAWS}-draw distribution)"
            + (
                f"; per-element {worst:.3e} (literal 1e-4 bar {met[worst <= FP32_ELEM_REL_MAX]})"
                if per_element
                else ""
            ),
            flush=True,
        )
        if regime == "tail":
            assert error <= TAIL_REL_MAX, (
                f"{label}: tail-clause rel-L2 {error:.3e} > {TAIL_REL_MAX:.0e}"
            )
            if per_element:
                assert worst <= TAIL_REL_MAX, (
                    f"{label}: tail-clause per-element relative error {worst:.3e} > {TAIL_REL_MAX:.0e}"
                )
        elif regime == "unit":
            assert error <= FP32_REL_L2_MAX, (
                f"{label}: rel-L2 {error:.3e} > {FP32_REL_L2_MAX:.0e}"
            )
            if per_element:
                assert worst <= FP32_ELEM_REL_MAX, (
                    f"{label}: per-element relative error {worst:.3e} > {FP32_ELEM_REL_MAX:.0e}"
                )
        return row
    floor = regime == "large_logits" and comparator is not None
    if floor:
        print(
            f"{label}: rel-L2 {error:.3e} (literal 1e-5 bar {met[error <= FP32_REL_L2_MAX]}), "
            f"IEEE-FP32 comparator {comparator:.3e}; comparator-relative regime",
            flush=True,
        )
    else:
        assert error <= FP32_REL_L2_MAX, (
            f"{label}: rel-L2 {error:.3e} > {FP32_REL_L2_MAX:.0e}"
        )
    if comparator is not None:
        assert ratio <= FP32_COMPARATOR_FACTOR, (
            f"{label}: rel-L2 {error:.3e} is {ratio:.2f}x the IEEE-FP32 comparator's {comparator:.3e} (> 2x)"
        )
    if per_element:
        bar = FP32_ELEM_REL_MAX
        if floor:
            bar = max(bar, FP32_COMPARATOR_FACTOR * comparator_worst)
        assert worst <= bar, (
            f"{label}: per-element relative error {worst:.3e} > {bar:.3e}"
        )
    return row


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else (0.0 if a == 0 else math.inf)


def _draw_row(out: torch.Tensor, ref: torch.Tensor, comp: torch.Tensor) -> dict:
    """One draw's figures of an FP32 output: the kernel's and the comparator's rel-L2 and per-element errors vs FP64."""
    return {
        "kernel": _rel_l2(out, ref), "comparator": _rel_l2(comp, ref),
        "kernel_elem": _max_rel_elem(out, ref), "comparator_elem": _max_rel_elem(comp, ref),
    }  # fmt: skip


def _parameter_rows(
    T: int, *, mode: str = "unit", fn_scale: float = 1.0, seeds
) -> dict:
    """The per-draw figures of the parameter gradients and of the per-token statistics over ``seeds`` draws of a
    case, through the package end to end; in the ``tail`` regime every draw also holds the tail-gradient clause."""
    rows = {k: [] for k in _PARAMETER + _STATS}
    for seed in seeds:
        i = _make_inputs(T, mode=mode, seed=seed, fn_scale=fn_scale)
        res = _run(i)
        ref = _oracle(i)
        comp = _oracle(i, dtype=F32, block_tokens=None)
        for k in _PARAMETER:
            rows[k].append(_draw_row(res[k], ref[k], comp[k]))
        for k, out in _stats_columns(res["stats"]).items():
            rows[k].append(_draw_row(out.contiguous(), ref[k], comp[k]))
        if mode == "tail":
            _assert_tail_gradients(res, ref, f"T={T} mode=tail seed={seed}")
        del i, res, ref, comp
    return rows


def _assert_parameter_distribution(rows: dict, label: str, *, regime: str) -> None:
    """The clause of record of the parameter gradients over >= 25 draws (the contract owner's decision of 2026-10-19):
    ``unit`` (and the cancellation inputs): rel-L2 <= 1e-5 and (``dscale`` / ``dbase``) per element <= 1e-4 on every
    draw, and median(kernel rel-L2) <= 2x median(comparator) AND max(kernel) <= 2x max(comparator); ``tail``: the tail
    clause on every draw (rel-L2 and per element <= 1e-3), the 2x figures printed and not applied (the comparator loses
    the tail); ``large_logits``: the comparator-relative clause alone (rel-L2 medians and maxima; per element
    <= max(1e-4, 2x the comparator's maximum)). The per-token statistics (``_STATS`` rows, forward quantities) hold the
    ``unit`` form on the unit, cancellation and tail inputs and the ``large_logits`` form on the ``fn x 1e3`` inputs.
    The single-draw tallies are printed beside as the literal record."""
    for k, r in rows.items():
        n = len(r)
        assert n >= PARAM_CLAUSE_DRAWS, (label, k, n)
        reg = (
            regime
            if k in _PARAMETER
            else ("large_logits" if regime == "large_logits" else "unit")
        )
        ks, cs = [d["kernel"] for d in r], [d["comparator"] for d in r]
        es, ces = [d["kernel_elem"] for d in r], [d["comparator_elem"] for d in r]
        med_k, med_c, max_k, max_c = (
            statistics.median(ks),
            statistics.median(cs),
            max(ks),
            max(cs),
        )
        per_elem = k in ("dscale", "dbase")
        print(
            f"{k} {label}: {n} draws; kernel rel-L2 median {med_k:.3e} / max {max_k:.3e}, comparator {med_c:.3e} / {max_c:.3e}; "
            f"ratio of medians {_ratio(med_k, med_c):.2f}, of maxima {_ratio(max_k, max_c):.2f}; draws whose single-draw ratio "
            f"exceeds 2x: {sum(1 for a, b in zip(ks, cs, strict=False) if a > FP32_COMPARATOR_FACTOR * b)} (literal record); draws above "
            f"1e-5: kernel {sum(1 for a in ks if a > FP32_REL_L2_MAX)} / comparator {sum(1 for b in cs if b > FP32_REL_L2_MAX)}"
            + (
                f"; per-element max {max(es):.3e} (comparator {max(ces):.3e}), draws above 1e-4: "
                f"{sum(1 for e in es if e > FP32_ELEM_REL_MAX)}"
                if per_elem
                else ""
            ),
            flush=True,
        )
        if reg == "tail":
            assert max_k <= TAIL_REL_MAX, (
                f"{k} {label}: tail-clause rel-L2 max over the draws {max_k:.3e} > {TAIL_REL_MAX:.0e}"
            )
            if per_elem:
                assert max(es) <= TAIL_REL_MAX, (
                    f"{k} {label}: tail-clause per-element max {max(es):.3e} > {TAIL_REL_MAX:.0e}"
                )
            continue
        if reg == "unit":
            assert max_k <= FP32_REL_L2_MAX, (
                f"{k} {label}: rel-L2 max over the draws {max_k:.3e} > {FP32_REL_L2_MAX:.0e}"
            )
            if per_elem:
                assert max(es) <= FP32_ELEM_REL_MAX, (
                    f"{k} {label}: per-element max over the draws {max(es):.3e} > {FP32_ELEM_REL_MAX:.0e}"
                )
        assert (
            med_k <= FP32_COMPARATOR_FACTOR * med_c
            and max_k <= FP32_COMPARATOR_FACTOR * max_c
        ), (
            f"{k} {label}: kernel rel-L2 median {med_k:.3e} / max {max_k:.3e} vs the IEEE-FP32 comparator's "
            f"{med_c:.3e} / {max_c:.3e} (> 2x over {n} draws)"
        )
        if per_elem and reg == "large_logits":
            bar = max(FP32_ELEM_REL_MAX, FP32_COMPARATOR_FACTOR * max(ces))
            assert max(es) <= bar, (
                f"{k} {label}: per-element max over the draws {max(es):.3e} > {bar:.3e}"
            )


def _cancellation_band_report(
    dX: torch.Tensor, ref: dict, comp: dict, label: str
) -> dict:
    """One draw's cancellation-band figures (see the module docstring), printed as the literal record: the resolvable
    band and the spec-literal mask of the kernel and of the IEEE-FP32 comparator (ulp error ``_bf16_ulp_error`` against
    the FP64 ``dX``), the strict band verdict and the single-draw refined verdict. The verdict of record is the >= 64-draw
    aggregate of ``test_single_cast_cancellation``."""
    total = ref["dX32"]
    mapping = total - ref["collapse"] - ref["mix"]
    biggest = torch.maximum(
        torch.maximum(ref["collapse"].abs(), ref["mix"].abs()), mapping.abs()
    )
    literal = total.abs() <= CANCEL_FACTOR * biggest
    band = literal & (total.abs() >= CANCEL_FLOOR * biggest)
    ulp = _bf16_ulp_error(dX, total)
    culp = _bf16_ulp_error(comp["dX32"].to(BF16), total)

    def dist(u: torch.Tensor, mask: torch.Tensor) -> dict:
        sel = u[mask]
        n = int(sel.numel())
        return {
            "n": n,
            "max": round(float(sel.max()), 3) if n else 0.0,
            "gt2": int((sel > CANCEL_ULP_MAX).sum()) if n else 0,
        }

    b, lt, cb, cl = (
        dist(ulp, band),
        dist(ulp, literal),
        dist(culp, band),
        dist(culp, literal),
    )
    assert lt["n"] > 0, (
        f"{label}: the cancellation inputs produced no element inside the contract's mask"
    )
    strict = b["max"] <= CANCEL_ULP_MAX
    within = b["gt2"] <= 2 * cb["gt2"] and b["max"] <= 2 * cb["max"]
    verdict = {True: "PASS", False: "FAIL"}
    print(
        f"{label}: resolvable band {b['n']} elements, max ulp {b['max']}, > 2 ulp: {b['gt2']} (IEEE-FP32 comparator max "
        f"{cb['max']}, > 2 ulp: {cb['gt2']}); spec-literal mask {lt['n']} elements, max ulp {lt['max']}, > 2 ulp: "
        f"{lt['gt2']} (comparator max {cl['max']}, > 2 ulp: {cl['gt2']}); {int((literal & ~band).sum())} mask elements "
        f"below FP32 resolution; strict band {verdict[strict]}, spec-literal 2 ulp {verdict[lt['max'] <= CANCEL_ULP_MAX]}, "
        f"single-draw refined gate {verdict[strict or within]} (literal record)",
        flush=True,
    )
    return {
        "band": b,
        "literal": lt,
        "comparator_band": cb,
        "comparator_literal": cl,
        "strict": strict,
        "within": within,
    }


def _assert_tail_gradients(res: dict, ref: dict, label: str) -> None:
    """Saturated sigmoid tails: the pre / post lanes (rows 0..7) of dbase, the pre / post entries of dscale and the
    pre / post rows of dfn are non-zero and within 1e-3 relative (L2) of FP64."""
    lanes = {
        "dbase[0:8]": (res["dbase"][:8], ref["dbase"][:8]),
        "dscale[0:2]": (res["dscale"][:2], ref["dscale"][:2]),
        "dfn[0:8]": (res["dfn"][:8], ref["dfn"][:8]),
    }
    for name, (out, want) in lanes.items():
        assert torch.isfinite(out).all(), f"{label} {name}: non-finite"
        assert (want != 0).all(), (
            f"{label} {name}: the FP64 tail gradient is zero; the case does not test the tail"
        )
        assert (out != 0).all() if out.ndim == 1 else (out != 0).any(dim=1).all(), (
            f"{label} {name}: a tail gradient is zero"
        )
        error = _rel_l2(out, want)
        assert error <= TAIL_REL_MAX, (
            f"{label} {name}: tail rel-L2 {error:.3e} > {TAIL_REL_MAX:.0e}"
        )


def _check_case(
    T: int,
    *,
    mode: str = "unit",
    fn_scale: float = 1.0,
    zero_rows=(),
    seed: int = SEED,
    rms_eps: float = RMS_EPS,
    hc_eps: float = HC_EPS,
) -> dict:
    label = (
        f"T={T} mode={mode}"
        + (f" fn_scale={fn_scale:g}" if fn_scale != 1.0 else "")
        + (f" zero_rows={tuple(zero_rows)}" if zero_rows else "")
        + (f" rms_eps={rms_eps:g}" if rms_eps != RMS_EPS else "")
        + (f" hc_eps={hc_eps:g}" if hc_eps != HC_EPS else "")
    )
    # The FP32 regime of the draw: the fn x 1e3 inputs (comparator-relative clauses only), the saturated tails (the tail
    # clause) or the contract's unit / cancellation inputs (the absolute bars; see _assert_fp32).
    regime = (
        "large_logits" if fn_scale != 1.0 else ("tail" if mode == "tail" else "unit")
    )
    floor = regime == "large_logits"
    i = _make_inputs(T, mode=mode, seed=seed, fn_scale=fn_scale, zero_rows=zero_rows)
    i["rms_eps"], i["hc_eps"] = rms_eps, hc_eps
    res = _run(i)
    ref = _oracle(i, want_contrib=(mode == "cancel"))
    comp = _oracle(i, dtype=F32, block_tokens=None)
    assert tuple(res["stats"].shape) == (T, 64) and tuple(res["gstats"].shape) == (
        T,
        32,
    )
    _assert_bf16(res["U"], ref["U32"], f"U {label}")
    _assert_bf16(res["Y"], ref["Y32"], f"Y {label}")
    _assert_bf16(res["dB"], ref["dB32"], f"dB {label}")
    _assert_bf16(res["dX"], ref["dX32"], f"dX {label}")
    stats_rows = {
        name: _assert_fp32(
            out.contiguous(),
            ref[name],
            comp[name],
            f"{name} {label}",
            regime="large_logits" if floor else "unit",
            parameter=True,
        )
        for name, out in _stats_columns(res["stats"]).items()
    }
    # The gradient statistics record is an FP32 output of the API too (dot products over 4096 exact products).
    _assert_fp32(
        res["gstats"][:, 0:4].contiguous(), ref["dpost"], None, f"dpost {label}"
    )
    _assert_fp32(
        res["gstats"][:, 4:20].reshape(T, STREAMS, STREAMS).contiguous(),
        ref["dC"],
        None,
        f"dC {label}",
    )
    assert torch.equal(
        res["gstats"][:, 20:], torch.zeros_like(res["gstats"][:, 20:])
    ), f"gstats pad {label}"
    out = {
        "inputs": i,
        "result": res,
        "oracle": ref,
        "comparator": comp,
        "rows": stats_rows,
    }
    for k in _PARAMETER:
        out["rows"][k] = _assert_fp32(
            res[k],
            ref[k],
            comp[k],
            f"{k} {label}",
            per_element=(k != "dfn"),
            regime=regime,
            parameter=True,
        )
    if mode == "cancel":
        out["cancel"] = _cancellation_band_report(
            res["dX"], ref, comp, f"dX cancellation {label} seed={seed}"
        )
    if mode == "tail":
        _assert_tail_gradients(res, ref, label)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("T", T_CHECKED)
def test_forward_backward_match_fp64(T: int) -> None:
    """The 15 non-zero correctness token counts and the 13 performance rows (T = 0 is ``test_zero_rows``)."""
    _check_case(T)


def test_zero_row_of_x_takes_the_eps_path() -> None:
    _check_case(7, zero_rows=(3,))


@pytest.mark.parametrize("T", (257, 2049))
def test_saturated_sigmoid_tails(T: int) -> None:
    """The fixed draw through every gate of the case, then the tail clause on every one of the 25 draws: tail
    gradients non-zero and within 1e-3 of FP64, the parameter gradients within 1e-3 (rel-L2 and per element); the
    1e-5 / 1e-4 / 2x figures are printed as the literal record."""
    _check_case(T, mode="tail")
    rows = _parameter_rows(T, mode="tail", seeds=range(SEED, SEED + PARAM_CLAUSE_DRAWS))
    _assert_parameter_distribution(rows, f"T={T} mode=tail", regime="tail")


@pytest.mark.parametrize("T", (257, 2049))
def test_single_cast_cancellation(T: int) -> None:
    """The cancellation band aggregated over 64 draws of the case (ulp error in units of the FP64 reference's BF16
    spacing): the resolvable band holds 2 ulp on every draw, or the pooled n(> 2 ulp) and the maximum ulp error over the
    draws are within 2x the IEEE-FP32 comparator's. The first draw
    runs every gate of the case; every draw prints its band figures and single-draw refined verdict as the literal
    record. The parameter gradients and the statistics of the same draws hold the distribution clause too."""
    first = _check_case(T, mode="cancel")
    reports = [first["cancel"]]
    rows = {k: [first["rows"][k]] for k in _PARAMETER + _STATS}
    for seed in range(SEED + 1, SEED + CANCEL_CLAUSE_DRAWS):
        i = _make_inputs(T, mode="cancel", seed=seed)
        res = _run(i)
        ref = _oracle(i, want_contrib=True)
        comp = _oracle(i, dtype=F32, block_tokens=None)
        reports.append(
            _cancellation_band_report(
                res["dX"], ref, comp, f"dX cancellation T={T} mode=cancel seed={seed}"
            )
        )
        for k in _PARAMETER:
            rows[k].append(_draw_row(res[k], ref[k], comp[k]))
        for k, out in _stats_columns(res["stats"]).items():
            rows[k].append(_draw_row(out.contiguous(), ref[k], comp[k]))
        del i, res, ref, comp
    total_k = sum(r["band"]["gt2"] for r in reports)
    total_c = sum(r["comparator_band"]["gt2"] for r in reports)
    max_k = max(r["band"]["max"] for r in reports)
    max_c = max(r["comparator_band"]["max"] for r in reports)
    strict_every_draw = max_k <= CANCEL_ULP_MAX
    pooled = total_k <= 2 * total_c and max_k <= 2 * max_c
    single_draw_fails = sum(1 for r in reports if not (r["strict"] or r["within"]))
    print(
        f"dX cancellation T={T}: {len(reports)} draws; kernel total n(> 2 ulp) {total_k}, max ulp {max_k}; IEEE-FP32 "
        f"comparator {total_c} / {max_c}; draws failing the single-draw refined gate {single_draw_fails} (literal record); "
        f"strict band on every draw {strict_every_draw}; aggregate verdict {'PASS' if strict_every_draw or pooled else 'FAIL'}",
        flush=True,
    )
    assert strict_every_draw or pooled, (
        f"dX cancellation T={T}: over {len(reports)} draws the resolvable band has {total_k} elements > 2 ulp (max {max_k}) "
        f"against the IEEE-FP32 comparator's {total_c} (max {max_c}): beyond 2x"
    )
    _assert_parameter_distribution(rows, f"T={T} mode=cancel", regime="unit")


@pytest.mark.parametrize("T", T_DISTRIBUTION)
def test_parameter_gradients_within_2x_comparator_over_draws(T: int) -> None:
    """The clause of record of dfn / dscale / dbase over 25 draws of the unit case: rel-L2 <= 1e-5 (dscale / dbase per
    element <= 1e-4) on every draw; median and maximum of the rel-L2 <= 2x the IEEE-FP32 comparator's median and
    maximum. The per-token statistics r / z / pre / post / C of the same draws hold the same form (rel-L2 <= 1e-5 on
    every draw; median and maximum within 2x the comparator's)."""
    rows = _parameter_rows(T, seeds=range(SEED, SEED + PARAM_CLAUSE_DRAWS))
    _assert_parameter_distribution(rows, f"T={T} mode=unit", regime="unit")


@pytest.mark.parametrize("T", (257, 2049))
def test_large_logits(T: int) -> None:
    """``fn x 1e3`` (|z| ~ 1e3: saturated sigmoids and softmax rows), an input regime of this suite beyond the
    contract's distributions: the BF16 gates hold unchanged; the FP32 outputs are held to the comparator-relative
    clauses (within 2x the IEEE-FP32 evaluation of the same function, which carries the same representation floor of
    the pre-activations) in the distribution form over 25 draws, with the literal absolute verdicts printed."""
    _check_case(T, fn_scale=1e3)
    rows = _parameter_rows(
        T, fn_scale=1e3, seeds=range(SEED, SEED + PARAM_CLAUSE_DRAWS)
    )
    _assert_parameter_distribution(rows, f"T={T} fn_scale=1000", regime="large_logits")


def test_zero_rows() -> None:
    i = _make_inputs(0)
    res = _run(i)
    assert tuple(res["U"].shape) == (0, HIDDEN) and tuple(res["stats"].shape) == (0, 64)
    assert tuple(res["Y"].shape) == (0, STREAMS, HIDDEN) and tuple(res["dB"].shape) == (
        0,
        HIDDEN,
    )
    assert tuple(res["gstats"].shape) == (0, 32) and tuple(res["dX"].shape) == (
        0,
        STREAMS,
        HIDDEN,
    )
    assert torch.equal(res["dfn"], torch.zeros_like(res["dfn"]))
    assert torch.equal(res["dscale"], torch.zeros_like(res["dscale"]))
    assert torch.equal(res["dbase"], torch.zeros_like(res["dbase"]))
    assert forward_stats(i["X"], i["fn"], i["scale"], i["base"]).shape == (0, 64)
    # accumulate=True leaves the caller's gradients untouched.
    dfn = torch.full((MAPPING_ROWS, N), 3.0, dtype=F32, device=_device())
    dscale = torch.full((3,), 5.0, dtype=F32, device=_device())
    dbase = torch.full((MAPPING_ROWS,), 7.0, dtype=F32, device=_device())
    out = backward_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], res["stats"], res["gstats"],
        accumulate=True, dfn=dfn, dscale=dscale, dbase=dbase,
    )  # fmt: skip
    assert out[1] is dfn and out[2] is dscale and out[3] is dbase
    assert (
        float(dfn.min()) == 3.0
        and float(dscale.min()) == 5.0
        and float(dbase.min()) == 7.0
    )
    assert (
        forward_workspace_bytes(0, _device()) == 0
        and backward_workspace_bytes(0, _device()) == 0
    )


@pytest.mark.parametrize("T", (129, 2049))
def test_checkpoint_replay_is_bitwise(T: int) -> None:
    i = _make_inputs(T, seed=SEED + 11)
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"])
    replay = forward_stats(i["X"], i["fn"], i["scale"], i["base"])
    assert torch.equal(stats, replay), (
        "forward_stats is not bitwise the forward_prepare record"
    )
    grads = []
    for save_stats in (True, False):
        X = i["X"].clone().requires_grad_(True)
        fn = i["fn"].clone().requires_grad_(True)
        scale = i["scale"].clone().requires_grad_(True)
        base = i["base"].clone().requires_grad_(True)
        U, handle = pkg.cake_mhc_train_prepare(
            X, fn, scale, base, save_stats=save_stats
        )
        B = (U.float() + i["B"].float()).to(BF16)
        Y = pkg.cake_mhc_train_finish(X, B, handle)
        (Y.float() * i["dY"].float()).sum().backward()
        grads.append(
            (X.grad.clone(), fn.grad.clone(), scale.grad.clone(), base.grad.clone())
        )
    for saved, recomputed in zip(*grads, strict=False):
        assert torch.equal(saved, recomputed), (
            "checkpoint replay changed a gradient bit"
        )


def _live_bytes() -> int | None:
    """Bytes the live CUDA tensors asked the caching allocator for (its ``requested_bytes``), not the bytes of the
    blocks they occupy (``torch.cuda.memory_allocated``): a tensor placed in a reused free block up to 1 MiB larger
    than itself (the allocator does not split a large block for a remainder of at most 1 MiB) would otherwise count
    that slack as memory kept alive, as the flashinfer-ci nightly did on torch 2.15 (186880 B on B200, 1047040 B on
    R200) for the checkpointed output below.  ``None`` under another allocator backend (``cudaMallocAsync`` keeps
    no ``requested_bytes``), where the retention check is skipped and the gradient checks still run."""
    torch.cuda.synchronize()
    if torch.cuda.get_allocator_backend() != "native":
        return None
    return int(torch.cuda.memory_stats()["requested_bytes.all.current"])


def _assert_nothing_kept(before: int | None, Y: torch.Tensor, what: str) -> None:
    if before is None:
        return
    kept = _live_bytes() - before - Y.untyped_storage().nbytes()
    assert kept == 0, f"{kept} bytes kept alive after a checkpointed {what}"


def _assert_kept(
    before: int | None, what: str, *alive: torch.Tensor, expected: int = 0
) -> None:
    """The live bytes beyond ``before`` are those of ``alive`` plus ``expected``."""
    if before is None:
        return
    kept = _live_bytes() - before - sum(t.untyped_storage().nbytes() for t in alive)
    assert kept == expected, f"{kept - expected} bytes kept alive after {what}"


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_keeps_nothing_alive(T: int, save_stats: bool) -> None:
    """A non-reentrant ``torch.utils.checkpoint`` around prepare / finish keeps no tensor of the layer alive after
    the forward (``stats`` and the inputs travel through the saved tensors; the handle holds no branch output), and
    the checkpointed backward is bitwise the plain one."""
    from torch.utils.checkpoint import checkpoint

    i = _make_inputs(T, seed=SEED + 41)

    def connection(X, fn, scale, base):
        U, handle = pkg.cake_mhc_train_prepare(
            X, fn, scale, base, save_stats=save_stats
        )
        B = (U.float() + i["B"].float()).to(BF16)
        return pkg.cake_mhc_train_finish(X, B, handle)

    def leaves() -> list:
        return [i[k].clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")]

    plain = leaves()
    (connection(*plain).float() * i["dY"].float()).sum().backward()

    checkpointed = leaves()
    before = _live_bytes()
    Y = checkpoint(connection, *checkpointed, use_reentrant=False)
    _assert_nothing_kept(before, Y, "forward")
    (Y.float() * i["dY"].float()).sum().backward()
    for p, c in zip(plain, checkpointed, strict=True):
        assert torch.equal(p.grad, c.grad), "checkpointing changed a gradient bit"


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_after_prepare_replays_statistics(T: int, save_stats: bool) -> None:
    """A non-reentrant ``torch.utils.checkpoint`` region that starts after ``cake_mhc_train_prepare`` (the branch and
    the finish inside it, the prepare outside) keeps nothing beyond its output after the forward; its backward reruns
    the finish on a handle that already released the statistics, which the finish forward replays bitwise, so the
    gradients equal the plain run bit for bit."""
    from torch.utils.checkpoint import checkpoint

    i = _make_inputs(T, seed=SEED + 42)

    def finish(X, U, handle):
        B = (U.float() + i["B"].float()).to(BF16)
        return pkg.cake_mhc_train_finish(X, B, handle)

    def leaves() -> list:
        return [i[k].clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")]

    plain = leaves()
    U, handle = pkg.cake_mhc_train_prepare(*plain, save_stats=save_stats)
    (finish(plain[0], U, handle).float() * i["dY"].float()).sum().backward()

    checkpointed = leaves()
    U, handle = pkg.cake_mhc_train_prepare(*checkpointed, save_stats=save_stats)
    # pinned: save_stats=False releases it in the finish forward
    stats = handle.state.stats
    before = _live_bytes()
    Y = checkpoint(finish, checkpointed[0], U, handle, use_reentrant=False)
    _assert_nothing_kept(before, Y, "finish")
    # The finish forward takes the prepare node's edge outputs off the handle.
    assert handle.edges == () and handle.state.stats is None, (
        "the handle kept the edges or the statistics after the forward"
    )
    del stats
    (Y.float() * i["dY"].float()).sum().backward()
    for p, c in zip(plain, checkpointed, strict=True):
        assert torch.equal(p.grad, c.grad), "checkpointed finish: gradient bit changed"


def _connection_leaves(i: dict) -> list:
    return [i[k].clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")]


def _two_layers(
    X, params: list, *, save_stats: bool, checkpointed: bool
) -> torch.Tensor:
    """Two layers of two connections each (the branch hands ``U`` on as ``B``), each layer a non-reentrant checkpoint
    region or not."""
    from torch.utils.checkpoint import checkpoint

    def connection(x, fn, scale, base):
        U, handle = pkg.cake_mhc_train_prepare(
            x, fn, scale, base, save_stats=save_stats
        )
        return pkg.cake_mhc_train_finish(x, U, handle)

    def layer(x, k):
        return connection(connection(x, *params[2 * k]), *params[2 * k + 1])

    if not checkpointed:
        return layer(layer(X, 0), 1)
    return checkpoint(
        layer, checkpoint(layer, X, 0, use_reentrant=False), 1, use_reentrant=False
    )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("checkpointed", (False, True))
def test_backward_keeps_nothing_alive(
    T: int, save_stats: bool, checkpointed: bool
) -> None:
    """Once a connection's backward has run, nothing of it stays alive: after ``torch.autograd.grad`` over two layers
    of two connections the only live tensor of the step is its output ``Y`` (which keeps the graph alive), since the
    finish node's backward hands ``dY`` and the gradient statistics to the prepare node's backward on autograd edges.
    A second step gives bitwise the gradients of the first."""
    i = _make_inputs(T, seed=SEED + 51)
    params = [
        [p.clone().requires_grad_(True) for p in (q["fn"], q["scale"], q["base"])]
        for q in (_make_inputs(1, seed=SEED + 52 + k) for k in range(4))
    ]
    X = i["X"].clone().requires_grad_(True)
    leaves = [X, *[p for ps in params for p in ps]]

    def step() -> tuple:
        Y = _two_layers(X, params, save_stats=save_stats, checkpointed=checkpointed)
        return Y, torch.autograd.grad(Y, leaves, i["dY"])

    reference = step()[1]  # also the warm-up: default workspaces and loaded modules
    before = _live_bytes()
    Y, grads = step()
    assert all(torch.equal(g, r) for g, r in zip(grads, reference, strict=True)), (
        "a second step changed a gradient bit"
    )
    del grads
    _assert_kept(before, "the backward", Y)


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_repeated_backward_with_retain_graph(T: int, save_stats: bool) -> None:
    """Backward passes over one retained graph are independent: two full passes give bitwise the gradients of a fresh
    graph, and a pass in between that reaches the prepare node but not the finish node (the loss is ``U``) takes no
    ``dY`` of the earlier pass: it equals the same pass on a fresh graph."""
    i = _make_inputs(T, seed=SEED + 56)

    def graph(leaves: list) -> tuple:
        U, handle = pkg.cake_mhc_train_prepare(*leaves, save_stats=save_stats)
        B = (U.float() + i["B"].float()).to(BF16)
        return U, pkg.cake_mhc_train_finish(leaves[0], B, handle)

    leaves = _connection_leaves(i)
    U, Y = graph(leaves)
    first = torch.autograd.grad(Y, leaves, i["dY"], retain_graph=True)
    u_only = torch.autograd.grad(U, leaves, i["dU"], retain_graph=True)
    second = torch.autograd.grad(Y, leaves, i["dY"])

    fresh = _connection_leaves(i)
    full = torch.autograd.grad(graph(fresh)[1], fresh, i["dY"])
    fresh = _connection_leaves(i)
    u_fresh = torch.autograd.grad(graph(fresh)[0], fresh, i["dU"])
    for a, b, f in zip(first, second, full, strict=True):
        assert torch.equal(a, f) and torch.equal(b, f), (
            "retain_graph: gradient bit changed"
        )
    for a, f in zip(u_only, u_fresh, strict=True):
        assert torch.equal(a, f), (
            "a backward that skipped the finish node took a stale dY"
        )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("region", ("connection", "after_prepare"))
def test_reentrant_checkpoint(T: int, save_stats: bool, region: str) -> None:
    """A reentrant ``torch.utils.checkpoint`` around the connection, or around the branch and the finish with the
    prepare outside (the rerun finish node has no edge to the prepare node, so its backward leaves ``dY`` and the
    gradient statistics on the shared state and the prepare node's backward empties it), gives bitwise the plain
    gradients and keeps nothing alive after the backward beyond ``Y`` and the leaves' gradients."""
    from torch.utils.checkpoint import checkpoint

    i = _make_inputs(T, seed=SEED + 57)

    def finish(X, U, handle):
        B = (U.float() + i["B"].float()).to(BF16)
        return pkg.cake_mhc_train_finish(X, B, handle)

    def connection(X, fn, scale, base):
        U, handle = pkg.cake_mhc_train_prepare(
            X, fn, scale, base, save_stats=save_stats
        )
        return finish(X, U, handle)

    plain = _connection_leaves(i)
    connection(*plain).backward(i["dY"])

    checkpointed = _connection_leaves(i)
    before = _live_bytes()
    if region == "connection":
        Y = checkpoint(connection, *checkpointed, use_reentrant=True)
    else:
        U, handle = pkg.cake_mhc_train_prepare(*checkpointed, save_stats=save_stats)
        Y = checkpoint(finish, checkpointed[0], U, handle, use_reentrant=True)
        del U
    Y.backward(i["dY"])
    for p, c in zip(plain, checkpointed, strict=True):
        assert torch.equal(p.grad, c.grad), (
            f"reentrant checkpoint ({region}): gradient bit changed"
        )
    if region == "after_prepare":
        assert all(t is None for t in handle.state.take()), "the state kept a gradient"
    _assert_kept(
        before, "a reentrant checkpointed backward", Y, *(c.grad for c in checkpointed)
    )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("prepare", ("skipped", "not_recorded"))
def test_backward_without_the_prepare_node(T: int, prepare: str) -> None:
    """A backward that runs the finish node but not the prepare node (``torch.autograd.grad`` for the branch output
    alone, or a prepare whose inputs need no gradient) leaves the finish node's ``dY`` and gradient statistics on no
    handle: the engine drops the edge gradients.  What stays alive is the prepare node's own saved statistics while
    its backward has not run."""
    i = _make_inputs(T, seed=SEED + 58)
    leaves = (
        _connection_leaves(i)
        if prepare == "skipped"
        else [i[k] for k in ("X", "fn", "scale", "base")]
    )

    def step() -> tuple:
        U, handle = pkg.cake_mhc_train_prepare(*leaves)
        B = (U.float() + i["B"].float()).to(BF16)
        if prepare == "not_recorded":
            B.requires_grad_(True)
        Y = pkg.cake_mhc_train_finish(leaves[0], B, handle)
        return Y, B, handle, torch.autograd.grad(Y, [B], i["dY"])

    step()
    before = _live_bytes()
    Y, B, handle, (dB,) = step()
    assert all(t is None for t in handle.state.take()), "the state kept a gradient"
    saved_stats = (
        T * 256 if prepare == "skipped" else 0
    )  # the 256-byte record per token
    _assert_kept(
        before, "a backward without the prepare node", Y, B, dB, expected=saved_stats
    )


@pytest.mark.parametrize("save_stats", (True, False))
def test_interleaved_connections(save_stats: bool) -> None:
    """Two connections of one shape whose prepares both run before either finish (their edge outputs alias the same
    stand-in buffers but are distinct autograd outputs) give bitwise the gradients of the two connections run one
    after the other."""
    T = 129
    a, b = _make_inputs(T, seed=SEED + 61), _make_inputs(T, seed=SEED + 62)

    def branch(i: dict, U: torch.Tensor) -> torch.Tensor:
        return (U.float() + i["B"].float()).to(BF16)

    def run(interleaved: bool) -> list:
        la, lb = _connection_leaves(a), _connection_leaves(b)
        Ua, ha = pkg.cake_mhc_train_prepare(*la, save_stats=save_stats)
        if interleaved:
            Ub, hb = pkg.cake_mhc_train_prepare(*lb, save_stats=save_stats)
            Yb = pkg.cake_mhc_train_finish(lb[0], branch(b, Ub), hb)
            Ya = pkg.cake_mhc_train_finish(la[0], branch(a, Ua), ha)
        else:
            Ya = pkg.cake_mhc_train_finish(la[0], branch(a, Ua), ha)
            Ub, hb = pkg.cake_mhc_train_prepare(*lb, save_stats=save_stats)
            Yb = pkg.cake_mhc_train_finish(lb[0], branch(b, Ub), hb)
        torch.autograd.backward([Ya, Yb], [a["dY"], b["dY"]])
        return [p.grad for p in la + lb]

    for s, t in zip(run(False), run(True), strict=True):
        assert torch.equal(s, t), "interleaved connections: gradient bit changed"


@pytest.mark.parametrize("T", (1000,))
def test_accumulate_four_times(T: int) -> None:
    """Four consecutive ``accumulate=True`` passes on four input sets against the FP64 sum of the four gradients plus
    the carried-in values, within the FP32 gates in their distribution form over 25 draws of the four-fold chain
    (comparator: the four IEEE-FP32 gradients summed in FP32 from the same carried-in values); the first draw also
    prints its single-draw figures."""
    rows = {k: [] for k in _PARAMETER}
    for draw in range(PARAM_CLAUSE_DRAWS):
        g = torch.Generator(device=_device()).manual_seed(SEED + 7919 + draw)
        carried = {
            "dfn": torch.randn((MAPPING_ROWS, N), generator=g, device=_device()) * 1e-3,
            "dscale": torch.randn((3,), generator=g, device=_device()) * 1e-3,
            "dbase": torch.randn((MAPPING_ROWS,), generator=g, device=_device()) * 1e-3,
        }
        acc = {k: v.clone() for k, v in carried.items()}
        want = {k: v.double() for k, v in carried.items()}
        comp = {k: v.clone() for k, v in carried.items()}
        for step in range(4):
            i = _make_inputs(T, seed=SEED + 100 + 4 * draw + step)
            _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"])
            _dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
            out = backward_finish(
                i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
                accumulate=True, dfn=acc["dfn"], dscale=acc["dscale"], dbase=acc["dbase"],
            )  # fmt: skip
            assert (
                out[1] is acc["dfn"]
                and out[2] is acc["dscale"]
                and out[3] is acc["dbase"]
            )
            ref = _oracle(i)
            comparator = _oracle(i, dtype=F32, block_tokens=None)
            for k in _PARAMETER:
                want[k] = want[k] + ref[k]
                comp[k] = comp[k] + comparator[k]
            del i, ref, comparator
        for k in _PARAMETER:
            if draw == 0:
                _assert_fp32(
                    acc[k],
                    want[k],
                    comp[k],
                    f"{k} accumulate x4 T={T} draw 0",
                    per_element=(k != "dfn"),
                    parameter=True,
                )
            rows[k].append(_draw_row(acc[k], want[k], comp[k]))
    _assert_parameter_distribution(rows, f"accumulate x4 T={T}", regime="unit")


@pytest.mark.parametrize("T", (129, 16263))
def test_outputs_are_deterministic(T: int) -> None:
    """Three launches of the whole forward / backward on the same inputs, outputs and the non-ticket workspace
    regions re-poisoned with NaN between them, give bitwise identical outputs."""
    i = _make_inputs(T, seed=SEED + 3)
    fwd_ws = forward_workspace(T, _device())
    bwd_ws = backward_workspace(T, _device())
    device_index = _device().index
    layouts = {
        "fwd": loader.workspace_layout("fwd", T, device_index=device_index),
        "bwd": loader.workspace_layout("bwd", T, device_index=device_index),
    }

    def poison(ws: torch.Tensor, kind: str) -> None:
        for name, region in layouts[kind]["regions"].items():
            if name != "ticket":
                ws[region["offset"] : region["offset"] + region["bytes"]].fill_(0xFF)

    results = []
    for _repeat in range(3):
        poison(fwd_ws, "fwd")
        poison(bwd_ws, "bwd")
        U = torch.full((T, HIDDEN), float("nan"), dtype=BF16, device=_device())
        stats = torch.full((T, 64), float("nan"), dtype=F32, device=_device())
        Y = torch.full((T, STREAMS, HIDDEN), float("nan"), dtype=BF16, device=_device())
        dB = torch.full((T, HIDDEN), float("nan"), dtype=BF16, device=_device())
        gstats = torch.full((T, 32), float("nan"), dtype=F32, device=_device())
        dX = torch.full(
            (T, STREAMS, HIDDEN), float("nan"), dtype=BF16, device=_device()
        )
        dfn = torch.full((MAPPING_ROWS, N), float("nan"), dtype=F32, device=_device())
        dscale = torch.full((3,), float("nan"), dtype=F32, device=_device())
        dbase = torch.full((MAPPING_ROWS,), float("nan"), dtype=F32, device=_device())
        forward_prepare(
            i["X"],
            i["fn"],
            i["scale"],
            i["base"],
            out=U,
            stats_out=stats,
            workspace=fwd_ws,
        )
        forward_finish(i["X"], i["B"], stats, out=Y)
        backward_prepare(i["dY"], i["B"], i["X"], stats, dB=dB, grad_stats_out=gstats)
        backward_finish(
            i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
            workspace=bwd_ws, dX=dX, dfn=dfn, dscale=dscale, dbase=dbase,
        )  # fmt: skip
        torch.cuda.synchronize()
        results.append(
            [t.clone() for t in (U, stats, Y, dB, gstats, dX, dfn, dscale, dbase)]
        )
        for t in results[-1]:
            assert torch.isfinite(t.float()).all()
    for later in results[1:]:
        for first, again in zip(results[0], later, strict=False):
            assert torch.equal(first, again), "a repeated launch changed an output bit"
    fresh = _run(i)
    assert torch.equal(fresh["dfn"], results[0][6]) and torch.equal(
        fresh["dX"], results[0][5]
    )
    # The ticket words self-reset: the caller-held workspaces are reusable without a reset.
    for ws, kind in ((fwd_ws, "fwd"), (bwd_ws, "bwd")):
        region = layouts[kind]["regions"]["ticket"]
        assert (
            torch.count_nonzero(
                ws[region["offset"] : region["offset"] + region["bytes"]]
            )
            == 0
        )


def test_changing_rows_in_one_process_does_not_rebuild() -> None:
    _check_case(64)
    loaded_before = loader.load.cache_info().misses
    for T in (1, 65, 2049, 7, 255):
        i = _make_inputs(T, seed=SEED + T)
        res = _run(i)
        ref = _oracle(i)
        _assert_bf16(res["dX"], ref["dX32"], f"dX T={T}")
        _assert_fp32(res["dfn"], ref["dfn"], None, f"dfn T={T}")
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed token count rebuilt a module"
    )


_EPS_CASES = ((2e-5, 2e-6), (1e-6, 1e-5), (1e-5, 4e-6))


@pytest.mark.parametrize("T", (64, 2049))
@pytest.mark.parametrize("rms_eps,hc_eps", _EPS_CASES)
def test_non_default_eps_match_fp64(T: int, rms_eps: float, hc_eps: float) -> None:
    """``rms_eps`` / ``hc_eps`` are launch arguments: the forward and backward at other values meet the same FP64
    bars as the defaults (the oracle is evaluated at the same values), and the statistics actually move."""
    out = _check_case(T, rms_eps=rms_eps, hc_eps=hc_eps)
    default = _run(_make_inputs(T))
    assert not torch.equal(out["result"]["stats"], default["stats"]), (
        f"stats unchanged by rms_eps={rms_eps:g} hc_eps={hc_eps:g}"
    )


def test_changing_eps_in_one_process_does_not_rebuild() -> None:
    """A different epsilon is a launch argument of the already compiled programs: no module is rebuilt or reloaded
    when the epsilons change between calls, and the outputs follow the FP64 oracle at each value."""
    _check_case(64)
    loaded_before = loader.load.cache_info().misses
    for rms_eps, hc_eps in _EPS_CASES + ((RMS_EPS, HC_EPS),):
        i = _make_inputs(255, seed=SEED + 3)
        i["rms_eps"], i["hc_eps"] = rms_eps, hc_eps
        res = _run(i)
        ref = _oracle(i)
        _assert_bf16(
            res["dX"], ref["dX32"], f"dX rms_eps={rms_eps:g} hc_eps={hc_eps:g}"
        )
        _assert_fp32(
            res["dfn"], ref["dfn"], None, f"dfn rms_eps={rms_eps:g} hc_eps={hc_eps:g}"
        )
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed epsilon rebuilt or reloaded a module"
    )


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_autograd_end_to_end(T: int, save_stats: bool) -> None:
    """``cake_mhc_train_prepare`` / ``cake_mhc_train_finish`` around a branch, against the FP64 reference with the
    branch output and the branch's ``dU`` the autograd graph produced."""
    i = _make_inputs(T, seed=SEED + 31)
    X = i["X"].clone().requires_grad_(True)
    fn = i["fn"].clone().requires_grad_(True)
    scale = i["scale"].clone().requires_grad_(True)
    base = i["base"].clone().requires_grad_(True)
    U, handle = pkg.cake_mhc_train_prepare(X, fn, scale, base, save_stats=save_stats)
    U.retain_grad()
    B = (U.float() + i["B"].float()).to(
        BF16
    )  # the branch: dB/dU = 1, so the branch backward hands dB on as dU
    Y = pkg.cake_mhc_train_finish(X, B, handle)
    (Y.float() * i["dY"].float()).sum().backward()
    assert U.grad is not None and U.grad.dtype == BF16
    j = {**i, "B": B.detach(), "dU": U.grad.detach()}
    ref = _oracle(j)
    comp = _oracle(j, dtype=F32, block_tokens=None)
    _assert_bf16(U.detach(), ref["U32"], f"autograd U T={T}")
    _assert_bf16(Y.detach(), ref["Y32"], f"autograd Y T={T}")
    _assert_bf16(U.grad, ref["dB32"], f"autograd dB T={T}")
    assert X.grad.dtype == BF16
    _assert_bf16(X.grad, ref["dX32"], f"autograd dX T={T}")
    # The absolute bars on this draw; the 2x clause of these gradients is the >= 25-draw distribution of the direct
    # tests (the autograd path launches the same programs), so its single-draw ratio is printed as the literal record.
    _assert_fp32(
        fn.grad, ref["dfn"], comp["dfn"], f"autograd dfn T={T}", parameter=True
    )
    _assert_fp32(
        scale.grad,
        ref["dscale"],
        comp["dscale"],
        f"autograd dscale T={T}",
        per_element=True,
        parameter=True,
    )
    _assert_fp32(
        base.grad,
        ref["dbase"],
        comp["dbase"],
        f"autograd dbase T={T}",
        per_element=True,
        parameter=True,
    )


def test_workspace_bytes_and_layout() -> None:
    small_f, small_b = (
        forward_workspace_bytes(1, _device()),
        backward_workspace_bytes(1, _device()),
    )
    large_f, large_b = (
        forward_workspace_bytes(65536, _device()),
        backward_workspace_bytes(65536, _device()),
    )
    assert (
        small_f > 0
        and small_b > 0
        and all(n % 256 == 0 for n in (small_f, small_b, large_f, large_b))
    )
    assert large_f >= small_f and large_b >= small_b
    previous = (0, 0)
    for T in (1, 7, 128, 129, 255, 1000, 2049, 16263, 20001, 65536):
        current = (
            forward_workspace_bytes(T, _device()),
            backward_workspace_bytes(T, _device()),
        )
        assert current[0] >= previous[0] and current[1] >= previous[1], T
        device_index = _device().index
        assert (
            current[0]
            >= loader.workspace_layout("fwd", T, device_index=device_index)[
                "total_bytes"
            ]
        )
        assert (
            current[1]
            >= loader.workspace_layout("bwd", T, device_index=device_index)[
                "total_bytes"
            ]
        )
        previous = current
    ws = backward_workspace(2049, _device())
    assert ws.dtype == torch.uint8 and ws.numel() == backward_workspace_bytes(
        2049, _device()
    )
    assert torch.count_nonzero(ws) == 0
    ws.fill_(1)
    pkg.cake_mhc_train_reset_workspace(ws)
    assert torch.count_nonzero(ws) == 0
    with pytest.raises(ValueError):
        forward_workspace_bytes(-1, _device())
    # A workspace sized for a larger token count serves a smaller one; a smaller one is refused.
    i = _make_inputs(129, seed=SEED + 5)
    big = forward_workspace(4096, _device())
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"], workspace=big)
    assert torch.equal(
        stats, forward_prepare(i["X"], i["fn"], i["scale"], i["base"])[1]
    )
    with pytest.raises(ValueError):
        forward_prepare(
            i["X"],
            i["fn"],
            i["scale"],
            i["base"],
            workspace=torch.zeros(256, dtype=torch.uint8, device=_device()),
        )


def test_argument_validation() -> None:
    i = _make_inputs(8, seed=SEED + 9)
    X, fn, scale, base = i["X"], i["fn"], i["scale"], i["base"]
    with pytest.raises(ValueError):
        forward_prepare(X.float(), fn, scale, base)
    with pytest.raises(ValueError):
        forward_prepare(X, fn.to(BF16), scale, base)
    with pytest.raises(ValueError):
        forward_prepare(X, fn[:, 1:], scale, base)
    with pytest.raises(ValueError):
        forward_prepare(X, fn, scale[:2], base)
    with pytest.raises(ValueError):
        forward_prepare(X.view(8, HIDDEN * STREAMS), fn, scale, base)
    # Padded rows (a [T, 4, 4096] view of a wider buffer) are not materialized silently: the caller decides.
    padded = torch.zeros(8, STREAMS, HIDDEN + 64, dtype=BF16, device=_device())[
        :, :, :HIDDEN
    ]
    with pytest.raises(ValueError):
        forward_prepare(padded, fn, scale, base)
    with pytest.raises(ValueError):
        forward_prepare(X.cpu(), fn.cpu(), scale.cpu(), base.cpu())
    # The epsilons are launch arguments of the compiled programs (any finite positive value runs); the Sinkhorn
    # iteration count is an integer from 1 to SINKHORN_ITERS_MAX (test_cake_mhc_train_iters.py runs the counts).
    for kwargs in (
        {"rms_eps": 1e-6},
        {"hc_eps": 1e-5},
        {"rms_eps": 2e-5, "hc_eps": 2e-6},
    ):
        forward_prepare(X, fn, scale, base, **kwargs)
    for kwargs in (
        {"rms_eps": 0.0},
        {"rms_eps": -1e-5},
        {"hc_eps": float("nan")},
        {"hc_eps": float("inf")},
    ):
        with pytest.raises(ValueError):
            forward_prepare(X, fn, scale, base, **kwargs)
    for kwargs in (
        {"sinkhorn_iters": 0},
        {"sinkhorn_iters": 20.5},
    ):
        with pytest.raises(ValueError):
            forward_prepare(X, fn, scale, base, **kwargs)
    with pytest.raises(NotImplementedError):
        forward_prepare(X, fn, scale, base, sinkhorn_iters=pkg.SINKHORN_ITERS_MAX + 1)
    assert pkg.SHIPPED_SINKHORN_ITERS == (SINKHORN_ITERS,)
    U, stats = forward_prepare(X, fn, scale, base)
    with pytest.raises(ValueError):
        forward_finish(X, i["B"][:, : HIDDEN // 2], stats)
    with pytest.raises(ValueError):
        forward_finish(X, i["B"], stats.double())
    dB, gstats = backward_prepare(i["dY"], i["B"], X, stats)
    with pytest.raises(ValueError):
        backward_finish(
            X, i["dY"], i["dU"], i["B"], fn, scale, base, stats, gstats, accumulate=True
        )
    with pytest.raises(ValueError):
        backward_finish(X, i["dY"], i["dU"], i["B"][:4], fn, scale, base, stats, gstats)
    with pytest.raises(ValueError):
        backward_finish(
            X,
            i["dY"],
            i["dU"],
            i["B"],
            fn,
            scale,
            base,
            stats,
            gstats[:, :16].contiguous(),
        )
    # B is optional in the backward finish (its contribution is in grad_stats): None gives the same bits.
    with_b = backward_finish(
        X, i["dY"], i["dU"], i["B"], fn, scale, base, stats, gstats
    )
    without_b = backward_finish(
        X, i["dY"], i["dU"], None, fn, scale, base, stats, gstats
    )
    assert all(torch.equal(a, b) for a, b in zip(with_b, without_b, strict=False))
    with pytest.raises(TypeError):
        pkg.cake_mhc_train_finish(X, i["B"], object())


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library, architecture) the registry
    references, each with the kernel and binding sources under ``csrc/`` of this package, their digests and a
    closure digest; the registry itself records the host plan only.
    """
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"], name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
