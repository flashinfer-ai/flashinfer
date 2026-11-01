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

Backend suite of the Cake four-stream mHC training package with the opt-in
non-deterministic backward
(``flashinfer.experimental.cake.cake_mhc_train_backend``; hidden 4096, 5120
and 7168 on SM100 / SM103, 4096 on SM107; 4 streams; FlashInfer issue #8068).

The package is standalone: its forward, statistics replay, residual mixing and
backward prepare run its own copies of the deterministic programs, and
``cake_mhc_train_backward_finish(..., deterministic=False)`` launches its copy
of the deterministic token-owner pass, its atomic fn split (beside the
token-owner pass as its programmatic dependent; it also clears ``dfn`` without
``accumulate``) and its atomic column-slab pass, which adds
the ``dfn`` partial of every (column slab, token chunk) unit into ``dfn`` with
FP32 reductions and folds ``dscale`` / ``dbase`` in fixed order; there is no
reduction kernel.  ``dX``, ``dscale`` and ``dbase`` are the deterministic
package's bits; each ``dfn`` element sums ``P`` per-chunk FP32 partials (``P``
the column slab's chunk count: ``cake_jit.k7_chunks``, or the two-size
geometry where ``cake_jit.lpt_chunks`` fires; at most 32) in an order that may
change from run to run.  Every entry point takes ``deterministic=False`` (any
other value is a ``ValueError``, checked without a device by
``test_cake_mhc_train_rules.py``).  The deterministic package
(``flashinfer.cake_mhc_det_train_backend``) is the same-input reference
throughout, never a source of tolerances.  This module checks:

* accuracy against the FP64 reference of ``test_cake_mhc_det_train.py`` under
  its bars, on the worst of 5 launches of every draw: BF16 ``dX`` at atol =
  rtol = 1e-2 with no mismatch; ``dfn`` / ``dscale`` / ``dbase`` rel-L2 <= 1e-5
  and (``dscale`` / ``dbase``) per element <= 1e-4; the "<= 2x the IEEE-FP32
  comparator" clause as the distribution over 25 draws (median and maximum);
  the saturated-sigmoid tail clause and the single-cast cancellation band in
  the deterministic suite's form;
* against the deterministic backward on the same inputs: ``dX``, ``dscale``,
  ``dbase`` bitwise; ``|dfn - dfn_det| <= F u (|carried| + 1.02 sum_t |dz'_t|
  |x_t|)`` per element (``u = 2^-24``; the FP64 absolute-value contraction of
  the reference bounds the magnitude of every partial sum either order forms):
  ``F = 2 P`` where the column slab runs ``dynr`` (both sides sum the same
  per-chunk partials, ``P`` roundings each) and ``F = 2 max(P, P_k7) + 10``
  where it runs ``lpt`` (the partials are another partition's; see
  :func:`_dfn_factor`), the largest ratio printed; run to run, ``dX`` /
  ``dscale`` / ``dbase`` identical and the ``dfn`` bits that differ counted and
  printed (never asserted);
* the two-size schedule at forced geometries (``cake_jit.lpt_chunks`` replaced
  for one token count: one size, one small chunk, several, small chunks past
  ``T``, empty small chunks, 32 chunks, one round of units, a short odd row,
  with and without ``accumulate``, and the screen's best geometry at 16263
  tokens): the ``lpt`` record launches over ``min(SMs, 64 P)`` CTAs, ``dX`` /
  ``dscale`` / ``dbase`` bitwise the deterministic package's on two launches
  (fresh outputs, then a poisoned caller workspace), ``dfn`` within the bound,
  the ticket word back to 0;
* the forward entry points (forward prepare, statistics replay, residual
  mixing, backward prepare) bitwise the deterministic package's, and every
  launch one of this package's records (the deterministic package launches
  nothing; the backward loads exactly the token-owner record, the atomic fn
  split and the column slab's form of the call, in that order);
* PyTorch's deterministic algorithms: ``backward_finish`` raises PyTorch's
  alert (``RuntimeError`` before any launch; under ``warn_only`` one
  ``UserWarning`` per call, then the non-deterministic backward runs, never a
  fallback), and so does the backward of the ``cake_mhc_train_prepare`` node,
  also when the mode is enabled between the forward and the backward; the
  forward entry points, the workspaces and the prepare node's forward never
  alert;
* four ``accumulate=True`` passes; the autograd functions plain and under a
  non-reentrant checkpoint, against the deterministic gradients;
* ``test_cake_mhc_det_train.py``'s checkpoint and lifetime cases with
  ``deterministic=False`` (the saved-tensor and autograd-edge contract of
  FlashInfer issues #7039 / #7547): the checkpoint replay of the statistics, a
  checkpoint around the connection or after the prepare keeping nothing alive,
  nothing alive once a backward has run, repeated backward passes with
  ``retain_graph``, reentrant checkpoints, a backward without the prepare node,
  interleaved connections; a zero row of ``X``, ``fn x 1e3`` logits, a changed
  token count without a rebuild and the argument validation;
* concurrent launches on several streams; ``T == 0``; the build manifest;
* the hidden sizes: the contract cases above that take ``hidden`` run at
  every size the device's architecture routes (5120 and 7168 on a
  representative subset of their token counts: small, a middle one and large
  ones where the token-owner grid reaches its per-SM bound; the forward
  entry points and the backward's exact outputs also at the 19 benchmark
  token counts), and the sizes mixed in one process (the host caches keyed by
  the hidden size) give each size's bits.

The siblings hold the rest of the deterministic suite's organization:
``test_cake_mhc_train_graphs.py`` (CUDA Graph capture and replay),
``test_cake_mhc_train_iters.py`` (the run-time Sinkhorn count and the
epsilons), ``test_cake_mhc_train_workspace.py`` (caller and default
workspaces across token-count sequences, their sizes and which sizing serves
which backward), ``test_cake_mhc_train_rules.py`` (the registry against the
deterministic one, the standalone package and the mode policy without a
device) and ``test_cake_mhc_train_api.py`` (the
``flashinfer.experimental.cake.cake_mhc_train`` facade).

The largest cases hold the FP64 reference in row blocks of 8192 tokens.  Like
``test_cake_mhc_det_train.py`` this suite exercises the backend packages alone
(both named through ``tests.test_helpers.cake_package``) and imports nothing
else from flashinfer.
"""

from __future__ import annotations

import contextlib
import functools
import math
import re
import statistics
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers.cake_package import cake_package

KEY = "cake.cake_mhc_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
# The deterministic package: the same-input comparison point (never a source of tolerances).
det = cake_package("cake_mhc_det_train")
det_loader = cake_package("cake_mhc_det_train", "cake_jit")
# The package's entry points with the mode they require.
backward_finish = pkg.cake_mhc_train_backward_finish
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
det_backward_finish = det.cake_mhc_det_train_backward_finish

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
# HIDDEN / N: the hidden size and flattened token of the cases that take no ``hidden`` (the two-size column-slab
# schedule and the argument checks run at H 4096 only).
STREAMS, HIDDEN, N, MAPPING_ROWS = 4, 4096, 16384, 24
# The package's hidden sizes; a case of a size runs where the device's architecture routes it (:func:`_require_hidden`).
HIDDEN_SIZES = (4096, 5120, 7168)
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
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
PARAM_CLAUSE_DRAWS = 25
CANCEL_CLAUSE_DRAWS = 64
# Launches of the non-deterministic backward per draw: every gate holds on each (the worst is reported).
REPEATS = 5
RUN_TO_RUN_REPEATS = 8
# The FP32 unit roundoff and the slack of the FP64 absolute-value contraction (the kernel's dz' is its own FP32
# value split into three BF16 pieces, within 1e-5 relative of the FP64 dz'; the pieces add at most 2^-8 relative).
FP32_UNIT_ROUNDOFF = 2.0**-24
ABS_SLACK = 1.02
TICKET_BYTES = (
    256  # the ticket region: first in the backward workspace for every token count
)
T_ACCURACY = (1, 7, 64, 257, 1024, 2049, 4096, 16263, 32768)
T_DISTRIBUTION = (1, 2, 3, 7, 63, 129, 255, 1000, 2049)
T_COMPARE = (1, 7, 64, 65, 257, 1000, 1024, 2049, 4096, 5000, 8192, 14928, 16263, 32768)
# The token counts of hidden 5120 and 7168 in the cases above: small ones, a middle one (2049: 257 token groups, below
# the token owner's 2 x SMs at 5120 and above its SM count at 7168) and large ones where the token owner's grid G6 =
# min(K6_CTAS_PER_SM x SMs, ceil(T / 8)) reaches its bound at both sizes (4096, 16263, 32768).
T_ACCURACY_NEW = (1, 257, 2049, 16263)
T_COMPARE_NEW = (1, 7, 65, 1000, 2049, 4096, 16263, 32768)
# The 19 benchmark token counts of the performance tables.
T_BENCH = (1024, 1536, 2048, 2304, 2560, 3072, 3584, 4096, 8192, 14928, 15288, 15566, 15590, 16114, 16132, 16164,
           16263, 32768, 65536)  # fmt: skip
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
_PARAMETER = ("dfn", "dscale", "dbase")
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")
_EXACT = ("dX", "dscale", "dbase")  # bitwise the deterministic package's
_ALERT = "does not have a deterministic implementation"
SLAB = "cake_mhc_train_bwd_slab_atomic__dynr"
LPT = "cake_mhc_train_bwd_slab_atomic__lpt"
# The lpt record's chunk scalars (cake_jit.slab_plan), in kernel-parameter order (P, rpc, p_big, rpc_small).
LPT_HOST_KEYS = (
    "lpt_parts",
    "lpt_rows_per_chunk",
    "lpt_big_parts",
    "lpt_small_rows_per_chunk",
)
# Forced two-size geometries (P, rpc, p_big, rpc_small) at T = 2049 (33 tiles, the last one token).
LPT_FORCED = (
    (7, 320, 7, 320),  # p_big = P: one size, 7 chunks of 5 tiles
    (5, 448, 4, 320),  # one small chunk: 4 x 7 tiles, then 5 tiles
    (6, 512, 3, 192),  # several small chunks: 3 x 8 tiles, then 3 x 3 tiles
    (8, 512, 3, 192),  # small chunks past T: 3 x 8 tiles, then 5 x 3 tiles of which the last 2 hold no token
    (6, 512, 4, 256),  # a small chunk longer than the tokens left: 4 x 8 tiles leave 1 token, the second small is empty
    (5, 768, 3, 128),  # big chunks that reach T (12, 12 and 9 tiles): both small chunks are empty
    (32, 128, 12, 64),  # 32 chunks: 12 x 2 tiles, then 20 x 1 tile of which the last 11 are empty
    (2, 1280, 1, 832),  # 128 units, one round on every board: 20 tiles, then 13
)  # fmt: skip


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
    # Build the local architecture's libraries of both packages that have no prebuilt file (this package's, and the
    # deterministic package's for the same-input comparisons) through the spec objects the loaders build on first
    # use; a prebuilt or cached library loads without compiling.
    if _supported():
        arch = _arch()
        for spec in [*det_loader.specs_for_arch(arch), *loader.specs_for_arch(arch)]:
            spec.build_and_load()
    yield


@pytest.fixture
def deterministic_algorithms():
    """``torch.use_deterministic_algorithms``; the process-wide flags are restored however the test ends."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield torch.use_deterministic_algorithms
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture routes every kind of."""
    arch = _arch()
    return tuple(loader.supported_hidden_sizes(arch)) if arch else ()


def _require_hidden(hidden: int) -> int:
    if hidden not in _hidden_sizes():
        pytest.skip(
            f"hidden {hidden} has no exported route on {_arch()} "
            f"(exported hidden sizes: {_hidden_sizes()})"
        )
    return hidden


def _hidden_cases(Ts: tuple, new_Ts: tuple) -> list:
    """``(hidden, T)`` parameters: the token counts ``Ts`` at hidden 4096, ``new_Ts`` at 5120 and 7168 (each case
    skipped where the architecture has no route of its size)."""
    return [
        pytest.param(hidden, T, id=f"H{hidden}-{T}")
        for hidden in HIDDEN_SIZES
        for T in (Ts if hidden == HIDDEN else new_Ts)
    ]


def _suffix(hidden: int) -> str:
    """The library suffix of this package's records of ``hidden``: none at 4096, ``__h<H>`` otherwise."""
    return "" if hidden == HIDDEN else f"__h{hidden}"


def _hidden_of(i: dict) -> int:
    return int(i["X"].shape[2])


def _parts(T: int, hidden: int = HIDDEN) -> int:
    """``P``: the chunk count of the column-slab pass, the number of partials each ``dfn`` element sums (the
    deterministic backward's chunk count too: both sum the partials of the same chunks)."""
    parts = loader.host_plan(T, _sm_count(), _arch(), hidden)["parts"]
    assert parts == loader.k7_chunks(T, _sm_count(), hidden=hidden)[0], (T, hidden)
    assert parts == det_loader.host_plan(T, _sm_count(), _arch(), hidden)["parts"]
    assert 1 <= parts <= 32, (T, hidden, parts)
    return parts


def _lpt_runs(T: int, hidden: int = HIDDEN) -> bool:
    """Whether the column slab runs its two-size ``lpt`` record at ``T`` on this device (read at call time, so a
    replaced ``cake_jit.lpt_chunks`` counts; never at hidden 5120 and 7168)."""
    return loader.bwd_slab_atomic_program(T, _sm_count(), _arch(), hidden) == "lpt"


def _slab_parts(T: int, hidden: int = HIDDEN) -> int:
    """The chunk count the column slab launches: ``slab_plan``'s ``lpt_parts`` under ``lpt``, else :func:`_parts`."""
    if not _lpt_runs(T, hidden):
        return _parts(T, hidden)
    parts = loader.slab_plan(T, _sm_count(), _arch(), hidden)["lpt_parts"]
    assert 1 <= parts <= 32, (T, parts)
    return parts


def _dfn_factor(T: int, hidden: int = HIDDEN) -> float:
    """F of the dfn bound F u (|carried| + ABS_SLACK sum_t |dz'_t| |x_t|) at T.  F = 2 P where the column slab runs
    dynr: both sides sum the same per-chunk partials, at most P roundings each.  Under lpt the partials are another
    partition's.  Every chunk starts on a 64-token tile and each tile's leading-piece product starts a fresh
    accumulator, so those products are bitwise the same on both sides.  Each side then differs from the exact sum of
    those products and the trailing terms by its own compensated summation of its tiles (2u), the rounding of
    acc - comp (u) and the FP32
    accumulation of the trailing dz' pieces (at most 2^-8 of each term) over 16-tile groups (at most 128 accumulate
    steps of at most 2u each: about 1.1u).  That is about 4.1u per side.  The cross-chunk roundings are P on this
    side and at most P_k7 on the deterministic side.  So F = 2 max(P, P_k7) + 10 >= P + P_k7 + 8.2, which also
    covers two launches of the same lpt partition (2 P)."""
    if not _lpt_runs(T, hidden):
        return 2.0 * _parts(T, hidden)
    return 2.0 * max(_slab_parts(T, hidden), _parts(T, hidden)) + 10.0


# ----------------------------------------------------------------------------------------------------------------
# Reference arithmetic (the FP64 oracle of test_cake_mhc_det_train.py, with the Sinkhorn count as a parameter; the
# same code in FP32 after _set_ieee_fp32 is the comparator)


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
    """The BF16 spacing at ``|x|`` (FP64): ``2^(floor(log2 |x|) - 7)``, the subnormal spacing below ``2^-126``."""
    e = torch.floor(torch.log2(x.to(F64).abs().clamp_min(2.0**-126)))
    return torch.exp2(e - 7)


def _bf16_ulp_error(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """``|out - ref| / ulp_bf16(ref)`` (FP64) of a BF16 output against the FP64 reference."""
    assert out.dtype == BF16, out.dtype
    ref64 = ref.to(F64)
    return (out.to(F64) - ref64).abs() / _bf16_ulp_spacing(ref64)


def _sinkhorn(logits: torch.Tensor, hc_eps: float, iters: int) -> torch.Tensor:
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _coefficients(z, scale, base, hc_eps: float = HC_EPS, iters: int = SINKHORN_ITERS):
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
    rms_eps: float,
    hc_eps: float,
    iters: int,
) -> dict:
    """One row block of the reference: the forward in ``dtype``, the backward by autograd of the FP arithmetic."""
    T = X.shape[0]
    with torch.enable_grad():
        Xd = X.detach().to(dtype).requires_grad_(True)
        fnd = fn.detach().to(dtype).requires_grad_(True)
        scaled = scale.detach().to(dtype).requires_grad_(True)
        based = base.detach().to(dtype).requires_grad_(True)
        u = Xd.reshape(T, Xd.shape[1] * Xd.shape[2])
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)
        z = (u * r) @ fnd.T
        pre, post, C = _coefficients(z, scaled, based, hc_eps, iters)
        U_pre = (pre.unsqueeze(-1) * Xd).sum(dim=1)
        Y_pre = post.unsqueeze(-1) * B.detach().to(dtype).unsqueeze(1) + torch.einsum(
            "tij,tid->tjd", C, Xd
        )
        dUd = dU.detach().to(dtype)
        dYd = dY.detach().to(dtype)
        dX32, dfn, dscale, dbase = torch.autograd.grad(
            [U_pre, Y_pre], [Xd, fnd, scaled, based], grad_outputs=[dUd, dYd]
        )
    out = {"dX32": dX32, "dfn": dfn, "dscale": dscale, "dbase": dbase}
    if want_contrib:
        with torch.no_grad():
            out["collapse"] = pre.detach().unsqueeze(-1) * dUd.unsqueeze(1)
            out["mix"] = torch.einsum("tij,tjd->tid", C.detach(), dYd)
    return out


def _oracle(
    i: dict,
    *,
    dtype=F64,
    block_tokens: int | None = BLOCK_TOKENS,
    want_contrib: bool = False,
) -> dict:
    """The reference gradients of one input set: FP64 in row blocks (the oracle) or IEEE FP32 in one block (the
    comparator), at the case's epsilons and Sinkhorn count."""
    T = i["X"].shape[0]
    if dtype == F32:
        _set_ieee_fp32()
    bt = T if not block_tokens else max(1, min(block_tokens, T))
    consts = {
        "rms_eps": i.get("rms_eps", RMS_EPS),
        "hc_eps": i.get("hc_eps", HC_EPS),
        "iters": i.get("sinkhorn_iters", SINKHORN_ITERS),
    }
    blocks = [
        _block(
            i["X"][lo : lo + bt], i["fn"], i["scale"], i["base"], i["B"][lo : lo + bt], i["dY"][lo : lo + bt],
            i["dU"][lo : lo + bt], dtype=dtype, want_contrib=want_contrib, **consts,
        )
        for lo in range(0, T, bt)
    ]  # fmt: skip
    if len(blocks) == 1:
        return blocks[0]
    out = {
        k: torch.cat([b[k] for b in blocks], dim=0)
        for k in ("dX32", "collapse", "mix")
        if k in blocks[0]
    }
    for k in _PARAMETER:
        acc = blocks[0][k].clone()
        for b in blocks[1:]:
            acc += b[k]
        out[k] = acc
    return out


def _dzp(i: dict) -> torch.Tensor:
    """The FP64 ``dz' = r dL/dz`` of the reference, ``[T, 24]``: ``dfn = dz'^T X_flat`` sums one product per token.

    ``dL/dz`` is the backward of the coefficients alone (sigmoids and Sinkhorn chain) from the per-token gradients
    of ``pre``, ``post`` and ``C``, so no ``[T, 4H]`` autograd graph is held."""
    T = i["X"].shape[0]
    rms_eps = i.get("rms_eps", RMS_EPS)
    hc_eps = i.get("hc_eps", HC_EPS)
    iters = i.get("sinkhorn_iters", SINKHORN_ITERS)
    fn, scale, base = (i[k].double() for k in ("fn", "scale", "base"))
    blocks = [torch.zeros((0, MAPPING_ROWS), dtype=F64, device=_device())]
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        X = i["X"][lo:hi].double()
        dY = i["dY"][lo:hi].double()
        dU = i["dU"][lo:hi].double()
        B = i["B"][lo:hi].double()
        u = X.reshape(hi - lo, -1)
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


def _abs_contraction(i: dict) -> torch.Tensor:
    """``sum_t |dz'[t, m]| |x[t, k]|`` (FP64 ``[24, 4H]``): the magnitudes of every term of ``dfn``, summed."""
    T = i["X"].shape[0]
    dzp = _dzp(i).abs()
    A = torch.zeros(
        (MAPPING_ROWS, STREAMS * _hidden_of(i)), dtype=F64, device=_device()
    )
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        A += dzp[lo:hi].T @ i["X"][lo:hi].reshape(hi - lo, -1).double().abs()
    return A


def _dfn_bound(
    i: dict, factor: float, *, carried: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-element bound of ``|dfn_a - dfn_b|`` for two sums of per-chunk partials in two orders:
    ``factor u (|carried| + sum_t |dz'_t| |x_t|)`` (``factor`` = :func:`_dfn_factor` of the token count).

    Each side adds ``P`` partials (onto the carried-in value with ``accumulate``) with at most ``P`` FP32 roundings,
    each at most ``u`` times the running sum, whose magnitude never exceeds ``|carried| + sum_j |partial_j| <=
    |carried| + sum_t |terms_t|``: two sums of the same partials differ by at most ``2 P u (|carried| + sum_t |dz'_t|
    |x_t|)``.  The bound holds for any partition of the tokens into the ``P`` chunks (``sum_j |partial_j| <= sum_t
    |terms_t|`` up to the within-chunk rounding, which ``ABS_SLACK`` absorbs); where the two sides form the partials
    of two partitions, ``factor`` adds both partitions' roundings and the within-chunk differences."""
    total = ABS_SLACK * _abs_contraction(i)
    if carried is not None:
        total = total + carried.double().abs()
    return float(factor) * FP32_UNIT_ROUNDOFF * total


# ----------------------------------------------------------------------------------------------------------------
# Inputs and launches


def _make_inputs(
    T: int,
    *,
    hidden: int = HIDDEN,
    mode: str = "unit",
    seed: int = SEED,
    fn_scale: float = 1.0,
    zero_rows=(),
) -> dict:
    """The inputs of one case (``test_cake_mhc_det_train.py``'s generator): ``X, B, dY, dU ~ N(0, 1)`` BF16;
    ``fn ~ 1e-2 N(0, 1)`` (``1e-4`` in the ``tail`` and ``cancel`` regimes, times ``fn_scale``); ``scale ~ 1 +
    0.1 N``; ``base ~ 0.1 N`` (``base[:8] = 20`` in ``tail``); the tokens of ``zero_rows`` zeroed in ``X``; in
    ``cancel`` ``dU := BF16(-(sum_j C_0j dY_j) / pre_0)`` from the FP64 statistics, so the collapse and
    residual-mixing contributions of stream 0 to ``dX`` cancel.  ``X`` / ``dY`` are ``[T, 4, hidden]``, ``B`` /
    ``dU`` ``[T, hidden]`` and ``fn`` ``[24, 4 hidden]``."""
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    fs = {"unit": 1e-2, "tail": 1e-4, "cancel": 1e-4}[mode]
    X = randn(T, STREAMS, hidden, dtype=BF16)
    fn = (randn(MAPPING_ROWS, STREAMS * hidden) * fs).contiguous()
    scale = 1.0 + 0.1 * randn(3)
    base = 0.1 * randn(MAPPING_ROWS)
    if mode == "tail":
        base[:8] = 20.0
    B = randn(T, hidden, dtype=BF16)
    dY = randn(T, STREAMS, hidden, dtype=BF16)
    dU = randn(T, hidden, dtype=BF16)
    if fn_scale != 1.0:
        fn = (fn * float(fn_scale)).contiguous()
    if zero_rows:
        X[list(zero_rows)] = 0
    if mode == "cancel" and T > 0:
        with torch.no_grad():
            dU_new = torch.empty_like(dU)
            for lo in range(0, T, BLOCK_TOKENS):
                hi = min(T, lo + BLOCK_TOKENS)
                u = X[lo:hi].double().reshape(hi - lo, -1)
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
    }


def _forward(i: dict) -> dict:
    """The package's forward prepare and backward prepare of a case: the ``stats`` / ``grad_stats`` records
    ``backward_finish`` takes (copies of the deterministic programs, so both backwards take the same records)."""
    consts = _constants(i)
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"], **consts)
    _dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
    return {"stats": stats, "gstats": gstats}


def _backward(entry, i: dict, prep: dict, **kwargs) -> dict:
    out = entry(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"],
        **_constants(i), **kwargs,
    )  # fmt: skip
    return dict(zip(_OUTPUTS, out, strict=True))


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    return _backward(backward_finish, i, prep, deterministic=False, **kwargs)


def _det_backward(i: dict, prep: dict, **kwargs) -> dict:
    return _backward(det_backward_finish, i, prep, **kwargs)


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _nd_repeats(i: dict, prep: dict, n: int = REPEATS, **kwargs) -> list[dict]:
    """``n`` launches of the non-deterministic backward, each output finite."""
    outs = []
    for repeat in range(n):
        out = _snapshot(_nd_backward(i, prep, **kwargs))
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{k} repeat {repeat}: non-finite values"
        outs.append(out)
    return outs


def _poisoned_outputs(T: int, hidden: int = HIDDEN) -> dict:
    """Caller-provided outputs holding NaN in every element (what the kernels must overwrite)."""
    nan = float("nan")
    return {
        "dX": torch.full((T, STREAMS, hidden), nan, dtype=BF16, device=_device()),
        "dfn": torch.full(
            (MAPPING_ROWS, STREAMS * hidden), nan, dtype=F32, device=_device()
        ),
        "dscale": torch.full((3,), nan, dtype=F32, device=_device()),
        "dbase": torch.full((MAPPING_ROWS,), nan, dtype=F32, device=_device()),
    }


def _nd_workspace_exact(T: int, hidden: int = HIDDEN) -> torch.Tensor:
    """A fresh zeroed non-deterministic backward workspace of exactly ``T`` tokens' layout at ``hidden``."""
    return torch.zeros(
        loader.workspace_bytes("bwd", T, _sm_count(), _arch(), hidden),
        dtype=torch.uint8,
        device=_device(),
    )


def _poison_workspace(ws: torch.Tensor) -> None:
    """Every byte but the ticket region (first for every token count) set to 0xFF."""
    ws[TICKET_BYTES:].fill_(0xFF)


def _assert_ticket_zero(ws: torch.Tensor, what: str) -> None:
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(ws[:TICKET_BYTES]))
    assert nonzero == 0, f"{what}: {nonzero} non-zero bytes in the ticket region"


# ----------------------------------------------------------------------------------------------------------------
# Bit comparisons


_INT_VIEW = {BF16: torch.int16, F32: torch.int32}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(_INT_VIEW[t.dtype])


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements whose bits differ."""
    return int((_bits(a) != _bits(b)).sum())


def _differing_bits(a: torch.Tensor, b: torch.Tensor) -> int:
    """Bits that differ over all elements (FP32)."""
    x = (_bits(a) ^ _bits(b)).to(torch.int64) & 0xFFFFFFFF
    return int(sum(int(((x >> s) & 1).sum()) for s in range(32)))


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """The largest distance in FP32 ulps (representable values between) over the elements."""
    if a.numel() == 0:
        return 0

    def ordered(t: torch.Tensor) -> torch.Tensor:
        bits = _bits(t).to(torch.int64)
        magnitude = bits & 0x7FFFFFFF
        return torch.where(bits < 0, -magnitude, magnitude)

    return int((ordered(a) - ordered(b)).abs().max())


def _assert_same_bits(got: torch.Tensor, want: torch.Tensor, label: str) -> None:
    assert got.dtype == want.dtype and tuple(got.shape) == tuple(want.shape), (
        label,
        got.dtype,
        tuple(got.shape),
        want.dtype,
        tuple(want.shape),
    )
    assert torch.equal(_bits(got), _bits(want)), (
        f"{label}: {_differing(got, want)} of {got.numel()} elements differ from the reference bits"
    )


def _assert_exact_outputs(got: dict, want: dict, label: str) -> None:
    """``dX``, ``dscale``, ``dbase``: bitwise the reference."""
    for k in _EXACT:
        _assert_same_bits(got[k], want[k], f"{k} {label}")


def _assert_dfn_within(
    got: torch.Tensor, want: torch.Tensor, bound: torch.Tensor, label: str
) -> dict:
    """``|got - want| <= bound`` on every ``dfn`` element; returns the figures (differing elements, the largest
    ulp distance, the largest ratio to the bound)."""
    assert got.dtype == F32 and tuple(got.shape) == tuple(want.shape), label
    assert got.shape[0] == MAPPING_ROWS and got.shape[1] % STREAMS == 0, label
    assert torch.isfinite(got).all(), f"{label}: non-finite dfn"
    diff = (got.double() - want.double()).abs()
    over = int((diff > bound).sum())
    ratio = float((diff / bound.clamp_min(torch.finfo(F64).tiny)).max())
    figures = {
        "differ": _differing(got, want),
        "max_ulp": _ulp_distance(got, want),
        "max_ratio": ratio,
    }
    assert over == 0, (
        f"{label}: {over} dfn elements beyond F u (|carried| + sum_t |dz'_t| |x_t|) (largest ratio {ratio:.3e})"
    )
    return figures


# ----------------------------------------------------------------------------------------------------------------
# Accuracy gates (test_cake_mhc_det_train.py's, on the worst of the repeated launches of a draw)


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


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else (0.0 if a == 0 else math.inf)


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
    n_bad = int((err > BF16_ATOL + BF16_RTOL * b.abs()).sum())
    assert n_bad == 0, (
        f"{label}: {n_bad} of {a.numel()} elements outside atol = rtol = 1e-2 (max abs err {float(err.max()):.3e})"
    )


def _worst_row(outs: list[dict], ref: dict, comp: dict, k: str) -> dict:
    """One draw's figures of ``k``: the worst kernel rel-L2 and per-element error over the launches, and the
    IEEE-FP32 comparator's."""
    return {
        "kernel": max(_rel_l2(o[k], ref[k]) for o in outs),
        "kernel_elem": max(_max_rel_elem(o[k], ref[k]) for o in outs),
        "comparator": _rel_l2(comp[k], ref[k]),
        "comparator_elem": _max_rel_elem(comp[k], ref[k]),
    }


def _assert_draw(rows: dict, label: str, *, regime: str) -> None:
    """The absolute bars of one draw on its worst launch (``unit``: rel-L2 <= 1e-5, ``dscale`` / ``dbase`` per element
    <= 1e-4; ``tail``: both <= 1e-3; ``large_logits``: none, the comparator-relative clause of the distribution
    governs); the single-draw comparator ratio is printed as the literal record (the clause of record is the >= 25-draw
    distribution)."""
    met = {True: "met", False: "NOT met"}
    bar = TAIL_REL_MAX if regime == "tail" else FP32_REL_L2_MAX
    elem_bar = TAIL_REL_MAX if regime == "tail" else FP32_ELEM_REL_MAX
    for k, row in rows.items():
        per_elem = k in ("dscale", "dbase")
        ratio = _ratio(row["kernel"], row["comparator"])
        print(
            f"{k} {label}: worst rel-L2 over {REPEATS} launches {row['kernel']:.3e}, IEEE-FP32 comparator "
            f"{row['comparator']:.3e}; single-draw 2x clause {met[ratio <= FP32_COMPARATOR_FACTOR]} (literal record)"
            + (f"; worst per-element {row['kernel_elem']:.3e}" if per_elem else ""),
            flush=True,
        )
        if regime == "large_logits":
            continue
        assert row["kernel"] <= bar, (
            f"{k} {label}: rel-L2 {row['kernel']:.3e} > {bar:.0e}"
        )
        if per_elem:
            assert row["kernel_elem"] <= elem_bar, (
                f"{k} {label}: per-element relative error {row['kernel_elem']:.3e} > {elem_bar:.0e}"
            )


def _assert_parameter_distribution(rows: dict, label: str, *, regime: str) -> None:
    """The clause of record of the parameter gradients over >= 25 draws (``test_cake_mhc_det_train.py``'s), on the
    worst launch of every draw: ``unit``: rel-L2 <= 1e-5 and (``dscale`` / ``dbase``) per element <= 1e-4 on every
    draw, median(kernel rel-L2) <= 2x median(comparator) AND max(kernel) <= 2x max(comparator); ``tail``: rel-L2 and
    per element <= 1e-3 on every draw, the 2x figures printed and not applied (the comparator loses the tail);
    ``large_logits``: the comparator-relative clause alone (rel-L2 medians and maxima; per element <= max(1e-4, 2x
    the comparator's maximum))."""
    for k, r in rows.items():
        n = len(r)
        assert n >= PARAM_CLAUSE_DRAWS, (label, k, n)
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
            f"{k} {label}: {n} draws x {REPEATS} launches; worst kernel rel-L2 median {med_k:.3e} / max {max_k:.3e}, "
            f"comparator {med_c:.3e} / {max_c:.3e}; ratio of medians {_ratio(med_k, med_c):.2f}, of maxima "
            f"{_ratio(max_k, max_c):.2f}; draws whose single-draw ratio exceeds 2x: "
            f"{sum(1 for a, b in zip(ks, cs, strict=True) if a > FP32_COMPARATOR_FACTOR * b)} (literal record)"
            + (f"; per-element max {max(es):.3e}" if per_elem else ""),
            flush=True,
        )
        if regime == "tail":
            assert max_k <= TAIL_REL_MAX, (
                f"{k} {label}: tail-clause rel-L2 max over the draws {max_k:.3e} > {TAIL_REL_MAX:.0e}"
            )
            if per_elem:
                assert max(es) <= TAIL_REL_MAX, (
                    f"{k} {label}: tail-clause per-element max {max(es):.3e} > {TAIL_REL_MAX:.0e}"
                )
            continue
        if regime == "unit":
            assert max_k <= FP32_REL_L2_MAX, (
                f"{k} {label}: rel-L2 max over the draws {max_k:.3e} > {FP32_REL_L2_MAX:.0e}"
            )
            if per_elem:
                assert max(es) <= FP32_ELEM_REL_MAX, (
                    f"{k} {label}: per-element max over the draws {max(es):.3e} > {FP32_ELEM_REL_MAX:.0e}"
                )
        elif per_elem:
            bar = max(FP32_ELEM_REL_MAX, FP32_COMPARATOR_FACTOR * max(ces))
            assert max(es) <= bar, (
                f"{k} {label}: per-element max over the draws {max(es):.3e} > {bar:.3e}"
            )
        assert (
            med_k <= FP32_COMPARATOR_FACTOR * med_c
            and max_k <= FP32_COMPARATOR_FACTOR * max_c
        ), (
            f"{k} {label}: kernel rel-L2 median {med_k:.3e} / max {max_k:.3e} vs the IEEE-FP32 comparator's "
            f"{med_c:.3e} / {max_c:.3e} (> 2x over {n} draws)"
        )


def _parameter_rows(
    T: int, *, mode: str = "unit", fn_scale: float = 1.0, seeds
) -> dict:
    """The per-draw figures (worst of REPEATS launches) of the parameter gradients over ``seeds`` draws; in the
    ``tail`` regime every launch also holds the tail-gradient clause."""
    rows = {k: [] for k in _PARAMETER}
    for seed in seeds:
        i = _make_inputs(T, mode=mode, seed=seed, fn_scale=fn_scale)
        prep = _forward(i)
        outs = _nd_repeats(i, prep)
        ref = _oracle(i)
        comp = _oracle(i, dtype=F32, block_tokens=None)
        for k in _PARAMETER:
            rows[k].append(_worst_row(outs, ref, comp, k))
        if mode == "tail":
            for repeat, out in enumerate(outs):
                _assert_tail_gradients(
                    out, ref, f"T={T} mode=tail seed={seed} launch {repeat}"
                )
        del i, prep, outs, ref, comp
    return rows


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


def _cancellation_band_report(
    dX: torch.Tensor, ref: dict, comp: dict, label: str
) -> dict:
    """One draw's cancellation-band figures (``test_cake_mhc_det_train.py``'s), printed as the literal record."""
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

    b, lt, cb = dist(ulp, band), dist(ulp, literal), dist(culp, band)
    assert lt["n"] > 0, (
        f"{label}: the cancellation inputs produced no element inside the contract's mask"
    )
    strict = b["max"] <= CANCEL_ULP_MAX
    within = b["gt2"] <= 2 * cb["gt2"] and b["max"] <= 2 * cb["max"]
    verdict = {True: "PASS", False: "FAIL"}
    print(
        f"{label}: resolvable band {b['n']} elements, max ulp {b['max']}, > 2 ulp: {b['gt2']} (IEEE-FP32 comparator "
        f"max {cb['max']}, > 2 ulp: {cb['gt2']}); spec-literal mask {lt['n']} elements, max ulp {lt['max']}; strict "
        f"band {verdict[strict]}, single-draw refined gate {verdict[strict or within]} (literal record)",
        flush=True,
    )
    return {"band": b, "comparator_band": cb, "strict": strict, "within": within}


# ----------------------------------------------------------------------------------------------------------------
# Accuracy against the FP64 reference


@pytest.mark.parametrize(("hidden", "T"), _hidden_cases(T_ACCURACY, T_ACCURACY_NEW))
def test_nondeterministic_backward_matches_fp64(hidden: int, T: int) -> None:
    """``deterministic=False`` against the FP64 reference on the worst of REPEATS launches: BF16 ``dX`` with no
    mismatch at atol = rtol = 1e-2 (the launches' ``dX`` bitwise equal); ``dfn`` / ``dscale`` / ``dbase`` rel-L2
    <= 1e-5, ``dscale`` / ``dbase`` per element <= 1e-4."""
    _require_hidden(hidden)
    i = _make_inputs(T, hidden=hidden)
    prep = _forward(i)
    outs = _nd_repeats(i, prep)
    ref = _oracle(i)
    comp = _oracle(i, dtype=F32, block_tokens=None)
    label = f"T={T} H={hidden} P={_slab_parts(T, hidden)} deterministic=False"
    _assert_bf16(outs[0]["dX"], ref["dX32"], f"dX {label}")
    for repeat, out in enumerate(outs[1:], start=1):
        _assert_same_bits(out["dX"], outs[0]["dX"], f"dX {label} launch {repeat}")
    _assert_draw(
        {k: _worst_row(outs, ref, comp, k) for k in _PARAMETER}, label, regime="unit"
    )


@pytest.mark.parametrize("T", T_DISTRIBUTION)
def test_parameter_gradients_within_2x_comparator_over_draws(T: int) -> None:
    """The clause of record of ``dfn`` / ``dscale`` / ``dbase`` over 25 draws of the unit case, each draw's worst
    launch: rel-L2 <= 1e-5 (``dscale`` / ``dbase`` per element <= 1e-4) on every draw; the median and the maximum of
    the rel-L2 <= 2x the IEEE-FP32 comparator's median and maximum."""
    rows = _parameter_rows(T, seeds=range(SEED, SEED + PARAM_CLAUSE_DRAWS))
    _assert_parameter_distribution(
        rows, f"T={T} mode=unit deterministic=False", regime="unit"
    )


@pytest.mark.parametrize("T", (257, 2049))
def test_saturated_sigmoid_tails(T: int) -> None:
    """Saturated sigmoid tails (``scale z + base >= 17``): the fixed draw through the BF16 gate and the tail bars, then
    the tail clause on every launch of 25 draws (tail gradients non-zero and within 1e-3 of FP64; the parameter
    gradients within 1e-3, rel-L2 and per element)."""
    i = _make_inputs(T, mode="tail")
    prep = _forward(i)
    outs = _nd_repeats(i, prep)
    ref = _oracle(i)
    comp = _oracle(i, dtype=F32, block_tokens=None)
    label = f"T={T} mode=tail deterministic=False"
    _assert_bf16(outs[0]["dX"], ref["dX32"], f"dX {label}")
    _assert_draw(
        {k: _worst_row(outs, ref, comp, k) for k in _PARAMETER}, label, regime="tail"
    )
    for repeat, out in enumerate(outs):
        _assert_tail_gradients(out, ref, f"{label} launch {repeat}")
    del i, prep, outs, ref, comp
    rows = _parameter_rows(T, mode="tail", seeds=range(SEED, SEED + PARAM_CLAUSE_DRAWS))
    _assert_parameter_distribution(
        rows, f"T={T} mode=tail deterministic=False", regime="tail"
    )


@pytest.mark.parametrize("T", (257, 2049))
def test_single_cast_cancellation(T: int) -> None:
    """The cancellation band over 64 draws in ``test_cake_mhc_det_train.py``'s aggregate form (the resolvable band
    within 2 ulp on every draw, or the pooled n(> 2 ulp) and the maximum within 2x the IEEE-FP32 comparator's); every
    launch's ``dX`` is bitwise the deterministic backward's, and the parameter gradients of the same draws hold the
    distribution clause on their worst launch."""
    reports = []
    rows = {k: [] for k in _PARAMETER}
    for seed in range(SEED, SEED + CANCEL_CLAUSE_DRAWS):
        i = _make_inputs(T, mode="cancel", seed=seed)
        prep = _forward(i)
        want = _snapshot(_det_backward(i, prep))
        outs = _nd_repeats(i, prep)
        label = f"T={T} mode=cancel seed={seed} deterministic=False"
        for repeat, out in enumerate(outs):
            _assert_same_bits(out["dX"], want["dX"], f"dX {label} launch {repeat}")
        ref = _oracle(i, want_contrib=True)
        comp = _oracle(i, dtype=F32, block_tokens=None)
        if seed == SEED:
            _assert_bf16(outs[0]["dX"], ref["dX32"], f"dX {label}")
        reports.append(
            _cancellation_band_report(
                outs[0]["dX"], ref, comp, f"dX cancellation {label}"
            )
        )
        for k in _PARAMETER:
            rows[k].append(_worst_row(outs, ref, comp, k))
        del i, prep, want, outs, ref, comp
    total_k = sum(r["band"]["gt2"] for r in reports)
    total_c = sum(r["comparator_band"]["gt2"] for r in reports)
    max_k = max(r["band"]["max"] for r in reports)
    max_c = max(r["comparator_band"]["max"] for r in reports)
    strict_every_draw = max_k <= CANCEL_ULP_MAX
    pooled = total_k <= 2 * total_c and max_k <= 2 * max_c
    print(
        f"dX cancellation T={T} deterministic=False: {len(reports)} draws; kernel total n(> 2 ulp) {total_k}, max ulp "
        f"{max_k}; IEEE-FP32 comparator {total_c} / {max_c}; aggregate verdict "
        f"{'PASS' if strict_every_draw or pooled else 'FAIL'}",
        flush=True,
    )
    assert strict_every_draw or pooled, (
        f"dX cancellation T={T}: over {len(reports)} draws the resolvable band has {total_k} elements > 2 ulp (max "
        f"{max_k}) against the IEEE-FP32 comparator's {total_c} (max {max_c}): beyond 2x"
    )
    _assert_parameter_distribution(
        rows, f"T={T} mode=cancel deterministic=False", regime="unit"
    )


# ----------------------------------------------------------------------------------------------------------------
# Against the deterministic backward


@pytest.mark.parametrize(("hidden", "T"), _hidden_cases(T_COMPARE, T_COMPARE_NEW))
def test_matches_the_deterministic_backward(hidden: int, T: int) -> None:
    """On the same inputs as the deterministic backward: ``dX``, ``dscale``, ``dbase`` bitwise; every ``dfn`` element
    within ``F u sum_t |dz'_t| |x_t|`` (``F`` = :func:`_dfn_factor`: ``2 P`` under ``dynr``, ``2 max(P, P_k7) + 10``
    where the column slab runs ``lpt`` at ``T`` on this device).  The launches alternate between the default workspace
    with fresh outputs and one caller workspace (non-ticket bytes 0xFF before each launch) with NaN-filled outputs;
    the largest ratio to the bound and the differing bits are printed."""
    _require_hidden(hidden)
    i = _make_inputs(T, hidden=hidden, seed=SEED + 1)
    prep = _forward(i)
    want = _snapshot(_det_backward(i, prep))
    parts, lpt = _slab_parts(T, hidden), _lpt_runs(T, hidden)
    bound = _dfn_bound(i, _dfn_factor(T, hidden))
    ws = pkg.cake_mhc_train_backward_workspace(
        T, _device(), hidden=hidden, deterministic=False
    )
    figures = []
    for repeat in range(REPEATS):
        label = f"T={T} H={hidden} P={parts} launch {repeat}"
        kwargs = {}
        if repeat % 2:
            _poison_workspace(ws)
            kwargs = {"workspace": ws, **_poisoned_outputs(T, hidden)}
        got = _snapshot(_nd_backward(i, prep, **kwargs))
        if repeat % 2:
            _assert_ticket_zero(ws, label)
        _assert_exact_outputs(got, want, f"{label} vs deterministic")
        figures.append(
            _assert_dfn_within(
                got["dfn"], want["dfn"], bound, f"dfn {label} vs deterministic"
            )
        )
    print(
        f"dfn T={T} H={hidden} P={parts}: vs the deterministic backward over {REPEATS} launches: largest |nd - det| / "
        f"bound {max(f['max_ratio'] for f in figures):.3e}, elements differing up to "
        f"{max(f['differ'] for f in figures)} of {want['dfn'].numel()} "
        f"(largest {max(f['max_ulp'] for f in figures)} ulp)"
        + (
            f"; P = 1: equal to the deterministic dfn on every launch: {all(f['max_ulp'] == 0 for f in figures)}"
            if parts == 1 and not lpt
            else ""
        ),
        flush=True,
    )


@pytest.mark.parametrize(("hidden", "T"), _hidden_cases((2049, 16263), (2049, 16263)))
def test_run_to_run(hidden: int, T: int) -> None:
    """Repeated launches on the same inputs: ``dX``, ``dscale``, ``dbase`` identical bits; the ``dfn`` elements and
    bits that differ from the first launch are counted and printed (the summation order of the atomic reductions may
    change; never asserted either way)."""
    _require_hidden(hidden)
    i = _make_inputs(T, hidden=hidden, seed=SEED + 2)
    prep = _forward(i)
    outs = _nd_repeats(i, prep, n=RUN_TO_RUN_REPEATS)
    tallies = []
    for repeat, out in enumerate(outs[1:], start=1):
        _assert_exact_outputs(
            out, outs[0], f"T={T} H={hidden} launch {repeat} vs launch 0"
        )
        tallies.append(
            (
                _differing(out["dfn"], outs[0]["dfn"]),
                _differing_bits(out["dfn"], outs[0]["dfn"]),
                _ulp_distance(out["dfn"], outs[0]["dfn"]),
            )
        )
    print(
        f"dfn T={T} H={hidden} P={_slab_parts(T, hidden)} run to run (launches 1..{RUN_TO_RUN_REPEATS - 1} vs launch "
        f"0; elements, bits, largest ulp distance): {tallies}",
        flush=True,
    )


@pytest.fixture
def forced_lpt():
    """``forced[T] = (P, rpc, p_big, rpc_small)``: the geometry ``cake_jit.lpt_chunks`` returns at ``T`` instead of
    the host rule's (every other token count keeps the rule), with the package's route-plan caches cleared on entry
    and exit (a plan binds the selection and the chunk scalars of its token count)."""
    forced: dict[int, tuple[int, int, int, int]] = {}
    rule = loader.lpt_chunks
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            loader,
            "lpt_chunks",
            lambda T, sm, arch: forced[T] if T in forced else rule(T, sm, arch),
        )
        pkg._PLANS.clear()
        pkg._FAST_ROUTES.clear()
        try:
            yield forced
        finally:
            pkg._PLANS.clear()
            pkg._FAST_ROUTES.clear()


# (T, forced geometry or None for the screen's best geometry on this device, accumulate)
_FORCED_CASES = (
    *((2049, geometry, False) for geometry in LPT_FORCED),
    (257, (3, 128, 2, 64), False),  # 5 tiles: 2 x 2 tiles, then one token
    (2049, LPT_FORCED[2], True),
    (2049, LPT_FORCED[6], True),
    (16263, None, False),
)


def _forced_id(case: tuple) -> str:
    T, geometry, accumulate = case
    shape = "screen" if geometry is None else "x".join(str(v) for v in geometry)
    return f"T{T}-{shape}" + ("-accumulate" if accumulate else "")


@pytest.mark.parametrize(
    ("T", "geometry", "accumulate"),
    _FORCED_CASES,
    ids=[_forced_id(case) for case in _FORCED_CASES],
)
def test_two_size_schedule_at_forced_geometries(
    forced_lpt, monkeypatch, T: int, geometry, accumulate: bool
) -> None:
    """The column slab's ``lpt`` record at a forced geometry (``geometry`` None: the screen's best two-size geometry
    at ``T`` on this device, ``_lpt_plan`` at ``o_tiles = 1``, also where the table of record is ``None``): the
    selection's column slab is the ``lpt`` record, the only column slab ``backward_finish`` loads, launched over
    ``min(SMs, 64 P)`` CTAs with ``slab_plan`` = the geometry.  On two launches -- fresh outputs on the default
    workspace, then NaN outputs (the carried-in values under ``accumulate``) on a caller workspace whose non-ticket
    bytes are 0xFF -- ``dX``, ``dscale``, ``dbase`` are bitwise the deterministic backward's, every ``dfn`` element
    lies within ``_dfn_factor(T) u (|carried| + sum_t |dz'_t| |x_t|)`` and the ticket region is zero; the largest
    ratio to the bound is printed."""
    arch, sm, index = _arch(), _sm_count(), _device().index
    if geometry is None:
        plan = loader._lpt_plan(T, sm, 1.0)
        assert plan is not None, (T, sm)
        geometry = plan[2]
    forced_lpt[T] = geometry
    P = geometry[0]
    assert loader.slab_plan(T, sm, arch) == dict(
        zip(LPT_HOST_KEYS, geometry, strict=True)
    )
    selection = loader.selected_modules(arch, HIDDEN, "bwd_finish", rows=T, sm_count=sm)
    assert selection[-1] == LPT and SLAB not in selection, selection
    assert loader.launch_grid(loader.MODULES[LPT], T, device_index=index) == (
        min(sm, 64 * P),
        1,
        1,
    )
    i = _make_inputs(T, seed=SEED + 71)
    prep = _forward(i)
    carried = None
    if accumulate:
        g = torch.Generator(device=_device()).manual_seed(SEED + 72)
        carried = {
            "dfn": torch.randn((MAPPING_ROWS, N), generator=g, device=_device()) * 1e-3,
            "dscale": torch.randn((3,), generator=g, device=_device()) * 1e-3,
            "dbase": torch.randn((MAPPING_ROWS,), generator=g, device=_device()) * 1e-3,
        }

    def outputs(poisoned: bool) -> dict:
        if carried is not None:
            return {k: v.clone() for k, v in carried.items()}
        return _poisoned_outputs(T) if poisoned else {}

    want = _snapshot(_det_backward(i, prep, accumulate=accumulate, **outputs(False)))
    factor = _dfn_factor(T)
    bound = _dfn_bound(i, factor, carried=None if carried is None else carried["dfn"])
    ws = pkg.cake_mhc_train_backward_workspace(T, _device(), deterministic=False)
    loaded: list[str] = []
    load = loader.load

    def recording_load(name: str, record_arch: str):
        loaded.append(name)
        return load(name, record_arch)

    monkeypatch.setattr(loader, "load", recording_load)
    figures = []
    for launch in range(2):
        label = f"T={T} lpt {geometry} accumulate={accumulate} launch {launch}"
        kwargs = {"accumulate": accumulate, **outputs(launch == 1)}
        if launch:
            _poison_workspace(ws)
            kwargs["workspace"] = ws
        got = _snapshot(_nd_backward(i, prep, **kwargs))
        if launch:
            _assert_ticket_zero(ws, label)
        _assert_exact_outputs(got, want, f"{label} vs deterministic")
        figures.append(
            _assert_dfn_within(
                got["dfn"], want["dfn"], bound, f"dfn {label} vs deterministic"
            )
        )
    assert loaded == list(selection), loaded
    print(
        f"dfn T={T} lpt {geometry} (P_k7 {_parts(T)}, F {factor:g}) accumulate={accumulate}: largest |nd - det| / "
        f"bound over two launches {max(f['max_ratio'] for f in figures):.3e}, elements differing up to "
        f"{max(f['differ'] for f in figures)} of {MAPPING_ROWS * N}",
        flush=True,
    )


# ----------------------------------------------------------------------------------------------------------------
# The forward entry points and the launches


def _leaves(i: dict) -> list:
    return [
        i[k].detach().clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")
    ]


def _grads(leaves: list) -> list:
    return [t.grad for t in leaves]


def _forward_outputs(i: dict, **workspace) -> dict:
    """The forward entry points and the backward prepare of this package on ``i`` (``workspace``: the forward
    workspace keyword of the forward prepare and the statistics replay)."""
    X, fn, scale, base = (i[k] for k in ("X", "fn", "scale", "base"))
    U, stats = forward_prepare(X, fn, scale, base, **workspace)
    dB, gstats = backward_prepare(i["dY"], i["B"], X, stats)
    return {
        "U": U,
        "stats": stats,
        "replay": forward_stats(X, fn, scale, base, **workspace),
        "Y": forward_finish(X, i["B"], stats),
        "dB": dB,
        "gstats": gstats,
    }


def _reference_forward_outputs(i: dict) -> dict:
    """:func:`_forward_outputs` of the deterministic package (its default workspaces)."""
    X, fn, scale, base = (i[k] for k in ("X", "fn", "scale", "base"))
    U, stats = det.cake_mhc_det_train_forward_prepare(X, fn, scale, base)
    dB, gstats = det.cake_mhc_det_train_backward_prepare(i["dY"], i["B"], X, stats)
    return {
        "U": U,
        "stats": stats,
        "replay": det.cake_mhc_det_train_forward_stats(X, fn, scale, base),
        "Y": det.cake_mhc_det_train_forward_finish(X, i["B"], stats),
        "dB": dB,
        "gstats": gstats,
    }


@pytest.mark.parametrize(
    ("hidden", "T"), _hidden_cases((1, 7, 129, 2049, 16263), (1, 7, 129, 2049, 16263))
)
def test_forward_entry_points_are_the_deterministic_programs(
    hidden: int, T: int
) -> None:
    """The package's forward prepare, statistics replay, residual mixing and backward prepare (its copies of the
    deterministic programs) give the deterministic package's bits, on the default workspaces and on a caller forward
    workspace of this package (the deterministic forward layout: the same size, zero-initialized)."""
    _require_hidden(hidden)
    i = _make_inputs(T, hidden=hidden, seed=SEED + 12)
    want = _snapshot(_reference_forward_outputs(i))
    nbytes = pkg.cake_mhc_train_forward_workspace_bytes(
        T, _device(), hidden=hidden, deterministic=False
    )
    assert nbytes == det.cake_mhc_det_train_forward_workspace_bytes(
        T, _device(), hidden=hidden
    )
    ws = pkg.cake_mhc_train_forward_workspace(
        T, _device(), hidden=hidden, deterministic=False
    )
    assert ws.numel() == nbytes and int(torch.count_nonzero(ws)) == 0
    for workspace in (None, ws):
        got = _forward_outputs(i, workspace=workspace)
        for k, v in want.items():
            _assert_same_bits(
                got[k],
                v,
                f"{k} T={T} H={hidden} caller workspace={workspace is not None}",
            )


@pytest.mark.parametrize(
    ("hidden", "T"),
    [pytest.param(h, T, id=f"H{h}-{T}") for h in HIDDEN_SIZES[1:] for T in T_BENCH],
)
def test_benchmark_token_counts_at_the_new_hidden_sizes(hidden: int, T: int) -> None:
    """At hidden 5120 and 7168 on the 19 benchmark token counts (their copies of the deterministic programs are
    renders of their own; H 4096 is the cases above): the forward entry points and the backward prepare give the
    deterministic package's bits, and one ``backward_finish`` gives its ``dX`` / ``dscale`` / ``dbase`` bits and a
    ``dfn`` within the bound of :func:`test_matches_the_deterministic_backward`, the ticket region of a caller
    workspace back to zero."""
    _require_hidden(hidden)
    i = _make_inputs(T, hidden=hidden, seed=SEED + 14)
    want = _snapshot(_reference_forward_outputs(i))
    got = _snapshot(_forward_outputs(i))
    label = f"T={T} H={hidden}"
    for k, v in want.items():
        _assert_same_bits(got[k], v, f"{k} {label}")
    prep = {"stats": got["stats"], "gstats": got["gstats"]}
    want = _snapshot(_det_backward(i, prep))
    ws = pkg.cake_mhc_train_backward_workspace(
        T, _device(), hidden=hidden, deterministic=False
    )
    got = _snapshot(_nd_backward(i, prep, workspace=ws))
    _assert_ticket_zero(ws, label)
    _assert_exact_outputs(got, want, f"{label} vs deterministic")
    figures = _assert_dfn_within(
        got["dfn"],
        want["dfn"],
        _dfn_bound(i, _dfn_factor(T, hidden)),
        f"dfn {label} vs deterministic",
    )
    print(
        f"dfn {label} P={_parts(T, hidden)}: largest |nd - det| / bound {figures['max_ratio']:.3e}",
        flush=True,
    )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES, ids=lambda h: f"H{h}")
def test_launches_are_this_packages_records(monkeypatch, hidden: int) -> None:
    """Every module the package's entry points load (both forwards, the residual mixing, both backward passes, the
    autograd connection with the statistics saved and replayed, a run-time Sinkhorn count) is one of this package's
    records for the local architecture and the call's hidden size, and the deterministic package launches nothing
    (its launch paths refuse); ``backward_finish`` loads exactly its ``bwd_finish`` selection, in launch order: one
    token-owner record, the atomic fn split of the architecture and size and the column slab's form of the call
    (``bwd_slab_atomic_program``; the size's ``dynr`` record at 5120 and 7168)."""
    _require_hidden(hidden)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the deterministic package launched a kernel")

    monkeypatch.setattr(det, "_launch_route", refuse)
    monkeypatch.setattr(det, "_run_route", refuse)
    # Fresh plan caches: every route below resolves (and loads) its records again.
    monkeypatch.setattr(pkg, "_PLANS", {})
    monkeypatch.setattr(pkg, "_FAST_ROUTES", {})
    loaded: list[tuple[str, str]] = []
    load = loader.load

    def recording_load(name: str, arch: str):
        loaded.append((name, arch))
        return load(name, arch)

    monkeypatch.setattr(loader, "load", recording_load)
    T, arch, sm = 2049, _arch(), _sm_count()
    i = _make_inputs(T, hidden=hidden, seed=SEED + 13)
    prep = _forward(i)
    forward_stats(i["X"], i["fn"], i["scale"], i["base"])
    forward_finish(i["X"], i["B"], prep["stats"])
    start = len(loaded)
    out = _nd_backward(i, prep)
    selection = loader.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
    assert [name for name, _ in loaded[start:]] == list(selection), loaded[start:]
    assert loader.MODULES[selection[0]]["role"] == "bwd_token", selection
    assert selection[1] == (
        f"cake_mhc_train_fn_split_atomic__{loader.fn_split_program(arch, 1, hidden)}_dep"
        + _suffix(hidden)
    )
    program = loader.bwd_slab_atomic_program(T, sm, arch, hidden)
    assert program == "dynr" or hidden == HIDDEN, (program, hidden)
    assert selection[-1] == {"dynr": SLAB, "lpt": LPT}[program] + _suffix(hidden), (
        program,
        selection,
    )
    assert len(selection) == 3, selection
    assert all(torch.isfinite(v).all() for v in out.values())
    for save_stats in (True, False):
        leaves = _leaves(i)
        U, handle = _nd_prepare(*leaves, save_stats=save_stats)
        Y = _nd_finish(leaves[0], (U.float() + i["B"].float()).to(BF16), handle)
        (Y.float() * i["dY"].float()).sum().backward()
        assert all(torch.isfinite(t.grad).all() for t in leaves), save_stats
    counted = {**i, "sinkhorn_iters": 7}
    out = _nd_backward(counted, _forward(counted))
    assert all(torch.isfinite(v).all() for v in out.values())
    assert loaded
    for name, record_arch in loaded:
        assert name in loader.MODULES and name.startswith("cake_mhc_train_"), name
        assert record_arch == arch and arch in loader.MODULES[name]["arches"], name
        assert loader.MODULES[name]["hidden"] == hidden, name
    assert selection[-1] in {name for name, _ in loaded}


# ----------------------------------------------------------------------------------------------------------------
# PyTorch's deterministic algorithms (the alert of the unordered reduction; the facade's fallback to the
# deterministic package is test_cake_mhc_train_api.py's)


def _our_alerts(caught) -> list[str]:
    """The entry names of this package's alerts among the recorded warnings, in order (each a ``UserWarning``)."""
    names = []
    for w in caught:
        message = str(w.message)
        if _ALERT not in message:
            continue
        name = message.split(f" {_ALERT}")[0]
        if name.startswith(("cake_mhc_train", "CakeMHCTrain")):
            assert w.category is UserWarning, (name, w.category)
            names.append(name)
    return names


def test_policy_strict_alerts_the_direct_backward_only(
    deterministic_algorithms,
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``: the forward prepare, the statistics replay, the residual
    mixing, the backward prepare and the workspace entry points run without an alert (warnings are errors) and give
    the bits they give with the mode off; ``backward_finish`` raises PyTorch's alert before any launch (the caller's
    outputs and workspace keep their bytes) and, once the mode is off, the same call runs."""
    T = 257
    i = _make_inputs(T, seed=SEED + 6)
    want_prep = _snapshot(_forward(i))
    want_replay = forward_stats(i["X"], i["fn"], i["scale"], i["base"]).clone()
    want = _snapshot(_det_backward(i, want_prep))
    bound = _dfn_bound(i, _dfn_factor(T))
    ws = pkg.cake_mhc_train_backward_workspace(T, _device(), deterministic=False)
    deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prep = _snapshot(_forward(i))
        replay = forward_stats(i["X"], i["fn"], i["scale"], i["base"]).clone()
        forward_finish(i["X"], i["B"], prep["stats"])
        assert pkg.cake_mhc_train_backward_workspace_bytes(
            T, _device(), deterministic=False
        ) == loader.workspace_bytes_upto("bwd", T, _sm_count(), _arch())
        pkg.cake_mhc_train_forward_workspace(T, _device(), deterministic=False)
    for k in ("stats", "gstats"):
        _assert_same_bits(prep[k], want_prep[k], f"{k} under the strict mode")
    _assert_same_bits(replay, want_replay, "statistics replay under the strict mode")
    outputs = {k: v.fill_(3.0) for k, v in _poisoned_outputs(T).items()}
    _poison_workspace(ws)
    before = ws.clone()
    with pytest.raises(RuntimeError) as info:
        _nd_backward(i, prep, workspace=ws, **outputs)
    assert str(info.value).startswith(
        f"cake_mhc_train_backward_finish {_ALERT}, but you set "
        "'torch.use_deterministic_algorithms(True)'"
    ), str(info.value)
    torch.cuda.synchronize()
    for k, v in outputs.items():
        assert bool((v == 3.0).all()), f"{k}: a refused call wrote its output"
    assert torch.equal(ws, before), "a refused call wrote its workspace"
    deterministic_algorithms(False)
    got = _snapshot(_nd_backward(i, prep, workspace=ws, **outputs))
    _assert_ticket_zero(ws, "the call after the strict-mode refusal")
    _assert_exact_outputs(got, want, "the call after the strict-mode refusal")
    _assert_dfn_within(
        got["dfn"], want["dfn"], bound, "dfn of the call after the strict-mode refusal"
    )


def test_policy_warn_only_warns_once_and_runs_the_nondeterministic_backward(
    deterministic_algorithms,
) -> None:
    """Under ``torch.use_deterministic_algorithms(True, warn_only=True)`` every ``backward_finish`` call warns once
    (PyTorch's alert as a ``UserWarning``) and runs the non-deterministic backward -- the package never falls back:
    on the default workspace, on a workspace of the deterministic package's sizing and on one of this package's
    sizing (which the deterministic backward refuses), ``dX`` / ``dscale`` / ``dbase`` are the deterministic bits,
    ``dfn`` is within the bound and the ticket region is zero."""
    T = 2049
    i = _make_inputs(T, seed=SEED + 7)
    prep = _forward(i)
    want = _snapshot(_det_backward(i, prep))
    bound = _dfn_bound(i, _dfn_factor(T))
    det_sized = det.cake_mhc_det_train_backward_workspace(T, _device())
    nd_sized = _nd_workspace_exact(T)
    with pytest.raises(ValueError, match="workspace holds"):
        _det_backward(i, prep, workspace=nd_sized)
    deterministic_algorithms(True, warn_only=True)
    for name, workspace in (
        ("default", None),
        ("deterministic-sized", det_sized),
        ("nd-sized", nd_sized),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = _snapshot(_nd_backward(i, prep, workspace=workspace))
        assert _our_alerts(caught) == ["cake_mhc_train_backward_finish"], (
            name,
            [str(w.message) for w in caught],
        )
        _assert_exact_outputs(got, want, f"warn_only workspace={name}")
        _assert_dfn_within(
            got["dfn"], want["dfn"], bound, f"dfn warn_only workspace={name}"
        )
        if workspace is not None:
            _assert_ticket_zero(workspace, f"warn_only workspace={name}")


@pytest.mark.parametrize("save_stats", (True, False))
def test_policy_prepare_node_alerts_its_backward(
    deterministic_algorithms, save_stats: bool
) -> None:
    """The prepare node's forward, the finish node and (``save_stats=False``) the statistics replay never alert; the
    prepare node's backward raises PyTorch's alert named ``CakeMHCTrainPrepareFunction.backward`` whenever the
    deterministic algorithms are on when it runs -- recorded under the strict mode, or recorded with the mode off and
    the mode enabled before ``backward()``.  Under ``warn_only`` that backward warns once (``backward_finish`` adds
    no second alert) and runs the non-deterministic backward: ``Y`` and the ``X`` / ``scale`` / ``base`` gradients
    bitwise the deterministic autograd's, the ``fn`` gradient within the bound, also on a caller backward workspace
    of this package's sizing."""
    T = 257
    i = _make_inputs(T, seed=SEED + 8)
    bound = _connection_bound(i, T)

    def graph(api, **kwargs) -> tuple:
        prepare, finish = api
        leaves = _leaves(i)
        U, handle = prepare(*leaves, save_stats=save_stats, **kwargs)
        Y = finish(leaves[0], (U.float() + i["B"].float()).to(BF16), handle)
        return leaves, Y, (Y.float() * i["dY"].float()).sum()

    leaves_det, Y_det, loss = graph(DET_API)
    loss.backward()
    refusal = f"CakeMHCTrainPrepareFunction.backward {_ALERT}"
    # recorded under the strict mode: the forward runs without an alert, the backward refuses
    deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _leaves_strict, Y, loss = graph(ND_API)
    _assert_same_bits(Y.detach(), Y_det.detach(), "Y recorded under the strict mode")
    with pytest.raises(RuntimeError, match=refusal):
        loss.backward()
    deterministic_algorithms(False)
    # recorded with the mode off, the strict mode enabled before backward()
    _leaves_off, _Y, loss = graph(ND_API)
    deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match=refusal):
        loss.backward()
    deterministic_algorithms(False)
    for workspace in (None, _nd_workspace_exact(T)):
        leaves, Y, loss = graph(ND_API, backward_workspace=workspace)
        deterministic_algorithms(True, warn_only=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss.backward()
        deterministic_algorithms(False)
        what = f"warn_only at backward, save_stats={save_stats}, workspace={workspace is not None}"
        assert _our_alerts(caught) == ["CakeMHCTrainPrepareFunction.backward"], (
            what,
            [str(w.message) for w in caught],
        )
        _assert_same_bits(Y.detach(), Y_det.detach(), f"Y {what}")
        _assert_grads(_grads(leaves), _grads(leaves_det), bound, what)


# ----------------------------------------------------------------------------------------------------------------
# accumulate


@pytest.mark.parametrize(("hidden", "T"), _hidden_cases((1000, 4096), (4096,)))
def test_accumulate_four_times(hidden: int, T: int) -> None:
    """Four consecutive ``accumulate=True`` passes on four input sets into caller buffers that start from non-zero
    values, beside the deterministic chain on the same values: the returned tensors are the caller's buffers; ``dX``,
    ``dscale``, ``dbase`` bitwise at every step; ``dfn`` within the sum of the per-pass bounds (each with the larger
    carried-in magnitude of the two chains)."""
    _require_hidden(hidden)
    n = STREAMS * hidden
    g = torch.Generator(device=_device()).manual_seed(SEED + 7919)
    carried = {
        "dfn": torch.randn((MAPPING_ROWS, n), generator=g, device=_device()) * 1e-3,
        "dscale": torch.randn((3,), generator=g, device=_device()) * 1e-3,
        "dbase": torch.randn((MAPPING_ROWS,), generator=g, device=_device()) * 1e-3,
    }
    acc_nd = {k: v.clone() for k, v in carried.items()}
    acc_det = {k: v.clone() for k, v in carried.items()}
    parts, factor = _slab_parts(T, hidden), _dfn_factor(T, hidden)
    bound = torch.zeros((MAPPING_ROWS, n), dtype=F64, device=_device())
    figures = []
    for step in range(4):
        i = _make_inputs(T, hidden=hidden, seed=SEED + 100 + step)
        prep = _forward(i)
        before = torch.maximum(
            acc_nd["dfn"].double().abs(), acc_det["dfn"].double().abs()
        )
        got = _nd_backward(i, prep, accumulate=True, **acc_nd)
        want = _det_backward(i, prep, accumulate=True, **acc_det)
        for k in _PARAMETER:
            assert got[k] is acc_nd[k] and want[k] is acc_det[k], (k, step)
        bound = bound + _dfn_bound(i, factor, carried=before)
        label = f"accumulate x4 T={T} H={hidden} P={parts} step {step}"
        _assert_exact_outputs(got, want, label)
        figures.append(
            _assert_dfn_within(acc_nd["dfn"], acc_det["dfn"], bound, f"dfn {label}")
        )
        del i, prep
    print(
        f"dfn accumulate x4 T={T} H={hidden} P={parts}: largest |nd - det| / bound per step "
        f"{[round(f['max_ratio'], 4) for f in figures]}, elements differing {[f['differ'] for f in figures]}",
        flush=True,
    )


def test_accumulate_needs_the_buffers() -> None:
    i = _make_inputs(8, seed=SEED + 11)
    prep = _forward(i)
    zeros = {
        "dfn": torch.zeros((MAPPING_ROWS, N), dtype=F32, device=_device()),
        "dscale": torch.zeros((3,), dtype=F32, device=_device()),
        "dbase": torch.zeros((MAPPING_ROWS,), dtype=F32, device=_device()),
    }
    with pytest.raises(ValueError, match="accumulate=True"):
        _nd_backward(i, prep, accumulate=True)
    for missing in _PARAMETER:
        buffers = {k: v for k, v in zeros.items() if k != missing}
        with pytest.raises(ValueError, match="accumulate=True"):
            _nd_backward(i, prep, accumulate=True, **buffers)


# ----------------------------------------------------------------------------------------------------------------
# Autograd and activation checkpointing


def _branch(name: str, i: dict):
    if name == "residual":  # dB/dU = 1: the branch backward hands dB on as dU
        return lambda U: (U.float() + i["B"].float()).to(BF16)
    W = i["W"]
    return lambda U: U @ W  # a BF16 linear layer


def _connection(prepare, finish, branch, *, save_stats: bool, record=None, **kwargs):
    def connection(X, fn, scale, base):
        U, handle = prepare(X, fn, scale, base, save_stats=save_stats, **kwargs)
        if record is not None:
            U.retain_grad()
            record["U"] = U
        B = branch(U)
        if record is not None:
            record["B"] = B
        return finish(X, B, handle)

    return connection


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("checkpointed", (False, True))
@pytest.mark.parametrize("branch", ("residual", "linear"))
def test_autograd_matches_the_deterministic_gradients(
    T: int, save_stats: bool, checkpointed: bool, branch: str
) -> None:
    """``cake_mhc_train_prepare(deterministic=False)`` / ``cake_mhc_train_finish`` around an external branch, plain or
    inside a non-reentrant ``torch.utils.checkpoint``, against the plain deterministic autograd on the same inputs:
    ``Y`` and the gradients of ``X``, ``scale`` and ``base`` bitwise; the ``fn`` gradient within the ``dfn`` bound
    (evaluated on the branch output and the branch's ``dU`` of the deterministic graph)."""
    i = _make_inputs(T, seed=SEED + 31)
    g = torch.Generator(device=_device()).manual_seed(SEED + 32)
    i["W"] = (torch.randn((HIDDEN, HIDDEN), generator=g, device=_device()) / 64.0).to(
        BF16
    )
    record: dict = {}
    reference = _connection(
        det.cake_mhc_det_train_prepare,
        det.cake_mhc_det_train_finish,
        _branch(branch, i),
        save_stats=save_stats,
        record=record,
    )
    leaves_det = _leaves(i)
    Y_det = reference(*leaves_det)
    (Y_det.float() * i["dY"].float()).sum().backward()
    assert record["U"].grad is not None and record["U"].grad.dtype == BF16
    bound = _dfn_bound(
        {**i, "B": record["B"].detach(), "dU": record["U"].grad.detach()},
        _dfn_factor(T),
    )
    connection = _connection(
        _nd_prepare,
        _nd_finish,
        _branch(branch, i),
        save_stats=save_stats,
    )
    leaves = _leaves(i)
    if checkpointed:
        Y = checkpoint(connection, *leaves, use_reentrant=False)
    else:
        Y = connection(*leaves)
    (Y.float() * i["dY"].float()).sum().backward()
    what = f"T={T} save_stats={save_stats} checkpointed={checkpointed} branch={branch}"
    _assert_same_bits(Y.detach(), Y_det.detach(), f"Y {what}")
    assert leaves[0].grad.dtype == BF16
    for index, name in ((0, "X"), (2, "scale"), (3, "base")):
        _assert_same_bits(
            leaves[index].grad, leaves_det[index].grad, f"grad {name} {what}"
        )
    figures = _assert_dfn_within(
        leaves[1].grad, leaves_det[1].grad, bound, f"grad fn {what}"
    )
    print(
        f"grad fn {what}: largest |nd - det| / bound {figures['max_ratio']:.3e}",
        flush=True,
    )


# ----------------------------------------------------------------------------------------------------------------
# The autograd connection with deterministic=False: test_cake_mhc_det_train.py's checkpoint and lifetime cases (the
# saved-tensor and autograd-edge contract of FlashInfer issues #7039 / #7547: nothing of a connection stays alive
# once its backward has run, nothing beyond the outputs under a checkpoint)


def _nd_prepare(*args, **kwargs):
    return pkg.cake_mhc_train_prepare(*args, deterministic=False, **kwargs)


def _nd_finish(*args, **kwargs):
    return pkg.cake_mhc_train_finish(*args, deterministic=False, **kwargs)


# (prepare, finish) of each package: a handle goes to the finish of the package that made it.
ND_API = (_nd_prepare, _nd_finish)
DET_API = (det.cake_mhc_det_train_prepare, det.cake_mhc_det_train_finish)


def _connection_bound(i: dict, T: int) -> torch.Tensor:
    """The ``dfn`` bound of one connection whose branch is ``B = U + B_0`` (its backward hands ``dB`` on as ``dU``)
    under the loss gradient ``dY``: ``B`` and ``dU`` from the deterministic direct entry points."""
    U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"])
    B = (U.float() + i["B"].float()).to(BF16)
    dB, _gstats = backward_prepare(i["dY"], B, i["X"], stats)
    return _dfn_bound({**i, "B": B, "dU": dB}, _dfn_factor(T, _hidden_of(i)))


def _assert_grads(got, want, bound: torch.Tensor, label: str) -> dict:
    """The leaf gradients ``(X, fn, scale, base)`` of one connection: ``X``, ``scale``, ``base`` bitwise; ``fn``
    within the ``dfn`` bound."""
    for index, name in ((0, "X"), (2, "scale"), (3, "base")):
        _assert_same_bits(got[index], want[index], f"grad {name} {label}")
    return _assert_dfn_within(got[1], want[1], bound, f"grad fn {label}")


def _plain_grads(i: dict, api, *, save_stats: bool = True) -> list:
    """The leaf gradients of one plain connection of ``api`` (branch ``B = U + B_0``, loss ``sum(Y * dY)``)."""
    prepare, finish = api
    leaves = _leaves(i)
    U, handle = prepare(*leaves, save_stats=save_stats)
    B = (U.float() + i["B"].float()).to(BF16)
    Y = finish(leaves[0], B, handle)
    (Y.float() * i["dY"].float()).sum().backward()
    return _grads(leaves)


def _live_bytes() -> int | None:
    """Bytes the live CUDA tensors asked the caching allocator for (``requested_bytes``; see
    ``test_cake_mhc_det_train.py``); ``None`` under another allocator backend, where the retention checks are
    skipped and the gradient checks still run."""
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


@pytest.mark.parametrize(("hidden", "T"), _hidden_cases((129, 2049), (129, 2049)))
def test_checkpoint_replay(hidden: int, T: int) -> None:
    """``forward_stats`` is bitwise the ``forward_prepare`` record, and the ``deterministic=False`` gradients with the
    statistics saved (``save_stats=True``) and replayed in the backward (``False``, the checkpoint path) agree with
    each other and with the deterministic gradients: ``X`` / ``scale`` / ``base`` bitwise, ``fn`` within the bound."""
    _require_hidden(hidden)
    i = _make_inputs(T, hidden=hidden, seed=SEED + 11)
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"])
    replay = forward_stats(i["X"], i["fn"], i["scale"], i["base"])
    assert torch.equal(stats, replay), (
        "forward_stats is not bitwise the forward_prepare record"
    )
    bound = _connection_bound(i, T)
    want = _plain_grads(i, DET_API)
    grads = {
        save_stats: _plain_grads(i, ND_API, save_stats=save_stats)
        for save_stats in (True, False)
    }
    for save_stats, got in grads.items():
        _assert_grads(
            got,
            want,
            bound,
            f"T={T} H={hidden} save_stats={save_stats} vs deterministic",
        )
    _assert_grads(
        grads[False],
        grads[True],
        bound,
        f"T={T} H={hidden} replayed vs saved statistics",
    )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_keeps_nothing_alive(T: int, save_stats: bool) -> None:
    """A non-reentrant ``torch.utils.checkpoint`` around a ``deterministic=False`` prepare / finish keeps no tensor of
    the layer alive after the forward, and the checkpointed backward equals the plain one and the deterministic one
    (``X`` / ``scale`` / ``base`` bitwise, ``fn`` within the bound)."""
    i = _make_inputs(T, seed=SEED + 41)

    def connection(X, fn, scale, base):
        U, handle = _nd_prepare(X, fn, scale, base, save_stats=save_stats)
        B = (U.float() + i["B"].float()).to(BF16)
        return _nd_finish(X, B, handle)

    bound = _connection_bound(i, T)
    want = _plain_grads(i, DET_API, save_stats=save_stats)
    plain = _leaves(i)
    (connection(*plain).float() * i["dY"].float()).sum().backward()

    checkpointed = _leaves(i)
    before = _live_bytes()
    Y = checkpoint(connection, *checkpointed, use_reentrant=False)
    _assert_nothing_kept(before, Y, "forward")
    (Y.float() * i["dY"].float()).sum().backward()
    label = f"T={T} save_stats={save_stats} checkpointed"
    _assert_grads(_grads(checkpointed), _grads(plain), bound, f"{label} vs plain")
    _assert_grads(_grads(checkpointed), want, bound, f"{label} vs deterministic")


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_after_prepare_replays_statistics(T: int, save_stats: bool) -> None:
    """A non-reentrant checkpoint region that starts after a ``deterministic=False`` prepare (the branch and the finish
    inside it) keeps nothing beyond its output after the forward, the finish forward takes the edges off the handle
    and its rerun replays the released statistics; the gradients equal the plain run's (``X`` / ``scale`` / ``base``
    bitwise, ``fn`` within the bound)."""
    i = _make_inputs(T, seed=SEED + 42)

    def finish(X, U, handle):
        B = (U.float() + i["B"].float()).to(BF16)
        return _nd_finish(X, B, handle)

    bound = _connection_bound(i, T)
    plain = _leaves(i)
    U, handle = _nd_prepare(*plain, save_stats=save_stats)
    (finish(plain[0], U, handle).float() * i["dY"].float()).sum().backward()

    checkpointed = _leaves(i)
    U, handle = _nd_prepare(*checkpointed, save_stats=save_stats)
    # pinned: save_stats=False releases it in the finish forward
    stats = handle.state.stats
    before = _live_bytes()
    Y = checkpoint(finish, checkpointed[0], U, handle, use_reentrant=False)
    _assert_nothing_kept(before, Y, "finish")
    assert handle.edges == () and handle.state.stats is None, (
        "the handle kept the edges or the statistics after the forward"
    )
    del stats
    (Y.float() * i["dY"].float()).sum().backward()
    _assert_grads(
        _grads(checkpointed),
        _grads(plain),
        bound,
        f"T={T} save_stats={save_stats} checkpointed finish",
    )


def _two_layers(
    X, params: list, api, *, save_stats: bool, checkpointed: bool
) -> torch.Tensor:
    """Two layers of two connections each of ``api`` (the branch hands ``U`` on as ``B``), each layer a non-reentrant
    checkpoint region or not."""
    prepare, finish = api

    def connection(x, fn, scale, base):
        U, handle = prepare(x, fn, scale, base, save_stats=save_stats)
        return finish(x, U, handle)

    def layer(x, k):
        return connection(connection(x, *params[2 * k]), *params[2 * k + 1])

    if not checkpointed:
        return layer(layer(X, 0), 1)
    return checkpoint(
        layer, checkpoint(layer, X, 0, use_reentrant=False), 1, use_reentrant=False
    )


def _chain_bounds(X: torch.Tensor, params: list, dY: torch.Tensor, T: int) -> list:
    """The ``dfn`` bound of each connection of :func:`_two_layers`, from the deterministic direct entry points: the
    input, branch output ``U`` and loss gradient of every connection (the non-deterministic chain hands each
    connection bitwise the same ``dX`` as its upstream gradient)."""
    xs, us, records = [], [], []
    x = X.detach()
    for fn, scale, base in params:
        U, stats = forward_prepare(x, fn.detach(), scale.detach(), base.detach())
        xs.append(x)
        us.append(U)
        records.append(stats)
        x = forward_finish(x, U, stats)
    bounds: list = [None] * len(params)
    g = dY
    for c in reversed(range(len(params))):
        fn, scale, base = (p.detach() for p in params[c])
        dB, gstats = backward_prepare(g, us[c], xs[c], records[c])
        case = {
            "X": xs[c],
            "fn": fn,
            "scale": scale,
            "base": base,
            "B": us[c],
            "dY": g,
            "dU": dB,
        }
        bounds[c] = _dfn_bound(case, _dfn_factor(T))
        g = det_backward_finish(
            xs[c], g, dB, us[c], fn, scale, base, records[c], gstats
        )[0]
    return bounds


def _assert_chain_grads(got, want, bounds: list, label: str) -> None:
    """The gradients of ``[X, *(fn, scale, base) per connection]``: ``X``, every ``scale`` and ``base`` bitwise; each
    connection's ``fn`` within its bound."""
    _assert_same_bits(got[0], want[0], f"grad X {label}")
    for c, bound in enumerate(bounds):
        fn, scale, base = 1 + 3 * c, 2 + 3 * c, 3 + 3 * c
        _assert_same_bits(got[scale], want[scale], f"grad scale[{c}] {label}")
        _assert_same_bits(got[base], want[base], f"grad base[{c}] {label}")
        _assert_dfn_within(got[fn], want[fn], bound, f"grad fn[{c}] {label}")


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("checkpointed", (False, True))
def test_backward_keeps_nothing_alive(
    T: int, save_stats: bool, checkpointed: bool
) -> None:
    """Once a ``deterministic=False`` connection's backward has run, nothing of it stays alive: after
    ``torch.autograd.grad`` over two layers of two connections the only live tensor of the step is its output ``Y``.
    A second step equals the first and the deterministic step (``X`` / ``scale`` / ``base`` bitwise, each ``fn``
    within its connection's bound)."""
    i = _make_inputs(T, seed=SEED + 51)
    params = [
        [p.clone().requires_grad_(True) for p in (q["fn"], q["scale"], q["base"])]
        for q in (_make_inputs(1, seed=SEED + 52 + k) for k in range(4))
    ]
    X = i["X"].clone().requires_grad_(True)
    leaves = [X, *[p for ps in params for p in ps]]

    def step(api) -> tuple:
        Y = _two_layers(
            X, params, api, save_stats=save_stats, checkpointed=checkpointed
        )
        return Y, torch.autograd.grad(Y, leaves, i["dY"])

    det_grads = step(DET_API)[1]
    reference = step(ND_API)[1]  # also the warm-up
    bounds = _chain_bounds(i["X"], params, i["dY"], T)
    before = _live_bytes()
    Y, grads = step(ND_API)
    label = f"T={T} save_stats={save_stats} checkpointed={checkpointed}"
    _assert_chain_grads(grads, reference, bounds, f"second step {label}")
    _assert_chain_grads(grads, det_grads, bounds, f"vs deterministic {label}")
    del grads
    _assert_kept(before, "the backward", Y)


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_repeated_backward_with_retain_graph(T: int, save_stats: bool) -> None:
    """Backward passes over one retained ``deterministic=False`` graph are independent: two full passes equal a fresh
    graph's, and a pass in between that reaches the prepare node but not the finish node (the loss is ``U``) takes
    no ``dY`` of the earlier pass: it equals the same pass on a fresh graph (``X`` / ``scale`` / ``base`` bitwise,
    ``fn`` within the bound of its pass)."""
    i = _make_inputs(T, seed=SEED + 56)

    def graph(leaves: list) -> tuple:
        U, handle = _nd_prepare(*leaves, save_stats=save_stats)
        B = (U.float() + i["B"].float()).to(BF16)
        return U, _nd_finish(leaves[0], B, handle)

    bound = _connection_bound(i, T)
    # The U-only pass: dY and the gradient statistics are zero, dU is the loss gradient.
    bound_u = _dfn_bound({**i, "dY": torch.zeros_like(i["dY"])}, _dfn_factor(T))
    leaves = _leaves(i)
    U, Y = graph(leaves)
    first = torch.autograd.grad(Y, leaves, i["dY"], retain_graph=True)
    u_only = torch.autograd.grad(U, leaves, i["dU"], retain_graph=True)
    second = torch.autograd.grad(Y, leaves, i["dY"])

    fresh = _leaves(i)
    full = torch.autograd.grad(graph(fresh)[1], fresh, i["dY"])
    fresh = _leaves(i)
    u_fresh = torch.autograd.grad(graph(fresh)[0], fresh, i["dU"])
    label = f"T={T} save_stats={save_stats} retain_graph"
    _assert_grads(first, full, bound, f"{label} first pass")
    _assert_grads(second, full, bound, f"{label} second pass")
    _assert_grads(u_only, u_fresh, bound_u, f"{label} U-only pass")


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("region", ("connection", "after_prepare"))
def test_reentrant_checkpoint(T: int, save_stats: bool, region: str) -> None:
    """A reentrant ``torch.utils.checkpoint`` around a ``deterministic=False`` connection, or around the branch and the
    finish with the prepare outside, equals the plain gradients (``X`` / ``scale`` / ``base`` bitwise, ``fn`` within
    the bound) and keeps nothing alive after the backward beyond ``Y`` and the leaves' gradients."""
    i = _make_inputs(T, seed=SEED + 57)

    def finish(X, U, handle):
        B = (U.float() + i["B"].float()).to(BF16)
        return _nd_finish(X, B, handle)

    def connection(X, fn, scale, base):
        U, handle = _nd_prepare(X, fn, scale, base, save_stats=save_stats)
        return finish(X, U, handle)

    bound = _connection_bound(i, T)
    plain = _leaves(i)
    connection(*plain).backward(i["dY"])

    checkpointed = _leaves(i)
    before = _live_bytes()
    if region == "connection":
        Y = checkpoint(connection, *checkpointed, use_reentrant=True)
    else:
        U, handle = _nd_prepare(*checkpointed, save_stats=save_stats)
        Y = checkpoint(finish, checkpointed[0], U, handle, use_reentrant=True)
        del U
    Y.backward(i["dY"])
    _assert_grads(
        _grads(checkpointed),
        _grads(plain),
        bound,
        f"T={T} save_stats={save_stats} reentrant checkpoint ({region})",
    )
    if region == "after_prepare":
        assert all(t is None for t in handle.state.take()), "the state kept a gradient"
    _assert_kept(
        before, "a reentrant checkpointed backward", Y, *(c.grad for c in checkpointed)
    )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("prepare", ("skipped", "not_recorded"))
def test_backward_without_the_prepare_node(T: int, prepare: str) -> None:
    """A backward that runs the finish node but not the ``deterministic=False`` prepare node leaves the finish node's
    ``dY`` and gradient statistics on no handle; what stays alive is the prepare node's own saved statistics while
    its backward has not run."""
    i = _make_inputs(T, seed=SEED + 58)
    leaves = (
        _leaves(i)
        if prepare == "skipped"
        else [i[k] for k in ("X", "fn", "scale", "base")]
    )

    def step() -> tuple:
        U, handle = _nd_prepare(*leaves)
        B = (U.float() + i["B"].float()).to(BF16)
        if prepare == "not_recorded":
            B.requires_grad_(True)
        Y = _nd_finish(leaves[0], B, handle)
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
    """Two ``deterministic=False`` connections of one shape whose prepares both run before either finish equal the two
    connections run one after the other (``X`` / ``scale`` / ``base`` bitwise, each ``fn`` within its bound)."""
    T = 129
    a, b = _make_inputs(T, seed=SEED + 61), _make_inputs(T, seed=SEED + 62)

    def branch(i: dict, U: torch.Tensor) -> torch.Tensor:
        return (U.float() + i["B"].float()).to(BF16)

    def run(interleaved: bool) -> list:
        la, lb = _leaves(a), _leaves(b)
        Ua, ha = _nd_prepare(*la, save_stats=save_stats)
        if interleaved:
            Ub, hb = _nd_prepare(*lb, save_stats=save_stats)
            Yb = _nd_finish(lb[0], branch(b, Ub), hb)
            Ya = _nd_finish(la[0], branch(a, Ua), ha)
        else:
            Ya = _nd_finish(la[0], branch(a, Ua), ha)
            Ub, hb = _nd_prepare(*lb, save_stats=save_stats)
            Yb = _nd_finish(lb[0], branch(b, Ub), hb)
        torch.autograd.backward([Ya, Yb], [a["dY"], b["dY"]])
        return _grads(la), _grads(lb)

    bounds = (_connection_bound(a, T), _connection_bound(b, T))
    serial, interleaved = run(False), run(True)
    for n, bound in enumerate(bounds):
        _assert_grads(
            interleaved[n], serial[n], bound, f"connection {n} save_stats={save_stats}"
        )


def _check_case(T: int, *, seed: int = SEED, zero_rows=()) -> None:
    """One draw through the gates of :func:`test_nondeterministic_backward_matches_fp64` and, on every launch, against
    the deterministic backward (``dX`` / ``dscale`` / ``dbase`` bitwise, ``dfn`` within the bound)."""
    i = _make_inputs(T, seed=seed, zero_rows=zero_rows)
    prep = _forward(i)
    outs = _nd_repeats(i, prep)
    want = _snapshot(_det_backward(i, prep))
    bound = _dfn_bound(i, _dfn_factor(T))
    label = f"T={T}" + (f" zero_rows={tuple(zero_rows)}" if zero_rows else "")
    for repeat, out in enumerate(outs):
        _assert_exact_outputs(out, want, f"{label} launch {repeat}")
        _assert_dfn_within(
            out["dfn"], want["dfn"], bound, f"dfn {label} launch {repeat}"
        )
    ref = _oracle(i)
    comp = _oracle(i, dtype=F32, block_tokens=None)
    _assert_bf16(outs[0]["dX"], ref["dX32"], f"dX {label}")
    _assert_draw(
        {k: _worst_row(outs, ref, comp, k) for k in _PARAMETER}, label, regime="unit"
    )


def test_zero_row_of_x_takes_the_eps_path() -> None:
    _check_case(7, zero_rows=(3,))


@pytest.mark.parametrize("T", (257, 2049))
def test_large_logits(T: int) -> None:
    """``fn x 1e3`` (``|z| ~ 1e3``), the deterministic suite's regime beyond the contract's distributions: ``dX`` /
    ``dscale`` / ``dbase`` bitwise the deterministic backward's on every launch; the BF16 ``dX`` gate; the parameter
    gradients held to the comparator-relative clause in the distribution form over 25 draws (each draw's worst launch).
    The ``dfn`` bound against the deterministic backward is not applied here: the FP64 ``dz'`` it is built from is
    not the kernel's FP32 ``dz'`` at these pre-activations (the rel-L2 to the deterministic ``dfn`` is printed)."""
    i = _make_inputs(T, fn_scale=1e3)
    prep = _forward(i)
    outs = _nd_repeats(i, prep)
    want = _snapshot(_det_backward(i, prep))
    label = f"T={T} fn_scale=1000 deterministic=False"
    for repeat, out in enumerate(outs):
        _assert_exact_outputs(out, want, f"{label} launch {repeat}")
    ref = _oracle(i)
    comp = _oracle(i, dtype=F32, block_tokens=None)
    _assert_bf16(outs[0]["dX"], ref["dX32"], f"dX {label}")
    _assert_draw(
        {k: _worst_row(outs, ref, comp, k) for k in _PARAMETER},
        label,
        regime="large_logits",
    )
    print(
        f"dfn {label}: rel-L2 to the deterministic dfn, largest over {REPEATS} launches "
        f"{max(_rel_l2(out['dfn'], want['dfn']) for out in outs):.3e} (literal record)",
        flush=True,
    )
    del i, prep, outs, want, ref, comp
    rows = _parameter_rows(
        T, fn_scale=1e3, seeds=range(SEED, SEED + PARAM_CLAUSE_DRAWS)
    )
    _assert_parameter_distribution(rows, label, regime="large_logits")


def test_changing_rows_in_one_process_does_not_rebuild() -> None:
    """A changed token count loads no module of either package after the first call, and each call follows the FP64
    reference (BF16 ``dX`` gate, ``dfn`` rel-L2 <= 1e-5)."""
    _check_case(64)
    loaded = (loader.load.cache_info().misses, det_loader.load.cache_info().misses)
    for T in (1, 65, 2049, 7, 255):
        i = _make_inputs(T, seed=SEED + T)
        prep = _forward(i)
        out = _nd_backward(i, prep)
        ref = _oracle(i)
        _assert_bf16(out["dX"], ref["dX32"], f"dX T={T}")
        error = _rel_l2(out["dfn"], ref["dfn"])
        assert error <= FP32_REL_L2_MAX, f"dfn T={T}: rel-L2 {error:.3e}"
    assert (
        loader.load.cache_info().misses,
        det_loader.load.cache_info().misses,
    ) == loaded, "a changed token count rebuilt or reloaded a module"


def test_argument_validation() -> None:
    """``deterministic=False`` validates like the deterministic backward (``test_cake_mhc_det_train.py``'s checks,
    on ``backward_finish`` and the autograd prepare / finish): a wrong shape, dtype, device or layout of an operand or
    an output is a ``ValueError``, as are out-of-range epsilons and Sinkhorn counts (a count above
    ``SINKHORN_ITERS_MAX`` a ``NotImplementedError``) and ``accumulate=True`` without its buffers; the workspace must
    be a 1-D, 256-byte aligned ``uint8`` CUDA tensor of the non-deterministic size (a non-tensor a ``TypeError``).
    An operand or output of another hidden size than ``X``, and a call whose hidden size the package does not list
    (6144) or the architecture does not route (5120 and 7168 on sm_107a), is a ``ValueError``.  ``B`` is optional:
    ``None`` gives the same ``dX`` / ``dscale`` / ``dbase`` bits and a ``dfn`` within the bound."""
    T = 8
    i = _make_inputs(T, seed=SEED + 9)
    X, fn, scale, base = i["X"], i["fn"], i["scale"], i["base"]
    prep = _forward(i)
    dev = _device()
    args = {
        "X": X,
        "dY": i["dY"],
        "dU": i["dU"],
        "B": i["B"],
        "fn": fn,
        "scale": scale,
        "base": base,
        "stats": prep["stats"],
        "grad_stats": prep["gstats"],
    }

    def call(**changes):
        operands = {**args, **{k: v for k, v in changes.items() if k in args}}
        keywords = {k: v for k, v in changes.items() if k not in args}
        return backward_finish(*operands.values(), deterministic=False, **keywords)

    # Padded rows (a [T, 4, 4096] view of a wider buffer) are not materialized silently: the caller decides.
    padded = torch.zeros(T, STREAMS, HIDDEN + 64, dtype=BF16, device=dev)[:, :, :HIDDEN]
    nbytes = pkg.cake_mhc_train_backward_workspace_bytes(T, dev, deterministic=False)
    shifted = torch.zeros(nbytes + 512, dtype=torch.uint8, device=dev)[1 : 1 + nbytes]
    for change in (
        {"X": X.float()},
        {"X": padded},
        {"fn": fn.to(BF16)},
        {"fn": fn[:, 1:]},
        {"scale": scale[:2]},
        {"dY": i["dY"][:4]},
        {"dU": i["dU"].float()},
        {"B": i["B"][:4]},
        {"stats": prep["stats"].double()},
        {"grad_stats": prep["gstats"][:, :16].contiguous()},
        {"accumulate": True},
        {"dX": torch.empty(T, STREAMS, HIDDEN, dtype=F32, device=dev)},
        {"dfn": torch.empty(MAPPING_ROWS, N - 1, dtype=F32, device=dev)},
        {"rms_eps": 0.0},
        {"rms_eps": -1e-5},
        {"hc_eps": float("nan")},
        {"hc_eps": float("inf")},
        {"sinkhorn_iters": 0},
        {"sinkhorn_iters": 20.5},
        {"workspace": torch.zeros(nbytes // 4, dtype=F32, device=dev)},
        {"workspace": torch.zeros(2, nbytes, dtype=torch.uint8, device=dev)},
        {"workspace": shifted},
        {"workspace": torch.zeros(256, dtype=torch.uint8, device=dev)},
        # operands and outputs of another hidden size than X's
        {"fn": torch.zeros(MAPPING_ROWS, STREAMS * 5120, dtype=F32, device=dev)},
        {"dY": torch.zeros(T, STREAMS, 7168, dtype=BF16, device=dev)},
        {"dU": torch.zeros(T, 5120, dtype=BF16, device=dev)},
        {"B": torch.zeros(T, 7168, dtype=BF16, device=dev)},
        {"dX": torch.empty(T, STREAMS, 5120, dtype=BF16, device=dev)},
        {"dfn": torch.empty(MAPPING_ROWS, STREAMS * 7168, dtype=F32, device=dev)},
    ):
        with pytest.raises(ValueError):
            call(**change)
    # every operand of a hidden size the package does not list, or that the architecture does not route
    for hidden in (6144, *(h for h in HIDDEN_SIZES if h not in _hidden_sizes())):
        j = _make_inputs(T, hidden=hidden, seed=SEED + 10)
        with pytest.raises(ValueError):
            backward_finish(
                j["X"], j["dY"], j["dU"], j["B"], j["fn"], j["scale"], j["base"], prep["stats"], prep["gstats"],
                deterministic=False,
            )  # fmt: skip
        with pytest.raises(ValueError):
            forward_prepare(j["X"], j["fn"], j["scale"], j["base"])
        with pytest.raises(ValueError):
            _nd_prepare(j["X"], j["fn"], j["scale"], j["base"])
    with pytest.raises(NotImplementedError):
        call(sinkhorn_iters=det.SINKHORN_ITERS_MAX + 1)
    with pytest.raises(TypeError):
        call(workspace=[0] * 256)
    with pytest.raises(ValueError):
        backward_finish(*(t.cpu() for t in args.values()), deterministic=False)
    # B is optional in the backward finish (its contribution is in grad_stats).
    with_b = _snapshot(dict(zip(_OUTPUTS, call(), strict=True)))
    without_b = _snapshot(dict(zip(_OUTPUTS, call(B=None), strict=True)))
    _assert_exact_outputs(without_b, with_b, "B=None")
    _assert_dfn_within(
        without_b["dfn"], with_b["dfn"], _dfn_bound(i, _dfn_factor(T)), "dfn B=None"
    )
    with pytest.raises(ValueError):
        _nd_prepare(X.float(), fn, scale, base)
    with pytest.raises(ValueError):
        _nd_prepare(padded, fn, scale, base)
    with pytest.raises(TypeError):
        _nd_finish(X, i["B"], object())
    with pytest.raises(TypeError):
        _nd_finish(X, i["B"], det.cake_mhc_det_train_prepare(X, fn, scale, base)[1])


# ----------------------------------------------------------------------------------------------------------------
# Concurrent streams


@pytest.mark.parametrize("Ts", ((2049, 2049), (4096, 1024, 257)))
def test_concurrent_streams(Ts: tuple) -> None:
    """Launches on several ``torch.cuda.Stream`` objects at once, each with its own workspace and outputs (non-ticket
    workspace bytes 0xFF and NaN outputs before every round): each equals its serial result (``dX``, ``dscale``,
    ``dbase`` bitwise, ``dfn`` within the bound) and the deterministic backward's, and leaves its ticket region zero."""
    cases = []
    for n, T in enumerate(Ts):
        i = _make_inputs(T, seed=SEED + 500 + n)
        prep = _forward(i)
        ws = pkg.cake_mhc_train_backward_workspace(T, _device(), deterministic=False)
        serial = _snapshot(_nd_backward(i, prep, workspace=ws))
        want = _snapshot(_det_backward(i, prep))
        _assert_exact_outputs(serial, want, f"serial T={T}")
        cases.append(
            {
                "T": T,
                "i": i,
                "prep": prep,
                "ws": ws,
                "serial": serial,
                "want": want,
                "bound": _dfn_bound(i, _dfn_factor(T)),
                "outs": _poisoned_outputs(T),
            }
        )
    streams = [torch.cuda.Stream() for _ in Ts]
    main = torch.cuda.current_stream()
    torch.cuda.synchronize()
    for rnd in range(3):
        for c in cases:
            _poison_workspace(c["ws"])
            for v in c["outs"].values():
                v.fill_(float("nan"))
        for stream, c in zip(streams, cases, strict=True):
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                _nd_backward(c["i"], c["prep"], workspace=c["ws"], **c["outs"])
        for stream in streams:
            main.wait_stream(stream)
        torch.cuda.synchronize()
        for n, c in enumerate(cases):
            what = f"round {rnd} stream {n} (T={c['T']}) of {Ts}"
            _assert_ticket_zero(c["ws"], what)
            _assert_exact_outputs(c["outs"], c["serial"], what + " vs serial")
            _assert_exact_outputs(c["outs"], c["want"], what + " vs deterministic")
            _assert_dfn_within(
                c["outs"]["dfn"],
                c["serial"]["dfn"],
                c["bound"],
                f"dfn {what} vs serial",
            )
            _assert_dfn_within(
                c["outs"]["dfn"],
                c["want"]["dfn"],
                c["bound"],
                f"dfn {what} vs deterministic",
            )


# ----------------------------------------------------------------------------------------------------------------
# Hidden sizes mixed in one process (test_cake_mhc_det_train.py's cases: the host caches -- layouts, default
# workspaces and their views, caller-workspace views, launch plans, fast routes, route checks -- are keyed by the
# hidden size; one keyed by the token count alone would serve a size's views or plan to another size's call)

_MIXED_T = (129, 2049)
_WORKSPACE_MODES = ("default", "per_hidden", "shared")


def _mixed_hidden_sizes() -> tuple[int, ...]:
    """Every routed hidden size, for the cases that mix sizes in one process (4096 and 7168 at least)."""
    sizes = _hidden_sizes()
    if not {4096, 7168} <= set(sizes):
        pytest.skip(
            f"mixing hidden sizes needs 4096 and 7168 on one device; {_arch()} exports {sizes}"
        )
    return sizes


def _mixed_workspaces(sizes: tuple[int, ...], T: int, mode: str) -> dict:
    """``{H: (forward workspace, backward workspace)}`` of a mixed-size case: ``default`` (``None``: the per-device
    default buffers), ``per_hidden`` (one caller buffer per size, ``hidden=H``) or ``shared`` (one zeroed caller
    buffer per kind serving every size, as large as the largest size's request)."""
    dev = _device()
    if mode == "default":
        return {H: (None, None) for H in sizes}
    nbytes = {
        "fwd": pkg.cake_mhc_train_forward_workspace_bytes,
        "bwd": pkg.cake_mhc_train_backward_workspace_bytes,
    }
    if mode == "per_hidden":
        return {
            H: tuple(
                torch.zeros(
                    size(T, dev, hidden=H, deterministic=False),
                    dtype=torch.uint8,
                    device=dev,
                )
                for size in nbytes.values()
            )
            for H in sizes
        }
    assert mode == "shared", mode
    shared = tuple(
        torch.zeros(
            max(size(T, dev, hidden=H, deterministic=False) for H in sizes),
            dtype=torch.uint8,
            device=dev,
        )
        for size in nbytes.values()
    )
    return {H: shared for H in sizes}


def _staged_run(cases: dict, order: tuple[int, ...], ws: dict) -> dict:
    """This package's entry points of every case stage by stage: each stage is issued for every hidden size (in
    ``order``, the next stage in the reverse order) before the next stage, so consecutive calls switch the hidden size
    at one token count.  ``replay`` is the ``forward_stats`` record of each size."""
    out: dict[int, dict] = {H: {} for H in order}
    back = tuple(reversed(order))
    for H in order:
        i = cases[H]
        out[H]["U"], out[H]["stats"] = forward_prepare(
            i["X"], i["fn"], i["scale"], i["base"], workspace=ws[H][0]
        )
    for H in back:
        i = cases[H]
        out[H]["replay"] = forward_stats(
            i["X"], i["fn"], i["scale"], i["base"], workspace=ws[H][0]
        )
    for H in order:
        i = cases[H]
        out[H]["Y"] = forward_finish(i["X"], i["B"], out[H]["stats"])
    for H in back:
        i, o = cases[H], out[H]
        o["dB"], o["gstats"] = backward_prepare(i["dY"], i["B"], i["X"], o["stats"])
    for H in order:
        i, o = cases[H], out[H]
        o["dX"], o["dfn"], o["dscale"], o["dbase"] = backward_finish(
            i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], o["stats"], o["gstats"],
            workspace=ws[H][1], deterministic=False,
        )  # fmt: skip
    return {H: _snapshot(o) for H, o in out.items()}


def _assert_mixed_case(got: dict, want: dict, bound: torch.Tensor, what: str) -> None:
    """Every output of ``want`` but ``dfn`` bitwise; ``dfn`` within the bound."""
    for k, v in want.items():
        if k == "dfn":
            _assert_dfn_within(got[k], v, bound, f"dfn {what}")
        else:
            _assert_same_bits(got[k], v, f"{k} {what}")


@pytest.mark.parametrize("T", _MIXED_T)
@pytest.mark.parametrize("workspaces", _WORKSPACE_MODES)
def test_mixed_hidden_sizes_same_token_count(T: int, workspaces: str) -> None:
    """Calls of every routed hidden size at the same token count, interleaved stage by stage through the direct
    entry points (default workspaces, one caller workspace per size, one caller workspace shared by the sizes), in
    two rounds of opposite size order: ``forward_stats`` is bitwise the ``forward_prepare`` record, every output but
    ``dfn`` is bitwise the sizes run one after the other and the deterministic package's on the same inputs, ``dfn``
    within the bound of both, and a caller backward workspace keeps its ticket region zero."""
    sizes = _mixed_hidden_sizes()
    cases = {H: _make_inputs(T, hidden=H, seed=SEED + 71 + H) for H in sizes}
    ws = _mixed_workspaces(sizes, T, workspaces)
    bounds = {H: _dfn_bound(cases[H], _dfn_factor(T, H)) for H in sizes}
    alone = {H: _staged_run({H: cases[H]}, (H,), ws)[H] for H in sizes}
    reference = {}
    for H in sizes:
        i = cases[H]
        want = _reference_forward_outputs(i)
        want.update(
            _det_backward(i, {"stats": want["stats"], "gstats": want["gstats"]})
        )
        reference[H] = _snapshot(want)
    for rnd, order in enumerate((sizes, tuple(reversed(sizes)))):
        got = _staged_run(cases, order, ws)
        for H in sizes:
            what = (
                f"T={T} H={H} mixed with {sizes} ({workspaces} workspaces, round {rnd})"
            )
            assert torch.equal(got[H]["replay"], got[H]["stats"]), (
                f"{what}: forward_stats is not bitwise the forward_prepare record"
            )
            _assert_mixed_case(got[H], alone[H], bounds[H], f"{what} vs alone")
            _assert_mixed_case(
                got[H], reference[H], bounds[H], f"{what} vs deterministic"
            )
            if ws[H][1] is not None:
                _assert_ticket_zero(ws[H][1], what)
        del got


@pytest.mark.parametrize("T", _MIXED_T)
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("workspaces", _WORKSPACE_MODES)
def test_mixed_hidden_sizes_autograd(T: int, save_stats: bool, workspaces: str) -> None:
    """The ``deterministic=False`` autograd connections with every routed hidden size at the same token count: the
    prepares of every size, then the finishes in the reverse order, then one backward over the sum of the losses.
    Each size's ``U`` and ``Y`` are bitwise those of the sizes run one after the other, its gradients equal theirs
    and the deterministic connection's (``X`` / ``scale`` / ``base`` bitwise, ``fn`` within the bound)."""
    sizes = _mixed_hidden_sizes()
    cases = {H: _make_inputs(T, hidden=H, seed=SEED + 81 + H) for H in sizes}
    ws = _mixed_workspaces(sizes, T, workspaces)
    bounds = {H: _connection_bound(cases[H], T) for H in sizes}

    def prepare(H: int) -> tuple:
        leaves = _leaves(cases[H])
        U, handle = _nd_prepare(
            *leaves,
            save_stats=save_stats,
            workspace=ws[H][0],
            backward_workspace=ws[H][1],
        )
        return leaves, U, handle

    def finish(H: int, connection: tuple) -> torch.Tensor:
        leaves, U, handle = connection
        return _nd_finish(
            leaves[0], (U.float() + cases[H]["B"].float()).to(BF16), handle
        )

    def loss(H: int, Y: torch.Tensor) -> torch.Tensor:
        return (Y.float() * cases[H]["dY"].float()).sum()

    alone = {}
    for H in sizes:
        connection = prepare(H)
        Y = finish(H, connection)
        loss(H, Y).backward()
        alone[H] = (
            connection[1].detach().clone(),
            Y.detach().clone(),
            _grads(connection[0]),
        )
        del connection, Y
    reference = {
        H: _plain_grads(cases[H], DET_API, save_stats=save_stats) for H in sizes
    }
    connections = {H: prepare(H) for H in sizes}
    Ys = {H: finish(H, connections[H]) for H in reversed(sizes)}
    torch.stack([loss(H, Ys[H]) for H in sizes]).sum().backward()
    for H in sizes:
        what = f"T={T} H={H} mixed with {sizes} ({save_stats=}, {workspaces=})"
        U_alone, Y_alone, grads_alone = alone[H]
        _assert_same_bits(connections[H][1].detach(), U_alone, f"U {what}")
        _assert_same_bits(Ys[H].detach(), Y_alone, f"Y {what}")
        grads = _grads(connections[H][0])
        _assert_grads(grads, grads_alone, bounds[H], f"{what} vs alone")
        _assert_grads(grads, reference[H], bounds[H], f"{what} vs deterministic")


# ----------------------------------------------------------------------------------------------------------------
# Zero tokens, the build manifest


@pytest.mark.parametrize("hidden", HIDDEN_SIZES, ids=lambda h: f"H{h}")
def test_zero_tokens_launch_nothing(hidden: int) -> None:
    """``T == 0`` behaves as the deterministic package: ``dX`` is empty; the parameter gradients are zeroed without
    ``accumulate`` (also caller buffers holding other values) and left untouched with it; the workspace needs no
    bytes and an empty one is accepted; PyTorch's deterministic mode raises no alert (nothing is reduced)."""
    _require_hidden(hidden)
    i = _make_inputs(0, hidden=hidden)
    prep = _forward(i)
    want = _det_backward(i, prep)
    empty_ws = pkg.cake_mhc_train_backward_workspace(
        0, _device(), hidden=hidden, deterministic=False
    )
    assert empty_ws.numel() == 0
    for workspace in (None, empty_ws):
        got = _nd_backward(i, prep, workspace=workspace)
        assert (
            tuple(got["dX"].shape) == (0, STREAMS, hidden) and got["dX"].dtype == BF16
        )
        assert tuple(got["dfn"].shape) == (MAPPING_ROWS, STREAMS * hidden)
        for k in _PARAMETER:
            assert torch.equal(got[k], torch.zeros_like(got[k])), k
            _assert_same_bits(got[k], want[k], f"{k} T=0 H={hidden}")
    for accumulate in (False, True):
        buffers = {
            "dfn": torch.full(
                (MAPPING_ROWS, STREAMS * hidden), 3.0, dtype=F32, device=_device()
            ),
            "dscale": torch.full((3,), 5.0, dtype=F32, device=_device()),
            "dbase": torch.full((MAPPING_ROWS,), 7.0, dtype=F32, device=_device()),
        }
        got = _nd_backward(i, prep, accumulate=accumulate, **buffers)
        for k, value in (("dfn", 3.0), ("dscale", 5.0), ("dbase", 7.0)):
            assert got[k] is buffers[k], (k, accumulate)
            expected = value if accumulate else 0.0
            assert bool((got[k] == expected).all()), (k, accumulate)
    # Nothing is reduced at ``T == 0``, so PyTorch's deterministic mode raises no alert there.
    prev = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = _nd_backward(i, prep)
        assert tuple(got["dX"].shape) == (0, STREAMS, hidden)
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record of the package names its ``build.json`` library (schema ``cake-build/1``): one entry
    per (library, architecture) with the kernel and binding sources under ``csrc/`` of this package."""
    cake_build = cake_package(KEY, "cake_build")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"], name
        assert record["library"].startswith("cake_mhc_train_"), name
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
