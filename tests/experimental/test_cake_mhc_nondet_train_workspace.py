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

Workspace reuse across token counts of the Cake four-stream mHC training
package ``flashinfer.experimental.cake.cake_mhc_nondet_train`` (hidden 4096,
5120 and 7168, 4 streams; SM100 / SM103 at the three hidden sizes, SM107 at
4096; FlashInfer issue #8068), in the
form of ``test_cake_mhc_det_train_workspace.py``.  Every case runs at each
hidden size the device's architecture routes, its buffers sized with
``hidden=``; a size it does not route is skipped.

The backward workspace of this package starts with a zero prefix of 768 bytes
at every token count and hidden size: the ticket words the column-slab pass
draws its work units from and the token owner's group-row form its groups from
(self-resetting ``atom.inc``; a word that is not zero at launch makes the launch
skip units or end away from zero), then the column slab's arrival counts
(``slab_cnt``: the grid's finished units and the CTAs past the wait for them;
the last CTA of a launch leaves both words zero).  The deterministic regions
follow (the token owner's records and parameter rows: the per-CTA partials, or
the group rows inside ``BWD_ROW_BANDS``), then the column slab's per-chunk
``dfn`` slots, so
the workspace is larger than the deterministic package's.  A caller-owned buffer
sized by ``cake_mhc_nondet_train_backward_workspace_bytes(T_max)`` serves
every token count up to ``T_max`` (at its ``hidden``) and CUDA Graph replays
at a smaller ``T``.
Before every call the bytes of the backward buffer past the zero prefix are
overwritten with 0xFF (no call may depend on what an earlier call left there).
Every call is compared with the same call on fresh zeroed exact-size
workspaces (every output bitwise, ``dfn`` included) and with the deterministic
package (``flashinfer.experimental.cake.cake_mhc_det_train``) on the same
inputs: every output but ``dfn`` bitwise; ``dfn`` within ``F u sum_t |dz'_t|
|x_t|`` per element (``F`` = ``test_cake_mhc_nondet_train.py``'s
``_dfn_factor``: ``2 P`` with ``P`` the chunk count of the column-slab pass,
``u = 2^-24``, ``dz'`` the FP64 reference's).  On one buffer reused over
realistic token-count sequences:

* every call finds the zero prefix of its layout zero and leaves it zero;
* a CUDA Graph captured at a smaller token count on the large buffers, its
  replays interleaved with eager calls at the large token count, keeps both
  properties at every replay;
* the default workspaces (``workspace=None``: one buffer per hidden size and
  power-of-two token-count bucket) across the token counts of one bucket;
* the sequence of FlashInfer issue #7763 (``backward_finish`` at 7832 tokens,
  then at 7435 tokens, on the default workspace of a fresh process, and the
  reverse order), each sequence in its own process;
* no default buffer is created during a CUDA graph capture (``ValueError``,
  nothing cached), and the eager call after it creates the buffer and matches
  the caller-owned workspace's call;
* with two devices, the capture check reads the workspace device's current
  stream, not the current device's: a capture on the workspace device raises
  while another device is current, and a capture on another device does not
  stop the call from creating the buffer and running eagerly;
* one buffer over a ragged sequence with caller outputs, and the sizes of the
  workspace and which sizing serves which backward (a buffer of the
  deterministic sizing is refused);
* the helpers' sizes per hidden size (an unrouted or unknown size is a
  ``ValueError``), and a buffer sized for H 4096 refused by an H 7168 call it
  is too small for.
"""

from __future__ import annotations

import gc
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import PACKAGES, cake_package

KEY = "cake.cake_mhc_nondet_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
# The deterministic package: the same-input comparison point.
det = cake_package("cake.cake_mhc_det_train")
det_loader = cake_package("cake.cake_mhc_det_train", "cake_jit")
backward_finish = pkg.cake_mhc_nondet_train_backward_finish

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, MAPPING_ROWS = 4, 4096, 24
HIDDEN_SIZES = (4096, 5120, 7168)
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
BLOCK_TOKENS = 8192
FP32_REL_L2_MAX = 1e-5
FP32_UNIT_ROUNDOFF = 2.0**-24
ABS_SLACK = 1.02
# The backward workspace's zero prefix: the ticket region (256 bytes) and the column slab's arrival counts (512 bytes),
# first for every token count and hidden size; every call finds it zero and leaves it zero.
ZERO_PREFIX_BYTES = 768
SLAB_CNT_WORDS = 128
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
SEED = 7046
# Token-count sequences of one buffer (the largest first: the buffer is sized for it).  The first two are the
# sequences of the deterministic suite's dynamic-schedule column-slab rows on sm_103a / sm_100a (this package's
# column slab draws its units from the ticket word on every architecture); the third is the per-step token count of a
# GLM-style training run.
SEQUENCES = {
    "slab_dynamic_sm103": (4096, 3000, 1024, 4096, 2049, 3000),
    "slab_dynamic_sm100": (8192, 6000, 4500, 8192, 5000),
    "glm_rows": (16263, 14928, 15566, 16263, 15288, 16164),
}
GRAPHS = ((4096, 3000), (8192, 4500), (16263, 14928))
DEFAULT_BUCKETS = ((4096, 3000, 2100, 4096), (8192, 6000, 4500, 8192))
# The backward token-count sequences of FlashInfer issue #7763 (one 8192-token default bucket).
REPORTED_SEQUENCES = ((7832, 7435), (7435, 7832))
# One caller workspace over token counts within and across the 64-token tiles, down to one token.
RAGGED_SEQUENCE = (4096, 1, 777, 2049, 4096)
OUTPUTS = ("U", "stats", "Y", "dB", "gstats", "dX", "dfn", "dscale", "dbase")
_BACKWARD = ("dX", "dfn", "dscale", "dbase")
_EXACT = ("dX", "dscale", "dbase")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_mhc_nondet_train_supported(_device())
        and loader.route_applies(
            device_capability=_capability(), hidden=HIDDEN, kind="bwd_finish"
        )
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_nondet_train needs an SM100 / SM103 / SM107 device with exported routes",
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


@pytest.fixture(autouse=True)
def collect_dead_graphs():
    # torch skips its gc.collect() before a capture by default (TORCH_CUDAGRAPH_GC): an earlier test's dead CUDA graph
    # that a cyclic collection destroys during a global-mode capture invalidates that capture.
    gc.collect()
    yield
    gc.collect()


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture routes every kind of."""
    arch = _arch()
    return tuple(loader.supported_hidden_sizes(arch)) if arch else ()


@pytest.fixture(params=HIDDEN_SIZES, ids=lambda h: f"H{h}")
def hidden(request) -> int:
    """The hidden size of a case (every size of the package; skipped where the architecture has no route)."""
    if request.param not in _hidden_sizes():
        pytest.skip(
            f"hidden {request.param} has no exported route on {_arch()} "
            f"(exported hidden sizes: {_hidden_sizes()})"
        )
    return request.param


def _parts(T: int, hidden: int = HIDDEN) -> int:
    """``P``: the chunk count of the column-slab pass, the number of partials each ``dfn`` element folds."""
    parts = loader.host_plan(T, _sm_count(), _arch(), hidden)["parts"]
    assert 1 <= parts <= 32, (T, hidden, parts)
    return parts


def _lpt_runs(T: int, hidden: int = HIDDEN) -> bool:
    """Whether the column slab runs a two-size record of ``hidden`` (``lpt`` or its slab-fastest copy ``lpt_os``) at
    ``T`` on this device (at hidden 5120 and 7168 only where the size's table calibrates the architecture)."""
    program = loader.bwd_slab_atomic_program(T, _sm_count(), _arch(), hidden)
    return program.startswith("lpt")


def _slab_parts(T: int, hidden: int = HIDDEN) -> int:
    """The chunk count the column slab launches: ``slab_plan``'s ``lpt_parts`` under ``lpt`` / ``lpt_os``, else
    :func:`_parts`."""
    if not _lpt_runs(T, hidden):
        return _parts(T, hidden)
    parts = loader.slab_plan(T, _sm_count(), _arch(), hidden)["lpt_parts"]
    assert 1 <= parts <= 32, (T, hidden, parts)
    return parts


def _dfn_factor(T: int, hidden: int = HIDDEN) -> float:
    """F of the dfn bound F u (|carried| + ABS_SLACK sum_t |dz'_t| |x_t|) at T (``test_cake_mhc_nondet_train.py``'s
    ``_dfn_factor``, which derives it): 2 P where the column slab runs dynr or dynr_os (both sides fold the same
    per-chunk partials), 2 max(P, P_k7) + 10 where it runs lpt or lpt_os (the partials are another partition's)."""
    if not _lpt_runs(T, hidden):
        return 2.0 * _parts(T, hidden)
    return 2.0 * max(_slab_parts(T, hidden), _parts(T, hidden)) + 10.0


def _round256(n: int) -> int:
    return 256 * (-(-n // 256))


# ----------------------------------------------------------------------------------------------------------------
# The dfn bound (test_cake_mhc_nondet_train.py's)


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


def _dzp(i: dict) -> torch.Tensor:
    """The FP64 ``dz' = r dL/dz`` of the reference, ``[T, 24]`` (the backward of the coefficients alone, so no
    ``[T, 4H]`` autograd graph is held)."""
    T, n = i["X"].shape[0], STREAMS * i["X"].shape[2]
    fn, scale, base = (i[k].double() for k in ("fn", "scale", "base"))
    blocks = [torch.zeros((0, MAPPING_ROWS), dtype=F64, device=_device())]
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        X = i["X"][lo:hi].double()
        dY = i["dY"][lo:hi].double()
        dU = i["dU"][lo:hi].double()
        B = i["B"][lo:hi].double()
        u = X.reshape(hi - lo, n)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + RMS_EPS)
        with torch.enable_grad():
            z = ((u * r) @ fn.T).requires_grad_(True)
            pre, post, C = _coefficients(z, scale, base)
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
    """Per-element bound of ``|dfn_a - dfn_b|`` for two folds of per-chunk partials:
    ``F u * 1.02 sum_t |dz'_t| |x_t|`` with ``F`` = :func:`_dfn_factor` (``test_cake_mhc_nondet_train.py``'s
    ``_dfn_bound``)."""
    T, hidden = i["X"].shape[0], i["X"].shape[2]
    dzp = _dzp(i).abs()
    A = torch.zeros((MAPPING_ROWS, STREAMS * hidden), dtype=F64, device=_device())
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        A += dzp[lo:hi].T @ i["X"][lo:hi].reshape(hi - lo, -1).double().abs()
    return _dfn_factor(T, hidden) * FP32_UNIT_ROUNDOFF * ABS_SLACK * A


# ----------------------------------------------------------------------------------------------------------------
# Inputs, steps and comparisons


_INPUTS: dict[tuple[int, int], dict] = {}


def _inputs(T: int, hidden: int = HIDDEN) -> dict:
    if (T, hidden) not in _INPUTS:
        g = torch.Generator(device=_device()).manual_seed(SEED + T)

        def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
            return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(
                dtype
            )

        # One (token count, hidden size)'s inputs at a time (16263 tokens hold ~1.3 GB at H 4096, ~2.3 GB at 7168).
        _INPUTS.clear()
        _INPUTS[(T, hidden)] = {
            "X": randn(T, STREAMS, hidden, dtype=BF16),
            "fn": (randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2).contiguous(),
            "scale": 1.0 + 0.1 * randn(3),
            "base": 0.1 * randn(MAPPING_ROWS),
            "B": randn(T, hidden, dtype=BF16),
            "dY": randn(T, STREAMS, hidden, dtype=BF16),
            "dU": randn(T, hidden, dtype=BF16),
        }
    return _INPUTS[(T, hidden)]


def _bound(i: dict) -> torch.Tensor:
    """The ``dfn`` bound of the inputs ``i`` (kept with them)."""
    if "bound" not in i:
        i["bound"] = _dfn_bound(i)
    return i["bound"]


def _step(i: dict, fwd_ws, bwd_ws, *, det_package: bool = False) -> dict:
    """One forward + backward of this package or of the deterministic package."""
    if det_package:
        entries = (
            det.cake_mhc_det_train_forward_prepare,
            det.cake_mhc_det_train_forward_finish,
            det.cake_mhc_det_train_backward_prepare,
            det.cake_mhc_det_train_backward_finish,
        )
    else:
        entries = (
            pkg.cake_mhc_nondet_train_forward_prepare,
            pkg.cake_mhc_nondet_train_forward_finish,
            pkg.cake_mhc_nondet_train_backward_prepare,
            backward_finish,
        )
    fwd_prepare, fwd_finish, bwd_prepare, bwd_finish = entries
    U, stats = fwd_prepare(i["X"], i["fn"], i["scale"], i["base"], workspace=fwd_ws)
    Y = fwd_finish(i["X"], i["B"], stats)
    dB, gstats = bwd_prepare(i["dY"], i["B"], i["X"], stats)
    dX, dfn, dscale, dbase = bwd_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
        workspace=bwd_ws,
    )  # fmt: skip
    return dict(
        zip(OUTPUTS, (U, stats, Y, dB, gstats, dX, dfn, dscale, dbase), strict=True)
    )


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _layout(kind: str, T: int, hidden: int) -> dict:
    return loader.workspace_layout(kind, T, device_index=_device().index, hidden=hidden)


def _workspaces(t_max: int, hidden: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Caller-owned forward and backward workspaces of this package's sizing for up to ``t_max`` tokens of hidden
    size ``hidden``."""
    return (
        pkg.cake_mhc_nondet_train_forward_workspace(t_max, _device(), hidden=hidden),
        pkg.cake_mhc_nondet_train_backward_workspace(t_max, _device(), hidden=hidden),
    )


def _nd_workspace_exact(T: int, hidden: int = HIDDEN) -> torch.Tensor:
    """A fresh zeroed backward workspace of this package of exactly ``T`` tokens' layout at ``hidden``."""
    return torch.zeros(
        loader.workspace_bytes("bwd", T, _sm_count(), _arch(), hidden),
        dtype=torch.uint8,
        device=_device(),
    )


def _fresh(T: int, hidden: int = HIDDEN) -> dict:
    """The outputs of one call of this package on fresh zeroed exact-size workspaces."""
    fwd_ws = torch.zeros(
        _layout("fwd", T, hidden)["total_bytes"], dtype=torch.uint8, device=_device()
    )
    out = _snapshot(_step(_inputs(T, hidden), fwd_ws, _nd_workspace_exact(T, hidden)))
    torch.cuda.synchronize()
    return out


def _deterministic(T: int, hidden: int = HIDDEN) -> dict:
    """The outputs of the deterministic package on its default workspaces."""
    out = _snapshot(_step(_inputs(T, hidden), None, None, det_package=True))
    torch.cuda.synchronize()
    return out


def _poison_workspace(ws: torch.Tensor) -> None:
    """Every byte past the zero prefix (the ticket region and the arrival counts, first for every token count) set
    to 0xFF."""
    ws[ZERO_PREFIX_BYTES:].fill_(0xFF)


def _assert_zero_prefix(ws: torch.Tensor, what: str) -> None:
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(ws[:ZERO_PREFIX_BYTES]))
    assert nonzero == 0, (
        f"{what}: {nonzero} non-zero bytes in the ticket region and the arrival counts"
    )


def _assert_zero_words(fwd_ws, bwd_ws, T: int, hidden: int, what: str) -> None:
    """The ticket words of the forward layout (the deterministic package's layout) and the zero prefix of this
    package's backward layout for ``T`` tokens of hidden size ``hidden`` (the ticket region at offset 0, the arrival
    counts right after it) are zero."""
    torch.cuda.synchronize()
    bwd = loader.workspace_regions("bwd", T, _sm_count(), _arch(), hidden)["regions"]
    assert bwd["ticket"]["offset"] == 0 and bwd["ticket"]["bytes"] == 256
    assert bwd["slab_cnt"]["offset"] == 256
    assert bwd["slab_cnt"]["bytes"] == 4 * SLAB_CNT_WORDS == ZERO_PREFIX_BYTES - 256
    fwd = _layout("fwd", T, hidden)["regions"]["ticket"]
    for ws, kind, lo, hi in (
        (fwd_ws, "fwd ticket words", fwd["offset"], fwd["offset"] + fwd["bytes"]),
        (bwd_ws, "bwd zero prefix", 0, ZERO_PREFIX_BYTES),
    ):
        nonzero = int(torch.count_nonzero(ws[lo:hi]))
        assert nonzero == 0, (
            f"{what}: {nonzero} non-zero bytes in the {kind} of T={T} H={hidden}"
        )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        a.detach().contiguous().reshape(-1).view(torch.uint8),
        b.detach().contiguous().reshape(-1).view(torch.uint8),
    )


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


def _assert_matches(ref: dict, got: dict, bound: torch.Tensor, what: str) -> None:
    """Every output of ``ref`` (the deterministic package's) but ``dfn`` bitwise; ``dfn`` within the bound."""
    for k, v in ref.items():
        if k == "dfn":
            _assert_dfn_within(got[k], v, bound, f"{what}: dfn")
        else:
            assert _same_bits(v, got[k]), (
                f"{what}: {k} differs in {int((v != got[k]).sum())} elements"
            )


def _assert_same_call(ref: dict, got: dict, what: str) -> None:
    """Every output of ``ref`` (the same call of this package), ``dfn`` included, bitwise."""
    for k, v in ref.items():
        assert _same_bits(v, got[k]), (
            f"{what}: {k} differs in {int((v != got[k]).sum())} elements"
        )


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_caller_workspace_reused_across_token_counts(name: str, hidden: int) -> None:
    Ts = SEQUENCES[name]
    t_max = max(Ts)
    fwd_ws, bwd_ws = _workspaces(t_max, hidden)
    for n, T in enumerate(Ts):
        bound = _bound(_inputs(T, hidden))
        ref, want = _fresh(T, hidden), _deterministic(T, hidden)
        what = f"{name} H={hidden} call {n} (T={T} on a buffer of T={t_max})"
        _poison_workspace(bwd_ws)
        _assert_zero_words(fwd_ws, bwd_ws, T, hidden, what + " before the call")
        got = _snapshot(_step(_inputs(T, hidden), fwd_ws, bwd_ws))
        _assert_zero_words(fwd_ws, bwd_ws, T, hidden, what + " after the call")
        _assert_same_call(ref, got, what + " vs the fresh workspaces")
        _assert_matches(want, got, bound, what + " vs deterministic")
        del ref, want, got


@pytest.mark.parametrize(("t_max", "t_small"), GRAPHS)
def test_graph_captured_at_a_smaller_token_count(
    t_max: int, t_small: int, hidden: int
) -> None:
    fwd_ws, bwd_ws = _workspaces(t_max, hidden)
    small = _inputs(t_small, hidden)
    bound_small = _bound(small)
    ref_small, want_small = _fresh(t_small, hidden), _deterministic(t_small, hidden)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _step(small, fwd_ws, bwd_ws)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _step(small, fwd_ws, bwd_ws)
    torch.cuda.synchronize()
    big = _inputs(t_max, hidden)
    bound_big = _bound(big)
    ref_big, want_big = _fresh(t_max, hidden), _deterministic(t_max, hidden)
    for r in range(3):
        what = f"H={hidden} replay {r} at T={t_small} after an eager call at T={t_max}"
        eager = f"H={hidden} eager T={t_max} before replay {r}"
        _poison_workspace(bwd_ws)
        got_big = _snapshot(_step(big, fwd_ws, bwd_ws))
        _assert_zero_words(fwd_ws, bwd_ws, t_max, hidden, eager)
        _assert_same_call(ref_big, got_big, eager)
        _assert_matches(want_big, got_big, bound_big, eager + " vs deterministic")
        del got_big
        _poison_workspace(bwd_ws)
        _assert_zero_words(fwd_ws, bwd_ws, t_small, hidden, what + " (before)")
        for t in out.values():
            t.fill_(float("nan"))
        graph.replay()
        _assert_zero_words(fwd_ws, bwd_ws, t_small, hidden, what + " (after)")
        got = _snapshot(out)
        _assert_same_call(ref_small, got, what)
        _assert_matches(want_small, got, bound_small, what + " vs deterministic")
    del graph


@pytest.mark.parametrize("Ts", DEFAULT_BUCKETS)
def test_default_workspace_bucket_across_token_counts(Ts: tuple, hidden: int) -> None:
    """``workspace=None`` over the token counts of one power-of-two bucket: the bucket's default buffer of the hidden
    size (keyed by device, kind, hidden size and bucket; allocated on the first call) has its bytes past the zero
    prefix overwritten with 0xFF before every later call and its zero prefix zero before and after every call."""
    index = _device().index
    for n, T in enumerate(Ts):
        bound = _bound(_inputs(T, hidden))
        ref, want = _fresh(T, hidden), _deterministic(T, hidden)
        what = f"default workspace call {n} (T={T}, H={hidden}) of {Ts}"
        key = (index, "bwd", hidden, 1 << (T - 1).bit_length())
        default = pkg._DEFAULT_WORKSPACES.get(key)
        if default is not None:
            _poison_workspace(default)
            _assert_zero_prefix(default, what + " before the call")
        got = _snapshot(_step(_inputs(T, hidden), None, None))
        _assert_zero_prefix(pkg._DEFAULT_WORKSPACES[key], what + " after the call")
        _assert_same_call(ref, got, what)
        _assert_matches(want, got, bound, what + " vs deterministic")


# One process per sequence: the inputs of the issue's reproduction (one CPU generator seeded 0, drawn in call order)
# at the hidden size of the case, each ``backward_finish(...)`` on the default workspace (keyed by
# device, kind, hidden size and power-of-two token bucket), then the same call on a fresh zeroed exact-size workspace
# (every output the same bits) and the deterministic call.  The inputs are not the records of a forward (``stats`` and
# ``grad_stats`` are random), so no FP64 reference applies: ``dfn`` is compared with the deterministic one by its
# rel-L2.
_REPORTED_CHILD = r"""
import importlib, json, sys
import torch
pkg = importlib.import_module(sys.argv[1])
loader = importlib.import_module(sys.argv[1] + ".cake_jit")
det = importlib.import_module(sys.argv[2])
H = int(sys.argv[4])
dev = torch.device("cuda:0")
arch = loader.arch_for_capability(tuple(torch.cuda.get_device_capability(dev)))
sm = loader.sm_count(0)
gen = torch.Generator(device="cpu").manual_seed(0)
def r(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(dtype).to(dev)
def rel_l2(a, b):
    num, den = (a.double() - b.double()).norm().item(), b.double().norm().item()
    return num / den if den > 0 else (0.0 if num == 0 else float("inf"))
names = ("dX", "dfn", "dscale", "dbase")
for T in (int(t) for t in sys.argv[3].split(",")):
    args = [r(T, 4, H, dtype=torch.bfloat16), r(T, 4, H, dtype=torch.bfloat16, scale=1e-3),
            r(T, H, dtype=torch.bfloat16, scale=1e-3), None, r(24, 4 * H, scale=0.05), r(3, scale=0.1),
            r(24), r(T, 64), r(T, 32, scale=1e-3)]
    layout = loader.workspace_regions("bwd", T, sm, arch, H)
    counts = layout["regions"]["slab_cnt"]
    prefix = counts["offset"] + counts["bytes"]
    key = (0, "bwd", H, 1 << (T - 1).bit_length())
    def nonzero():
        torch.cuda.synchronize()
        ws = pkg._DEFAULT_WORKSPACES.get(key)
        return 0 if ws is None else int(torch.count_nonzero(ws[:prefix]))
    call = {"T": T, "prefix": prefix, "prefix_before": nonzero()}
    print(json.dumps(call), flush=True)
    got = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args)]
    call["prefix_after"] = nonzero()
    call["default_found"] = key in pkg._DEFAULT_WORKSPACES
    fresh = torch.zeros(layout["total_bytes"], dtype=torch.uint8, device=dev)
    ref = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args, workspace=fresh)]
    want = det.cake_mhc_det_train_backward_finish(*args)
    torch.cuda.synchronize()
    call["finite"] = all(bool(torch.isfinite(t).all()) for t in got)
    call["differs"] = [n for n, a, b in zip(names, got, ref) if not torch.equal(a.view(-1).view(torch.uint8), b.view(-1).view(torch.uint8))]
    call["differs_det"] = [n for n, a, b in zip(names, got, want) if n != "dfn" and not torch.equal(a, b)]
    call["dfn_rel_l2_det"] = rel_l2(got[1], want[1])
    print(json.dumps(call), flush=True)
"""


@pytest.mark.parametrize(
    "Ts", REPORTED_SEQUENCES, ids=lambda Ts: "-".join(map(str, Ts))
)
def test_default_workspace_reported_sequence(Ts: tuple, hidden: int) -> None:
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            _REPORTED_CHILD,
            PACKAGES[KEY],
            PACKAGES["cake.cake_mhc_det_train"],
            ",".join(map(str, Ts)),
            str(hidden),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
    )
    calls = [
        json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")
    ]
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-6:])
    assert run.returncode == 0, (
        f"backward_finish() over {Ts} on the default workspace: exit {run.returncode}\n{tail}"
    )
    done = [c for c in calls if "differs" in c]
    assert [c["T"] for c in done] == list(Ts), tail
    for c in done:
        what = f"default workspace, backward_finish() at T={c['T']} H={hidden} of {Ts}"
        # The zero-prefix checks read the call's default buffer, not a missing one (a vacuous zero).
        assert c["default_found"], (
            f"{what}: no default workspace under the probed key (device, kind, hidden, bucket)"
        )
        assert c["prefix"] == ZERO_PREFIX_BYTES, (what, c["prefix"])
        assert c["prefix_before"] == 0, (
            f"{what}: {c['prefix_before']} non-zero bytes in the zero prefix before the call"
        )
        assert c["prefix_after"] == 0, (
            f"{what}: {c['prefix_after']} non-zero bytes in the zero prefix after the call"
        )
        assert c["finite"], f"{what}: non-finite outputs"
        assert not c["differs"], (
            f"{what}: {c['differs']} differ from the fresh-workspace call"
        )
        assert not c["differs_det"], (
            f"{what}: {c['differs_det']} differ from the deterministic call"
        )
        assert (
            math.isfinite(c["dfn_rel_l2_det"])
            and c["dfn_rel_l2_det"] <= FP32_REL_L2_MAX
        ), (
            f"{what}: dfn rel-L2 {c['dfn_rel_l2_det']:.3e} to the deterministic call > {FP32_REL_L2_MAX:.0e}"
        )


# A fresh process at the hidden size of the case: ``backward_finish(...)`` on a caller-owned
# workspace (every module loaded), then the same call with ``workspace=None`` inside a CUDA graph capture, before any
# default buffer of its bucket exists, then the same call eagerly.
_CAPTURE_CHILD = r"""
import importlib, json, sys
import torch
pkg = importlib.import_module(sys.argv[1])
dev = torch.device("cuda:0")
T, H = int(sys.argv[2]), int(sys.argv[3])
gen = torch.Generator(device="cpu").manual_seed(0)
def r(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(dtype).to(dev)
args = [r(T, 4, H, dtype=torch.bfloat16), r(T, 4, H, dtype=torch.bfloat16, scale=1e-3),
        r(T, H, dtype=torch.bfloat16, scale=1e-3), None, r(24, 4 * H, scale=0.05), r(3, scale=0.1),
        r(24), r(T, 64), r(T, 32, scale=1e-3)]
key = (0, "bwd", H, 1 << (T - 1).bit_length())
ws = pkg.cake_mhc_nondet_train_backward_workspace(T, dev, hidden=H)
ref = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args, workspace=ws)]
torch.cuda.synchronize()
out = {"cached_before": key in pkg._DEFAULT_WORKSPACES, "raised": None}
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        pkg.cake_mhc_nondet_train_backward_finish(*args)
except ValueError as e:
    out["raised"] = str(e)
del graph
torch.cuda.synchronize()
out["cached_after_capture"] = key in pkg._DEFAULT_WORKSPACES
got = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args)]
torch.cuda.synchronize()
out["cached_after_eager"] = key in pkg._DEFAULT_WORKSPACES
out["differs"] = [n for n, a, b in zip(("dX", "dfn", "dscale", "dbase"), got, ref) if not torch.equal(a.view(-1).view(torch.uint8), b.view(-1).view(torch.uint8))]
out["finite"] = all(bool(torch.isfinite(t).all()) for t in got)
print(json.dumps(out), flush=True)
"""


def test_default_workspace_is_not_created_during_capture(hidden: int) -> None:
    T = 300
    run = subprocess.run(
        [sys.executable, "-c", _CAPTURE_CHILD, PACKAGES[KEY], str(T), str(hidden)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-6:])
    assert run.returncode == 0, f"capture child: exit {run.returncode}\n{tail}"
    lines = [
        json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")
    ]
    assert len(lines) == 1, tail
    out = lines[0]
    what = f"backward_finish() at T={T} H={hidden} on the default workspace"
    assert not out["cached_before"], (
        f"{what}: a default buffer existed before the capture"
    )
    assert out["raised"] is not None and "CUDA graph capture" in out["raised"], (
        f"{what}: no ValueError inside the capture ({out['raised']!r})"
    )
    assert not out["cached_after_capture"], (
        f"{what}: the capture cached a default buffer"
    )
    assert out["cached_after_eager"], f"{what}: the eager call cached no default buffer"
    assert out["finite"], f"{what}: non-finite outputs"
    assert not out["differs"], (
        f"{what}: {out['differs']} differ from the caller-owned workspace's call"
    )


# A fresh process with two devices at the hidden size of the case: ``backward_finish(...)`` on a
# caller-owned workspace of each device (every module loaded), then the call with ``workspace=None`` on cuda:1
# tensors, before any default buffer of its bucket exists, (a) inside a capture on cuda:1 while cuda:0 is the current
# device and (b) inside a relaxed-mode capture on cuda:0 (the call's stream, cuda:1's current stream, is not capturing).
_CAPTURE_DEVICE_CHILD = r"""
import importlib, json, sys
import torch
pkg = importlib.import_module(sys.argv[1])
T, H = int(sys.argv[2]), int(sys.argv[3])
dev0, dev1 = torch.device("cuda:0"), torch.device("cuda:1")
gen = torch.Generator(device="cpu").manual_seed(0)
def r(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(dtype)
host = [r(T, 4, H, dtype=torch.bfloat16), r(T, 4, H, dtype=torch.bfloat16, scale=1e-3),
        r(T, H, dtype=torch.bfloat16, scale=1e-3), None, r(24, 4 * H, scale=0.05), r(3, scale=0.1),
        r(24), r(T, 64), r(T, 32, scale=1e-3)]
def on(dev):
    return [None if t is None else t.to(dev) for t in host]
names = ("dX", "dfn", "dscale", "dbase")
bucket = 1 << (T - 1).bit_length()
for dev in (dev0, dev1):
    ws = pkg.cake_mhc_nondet_train_backward_workspace(T, dev, hidden=H)
    pkg.cake_mhc_nondet_train_backward_finish(*on(dev), workspace=ws)
args1 = on(dev1)
ws1 = pkg.cake_mhc_nondet_train_backward_workspace(T, dev1, hidden=H)
ref = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args1, workspace=ws1)]
torch.cuda.synchronize(dev0)
torch.cuda.synchronize(dev1)
out = {"cached_before": [(i, "bwd", H, bucket) in pkg._DEFAULT_WORKSPACES for i in (0, 1)], "raised_a": None, "raised_b": None}
# Each capture gets its own stream: torch.cuda.graph's default capture stream is created once, on the device that
# was current at the first capture, and reused by later captures on any device.
capture1 = torch.cuda.Stream(device=dev1)
with torch.cuda.device(dev1):
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=capture1):
            with torch.cuda.device(dev0):
                out["stream_a"] = [torch.cuda.current_device(), torch.cuda.current_stream(dev1).cuda_stream == capture1.cuda_stream]
                pkg.cake_mhc_nondet_train_backward_finish(*args1)
    except ValueError as e:
        out["raised_a"] = str(e)
    del graph
torch.cuda.synchronize(dev1)
out["cached_after_a"] = (1, "bwd", H, bucket) in pkg._DEFAULT_WORKSPACES
y = torch.zeros(16, device=dev0)
got = None
capture0 = torch.cuda.Stream(device=dev0)
with torch.cuda.device(dev0):
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=capture0, capture_error_mode="relaxed"):
            y.add_(1)
            with torch.cuda.device(dev1):
                capturing1 = torch.cuda.is_current_stream_capturing()
            out["stream_b"] = [torch.cuda.current_device(), torch.cuda.current_stream(dev0).cuda_stream == capture0.cuda_stream, capturing1]
            got = [t.clone() for t in pkg.cake_mhc_nondet_train_backward_finish(*args1)]
    except ValueError as e:
        out["raised_b"] = str(e)
    del graph
torch.cuda.synchronize(dev0)
torch.cuda.synchronize(dev1)
out["cached_after_b"] = [(i, "bwd", H, bucket) in pkg._DEFAULT_WORKSPACES for i in (0, 1)]
out["ran_b"] = got is not None
if got is not None:
    out["differs"] = [n for n, a, b in zip(names, got, ref) if not torch.equal(a.view(-1).view(torch.uint8), b.view(-1).view(torch.uint8))]
    out["finite"] = all(bool(torch.isfinite(t).all()) for t in got)
print(json.dumps(out), flush=True)
"""


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_default_workspace_capture_check_uses_workspace_device(hidden: int) -> None:
    T = 300
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            _CAPTURE_DEVICE_CHILD,
            PACKAGES[KEY],
            str(T),
            str(hidden),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-6:])
    assert run.returncode == 0, (
        f"two-device capture child: exit {run.returncode}\n{tail}"
    )
    lines = [
        json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")
    ]
    assert len(lines) == 1, tail
    out = lines[0]
    what = f"backward_finish() at T={T} H={hidden} on cuda:1's default workspace"
    assert out["cached_before"] == [False, False], (
        f"{what}: a default buffer existed before the captures"
    )
    assert out.get("stream_a") == [0, True], (
        f"{what}: capture (a) did not run on cuda:1's stream with cuda:0 current ({out.get('stream_a')})"
    )
    assert out.get("stream_b") == [0, True, False], (
        f"{what}: capture (b) did not run on cuda:0's stream with cuda:1 idle ({out.get('stream_b')})"
    )
    assert out["raised_a"] is not None and "CUDA graph capture" in out["raised_a"], (
        f"{what}: no ValueError inside a capture on cuda:1 with cuda:0 current ({out['raised_a']!r})"
    )
    assert not out["cached_after_a"], (
        f"{what}: the capture on cuda:1 cached a default buffer"
    )
    assert out["raised_b"] is None, (
        f"{what}: ValueError inside a capture on cuda:0 ({out['raised_b']!r})"
    )
    assert out["ran_b"] and out["cached_after_b"] == [False, True], (
        f"{what}: the call during a capture on cuda:0 cached {out['cached_after_b']}"
    )
    assert out["finite"], f"{what}: non-finite outputs"
    assert not out["differs"], (
        f"{what}: {out['differs']} differ from the caller-owned workspace's call"
    )


# ----------------------------------------------------------------------------------------------------------------
# One buffer over a ragged sequence; the sizes


def _forward(i: dict) -> dict:
    """The forward prepare and the backward prepare of a case: the records ``backward_finish`` takes."""
    _U, stats = pkg.cake_mhc_nondet_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    _dB, gstats = pkg.cake_mhc_nondet_train_backward_prepare(
        i["dY"], i["B"], i["X"], stats
    )
    return {"stats": stats, "gstats": gstats}


def _backward(entry, i: dict, prep: dict, **kwargs) -> dict:
    out = entry(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"], **kwargs,
    )  # fmt: skip
    return dict(zip(_BACKWARD, out, strict=True))


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    return _backward(backward_finish, i, prep, **kwargs)


def _det_backward(i: dict, prep: dict) -> dict:
    return _backward(det.cake_mhc_det_train_backward_finish, i, prep)


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


def _assert_exact_outputs(got: dict, want: dict, label: str) -> None:
    """``dX``, ``dscale``, ``dbase``: bitwise the reference."""
    for k in _EXACT:
        assert _same_bits(got[k], want[k]), f"{k} {label}: bits differ"


def test_one_workspace_serves_a_ragged_token_sequence(hidden: int) -> None:
    """One caller workspace of this package's sizing for the largest token count, its bytes past the zero prefix
    0xFF before every call, over ``RAGGED_SEQUENCE`` with caller outputs holding NaN: every call finds the zero
    prefix zero and leaves it zero; every output is bitwise the fresh exact-size workspace's; ``dX``, ``dscale``,
    ``dbase`` are bitwise the deterministic backward's and ``dfn`` is within the bound of it."""
    t_max = max(RAGGED_SEQUENCE)
    _fwd_ws, ws = _workspaces(t_max, hidden)
    assert ws.numel() == loader.workspace_bytes_upto(
        "bwd", t_max, _sm_count(), _arch(), hidden
    )
    for n, T in enumerate(RAGGED_SEQUENCE):
        i = _inputs(T, hidden)
        prep = _forward(i)
        fresh = _snapshot(
            _nd_backward(i, prep, workspace=_nd_workspace_exact(T, hidden))
        )
        want = _snapshot(_det_backward(i, prep))
        bound = _bound(i)
        what = f"call {n} (T={T} H={hidden} on a buffer of T={t_max})"
        _poison_workspace(ws)
        _assert_zero_prefix(ws, what + " before the call")
        got = _snapshot(
            _nd_backward(i, prep, workspace=ws, **_poisoned_outputs(T, hidden))
        )
        _assert_zero_prefix(ws, what + " after the call")
        _assert_same_call(fresh, got, what + " vs the fresh workspace")
        _assert_exact_outputs(got, want, what + " vs deterministic")
        _assert_dfn_within(
            got["dfn"], want["dfn"], bound, f"dfn {what} vs deterministic"
        )


def test_workspace_sizes_and_which_sizing_serves_which_backward(hidden: int) -> None:
    """At the hidden size ``H`` of the case, this package's backward workspace is the ticket region, the column slab's
    arrival counts (``SLAB_CNT_WORDS`` uint32 at offset 256: the zero prefix), the deterministic regions but the
    per-chunk ``dfn`` partials (their sizes and shapes, the parameter region the deterministic one: the group rows
    inside ``BWD_ROW_BANDS``) and the per-chunk ``dfn`` slots (``min(32, ceil(T / 64)) x 24 x 4H`` FP32); its
    public size is the envelope over every token count up to ``T``, above the deterministic package's; the forward
    workspace is the deterministic one.  The size helper takes no ``deterministic`` keyword (``TypeError``).  A
    workspace of the deterministic sizing (``cake_mhc_det_train_backward_workspace``) is smaller than this package's
    exact layout and refused by its backward, as is one of this package's sizing for a smaller token count; one of this
    package's sizing serves the backward."""
    dev, sm, arch = _device(), _sm_count(), _arch()
    shift = ZERO_PREFIX_BYTES - 256
    for T in (1, 7, 63, 64, 65, 129, 257, 1000, 2049, 4096, 8192, 16263, 32768, 65536):
        ndl = loader.workspace_regions("bwd", T, sm, arch, hidden)
        dtl = det_loader.workspace_regions("bwd", T, sm, arch, hidden)
        names = list(dtl["regions"])
        assert names[-1] == "dfn_part", T
        assert list(ndl["regions"]) == [
            names[0],
            "slab_cnt",
            *names[1:-1],
            "dfn_slot",
        ], T
        assert ndl["regions"]["ticket"] == dtl["regions"]["ticket"], T
        assert ndl["regions"]["ticket"]["offset"] == 0
        assert ndl["regions"]["ticket"]["bytes"] == 4 * loader.TICKET_WORDS == 256
        assert ndl["regions"]["slab_cnt"] == {
            "offset": 256,
            "bytes": 4 * SLAB_CNT_WORDS,
            "dtype": "uint32",
            "shape": (SLAB_CNT_WORDS,),
        }, T
        for name in names[1:-1]:
            region = dtl["regions"][name]
            assert ndl["regions"][name] == {
                **region,
                "offset": region["offset"] + shift,
            }, (T, name)
        part, slots = dtl["regions"]["dfn_part"], ndl["regions"]["dfn_slot"]
        slot_parts = min(32, -(-T // 64))
        assert part["bytes"] == _parts(T, hidden) * MAPPING_ROWS * STREAMS * hidden * 4
        assert slots["shape"] == (slot_parts, MAPPING_ROWS, STREAMS * hidden), T
        assert slots["dtype"] == "float32" and slot_parts >= _parts(T, hidden), T
        assert ndl["total_bytes"] == (
            dtl["total_bytes"]
            - _round256(part["bytes"])
            + shift
            + _round256(slots["bytes"])
        ), T
        nd_bytes = pkg.cake_mhc_nondet_train_backward_workspace_bytes(
            T, dev, hidden=hidden
        )
        det_bytes = det.cake_mhc_det_train_backward_workspace_bytes(
            T, dev, hidden=hidden
        )
        assert nd_bytes == loader.workspace_bytes_upto("bwd", T, sm, arch, hidden), T
        assert ndl["total_bytes"] <= nd_bytes and nd_bytes % 256 == 0, T
        assert det_bytes < ndl["total_bytes"], (T, det_bytes, ndl["total_bytes"])
        assert pkg.cake_mhc_nondet_train_forward_workspace_bytes(
            T, dev, hidden=hidden
        ) == det.cake_mhc_det_train_forward_workspace_bytes(T, dev, hidden=hidden), T
    assert (
        pkg.cake_mhc_nondet_train_backward_workspace_bytes(0, dev, hidden=hidden) == 0
    )
    with pytest.raises(ValueError):
        pkg.cake_mhc_nondet_train_backward_workspace_bytes(-1, dev, hidden=hidden)
    for value in (False, True):
        with pytest.raises(TypeError, match="deterministic"):
            pkg.cake_mhc_nondet_train_backward_workspace_bytes(
                2049, dev, hidden=hidden, deterministic=value
            )
    allocated = pkg.cake_mhc_nondet_train_backward_workspace(2049, dev, hidden=hidden)
    assert allocated.dtype == torch.uint8 and allocated.device == dev
    assert allocated.numel() == pkg.cake_mhc_nondet_train_backward_workspace_bytes(
        2049, dev, hidden=hidden
    )
    assert int(torch.count_nonzero(allocated)) == 0

    T = 2049
    i = _inputs(T, hidden)
    prep = _forward(i)
    want = _snapshot(_det_backward(i, prep))
    bound = _bound(i)
    det_sized = det.cake_mhc_det_train_backward_workspace(T, dev, hidden=hidden)
    nd_sized = pkg.cake_mhc_nondet_train_backward_workspace(T, dev, hidden=hidden)
    got = _snapshot(_nd_backward(i, prep, workspace=nd_sized))
    _assert_exact_outputs(got, want, f"workspace nd-sized H={hidden}")
    _assert_dfn_within(
        got["dfn"], want["dfn"], bound, f"dfn workspace nd-sized H={hidden}"
    )
    _assert_zero_prefix(nd_sized, f"workspace nd-sized H={hidden}")
    with pytest.raises(ValueError, match="workspace holds"):
        _nd_backward(i, prep, workspace=det_sized)
    small = pkg.cake_mhc_nondet_train_backward_workspace(64, dev, hidden=hidden)
    with pytest.raises(ValueError, match="workspace holds"):
        _nd_backward(i, prep, workspace=small)


def test_workspace_bytes_per_hidden_size() -> None:
    """The workspace helpers size per hidden size (``test_cake_mhc_det_train.py``'s test of the same name): at each
    routed size the request is a multiple of 256 bytes, non-decreasing in the token count and at least the exact
    layout of the call; the forward request is the deterministic package's and the backward request above it at the
    same size (the backward requests are printed per size); an unrouted or unknown hidden size is a ``ValueError``;
    the default is 4096."""
    dev = _device()
    sizes = _hidden_sizes()
    unrouted = tuple(H for H in HIDDEN_SIZES if H not in sizes)
    for kind, layout in (("forward", "fwd"), ("backward", "bwd")):
        nbytes = getattr(pkg, f"cake_mhc_nondet_train_{kind}_workspace_bytes")
        allocate = getattr(pkg, f"cake_mhc_nondet_train_{kind}_workspace")
        reference_nbytes = getattr(det, f"cake_mhc_det_train_{kind}_workspace_bytes")
        for H in sizes:
            previous = 0
            for T in (1, 7, 128, 129, 1000, 2049, 8192, 16263, 65536):
                current = nbytes(T, dev, hidden=H)
                exact = _layout(layout, T, H)["total_bytes"]
                assert current % 256 == 0 and current >= max(previous, exact), (
                    kind,
                    H,
                    T,
                    current,
                    previous,
                    exact,
                )
                want = reference_nbytes(T, dev, hidden=H)
                if kind == "forward":
                    assert current == want, (kind, H, T, current, want)
                else:
                    assert current > want, (kind, H, T, current, want)
                previous = current
        assert nbytes(129, dev) == nbytes(129, dev, hidden=4096), kind
        for bad in (6144, 0, -4096, 4097, 2 * 7168, True) + unrouted:
            with pytest.raises(ValueError, match="hidden"):
                nbytes(129, dev, hidden=bad)
        for bad in (6144,) + unrouted:
            with pytest.raises(ValueError, match="hidden"):
                allocate(129, dev, hidden=bad)
    for T in (1, 129, 2049, 8192):
        backward = {
            H: pkg.cake_mhc_nondet_train_backward_workspace_bytes(T, dev, hidden=H)
            for H in sizes
        }
        print(f"backward workspace bytes T={T}: {backward}")


@pytest.mark.parametrize("T", (129, 2049))
def test_workspace_sized_for_another_hidden_size(T: int) -> None:
    """A caller workspace serves a call of this package when it holds the call's ``(T, H)`` layout and is refused
    (``ValueError``) when it is too small (``test_cake_mhc_det_train.py``'s test of the same name): buffers sized
    for ``(T, 4096)`` used by an H 7168 call at the same ``T`` are refused where smaller than its layout and
    otherwise give the default-workspace outputs; buffers one 256-byte block short of the H 7168 layout are refused
    and exact-size buffers serve it.  Every output, ``dfn`` included, is bitwise the default-workspace call's, and
    the zero prefix is zero after every backward."""
    if not {4096, 7168} <= set(_hidden_sizes()):
        pytest.skip(f"hidden 4096 and 7168 are not both routed on {_arch()}")
    dev = _device()
    i = _inputs(T, 7168)
    ref = _snapshot(_step(i, None, None))
    need = {kind: _layout(kind, T, 7168)["total_bytes"] for kind in ("fwd", "bwd")}
    sized_4096 = dict(zip(("fwd", "bwd"), _workspaces(T, 4096), strict=True))
    if T == 129:
        # The N-sized split of fn alone grows by more than the token-dependent regions of 129 tokens hold.
        held = sized_4096["fwd"].numel()
        assert held < need["fwd"], (held, need)
    print(
        f"T={T}: H 4096 buffers {sized_4096['fwd'].numel()} / {sized_4096['bwd'].numel()} bytes, H 7168 layouts "
        f"{need['fwd']} / {need['bwd']} bytes"
    )
    short = {
        kind: torch.zeros(need[kind] - 256, dtype=torch.uint8, device=dev)
        for kind in need
    }
    exact = {
        kind: torch.zeros(need[kind], dtype=torch.uint8, device=dev) for kind in need
    }
    args = (i["X"], i["fn"], i["scale"], i["base"])
    prep = {"stats": ref["stats"], "gstats": ref["gstats"]}
    for label, buffers in (("H 4096", sized_4096), ("short", short), ("exact", exact)):
        fwd_ws, bwd_ws = buffers["fwd"], buffers["bwd"]
        what = f"T={T} H=7168 on a {label} buffer"
        if fwd_ws.numel() < need["fwd"]:
            for entry in (
                pkg.cake_mhc_nondet_train_forward_prepare,
                pkg.cake_mhc_nondet_train_forward_stats,
            ):
                with pytest.raises(ValueError, match="workspace holds"):
                    entry(*args, workspace=fwd_ws)
        else:
            _U, stats = pkg.cake_mhc_nondet_train_forward_prepare(
                *args, workspace=fwd_ws
            )
            assert _same_bits(stats, ref["stats"]), f"{what}: forward stats"
        if bwd_ws.numel() < need["bwd"]:
            with pytest.raises(ValueError, match="workspace holds"):
                _nd_backward(i, prep, workspace=bwd_ws)
        else:
            got = _snapshot(_nd_backward(i, prep, workspace=bwd_ws))
            _assert_zero_prefix(bwd_ws, f"{what}: backward")
            _assert_same_call({k: ref[k] for k in _BACKWARD}, got, f"{what}: backward")
