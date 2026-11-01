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

CUDA-graph capture of the entry points of the Cake four-stream mHC training
package with the opt-in non-deterministic backward
(``flashinfer.experimental.cake.cake_mhc_train_backend``, every entry point with
``deterministic=False``; hidden 4096, 4 streams; SM100 / SM103 / SM107;
FlashInfer issue #8068), in the form of ``test_cake_mhc_det_train_graphs.py``.

One training iteration of a layer -- prepare, an external branch ``B = U @ W``
(BF16, ``W`` fixed), finish, the loss and its backward -- is captured whole
with ``torch.cuda.graph`` after a warm-up on a side stream (the JIT modules and
the per-device default workspaces exist before the capture), in three forms:

* the package's direct entry points ``forward_prepare`` -> ``forward_finish``
  -> ``backward_prepare`` -> ``backward_finish`` (the branch backward
  ``dU = dB @ W^T`` in between);
* its autograd entry points ``cake_mhc_train_prepare`` /
  ``cake_mhc_train_finish`` and ``loss.backward()`` (``save_stats`` both
  ways);
* the same layer inside a non-reentrant ``torch.utils.checkpoint``
  (``preserve_rng_state=False``: the layer draws no random numbers);

and the autograd / checkpointed layer through
``torch.cuda.make_graphed_callables``.  The captured forms run with the
default workspaces and with caller-owned workspaces of the package (the
backward workspace of the non-deterministic sizing; its bytes past the ticket
region overwritten with 0xFF before every replay, its ticket region zero after
every replay).  Three replays, each after the static outputs and gradients were
overwritten with 0xFF bytes, are compared with the eager non-deterministic
iteration and the eager iteration of the deterministic package
(``flashinfer.cake_mhc_det_train_backend``) on the same inputs: every output and
gradient but ``dfn`` bitwise; ``dfn`` within ``2 P u (sum_t |dz'_t| |x_t|)``
per element (``P`` the chunk count of the column-slab pass, ``u = 2^-24``,
``dz'`` the FP64 reference's).  The graphed callables get new input values on
each of three calls, each compared the same way with eager calls on the same
values.  A captured ``backward_finish`` on a caller workspace sized for a
larger token count, replayed with new inputs copied in place between eager
calls at the larger count, closes the module.
"""

from __future__ import annotations

import functools
import gc

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers.cake_package import cake_package

KEY = "cake.cake_mhc_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
# The deterministic package: the same-input comparison point.
det = cake_package("cake_mhc_det_train")
det_loader = cake_package("cake_mhc_det_train", "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, N, MAPPING_ROWS = 4, 4096, 16384, 24
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
BLOCK_TOKENS = 8192
SEED = 2032
REPLAYS = 3
FP32_UNIT_ROUNDOFF = 2.0**-24
ABS_SLACK = 1.02
TICKET_BYTES = (
    256  # the ticket region: first in the backward workspace for every token count
)
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")
_EXACT = ("dX", "dscale", "dbase")
_FN_KEYS = ("dfn", "grad_fn")  # the outputs summed with atomic reductions


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


def _parts(T: int) -> int:
    """``P``: the chunk count of the column-slab pass, the number of partials each ``dfn`` element sums."""
    parts = loader.host_plan(T, _sm_count(), _arch())["parts"]
    assert 1 <= parts <= 32, (T, parts)
    return parts


def _lpt_runs(T: int) -> bool:
    """Whether the column slab runs its two-size ``lpt`` record at ``T`` on this device."""
    return loader.bwd_slab_atomic_program(T, _sm_count(), _arch()) == "lpt"


def _slab_parts(T: int) -> int:
    """The chunk count the column slab launches: ``slab_plan``'s ``lpt_parts`` under ``lpt``, else :func:`_parts`."""
    if not _lpt_runs(T):
        return _parts(T)
    parts = loader.slab_plan(T, _sm_count(), _arch())["lpt_parts"]
    assert 1 <= parts <= 32, (T, parts)
    return parts


def _dfn_factor(T: int) -> float:
    """F of the dfn bound F u (|carried| + ABS_SLACK sum_t |dz'_t| |x_t|) at T (``test_cake_mhc_train.py``'s
    ``_dfn_factor``, which derives it): 2 P where the column slab runs dynr (both sides sum the same per-chunk
    partials), 2 max(P, P_k7) + 10 where it runs lpt (the partials are another partition's)."""
    if not _lpt_runs(T):
        return 2.0 * _parts(T)
    return 2.0 * max(_slab_parts(T), _parts(T)) + 10.0


# ----------------------------------------------------------------------------------------------------------------
# The dfn bound (test_cake_mhc_train.py's)


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
    """The FP64 ``dz' = r dL/dz`` of the reference, ``[T, 24]``: ``dfn = dz'^T X_flat`` sums one product per token
    (the backward of the coefficients alone, so no ``[T, 16384]`` autograd graph is held)."""
    T = i["X"].shape[0]
    fn, scale, base = (i[k].double() for k in ("fn", "scale", "base"))
    blocks = [torch.zeros((0, MAPPING_ROWS), dtype=F64, device=_device())]
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        X = i["X"][lo:hi].double()
        dY = i["dY"][lo:hi].double()
        dU = i["dU"][lo:hi].double()
        B = i["B"][lo:hi].double()
        u = X.reshape(hi - lo, N)
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
    """Per-element bound of ``|dfn_a - dfn_b|`` for two sums of per-chunk partials in two orders:
    ``F u * 1.02 sum_t |dz'_t| |x_t|`` with ``F`` = :func:`_dfn_factor` (``test_cake_mhc_train.py``'s
    ``_dfn_bound``)."""
    T = i["X"].shape[0]
    dzp = _dzp(i).abs()
    A = torch.zeros((MAPPING_ROWS, N), dtype=F64, device=_device())
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        A += dzp[lo:hi].T @ i["X"][lo:hi].reshape(hi - lo, N).double().abs()
    return _dfn_factor(T) * FP32_UNIT_ROUNDOFF * ABS_SLACK * A


def _layer_bound(i: dict, W: torch.Tensor) -> torch.Tensor:
    """The ``dfn`` bound of one layer iteration (branch ``B = U @ W``, loss gradient ``dY``): ``B`` and ``dU = dB @
    W^T`` from the deterministic direct entry points."""
    X, fn, scale, base = (i[k] for k in ("X", "fn", "scale", "base"))
    U, stats = det.cake_mhc_det_train_forward_prepare(X, fn, scale, base)
    B = U @ W
    dB, _gstats = det.cake_mhc_det_train_backward_prepare(i["dY"], B, X, stats)
    return _dfn_bound({**i, "B": B, "dU": dB @ W.t()})


# ----------------------------------------------------------------------------------------------------------------
# Inputs, steps and comparisons


def _inputs(T: int, seed: int) -> dict:
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    return {
        "X": randn(T, STREAMS, HIDDEN, dtype=BF16),
        "fn": (randn(MAPPING_ROWS, N) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(MAPPING_ROWS),
        "W": (randn(HIDDEN, HIDDEN) / 64.0).to(BF16),
        "dY": randn(T, STREAMS, HIDDEN, dtype=BF16),
        # the branch output and its gradient of the backward_finish-only capture
        "B": randn(T, HIDDEN, dtype=BF16),
        "dU": randn(T, HIDDEN, dtype=BF16),
    }


def _workspaces(T: int, caller_owned: bool):
    if not caller_owned:
        return None, None
    return (
        pkg.cake_mhc_train_forward_workspace(T, _device(), deterministic=False),
        pkg.cake_mhc_train_backward_workspace(T, _device(), deterministic=False),
    )


def _entries(deterministic: bool) -> dict:
    """The entry points of one package: this package's with ``deterministic=False``, or the deterministic package's
    (the eager reference)."""
    if deterministic:
        return {
            "forward_prepare": det.cake_mhc_det_train_forward_prepare,
            "forward_finish": det.cake_mhc_det_train_forward_finish,
            "backward_prepare": det.cake_mhc_det_train_backward_prepare,
            "backward_finish": det.cake_mhc_det_train_backward_finish,
            "prepare": det.cake_mhc_det_train_prepare,
            "finish": det.cake_mhc_det_train_finish,
        }
    names = (
        "forward_prepare",
        "forward_finish",
        "backward_prepare",
        "backward_finish",
        "prepare",
        "finish",
    )
    return {
        name: functools.partial(
            getattr(pkg, f"cake_mhc_train_{name}"), deterministic=False
        )
        for name in names
    }


def _leaves(i: dict) -> list:
    return [
        i[k].detach().clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")
    ]


def _layer(i: dict, wsf, wsb, save_stats: bool, deterministic: bool = False):
    api = _entries(deterministic)

    def layer(X, fn, scale, base):
        U, handle = api["prepare"](
            X,
            fn,
            scale,
            base,
            save_stats=save_stats,
            workspace=wsf,
            backward_workspace=wsb,
        )
        return api["finish"](X, U @ i["W"], handle)

    return layer


def _direct_step(i: dict, wsf, wsb, deterministic: bool = False):
    X, fn, scale, base, W, dY = (i[k] for k in ("X", "fn", "scale", "base", "W", "dY"))
    api = _entries(deterministic)

    def step() -> dict:
        U, stats = api["forward_prepare"](X, fn, scale, base, workspace=wsf)
        B = U @ W
        Y = api["forward_finish"](X, B, stats)
        dB, gstats = api["backward_prepare"](dY, B, X, stats)
        dX, dfn, dscale, dbase = api["backward_finish"](
            X, dY, dB @ W.t(), B, fn, scale, base, stats, gstats, workspace=wsb,
        )  # fmt: skip
        return {
            "U": U,
            "stats": stats,
            "Y": Y,
            "dB": dB,
            "grad_stats": gstats,
            "dX": dX,
            "dfn": dfn,
            "dscale": dscale,
            "dbase": dbase,
        }

    return step, ()


def _autograd_step(
    i: dict, wsf, wsb, *, save_stats: bool, ckpt: bool, deterministic: bool = False
):
    layer = _layer(i, wsf, wsb, save_stats, deterministic)
    leaves = _leaves(i)
    dY = i["dY"].float()

    def step() -> dict:
        if ckpt:
            Y = checkpoint(
                layer, *leaves, use_reentrant=False, preserve_rng_state=False
            )
        else:
            Y = layer(*leaves)
        (Y.float() * dY).sum().backward()
        return {"Y": Y, **_grads(leaves)}

    return step, leaves


def _grads(leaves) -> dict:
    return {
        f"grad_{n}": t.grad
        for n, t in zip(("X", "fn", "scale", "base"), leaves, strict=True)
    }


def _reset(leaves) -> None:
    for t in leaves:
        t.grad = None


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _poison(out: dict) -> None:
    for v in out.values():
        v.detach().reshape(-1).view(torch.uint8).fill_(0xFF)


def _poisoned_outputs(T: int) -> dict:
    """Caller-provided outputs holding NaN in every element (what the kernels must overwrite)."""
    nan = float("nan")
    return {
        "dX": torch.full((T, STREAMS, HIDDEN), nan, dtype=BF16, device=_device()),
        "dfn": torch.full((MAPPING_ROWS, N), nan, dtype=F32, device=_device()),
        "dscale": torch.full((3,), nan, dtype=F32, device=_device()),
        "dbase": torch.full((MAPPING_ROWS,), nan, dtype=F32, device=_device()),
    }


def _poison_workspace(ws: torch.Tensor) -> None:
    """Every byte but the ticket region (first for every token count) set to 0xFF."""
    ws[TICKET_BYTES:].fill_(0xFF)


def _assert_ticket_zero(ws: torch.Tensor, what: str) -> None:
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(ws[:TICKET_BYTES]))
    assert nonzero == 0, f"{what}: {nonzero} non-zero bytes in the ticket region"


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        a.detach().reshape(-1).view(torch.uint8),
        b.detach().reshape(-1).view(torch.uint8),
    )


def _assert_dfn_within(
    got: torch.Tensor, want: torch.Tensor, bound: torch.Tensor, label: str
) -> None:
    assert got.dtype == F32 and tuple(got.shape) == (MAPPING_ROWS, N), label
    assert torch.isfinite(got).all(), f"{label}: non-finite dfn"
    diff = (got.double() - want.double()).abs()
    over = int((diff > bound).sum())
    ratio = float((diff / bound.clamp_min(torch.finfo(F64).tiny)).max())
    assert over == 0, (
        f"{label}: {over} dfn elements beyond 2 P u sum_t |dz'_t| |x_t| (largest ratio {ratio:.3e})"
    )


def _assert_matches(ref: dict, got: dict, bound: torch.Tensor, what: str) -> None:
    """Every output and gradient but ``dfn`` bitwise ``ref``'s; ``dfn`` within the bound."""
    for k, v in ref.items():
        if k in _FN_KEYS:
            _assert_dfn_within(got[k], v, bound, f"{what}: {k}")
        else:
            assert _same_bits(v, got[k]), (
                f"{what}: {k} differs from the eager iteration"
            )


def _eager(make) -> dict:
    step, leaves = make()
    _reset(leaves)
    return _snapshot(step())


def _capture_and_replay(make, ws=None, what: str = "") -> list:
    """``REPLAYS`` replays of the captured iteration, each after its static outputs (and the bytes of ``ws`` past the
    ticket region) were overwritten with 0xFF; ``ws``'s ticket region is zero after every replay."""
    step, leaves = make()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _reset(leaves)
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    _reset(leaves)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    replays = []
    for r in range(REPLAYS):
        _poison(out)
        if ws is not None:
            _poison_workspace(ws)
        graph.replay()
        torch.cuda.synchronize()
        if ws is not None:
            _assert_ticket_zero(ws, f"{what} replay {r}")
        replays.append(_snapshot(out))
    return replays


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("T", (1, 129, 2049))
@pytest.mark.parametrize("caller_owned", (False, True))
def test_direct_entry_points_capture(T: int, caller_owned: bool) -> None:
    i = _inputs(T, seed=7046 + T)
    wsf, wsb = _workspaces(T, caller_owned)
    bound = _layer_bound(i, i["W"])
    eager = _eager(lambda: _direct_step(i, wsf, wsb))
    eager_det = _eager(lambda: _direct_step(i, None, None, deterministic=True))
    what = f"direct T={T} caller_owned={caller_owned}"
    replays = _capture_and_replay(lambda: _direct_step(i, wsf, wsb), wsb, what)
    for r, got in enumerate(replays):
        _assert_matches(eager, got, bound, f"{what} replay {r} vs eager")
        _assert_matches(eager_det, got, bound, f"{what} replay {r} vs deterministic")


@pytest.mark.parametrize("T", (1, 129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("ckpt", (False, True))
@pytest.mark.parametrize("caller_owned", (False, True))
def test_autograd_iteration_capture(
    T: int, save_stats: bool, ckpt: bool, caller_owned: bool
) -> None:
    i = _inputs(T, seed=7146 + T)
    wsf, wsb = _workspaces(T, caller_owned)
    bound = _layer_bound(i, i["W"])

    def make(deterministic: bool = False):
        if deterministic:
            return _autograd_step(
                i, None, None, save_stats=save_stats, ckpt=ckpt, deterministic=True
            )
        return _autograd_step(i, wsf, wsb, save_stats=save_stats, ckpt=ckpt)

    eager = _eager(make)
    eager_det = _eager(lambda: make(deterministic=True))
    what = f"autograd ckpt={ckpt} save_stats={save_stats} caller_owned={caller_owned} T={T}"
    for r, got in enumerate(_capture_and_replay(make, wsb, what)):
        _assert_matches(eager, got, bound, f"{what} replay {r} vs eager")
        _assert_matches(eager_det, got, bound, f"{what} replay {r} vs deterministic")


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("ckpt", (False, True))
def test_make_graphed_callables(T: int, save_stats: bool, ckpt: bool) -> None:
    i = _inputs(T, seed=7246 + T)

    def forward(deterministic: bool):
        layer = _layer(i, None, None, save_stats, deterministic)
        if not ckpt:
            return layer

        def fwd(X, fn, scale, base):
            return checkpoint(
                layer, X, fn, scale, base, use_reentrant=False, preserve_rng_state=False
            )

        return fwd

    def run(f, inp: dict) -> dict:
        leaves = _leaves(inp)
        Y = f(*leaves)
        (Y.float() * inp["dY"].float()).sum().backward()
        return _snapshot({"Y": Y, **_grads(leaves)})

    fwd, fwd_det = forward(False), forward(True)
    graphed = torch.cuda.make_graphed_callables(
        fwd, tuple(_leaves(i)), num_warmup_iters=3
    )
    for r in range(REPLAYS):
        # New values of the same shapes on every call: a call whose graphs did
        # not replay would return the previous call's outputs and fail.
        ir = _inputs(T, seed=7246 + T + 1000 * (r + 1))
        bound = _layer_bound(ir, i["W"])  # the layer's W is the first draw's
        ref, ref_det = run(fwd, ir), run(fwd_det, ir)
        got = run(graphed, ir)
        what = (
            f"make_graphed_callables ckpt={ckpt} save_stats={save_stats} T={T} call {r}"
        )
        _assert_matches(ref, got, bound, f"{what} vs eager")
        _assert_matches(ref_det, got, bound, f"{what} vs deterministic")


def _backward(entry, i: dict, prep: dict, **kwargs) -> dict:
    out = entry(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"], **kwargs,
    )  # fmt: skip
    return dict(zip(_OUTPUTS, out, strict=True))


def _forward(i: dict) -> dict:
    _U, stats = det.cake_mhc_det_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    _dB, gstats = det.cake_mhc_det_train_backward_prepare(
        i["dY"], i["B"], i["X"], stats
    )
    return {"stats": stats, "gstats": gstats}


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    return _backward(
        pkg.cake_mhc_train_backward_finish, i, prep, deterministic=False, **kwargs
    )


def _det_backward(i: dict, prep: dict) -> dict:
    return _backward(det.cake_mhc_det_train_backward_finish, i, prep)


def _assert_exact_outputs(got: dict, want: dict, label: str) -> None:
    for k in _EXACT:
        assert _same_bits(got[k], want[k]), f"{k} {label}: bits differ"


@pytest.mark.parametrize(
    ("T", "t_ws"), ((129, 129), (2049, 2049), (2049, 16263), (1, 4096))
)
def test_cuda_graph_capture_and_replay(T: int, t_ws: int) -> None:
    """A ``deterministic=False`` ``backward_finish`` with a caller workspace (sized for ``t_ws >= T``) and caller
    outputs, captured in a ``torch.cuda.CUDAGraph`` after a warm-up on a side stream, then replayed with new inputs
    copied in place (and, on a larger buffer, an eager call at ``t_ws`` on the same workspace before each replay):
    every replay leaves the ticket region zero and matches the eager deterministic and non-deterministic calls on the
    same inputs (``dX``, ``dscale``, ``dbase`` bitwise; ``dfn`` within the bound)."""
    dev = _device()
    ws = pkg.cake_mhc_train_backward_workspace(t_ws, dev, deterministic=False)
    static = _inputs(T, seed=SEED + 600)
    sprep = _forward(static)
    outs = _poisoned_outputs(T)

    def step() -> None:
        pkg.cake_mhc_train_backward_finish(
            static["X"], static["dY"], static["dU"], static["B"], static["fn"], static["scale"], static["base"],
            sprep["stats"], sprep["gstats"], workspace=ws, deterministic=False, **outs,
        )  # fmt: skip

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()
    big = None
    if t_ws > T:
        bi = _inputs(t_ws, seed=SEED + 601)
        bprep = _forward(bi)
        big = {
            "i": bi,
            "prep": bprep,
            "want": _snapshot(_det_backward(bi, bprep)),
            "bound": _dfn_bound(bi),
        }
    for r in range(REPLAYS):
        new = _inputs(T, seed=SEED + 610 + r)
        nprep = _forward(new)
        want = _snapshot(_det_backward(new, nprep))
        eager = _snapshot(_nd_backward(new, nprep))
        bound = _dfn_bound(new)
        for k in ("X", "dY", "dU", "B", "fn", "scale", "base"):
            static[k].copy_(new[k])
        sprep["stats"].copy_(nprep["stats"])
        sprep["gstats"].copy_(nprep["gstats"])
        what = f"replay {r} at T={T} on a workspace of T={t_ws}"
        if big is not None:
            _poison_workspace(ws)
            got_big = _snapshot(_nd_backward(big["i"], big["prep"], workspace=ws))
            _assert_ticket_zero(ws, f"eager T={t_ws} before {what}")
            _assert_exact_outputs(got_big, big["want"], f"eager T={t_ws} before {what}")
            _assert_dfn_within(
                got_big["dfn"],
                big["want"]["dfn"],
                big["bound"],
                f"dfn eager T={t_ws} before {what}",
            )
        _poison_workspace(ws)
        for v in outs.values():
            v.fill_(float("nan"))
        graph.replay()
        _assert_ticket_zero(ws, what)
        got = _snapshot(outs)
        _assert_exact_outputs(got, want, what + " vs eager deterministic")
        _assert_exact_outputs(got, eager, what + " vs eager non-deterministic")
        _assert_dfn_within(
            got["dfn"], want["dfn"], bound, f"dfn {what} vs eager deterministic"
        )
        _assert_dfn_within(
            got["dfn"], eager["dfn"], bound, f"dfn {what} vs eager non-deterministic"
        )
    del graph
