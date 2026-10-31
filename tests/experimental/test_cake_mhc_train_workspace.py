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
package with the opt-in non-deterministic backward
(``flashinfer.experimental.cake.cake_mhc_train_backend``, every entry point with
``deterministic=False``; hidden 4096, 4 streams; SM100 / SM103 / SM107;
FlashInfer issue #8068), in the form of
``test_cake_mhc_det_train_workspace.py``.

The non-deterministic backward workspace is the deterministic layout without
its per-chunk ``dfn`` partials; its first region is the ticket word the
column-slab pass draws its work units from with a self-resetting ``atom.inc``
(a word that is not zero at launch makes the launch skip units or end away
from zero), at offset 0 for every token count.  A caller-owned buffer sized by
``cake_mhc_train_backward_workspace_bytes(T_max, deterministic=False)`` serves
every token count up to ``T_max`` and CUDA Graph replays at a smaller ``T``.
Before every call the bytes of the backward buffer past the ticket region are
overwritten with 0xFF (no call may depend on what an earlier call left there).
Every call is compared with the same call on fresh zeroed exact-size
workspaces and with the deterministic package
(``flashinfer.cake_mhc_det_train_backend``) on the same inputs: every output but
``dfn`` bitwise; ``dfn`` within ``2 P u sum_t |dz'_t| |x_t|`` per element
(``P`` the chunk count of the column-slab pass, ``u = 2^-24``, ``dz'`` the FP64
reference's).  On one buffer reused over realistic token-count sequences:

* every call finds the ticket words of its layout zero and leaves them zero;
* a CUDA Graph captured at a smaller token count on the large buffers, its
  replays interleaved with eager calls at the large token count, keeps both
  properties at every replay;
* the default workspaces (``workspace=None``: one buffer per power-of-two
  token-count bucket) across the token counts of one bucket;
* the sequence of FlashInfer issue #7763 (``backward_finish`` at 7832 tokens,
  then at 7435 tokens, on the default workspace of a fresh process, and the
  reverse order), each sequence in its own process;
* one buffer over a ragged sequence with caller outputs, and the sizes of the
  workspace and which sizing serves which backward.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import PACKAGES, cake_package

KEY = "cake.cake_mhc_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
# The deterministic package: the same-input comparison point.
det = cake_package("cake_mhc_det_train")
det_loader = cake_package("cake_mhc_det_train", "cake_jit")
backward_finish = pkg.cake_mhc_train_backward_finish

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, N, MAPPING_ROWS = 4, 4096, 16384, 24
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
BLOCK_TOKENS = 8192
FP32_REL_L2_MAX = 1e-5
FP32_UNIT_ROUNDOFF = 2.0**-24
ABS_SLACK = 1.02
TICKET_BYTES = (
    256  # the ticket region: first in the backward workspace for every token count
)
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


def _parts(T: int) -> int:
    """``P``: the chunk count of the column-slab pass, the number of partials each ``dfn`` element sums."""
    parts = loader.host_plan(T, _sm_count(), _arch())["parts"]
    assert 1 <= parts <= 32, (T, parts)
    return parts


def _round256(n: int) -> int:
    return 256 * (-(-n // 256))


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
    """The FP64 ``dz' = r dL/dz`` of the reference, ``[T, 24]`` (the backward of the coefficients alone, so no
    ``[T, 16384]`` autograd graph is held)."""
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
    """Per-element bound of ``|dfn_a - dfn_b|`` for two sums of the same ``P`` per-chunk partials in two orders:
    ``2 P u * 1.02 sum_t |dz'_t| |x_t|`` (``test_cake_mhc_train.py``'s ``_dfn_bound``)."""
    T = i["X"].shape[0]
    dzp = _dzp(i).abs()
    A = torch.zeros((MAPPING_ROWS, N), dtype=F64, device=_device())
    for lo in range(0, T, BLOCK_TOKENS):
        hi = min(T, lo + BLOCK_TOKENS)
        A += dzp[lo:hi].T @ i["X"][lo:hi].reshape(hi - lo, N).double().abs()
    return 2.0 * _parts(T) * FP32_UNIT_ROUNDOFF * ABS_SLACK * A


# ----------------------------------------------------------------------------------------------------------------
# Inputs, steps and comparisons


_INPUTS: dict[int, dict] = {}


def _inputs(T: int) -> dict:
    if T not in _INPUTS:
        g = torch.Generator(device=_device()).manual_seed(SEED + T)

        def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
            return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(
                dtype
            )

        _INPUTS.clear()  # one token count's inputs at a time (16263 tokens hold ~1.3 GB)
        _INPUTS[T] = {
            "X": randn(T, STREAMS, HIDDEN, dtype=BF16),
            "fn": (randn(MAPPING_ROWS, N) * 1e-2).contiguous(),
            "scale": 1.0 + 0.1 * randn(3),
            "base": 0.1 * randn(MAPPING_ROWS),
            "B": randn(T, HIDDEN, dtype=BF16),
            "dY": randn(T, STREAMS, HIDDEN, dtype=BF16),
            "dU": randn(T, HIDDEN, dtype=BF16),
        }
    return _INPUTS[T]


def _bound(i: dict) -> torch.Tensor:
    """The ``dfn`` bound of the inputs ``i`` (kept with them)."""
    if "bound" not in i:
        i["bound"] = _dfn_bound(i)
    return i["bound"]


def _step(i: dict, fwd_ws, bwd_ws, *, deterministic: bool = False) -> dict:
    """One forward + backward of this package (``deterministic=False``) or of the deterministic package."""
    if deterministic:
        entries = (
            det.cake_mhc_det_train_forward_prepare,
            det.cake_mhc_det_train_forward_finish,
            det.cake_mhc_det_train_backward_prepare,
            det.cake_mhc_det_train_backward_finish,
        )
        mode = {}
    else:
        entries = (
            pkg.cake_mhc_train_forward_prepare,
            pkg.cake_mhc_train_forward_finish,
            pkg.cake_mhc_train_backward_prepare,
            backward_finish,
        )
        mode = {"deterministic": False}
    fwd_prepare, fwd_finish, bwd_prepare, bwd_finish = entries
    U, stats = fwd_prepare(
        i["X"], i["fn"], i["scale"], i["base"], workspace=fwd_ws, **mode
    )
    Y = fwd_finish(i["X"], i["B"], stats, **mode)
    dB, gstats = bwd_prepare(i["dY"], i["B"], i["X"], stats, **mode)
    dX, dfn, dscale, dbase = bwd_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
        workspace=bwd_ws, **mode,
    )  # fmt: skip
    return dict(
        zip(OUTPUTS, (U, stats, Y, dB, gstats, dX, dfn, dscale, dbase), strict=True)
    )


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _nd_workspace_exact(T: int) -> torch.Tensor:
    """A fresh zeroed non-deterministic backward workspace of exactly ``T`` tokens' layout."""
    return torch.zeros(
        loader.workspace_bytes("bwd", T, _sm_count(), _arch()),
        dtype=torch.uint8,
        device=_device(),
    )


def _fresh(T: int) -> dict:
    """The outputs of one ``deterministic=False`` call on fresh zeroed exact-size workspaces."""
    fwd_ws = torch.zeros(
        loader.workspace_layout("fwd", T, device_index=_device().index)["total_bytes"],
        dtype=torch.uint8,
        device=_device(),
    )
    out = _snapshot(_step(_inputs(T), fwd_ws, _nd_workspace_exact(T)))
    torch.cuda.synchronize()
    return out


def _deterministic(T: int) -> dict:
    """The outputs of the deterministic package on its default workspaces."""
    out = _snapshot(_step(_inputs(T), None, None, deterministic=True))
    torch.cuda.synchronize()
    return out


def _poison_workspace(ws: torch.Tensor) -> None:
    """Every byte but the ticket region (first for every token count) set to 0xFF."""
    ws[TICKET_BYTES:].fill_(0xFF)


def _assert_ticket_zero(ws: torch.Tensor, what: str) -> None:
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(ws[:TICKET_BYTES]))
    assert nonzero == 0, f"{what}: {nonzero} non-zero bytes in the ticket region"


def _assert_tickets_zero(fwd_ws, bwd_ws, T: int, what: str) -> None:
    """The ticket words of the forward layout (the deterministic package's layout) and of the non-deterministic
    backward layout for ``T`` tokens are zero."""
    torch.cuda.synchronize()
    regions = {
        "fwd": loader.workspace_layout("fwd", T, device_index=_device().index)[
            "regions"
        ]["ticket"],
        "bwd": loader.workspace_regions("bwd", T, _sm_count(), _arch())["regions"][
            "ticket"
        ],
    }
    assert regions["bwd"]["offset"] == 0 and regions["bwd"]["bytes"] == TICKET_BYTES
    for ws, kind in ((fwd_ws, "fwd"), (bwd_ws, "bwd")):
        region = regions[kind]
        nonzero = int(
            torch.count_nonzero(
                ws[region["offset"] : region["offset"] + region["bytes"]]
            )
        )
        assert nonzero == 0, (
            f"{what}: {nonzero} non-zero bytes in the {kind} ticket words of T={T}"
        )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        a.detach().contiguous().reshape(-1).view(torch.uint8),
        b.detach().contiguous().reshape(-1).view(torch.uint8),
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
    """Every output of ``ref`` but ``dfn`` bitwise; ``dfn`` within the bound."""
    for k, v in ref.items():
        if k == "dfn":
            _assert_dfn_within(got[k], v, bound, f"{what}: dfn")
        else:
            assert _same_bits(v, got[k]), (
                f"{what}: {k} differs in {int((v != got[k]).sum())} elements"
            )


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_caller_workspace_reused_across_token_counts(name: str) -> None:
    Ts = SEQUENCES[name]
    t_max = max(Ts)
    fwd_ws = pkg.cake_mhc_train_forward_workspace(t_max, _device(), deterministic=False)
    bwd_ws = pkg.cake_mhc_train_backward_workspace(
        t_max, _device(), deterministic=False
    )
    for n, T in enumerate(Ts):
        bound = _bound(_inputs(T))
        ref, want = _fresh(T), _deterministic(T)
        what = f"{name} call {n} (T={T} on a buffer of T={t_max})"
        _poison_workspace(bwd_ws)
        _assert_tickets_zero(fwd_ws, bwd_ws, T, what + " before the call")
        got = _snapshot(_step(_inputs(T), fwd_ws, bwd_ws))
        _assert_tickets_zero(fwd_ws, bwd_ws, T, what + " after the call")
        _assert_matches(ref, got, bound, what + " vs the fresh workspaces")
        _assert_matches(want, got, bound, what + " vs deterministic")
        del ref, want, got


@pytest.mark.parametrize(("t_max", "t_small"), GRAPHS)
def test_graph_captured_at_a_smaller_token_count(t_max: int, t_small: int) -> None:
    fwd_ws = pkg.cake_mhc_train_forward_workspace(t_max, _device(), deterministic=False)
    bwd_ws = pkg.cake_mhc_train_backward_workspace(
        t_max, _device(), deterministic=False
    )
    small = _inputs(t_small)
    bound_small = _bound(small)
    ref_small, want_small = _fresh(t_small), _deterministic(t_small)
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
    big = _inputs(t_max)
    bound_big = _bound(big)
    ref_big, want_big = _fresh(t_max), _deterministic(t_max)
    for r in range(3):
        what = f"replay {r} at T={t_small} after an eager call at T={t_max}"
        _poison_workspace(bwd_ws)
        got_big = _snapshot(_step(big, fwd_ws, bwd_ws))
        _assert_tickets_zero(
            fwd_ws, bwd_ws, t_max, f"eager T={t_max} before replay {r}"
        )
        _assert_matches(
            ref_big, got_big, bound_big, f"eager T={t_max} before replay {r}"
        )
        _assert_matches(
            want_big,
            got_big,
            bound_big,
            f"eager T={t_max} before replay {r} vs deterministic",
        )
        del got_big
        _poison_workspace(bwd_ws)
        _assert_tickets_zero(fwd_ws, bwd_ws, t_small, what + " (before)")
        for t in out.values():
            t.fill_(float("nan"))
        graph.replay()
        _assert_tickets_zero(fwd_ws, bwd_ws, t_small, what + " (after)")
        got = _snapshot(out)
        _assert_matches(ref_small, got, bound_small, what)
        _assert_matches(want_small, got, bound_small, what + " vs deterministic")
    del graph


@pytest.mark.parametrize("Ts", DEFAULT_BUCKETS)
def test_default_workspace_bucket_across_token_counts(Ts: tuple) -> None:
    """``workspace=None`` over the token counts of one power-of-two bucket: the bucket's default buffer (allocated on
    the first call) has its bytes past the ticket region overwritten with 0xFF before every later call and its ticket
    region zero before and after every call."""
    index = _device().index
    for n, T in enumerate(Ts):
        bound = _bound(_inputs(T))
        ref, want = _fresh(T), _deterministic(T)
        what = f"default workspace call {n} (T={T}) of {Ts}"
        key = (index, "bwd", 1 << (T - 1).bit_length())
        default = pkg._DEFAULT_WORKSPACES.get(key)
        if default is not None:
            _poison_workspace(default)
            _assert_ticket_zero(default, what + " before the call")
        got = _snapshot(_step(_inputs(T), None, None))
        _assert_ticket_zero(pkg._DEFAULT_WORKSPACES[key], what + " after the call")
        _assert_matches(ref, got, bound, what)
        _assert_matches(want, got, bound, what + " vs deterministic")


# One process per sequence: the inputs of the issue's reproduction (one CPU generator seeded 0, drawn in call order),
# each ``backward_finish(..., deterministic=False)`` on the default workspace, then the same call on a fresh zeroed
# exact-size workspace and the deterministic call.  The inputs are not the records of a forward (``stats`` and
# ``grad_stats`` are random), so no FP64 reference applies: ``dfn`` is compared by its rel-L2.
_REPORTED_CHILD = r"""
import importlib, json, sys
import torch
pkg = importlib.import_module(sys.argv[1])
loader = importlib.import_module(sys.argv[1] + ".cake_jit")
det = importlib.import_module(sys.argv[2])
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
    args = [r(T, 4, 4096, dtype=torch.bfloat16), r(T, 4, 4096, dtype=torch.bfloat16, scale=1e-3),
            r(T, 4096, dtype=torch.bfloat16, scale=1e-3), None, r(24, 16384, scale=0.05), r(3, scale=0.1),
            r(24), r(T, 64), r(T, 32, scale=1e-3)]
    layout = loader.workspace_regions("bwd", T, sm, arch)
    ticket = layout["regions"]["ticket"]
    key = (0, "bwd", 1 << (T - 1).bit_length())
    def nonzero():
        torch.cuda.synchronize()
        ws = pkg._DEFAULT_WORKSPACES.get(key)
        return 0 if ws is None else int(torch.count_nonzero(ws[ticket["offset"] : ticket["offset"] + ticket["bytes"]]))
    call = {"T": T, "ticket_before": nonzero()}
    print(json.dumps(call), flush=True)
    got = [t.clone() for t in pkg.cake_mhc_train_backward_finish(*args, deterministic=False)]
    call["ticket_after"] = nonzero()
    fresh = torch.zeros(layout["total_bytes"], dtype=torch.uint8, device=dev)
    ref = [t.clone() for t in pkg.cake_mhc_train_backward_finish(*args, workspace=fresh, deterministic=False)]
    want = det.cake_mhc_det_train_backward_finish(*args)
    torch.cuda.synchronize()
    call["finite"] = all(bool(torch.isfinite(t).all()) for t in got)
    call["differs"] = [n for n, a, b in zip(names, got, ref) if n != "dfn" and not torch.equal(a, b)]
    call["differs_det"] = [n for n, a, b in zip(names, got, want) if n != "dfn" and not torch.equal(a, b)]
    call["dfn_rel_l2"] = rel_l2(got[1], ref[1])
    call["dfn_rel_l2_det"] = rel_l2(got[1], want[1])
    print(json.dumps(call), flush=True)
"""


@pytest.mark.parametrize(
    "Ts", REPORTED_SEQUENCES, ids=lambda Ts: "-".join(map(str, Ts))
)
def test_default_workspace_reported_sequence(Ts: tuple) -> None:
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            _REPORTED_CHILD,
            PACKAGES[KEY],
            PACKAGES["cake_mhc_det_train"],
            ",".join(map(str, Ts)),
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
        f"backward_finish(deterministic=False) over {Ts} on the default workspace: exit {run.returncode}\n{tail}"
    )
    done = [c for c in calls if "differs" in c]
    assert [c["T"] for c in done] == list(Ts), tail
    for c in done:
        what = f"default workspace, backward_finish(deterministic=False) at T={c['T']} of {Ts}"
        assert c["ticket_before"] == 0, (
            f"{what}: {c['ticket_before']} non-zero ticket bytes before the call"
        )
        assert c["ticket_after"] == 0, (
            f"{what}: {c['ticket_after']} non-zero ticket bytes after the call"
        )
        assert c["finite"], f"{what}: non-finite outputs"
        assert not c["differs"], (
            f"{what}: {c['differs']} differ from the fresh-workspace call"
        )
        assert not c["differs_det"], (
            f"{what}: {c['differs_det']} differ from the deterministic call"
        )
        for key, against in (
            ("dfn_rel_l2", "the fresh-workspace call"),
            ("dfn_rel_l2_det", "the deterministic call"),
        ):
            assert math.isfinite(c[key]) and c[key] <= FP32_REL_L2_MAX, (
                f"{what}: dfn rel-L2 {c[key]:.3e} to {against} > {FP32_REL_L2_MAX:.0e}"
            )


# ----------------------------------------------------------------------------------------------------------------
# One buffer over a ragged sequence; the sizes


def _forward(i: dict) -> dict:
    """The forward prepare and the backward prepare of a case: the records ``backward_finish`` takes."""
    _U, stats = pkg.cake_mhc_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], deterministic=False
    )
    _dB, gstats = pkg.cake_mhc_train_backward_prepare(
        i["dY"], i["B"], i["X"], stats, deterministic=False
    )
    return {"stats": stats, "gstats": gstats}


def _backward(entry, i: dict, prep: dict, **kwargs) -> dict:
    out = entry(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"], **kwargs,
    )  # fmt: skip
    return dict(zip(_BACKWARD, out, strict=True))


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    return _backward(backward_finish, i, prep, deterministic=False, **kwargs)


def _det_backward(i: dict, prep: dict) -> dict:
    return _backward(det.cake_mhc_det_train_backward_finish, i, prep)


def _poisoned_outputs(T: int) -> dict:
    """Caller-provided outputs holding NaN in every element (what the kernels must overwrite)."""
    nan = float("nan")
    return {
        "dX": torch.full((T, STREAMS, HIDDEN), nan, dtype=BF16, device=_device()),
        "dfn": torch.full((MAPPING_ROWS, N), nan, dtype=F32, device=_device()),
        "dscale": torch.full((3,), nan, dtype=F32, device=_device()),
        "dbase": torch.full((MAPPING_ROWS,), nan, dtype=F32, device=_device()),
    }


def _assert_exact_outputs(got: dict, want: dict, label: str) -> None:
    """``dX``, ``dscale``, ``dbase``: bitwise the reference."""
    for k in _EXACT:
        assert _same_bits(got[k], want[k]), f"{k} {label}: bits differ"


def test_one_workspace_serves_a_ragged_token_sequence() -> None:
    """One caller workspace of the non-deterministic sizing for the largest token count, its non-ticket bytes 0xFF
    before every call, over ``RAGGED_SEQUENCE`` with caller outputs holding NaN: every call finds the ticket region
    zero and leaves it zero; ``dX``, ``dscale``, ``dbase`` are bitwise the fresh exact-size workspace's and the
    deterministic backward's; ``dfn`` is within the bound of both."""
    t_max = max(RAGGED_SEQUENCE)
    ws = pkg.cake_mhc_train_backward_workspace(t_max, _device(), deterministic=False)
    assert ws.numel() == loader.workspace_bytes_upto("bwd", t_max, _sm_count(), _arch())
    for n, T in enumerate(RAGGED_SEQUENCE):
        i = _inputs(T)
        prep = _forward(i)
        fresh = _snapshot(_nd_backward(i, prep, workspace=_nd_workspace_exact(T)))
        want = _snapshot(_det_backward(i, prep))
        bound = _bound(i)
        what = f"call {n} (T={T} on a buffer of T={t_max})"
        _poison_workspace(ws)
        _assert_ticket_zero(ws, what + " before the call")
        got = _snapshot(_nd_backward(i, prep, workspace=ws, **_poisoned_outputs(T)))
        _assert_ticket_zero(ws, what + " after the call")
        _assert_exact_outputs(got, fresh, what + " vs the fresh workspace")
        _assert_exact_outputs(got, want, what + " vs deterministic")
        _assert_dfn_within(
            got["dfn"], fresh["dfn"], bound, f"dfn {what} vs the fresh workspace"
        )
        _assert_dfn_within(
            got["dfn"], want["dfn"], bound, f"dfn {what} vs deterministic"
        )


def test_workspace_sizes_and_which_sizing_serves_which_backward() -> None:
    """The non-deterministic backward workspace is the deterministic layout without its last region, the per-chunk
    ``dfn`` partials (``P x 24 x 16384`` FP32): the same regions at the same offsets; its public size is the envelope
    over every token count up to ``T``, below the deterministic package's; the forward workspace is the
    deterministic one.  A workspace of the deterministic sizing serves the non-deterministic backward; one of the
    non-deterministic sizing is refused by the deterministic backward (and by a larger token count)."""
    dev, sm, arch = _device(), _sm_count(), _arch()
    for T in (1, 7, 63, 64, 65, 129, 257, 1000, 2049, 4096, 8192, 16263, 32768, 65536):
        ndl = loader.workspace_regions("bwd", T, sm, arch)
        dtl = det_loader.workspace_regions("bwd", T, sm, arch)
        assert list(dtl["regions"])[-1] == "dfn_part", T
        part = dtl["regions"]["dfn_part"]
        assert list(ndl["regions"]) == list(dtl["regions"])[:-1], T
        for name, region in ndl["regions"].items():
            assert region == dtl["regions"][name], (T, name)
        assert ndl["regions"]["ticket"]["offset"] == 0
        assert (
            ndl["regions"]["ticket"]["bytes"] == TICKET_BYTES == 4 * loader.TICKET_WORDS
        )
        assert part["bytes"] == _parts(T) * MAPPING_ROWS * N * 4, T
        assert ndl["total_bytes"] == dtl["total_bytes"] - _round256(part["bytes"]), T
        nd_bytes = pkg.cake_mhc_train_backward_workspace_bytes(
            T, dev, deterministic=False
        )
        det_bytes = det.cake_mhc_det_train_backward_workspace_bytes(T, dev)
        assert nd_bytes == loader.workspace_bytes_upto("bwd", T, sm, arch), T
        assert ndl["total_bytes"] <= nd_bytes < det_bytes and nd_bytes % 256 == 0, T
        assert pkg.cake_mhc_train_forward_workspace_bytes(
            T, dev, deterministic=False
        ) == det.cake_mhc_det_train_forward_workspace_bytes(T, dev), T
    assert pkg.cake_mhc_train_backward_workspace_bytes(0, dev, deterministic=False) == 0
    with pytest.raises(ValueError):
        pkg.cake_mhc_train_backward_workspace_bytes(-1, dev, deterministic=False)
    for kwargs in ({}, {"deterministic": True}):
        with pytest.raises(ValueError, match="takes deterministic=False"):
            pkg.cake_mhc_train_backward_workspace_bytes(2049, dev, **kwargs)
    allocated = pkg.cake_mhc_train_backward_workspace(2049, dev, deterministic=False)
    assert allocated.dtype == torch.uint8 and allocated.device == dev
    assert allocated.numel() == pkg.cake_mhc_train_backward_workspace_bytes(
        2049, dev, deterministic=False
    )
    assert int(torch.count_nonzero(allocated)) == 0

    T = 2049
    i = _inputs(T)
    prep = _forward(i)
    want = _snapshot(_det_backward(i, prep))
    bound = _bound(i)
    det_sized = det.cake_mhc_det_train_backward_workspace(T, dev)
    nd_sized = pkg.cake_mhc_train_backward_workspace(T, dev, deterministic=False)
    for name, ws in (("deterministic-sized", det_sized), ("nd-sized", nd_sized)):
        got = _snapshot(_nd_backward(i, prep, workspace=ws))
        _assert_exact_outputs(got, want, f"workspace {name}")
        _assert_dfn_within(got["dfn"], want["dfn"], bound, f"dfn workspace {name}")
        _assert_ticket_zero(ws, f"workspace {name}")
    with pytest.raises(ValueError, match="workspace holds"):
        _backward(det.cake_mhc_det_train_backward_finish, i, prep, workspace=nd_sized)
    small = pkg.cake_mhc_train_backward_workspace(64, dev, deterministic=False)
    with pytest.raises(ValueError, match="workspace holds"):
        _nd_backward(i, prep, workspace=small)
