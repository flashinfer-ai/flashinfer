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

Workspace reuse across token counts for the Cake four-stream mHC training
package (hidden 4096, 4 streams; SM100 / SM103 / SM107).

A caller-owned workspace sized by ``cake_mhc_train_{forward,backward}_workspace_bytes(T_max)``
serves every token count up to ``T_max`` and CUDA Graph replays at a smaller
``T``; its ticket words must be zero on first use and the kernels leave them
zero.  The dynamic-schedule column-slab form (sm_103a up to 4096 tokens,
sm_100a above 4096 up to 8192 tokens) draws its work units from the backward
workspace's ticket word with a self-resetting ``atom.inc``: a word that is not
zero at launch makes the launch skip units (wrong ``dX`` / ``dfn``) or end away
from zero.  The ticket region therefore sits at the same offset for every
token count.  On one buffer reused over realistic token-count sequences:

* every call finds the ticket words of its layout zero and leaves them zero,
  and every output is bitwise the output of the same call on a fresh
  exact-size workspace;
* a CUDA Graph captured at a smaller token count on the large buffers, its
  replays interleaved with eager calls at the large token count, keeps both
  properties at every replay;
* the default workspaces (``workspace=None``: one buffer per power-of-two
  token-count bucket) give bitwise the fresh-workspace outputs across the
  token counts of one bucket;
* the sequence of FlashInfer issue #7763 (``backward_finish`` at 7832 tokens,
  then at 7435 tokens, on the default workspace of a fresh process, and the
  reverse order) finds the ticket words of every call zero before and after
  the call and gives bitwise the fresh-workspace outputs.  Each sequence runs
  in its own process, so its default workspaces start unused and a sticky
  CUDA error cannot reach the other tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import PACKAGES, cake_package

pkg = cake_package("cake_mhc_train")
loader = cake_package("cake_mhc_train", "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, N, MAPPING_ROWS = 4, 4096, 16384, 24
BF16, F32 = torch.bfloat16, torch.float32
SEED = 7046
# Token-count sequences of one buffer (the largest first: the buffer is sized for it).  The first two run the
# dynamic-schedule column-slab form on sm_103a / sm_100a at every token count but 1024; the third is the per-step
# token count of a GLM-style training run.
SEQUENCES = {
    "slab_dynamic_sm103": (4096, 3000, 1024, 4096, 2049, 3000),
    "slab_dynamic_sm100": (8192, 6000, 4500, 8192, 5000),
    "glm_rows": (16263, 14928, 15566, 16263, 15288, 16164),
}
GRAPHS = ((4096, 3000), (8192, 4500), (16263, 14928))
DEFAULT_BUCKETS = ((4096, 3000, 2100, 4096), (8192, 6000, 4500, 8192))
# The backward token-count sequences of FlashInfer issue #7763 (one 8192-token default bucket).
REPORTED_SEQUENCES = ((7832, 7435), (7435, 7832))
OUTPUTS = ("U", "stats", "Y", "dB", "gstats", "dX", "dfn", "dscale", "dbase")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(_device()) in SUPPORTED_CAPABILITIES
        and pkg.is_cake_mhc_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(torch.cuda.get_device_capability(_device()))
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


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


def _step(i: dict, fwd_ws, bwd_ws) -> dict:
    U, stats = pkg.cake_mhc_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], workspace=fwd_ws
    )
    Y = pkg.cake_mhc_train_forward_finish(i["X"], i["B"], stats)
    dB, gstats = pkg.cake_mhc_train_backward_prepare(i["dY"], i["B"], i["X"], stats)
    dX, dfn, dscale, dbase = pkg.cake_mhc_train_backward_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
        workspace=bwd_ws,
    )  # fmt: skip
    return dict(
        zip(OUTPUTS, (U, stats, Y, dB, gstats, dX, dfn, dscale, dbase), strict=True)
    )


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _fresh(T: int) -> dict:
    """The outputs of one call on fresh zeroed exact-size workspaces."""
    fwd_ws = torch.zeros(
        loader.workspace_layout("fwd", T, device_index=_device().index)["total_bytes"],
        dtype=torch.uint8,
        device=_device(),
    )
    bwd_ws = torch.zeros(
        loader.workspace_layout("bwd", T, device_index=_device().index)["total_bytes"],
        dtype=torch.uint8,
        device=_device(),
    )
    out = _snapshot(_step(_inputs(T), fwd_ws, bwd_ws))
    torch.cuda.synchronize()
    return out


def _assert_bitwise(ref: dict, got: dict, what: str) -> None:
    for k in OUTPUTS:
        a, b = ref[k].contiguous(), got[k].contiguous()
        same = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        assert same, (
            f"{what}: {k} differs from the fresh-workspace call in {int((a != b).sum())} elements"
        )


def _ticket_words(ws: torch.Tensor, kind: str, T: int) -> int:
    """Non-zero bytes in the ticket region of ``kind``'s layout for ``T`` tokens."""
    region = loader.workspace_layout(kind, T, device_index=_device().index)["regions"][
        "ticket"
    ]
    return int(
        torch.count_nonzero(ws[region["offset"] : region["offset"] + region["bytes"]])
    )


def _assert_tickets_zero(fwd_ws, bwd_ws, T: int, what: str) -> None:
    torch.cuda.synchronize()
    for ws, kind in ((fwd_ws, "fwd"), (bwd_ws, "bwd")):
        nonzero = _ticket_words(ws, kind, T)
        assert nonzero == 0, (
            f"{what}: {nonzero} non-zero bytes in the {kind} ticket words of T={T}"
        )


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_caller_workspace_reused_across_token_counts(name: str) -> None:
    Ts = SEQUENCES[name]
    t_max = max(Ts)
    fwd_ws = pkg.cake_mhc_train_forward_workspace(t_max, _device())
    bwd_ws = pkg.cake_mhc_train_backward_workspace(t_max, _device())
    for n, T in enumerate(Ts):
        ref = _fresh(T)
        what = f"{name} call {n} (T={T} on a buffer of T={t_max})"
        _assert_tickets_zero(fwd_ws, bwd_ws, T, what + " before the call")
        got = _snapshot(_step(_inputs(T), fwd_ws, bwd_ws))
        _assert_tickets_zero(fwd_ws, bwd_ws, T, what + " after the call")
        _assert_bitwise(ref, got, what)


@pytest.mark.parametrize(("t_max", "t_small"), GRAPHS)
def test_graph_captured_at_a_smaller_token_count(t_max: int, t_small: int) -> None:
    fwd_ws = pkg.cake_mhc_train_forward_workspace(t_max, _device())
    bwd_ws = pkg.cake_mhc_train_backward_workspace(t_max, _device())
    ref_small = _fresh(t_small)
    small = _inputs(t_small)
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
    ref_big = _fresh(t_max)
    big = _inputs(t_max)
    for r in range(3):
        what = f"replay {r} at T={t_small} after an eager call at T={t_max}"
        got_big = _snapshot(_step(big, fwd_ws, bwd_ws))
        _assert_tickets_zero(
            fwd_ws, bwd_ws, t_max, f"eager T={t_max} before replay {r}"
        )
        _assert_bitwise(ref_big, got_big, f"eager T={t_max} before replay {r}")
        _assert_tickets_zero(fwd_ws, bwd_ws, t_small, what + " (before)")
        for t in out.values():
            t.fill_(float("nan"))
        graph.replay()
        _assert_tickets_zero(fwd_ws, bwd_ws, t_small, what + " (after)")
        _assert_bitwise(ref_small, _snapshot(out), what)
    del graph


@pytest.mark.parametrize("Ts", DEFAULT_BUCKETS)
def test_default_workspace_bucket_across_token_counts(Ts: tuple) -> None:
    for n, T in enumerate(Ts):
        ref = _fresh(T)
        got = _snapshot(_step(_inputs(T), None, None))
        torch.cuda.synchronize()
        _assert_bitwise(ref, got, f"default workspace call {n} (T={T}) of {Ts}")


# One process per sequence: the inputs of the issue's reproduction (one CPU generator seeded 0, drawn in call order),
# each ``backward_finish`` on the default workspace, then the same call on a fresh zeroed exact-size workspace.
_REPORTED_CHILD = r"""
import importlib, json, sys
import torch
pkg = importlib.import_module(sys.argv[1])
loader = importlib.import_module(sys.argv[1] + ".cake_jit")
dev = torch.device("cuda:0")
gen = torch.Generator(device="cpu").manual_seed(0)
def r(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(dtype).to(dev)
for T in (int(t) for t in sys.argv[2].split(",")):
    args = [r(T, 4, 4096, dtype=torch.bfloat16), r(T, 4, 4096, dtype=torch.bfloat16, scale=1e-3),
            r(T, 4096, dtype=torch.bfloat16, scale=1e-3), None, r(24, 16384, scale=0.05), r(3, scale=0.1),
            r(24), r(T, 64), r(T, 32, scale=1e-3)]
    layout = loader.workspace_layout("bwd", T, device_index=0)
    ticket = layout["regions"]["ticket"]
    key = (0, "bwd", 1 << (T - 1).bit_length())
    def nonzero():
        torch.cuda.synchronize()
        ws = pkg._DEFAULT_WORKSPACES.get(key)
        return 0 if ws is None else int(torch.count_nonzero(ws[ticket["offset"] : ticket["offset"] + ticket["bytes"]]))
    call = {"T": T, "ticket_before": nonzero()}
    print(json.dumps(call), flush=True)
    got = [t.clone() for t in pkg.cake_mhc_train_backward_finish(*args)]
    call["ticket_after"] = nonzero()
    fresh = torch.zeros(layout["total_bytes"], dtype=torch.uint8, device=dev)
    ref = pkg.cake_mhc_train_backward_finish(*args, workspace=fresh)
    torch.cuda.synchronize()
    call["differs"] = [n for n, a, b in zip(("dX", "dfn", "dscale", "dbase"), got, ref) if not torch.equal(a, b)]
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
            PACKAGES["cake_mhc_train"],
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
        f"backward_finish over {Ts} on the default workspace: exit {run.returncode}\n{tail}"
    )
    done = [c for c in calls if "differs" in c]
    assert [c["T"] for c in done] == list(Ts), tail
    for c in done:
        what = f"default workspace, backward_finish at T={c['T']} of {Ts}"
        assert c["ticket_before"] == 0, (
            f"{what}: {c['ticket_before']} non-zero ticket bytes before the call"
        )
        assert c["ticket_after"] == 0, (
            f"{what}: {c['ticket_after']} non-zero ticket bytes after the call"
        )
        assert not c["differs"], (
            f"{what}: {c['differs']} differ from the fresh-workspace call"
        )
