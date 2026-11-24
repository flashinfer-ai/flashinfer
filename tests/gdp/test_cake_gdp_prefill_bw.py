# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``chunk_gated_delta_product(..., backend="cake")`` on SM100 / SM103: the Blackwell prefill body.

The reference is cuDNN's GDP engine (``backend="cudnn"``) on the same inputs:
the output and the final state agree within FlashInfer's kernel tolerance
(Frobenius relative error), per call and per sequence.  The batches are the
prefill request's shapes at reduced lengths: one-token sequences with a state
ahead of a long fresh sequence, one long sequence with a state, two long
sequences.  The call issues no synchronization when the host offsets are given,
so it captures into a CUDA graph; graphs of different offsets replay
independently.  State pools and BF16 / FP16 state: ``test_cake_gdp_prefill_state.py``.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import flashinfer
import flashinfer.cake_gdp_backend as backend
from flashinfer.cake_gdp_backend import cake_jit as loader
from flashinfer.gdp_prefill import chunk_gated_delta_product
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability
from tests.gdn.test_helpers.cudnn_linear_attention import (
    CUDNN_AVAILABLE,
    rel_err,
)

K = 128
V = 64
N_HH = 3
TOLERANCE = 1e-2
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and backend.is_cake_gdp_prefill_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the Cake GDP Blackwell prefill needs an SM100 / SM103 device with exported routes",
)
needs_cudnn = pytest.mark.skipif(
    not CUDNN_AVAILABLE, reason="the cudnn python frontend is not installed"
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        prefixes = tuple(
            f"{loader.library_name(name, arch)}_"
            for name in loader.ROUTES[arch][loader.PREFILL_ROUTE]
        )
        specs = [
            spec
            for spec in loader.specs_for_arch(arch)
            if any(spec.name.startswith(p) for p in prefixes)
        ]
        flashinfer.jit.build_jit_specs(specs, verbose=False)
    yield


def _inputs(lengths, H, with_state, *, seed=0, n=N_HH):
    """The request's inputs: L2-normalized q / k, BF16 v, FP32 gates in (0, 1), FP32 betas, a random initial
    state for the sequences of ``with_state`` and a zero one for the others."""
    g = torch.Generator(device=_device()).manual_seed(seed)
    device = _device()
    T = sum(lengths)

    def unit(rows):
        x = torch.randn(rows, H, K, generator=g, device=device)
        return F.normalize(x, dim=-1).to(torch.bfloat16)

    state = 0.1 * torch.randn(len(lengths), H, V, K, generator=g, device=device)
    state *= torch.tensor(with_state, dtype=torch.float32, device=device)[
        :, None, None, None
    ]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return {
        "q": unit(T),
        "k": unit(T * n),
        "v": torch.randn(T * n, H, V, generator=g, device=device).to(torch.bfloat16),
        "g": torch.exp(
            -F.softplus(0.5 * torch.randn(T, H, generator=g, device=device) - 2.0)
        ),
        "beta": torch.rand(T * n, H, generator=g, device=device),
        "num_householder": n,
        "initial_state": state.contiguous(),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "cu_seqlens_cpu": torch.tensor(cu, dtype=torch.int32),
    }


def _call(inputs, backend_name, **kwargs):
    args = dict(inputs)
    host = args.pop("cu_seqlens_cpu")
    if backend_name == "cake":
        kwargs.setdefault("cu_seqlens_cpu", host)
    args.update(kwargs)
    return chunk_gated_delta_product(
        **{"output_final_state": True, **args}, backend=backend_name
    )


def _check_close(name, actual, expected, cu):
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"
    error = rel_err(actual, expected)
    assert error < TOLERANCE, f"{name}: relative error {error:.3e}"


# The request's batches at reduced lengths: (H, lengths, which sequences carry a state).
REQUEST_SHAPES = [
    *[
        (H, [1] * d + [L - d], [1] * d + [0])
        for H in (10, 80)
        for L in (2048, 8192)
        for d in (0, 1, 4, 15)
    ],
    *[(H, [8192], [1]) for H in (10, 20, 40, 80)],
    *[(H, [4608, 3584], [1, 0]) for H in (10, 40)],
]


@needs_cudnn
@pytest.mark.parametrize(
    "H,lengths,with_state",
    REQUEST_SHAPES,
    ids=[
        f"H{H}_N{len(lengths)}_T{sum(lengths)}_{''.join(map(str, s))}"
        for H, lengths, s in REQUEST_SHAPES
    ],
)
def test_matches_cudnn_on_the_request_shapes(H, lengths, with_state):
    inputs = _inputs(lengths, H, with_state, seed=sum(lengths) + H)
    out, state = _call(inputs, "cake")
    ref_out, ref_state = _call(inputs, "cudnn")
    assert out.shape == ref_out.shape and state.shape == ref_state.shape
    _check_close("output", out, ref_out, lengths)
    _check_close("final state", state, ref_state, lengths)
    # per sequence: a long sequence must not hide a wrong short one
    start = 0
    for seq, length in enumerate(lengths):
        _check_close(
            f"output of sequence {seq}",
            out[start : start + length],
            ref_out[start : start + length],
            lengths,
        )
        _check_close(f"state of sequence {seq}", state[seq], ref_state[seq], lengths)
        start += length


def _plan(lengths, H):
    return loader.prefill_v6_plan(
        [0, *itertools.accumulate(lengths)],
        num_heads=H,
        num_householder=N_HH,
        sm_count=loader.sm_count(_device().index or 0),
        arch=loader.arch_for_capability(get_compute_capability(_device())),
    )


@needs_cudnn
@pytest.mark.parametrize(
    "H,lengths,split",
    [
        (
            40,
            [1] * 15 + [1024],
            False,
        ),  # an uncut sequence: the block build, rows past the grid
        (
            10,
            [1] * 40 + [20000] + [1] * 3,
            True,
        ),  # a cut sequence: the 1-token rows stack on the CTAs with room
        (20, [0, 1, 0, 9000], True),  # empty sequences: their state copies
    ],
    ids=["block-wrap", "split-lists", "split-empty"],
)
def test_mixed_batches_of_both_builds(H, lengths, split):
    """One-token, empty and long sequences in one call, on the build the plan picks, against cuDNN."""
    plan = _plan(lengths, H)
    assert plan.split == split and plan.grid <= loader.sm_count(_device().index or 0)
    if split:
        table = loader.split_table(
            [0, *itertools.accumulate(lengths)],
            N_HH,
            H,
            loader.sm_count(_device().index or 0),
            arch=loader.arch_for_capability(get_compute_capability(_device())),
        )
        # every row of the table once, as per-CTA lists (no pad row)
        assert sorted(plan.rows) == sorted(table)
        assert len(plan.offsets) == plan.grid + 1
        assert plan.offsets[-1] == len(plan.rows)
    else:
        assert len(plan.rows) > plan.grid
    with_state = [int(length == 1) for length in lengths]
    inputs = _inputs(lengths, H, with_state, seed=7)
    out, state = _call(inputs, "cake")
    ref_out, ref_state = _call(inputs, "cudnn")
    _check_close("output", out, ref_out, lengths)
    _check_close("final state", state, ref_state, lengths)


def _graph(inputs, out, state):
    stream = torch.cuda.Stream(device=_device())
    stream.wait_stream(torch.cuda.current_stream(_device()))
    with torch.cuda.stream(stream):
        _call(inputs, "cake", output=out, output_state=state)  # warm-up, eager
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _call(inputs, "cake", output=out, output_state=state)
    return graph


def test_graphs_of_different_offsets_replay_independently():
    """Graphs captured over the same token count with different sequence offsets (a long sequence alone, 8 and 15
    one-token sequences ahead of it, two long sequences, short sequences on the block build) replay in any order,
    and a replay after new input values matches an eager call on those values."""
    H, T = 20, 4096
    batches = {
        "alone": ([T], [0]),
        "mixed": ([1] * 8 + [T - 8], [1] * 8 + [0]),
        "d15": ([1] * 15 + [T - 15], [1] * 15 + [0]),
        "two": ([2500, T - 2500], [1, 0]),
        "block": ([64] * 64, [1] * 64),
    }
    assert not _plan(*batches["block"][:1], H).split
    runs = {}
    for name, (lengths, with_state) in batches.items():
        inputs = _inputs(lengths, H, with_state, seed=11)
        out = torch.empty(T, H, V, dtype=torch.bfloat16, device=_device())
        state = torch.empty(len(lengths), H, V, K, device=_device())
        runs[name] = (inputs, out, state, _graph(inputs, out, state))
    for name in (
        "alone",
        "mixed",
        "d15",
        "two",
        "block",
        "alone",
        "d15",
        "mixed",
        "block",
        "two",
    ):
        inputs, out, state, graph = runs[name]
        out.fill_(float("nan"))
        state.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        eager_out, eager_state = _call(inputs, "cake")
        assert torch.equal(out, eager_out), name
        assert torch.equal(state, eager_state), name
    # new values in the captured input buffers
    inputs, out, state, graph = runs["mixed"]
    fresh = _inputs(batches["mixed"][0], H, batches["mixed"][1], seed=12)
    for key in ("q", "k", "v", "g", "beta", "initial_state"):
        inputs[key].copy_(fresh[key])
    graph.replay()
    torch.cuda.synchronize()
    eager_out, eager_state = _call(inputs, "cake")
    assert torch.equal(out, eager_out) and torch.equal(state, eager_state)


def test_a_graph_keeps_its_item_table_after_many_eager_batches():
    """A graph captured after an eager call of its batch reuses that call's device item table; eager calls of
    more distinct batches than the eager cache holds afterwards leave the graph's table intact."""
    H, T = 10, 3000
    lengths = [1] * 3 + [T - 3]
    inputs = _inputs(lengths, H, [1] * 3 + [0], seed=21)
    out = torch.empty(T, H, V, dtype=torch.bfloat16, device=_device())
    state = torch.empty(len(lengths), H, V, K, device=_device())
    graph = _graph(inputs, out, state)
    for extra in range(backend._ITEM_TABLES_EAGER_MAX + 8):
        other = [1] * (extra % 5) + [T - extra % 5 - extra - 1, extra + 1]
        _call(_inputs(other, H, [0] * len(other), seed=extra), "cake")
    out.fill_(float("nan"))
    state.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    eager_out, eager_state = _call(inputs, "cake")
    assert torch.equal(out, eager_out) and torch.equal(state, eager_state)


@needs_cudnn
@pytest.mark.parametrize(
    "H,lengths",
    [(10, [1] * 4 + [8192]), (20, [1] * 15 + [4085])],
    ids=["H10_d4", "H20_d15"],
)
def test_an_eager_call_after_a_cold_capture_of_its_batch(H, lengths):
    """The first call of a split batch is a CUDA graph capture (no eager warm-up of that batch; the program was
    loaded by another batch), so its item table is first seen during capture and filled only by a replay.  An
    eager call of that batch before any replay must not read that table: it matches cuDNN, then the replay and
    a later eager call equal it bitwise."""
    with_state = [1] * (len(lengths) - 1) + [0]
    _call(_inputs([1, 1, 2046], 10, [1, 1, 0], seed=99), "cake")  # another batch
    inputs = _inputs(lengths, H, with_state, seed=7)
    assert _plan(lengths, H).split
    T = sum(lengths)
    out = torch.empty(T, H, V, dtype=torch.bfloat16, device=_device())
    state = torch.empty(len(lengths), H, V, K, device=_device())
    stream = torch.cuda.Stream(device=_device())
    stream.wait_stream(torch.cuda.current_stream(_device()))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):  # the first call of this batch
        _call(inputs, "cake", output=out, output_state=state)
    torch.cuda.synchronize()
    eager_out, eager_state = _call(inputs, "cake")  # before any replay
    ref_out, ref_state = _call(inputs, "cudnn")
    _check_close("eager output before replay", eager_out, ref_out, lengths)
    _check_close("eager final state before replay", eager_state, ref_state, lengths)
    out.fill_(float("nan"))
    state.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_out) and torch.equal(state, eager_state)
    again_out, again_state = _call(inputs, "cake")
    assert torch.equal(again_out, eager_out) and torch.equal(again_state, eager_state)


# A graph replayed after the caller rewrote cu_seqlens in place with other lengths (same N and T): the captured
# plan no longer matches the batch.  The kernel compares the planned offsets with the device ones and then runs no
# item: it writes NaN to o[0, 0, :2] (one 32-bit word) and final state [0, 0, 0, 0] and leaves every other value as it was,
# with no out-of-bounds read and the CUDA context intact; replayed again on the restored batch, the same graph
# gives the eager call's outputs bit for bit.  Each case runs in its own process with CUDA_LAUNCH_BLOCKING=1.
_STALE_CHILD = """
import json, sys, torch
from tests.gdp.test_cake_gdp_prefill_bw import K, V, _call, _device, _graph, _inputs
x, y = json.loads(sys.argv[1])
H, T = 10, sum(x)
ix, iy = _inputs(x, H, [1] * len(x), seed=31), _inputs(y, H, [1] * len(y), seed=32)
keys = ("q", "k", "v", "g", "beta", "initial_state", "cu_seqlens", "cu_seqlens_cpu")
kept = {key: ix[key].clone() for key in keys}
out = torch.zeros(T, H, V, dtype=torch.bfloat16, device=_device())
state = torch.zeros(len(x), H, V, K, device=_device())
graph = _graph(ix, out, state)
for key in keys:
    ix[key].copy_(iy[key])
out.fill_(0.5)
state.fill_(0.5)
result = {"error": None}
try:
    graph.replay()
    torch.cuda.synchronize()
    result["marker"] = bool(out[0, 0, :2].isnan().all() and state[0, 0, 0, 0].isnan())
    out[0, 0, :2] = 0.5
    state[0, 0, 0, 0] = 0.5
    result["rest_kept"] = bool((out == 0.5).all() and (state == 0.5).all())
    for key in keys:
        ix[key].copy_(kept[key])
    graph.replay()
    torch.cuda.synchronize()
    eager_out, eager_state = _call(ix, "cake")
    result["restored"] = bool(torch.equal(out, eager_out) and torch.equal(state, eager_state))
except Exception as exc:
    result["error"] = str(exc).splitlines()[0]
try:
    result["context_alive"] = float(torch.ones(4, device=_device()).sum()) == 4.0
except Exception:
    result["context_alive"] = False
print("STALE " + json.dumps(result))
"""


@pytest.mark.parametrize(
    "captured,replayed",
    [
        ([1] * 4 + [4092], [4092] + [1] * 4),
        ([3996, 100], [100, 3996]),
        ([2048, 2048], [3000, 1096]),
        ([2048, 2048], [0, 4096]),
        ([256] * 16, [255] + [256] * 14 + [257]),
    ],
    ids=["tiny-moves", "long-shrinks", "rebalance", "to-empty", "block-shift"],
)
def test_a_graph_replayed_on_other_lengths_fails_explicitly(captured, replayed):
    env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-c", _STALE_CHILD, json.dumps([captured, replayed])],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=root,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("STALE ")]
    assert lines, (
        f"the child process died (exit {proc.returncode}): {proc.stderr[-2000:]}"
    )
    result = json.loads(lines[-1][len("STALE ") :])
    assert result == {
        "error": None,
        "marker": True,
        "rest_kept": True,
        "restored": True,
        "context_alive": True,
    }, result


def test_repeated_calls_are_bitwise_equal():
    inputs = _inputs([1] * 4 + [3000], 10, [1] * 4 + [0], seed=3)
    first = _call(inputs, "cake")
    second = _call(inputs, "cake")
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


@needs_cudnn
def test_flag_sets_and_input_forms():
    """No initial state, no final state, unit gates / betas, int64 offsets and an FP16 output all take the one
    launch of the plan's build."""
    lengths = [1, 1, 2000]
    inputs = _inputs(lengths, 8, [1, 1, 0], seed=5)
    ref_out, ref_state = _call(inputs, "cudnn")
    # int64 offsets
    out, state = _call(
        {
            **inputs,
            "cu_seqlens": inputs["cu_seqlens"].long(),
            "cu_seqlens_cpu": inputs["cu_seqlens_cpu"].long(),
        },
        "cake",
    )
    _check_close("output (int64 offsets)", out, ref_out, lengths)
    _check_close("state (int64 offsets)", state, ref_state, lengths)
    # no final state: the output alone, the same values
    alone = _call(inputs, "cake", output_final_state=False)
    assert torch.equal(alone, out)
    # no initial state == a zero initial state
    zero = {**inputs, "initial_state": torch.zeros_like(inputs["initial_state"])}
    none_out, none_state = _call({**inputs, "initial_state": None}, "cake")
    zero_out, zero_state = _call(zero, "cake")
    assert torch.equal(none_out, zero_out) and torch.equal(none_state, zero_state)
    # unit gates / unit betas == tensors of ones
    ones_g = torch.ones_like(inputs["g"])
    ones_b = torch.ones_like(inputs["beta"])
    for kwargs, given in (
        ({"g": None}, {"g": ones_g}),
        ({"beta": None}, {"beta": ones_b}),
    ):
        a = _call({**inputs, **kwargs}, "cake")
        b = _call({**inputs, **given}, "cake")
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # an FP16 call: converted in and out
    half = {
        **inputs,
        **{key: inputs[key].to(torch.float16) for key in ("q", "k", "v")},
    }
    half_out, _ = _call(half, "cake")
    assert half_out.dtype == torch.float16
    _check_close("output (fp16)", half_out, ref_out, lengths)


def test_empty_sequences():
    inputs = _inputs([0, 300, 0], 4, [1, 1, 1], seed=9)
    out, state = _call(inputs, "cake")
    assert torch.equal(state[0], inputs["initial_state"][0])
    assert torch.equal(state[2], inputs["initial_state"][2])
    assert torch.isfinite(out).all() and torch.isfinite(state).all()
    empty = _inputs([0, 0], 4, [1, 1], seed=10)
    out, state = _call(empty, "cake")
    assert out.shape[0] == 0 and torch.equal(state, empty["initial_state"])


@pytest.mark.parametrize("n", [1, 2, 4])
def test_other_householder_counts_are_refused(n):
    """The Blackwell programs are built for ``num_householder=3``; any other count is refused, not approximated."""
    inputs = _inputs([64], 4, [1], seed=2, n=n)
    with pytest.raises(ValueError, match=f"num_householder={n}"):
        _call(inputs, "cake")
