# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``SSDCombined(backend="cudnn")`` / ``ssd_combined_fwd(backend="cudnn")``.

Mamba-2 on cuDNN's fused SM100 engine, checked against two oracles:

* ``serial_mamba2``, an fp32 token-at-a-time recurrence that shares no code
  with any kernel.
* FlashInfer's own CuTe SSD kernel on the shapes it serves (chunk-aligned,
  bfloat16), the tightest bound: two chunked scans over the same inputs.

The feature-parity tests mirror the Cake backend's contract: dense and
packed calls (``cu_seqlens`` and the ``seq_idx`` triple), ``D`` per head and
per channel, the ``z`` gate, ``dt_bias``, ``dt_softplus``, ``dt_limit``, an
incoming state at bfloat16 or float32, caller-owned ``out``, no final state,
strided framework views, CUDA-graph replay. What cuDNN cannot serve (float32
activations, selective ``checkpoint_*`` capture) is rejected up front with a
message naming it.

Whether the engine can serve a call at all -- architecture, head dims --
is cuDNN's decision; the module gate only covers what a test can know
without launching.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.cudnn import cudnn_ssd_combined_fwd
from flashinfer.mamba import SSDCombined, ssd_combined_fwd
from tests.test_helpers.cudnn_linear_attention import (
    CUDNN_FRONTEND_VERSION,
    assert_rel_close,
    packed_offsets,
    rel_err,
    requires_cudnn_linear_attention,
    serial_mamba2,
)

pytestmark = [
    requires_cudnn_linear_attention,
    pytest.mark.skipif(
        CUDNN_FRONTEND_VERSION < (1, 32),
        reason="the cuDNN Mamba-2 node requires cudnn-frontend 1.32+",
    ),
]

HEADDIM = 64
DSTATE = 128
SERIAL_TOLERANCE = 5e-2
KERNEL_TOLERANCE = 2e-2

HEAD_CONFIGS = [(4, 4), (8, 2), (2, 1)]


def _make_inputs(
    seq_lens,
    nheads,
    ngroups,
    *,
    seed=0,
    dtype=torch.bfloat16,
    dt_dtype=torch.float32,
    dt_softplus=True,
    initial_state=False,
    state_dtype=torch.bfloat16,
    d_layout="none",
    z=False,
    dt_bias=True,
    cu_seqlens_dtype=torch.int32,
):
    """Dense ``[1, total, ...]`` tensors over the packed tokens, plus ``cu_seqlens``."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    total = sum(seq_lens)
    x = torch.randn(1, total, nheads, HEADDIM, device=device).to(dtype)
    b = torch.nn.functional.normalize(
        torch.randn(1, total, ngroups, DSTATE, device=device), dim=-1
    ).to(dtype)
    c = (0.1 * torch.randn(1, total, ngroups, DSTATE, device=device)).to(dtype)
    if dt_softplus:
        dt = (torch.rand(1, total, nheads, device=device) * 4.0 - 4.0).to(dt_dtype)
    else:
        dt = (torch.rand(1, total, nheads, device=device) * 0.1 + 0.01).to(dt_dtype)
    inputs = {
        "x": x.contiguous(),
        "dt": dt.contiguous(),
        "A": -(torch.rand(nheads, device=device) * 7.0 + 1.0),
        "B": b.contiguous(),
        "C": c.contiguous(),
        "dt_bias": (torch.rand(nheads, device=device) * 0.5) if dt_bias else None,
        "D": None,
        "z": None,
        "initial_states": None,
        "dt_softplus": dt_softplus,
        "cu_seqlens": packed_offsets(seq_lens, device, cu_seqlens_dtype),
    }
    if d_layout == "vector":
        inputs["D"] = torch.randn(nheads, device=device).to(dtype)
    elif d_layout == "matrix":
        inputs["D"] = torch.randn(nheads, HEADDIM, device=device).to(dtype)
    if z:
        inputs["z"] = torch.randn(1, total, nheads, HEADDIM, device=device).to(dtype)
    if initial_state:
        inputs["initial_states"] = (
            0.01 * torch.randn(len(seq_lens), nheads, HEADDIM, DSTATE, device=device)
        ).to(state_dtype)
    return inputs


def _run_cudnn(inputs, **overrides):
    call = dict(
        D=inputs["D"],
        z=inputs["z"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=inputs["dt_softplus"],
        initial_states=inputs["initial_states"],
        cu_seqlens=inputs["cu_seqlens"],
        return_final_states=True,
        backend="cudnn",
    )
    call.update(overrides)
    return ssd_combined_fwd(
        inputs["x"], inputs["dt"], inputs["A"], inputs["B"], inputs["C"], **call
    )


def _serial(inputs, *, dt_limit=(0.0, float("inf")), initial_state=...):
    if initial_state is ...:
        initial_state = inputs["initial_states"]
    out, state = serial_mamba2(
        inputs["x"][0],
        inputs["dt"][0],
        inputs["A"],
        inputs["B"][0],
        inputs["C"][0],
        inputs["cu_seqlens"],
        D=inputs["D"],
        z=None if inputs["z"] is None else inputs["z"][0],
        dt_bias=inputs["dt_bias"],
        dt_softplus=inputs["dt_softplus"],
        dt_min=dt_limit[0],
        dt_max=dt_limit[1],
        initial_state=initial_state,
    )
    return out[None], state


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lens", [[256], [192, 320], [64, 1, 1024]])
@pytest.mark.parametrize("nheads,ngroups", HEAD_CONFIGS)
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cudnn_matches_serial_reference(
    seq_lens, nheads, ngroups, use_initial_state, dtype
):
    """Token-at-a-time fp32 recurrence as the oracle, independent of any kernel."""
    inputs = _make_inputs(
        seq_lens, nheads, ngroups, seed=23, dtype=dtype, initial_state=use_initial_state
    )
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run_cudnn(inputs)
    assert out.shape == inputs["x"].shape and out.dtype == dtype
    assert final_state.shape == (len(seq_lens), nheads, HEADDIM, DSTATE)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


@pytest.mark.parametrize("d_layout", ["vector", "matrix"])
@pytest.mark.parametrize("z", [False, True])
def test_cudnn_applies_skip_and_output_gate(d_layout, z):
    """``D`` per head or per channel and the fused ``z sigmoid(z)`` gate."""
    inputs = _make_inputs([128, 96], 4, 4, seed=29, d_layout=d_layout, z=z)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run_cudnn(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)
    plain, _ = _run_cudnn(inputs, D=None, z=None)
    assert rel_err(out, plain) > 0.1, "D / z changed nothing"


def test_cudnn_reads_a_per_head_skip_with_a_trailing_singleton():
    """``D`` as ``[nheads, 1]`` is the per-head skip, as the CuTe backend reads it."""
    inputs = _make_inputs([256], 4, 4, seed=30, d_layout="vector")
    vector, _ = _run_cudnn(inputs)
    column, _ = _run_cudnn(inputs, D=inputs["D"][:, None])
    assert torch.equal(vector, column)


@pytest.mark.parametrize("dt_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dt_softplus", [False, True])
def test_cudnn_honors_dt_transform(dt_dtype, dt_softplus):
    """Raw or pre-activated ``dt`` at float32 or bfloat16, with and without the softplus."""
    inputs = _make_inputs(
        [256, 128], 4, 4, seed=31, dt_dtype=dt_dtype, dt_softplus=dt_softplus
    )
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run_cudnn(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_honors_dt_limit_and_keys_the_graph_cache_on_it():
    """Same shapes, three clamps: each call matches its own oracle.

    ``dt_min`` / ``dt_max`` are baked into the built graph, so a key that
    omitted them would replay the unclamped graph for the clamped call.
    """
    inputs = _make_inputs([256, 128], 4, 4, seed=37)
    for dt_limit in ((0.0, float("inf")), (0.05, 0.3), (0.1, float("inf"))):
        out, final_state = _run_cudnn(inputs, dt_limit=dt_limit)
        ref_out, ref_state = _serial(inputs, dt_limit=dt_limit)
        assert_rel_close(f"output dt_limit={dt_limit}", out, ref_out, SERIAL_TOLERANCE)
        assert_rel_close(
            f"final_state dt_limit={dt_limit}", final_state, ref_state, SERIAL_TOLERANCE
        )


def test_cudnn_keys_the_graph_cache_on_dt_softplus():
    """Same shapes, the softplus off then on, each against its own oracle."""
    inputs = _make_inputs([256], 4, 4, seed=41, dt_softplus=False)
    for dt_softplus in (False, True):
        inputs["dt_softplus"] = dt_softplus
        out, _ = _run_cudnn(inputs)
        ref_out, _ = _serial(inputs)
        assert_rel_close(f"dt_softplus={dt_softplus}", out, ref_out, SERIAL_TOLERANCE)


def test_cudnn_runs_without_dt_bias():
    inputs = _make_inputs([256], 4, 4, seed=43, dt_bias=False)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run_cudnn(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_cudnn_carries_state_dtype(state_dtype):
    """The incoming state's dtype is the outgoing state's dtype."""
    inputs = _make_inputs(
        [192, 64], 4, 4, seed=47, initial_state=True, state_dtype=state_dtype
    )
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run_cudnn(inputs)
    assert final_state.dtype == state_dtype
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_default_state_is_bfloat16_like_the_other_backends():
    inputs = _make_inputs([256], 4, 4, seed=49)
    _, final_state = _run_cudnn(inputs)
    assert final_state.dtype == torch.bfloat16


def test_cudnn_handles_zero_length_sequences():
    """Empty rows in a packed batch pass their incoming state through."""
    seq_lens = [0, 65, 0, 33]
    inputs = _make_inputs(seq_lens, 4, 4, seed=53, initial_state=True)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run_cudnn(inputs)
    assert out.shape[1] == sum(seq_lens)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)
    for empty in (0, 2):
        assert_rel_close(
            f"state[{empty}]",
            final_state[empty],
            inputs["initial_states"][empty],
            1e-6,
        )


# ---------------------------------------------------------------------------
# Call forms: dense, cu_seqlens, seq_idx triple
# ---------------------------------------------------------------------------


def _dense_view(inputs, batch, seqlen):
    """The packed tensors of equal-length sequences as dense ``[batch, seqlen, ...]``."""
    dense = dict(inputs)
    for name in ("x", "dt", "B", "C", "z"):
        if inputs[name] is not None:
            dense[name] = inputs[name].reshape(batch, seqlen, *inputs[name].shape[2:])
    dense["cu_seqlens"] = None
    return dense


@pytest.mark.parametrize("use_initial_state", [False, True])
def test_cudnn_dense_call_is_bitwise_the_packed_call(use_initial_state):
    """A dense ``[batch, seqlen]`` call is the packed call with equal lengths."""
    batch, seqlen = 3, 256
    inputs = _make_inputs(
        [seqlen] * batch, 4, 4, seed=59, initial_state=use_initial_state, z=True
    )
    packed_out, packed_state = _run_cudnn(inputs)
    dense = _dense_view(inputs, batch, seqlen)
    dense_out, dense_state = _run_cudnn(dense, cu_seqlens=None)
    assert dense_out.shape == (batch, seqlen, 4, HEADDIM)
    assert torch.equal(dense_out.reshape(1, batch * seqlen, 4, HEADDIM), packed_out)
    assert torch.equal(dense_state, packed_state)


@pytest.mark.parametrize("count_source", ["initial_states", "num_seqs"])
@pytest.mark.parametrize("seq_idx_dtype", [torch.int32, torch.int64])
def test_cudnn_seq_idx_form_is_bitwise_the_cu_seqlens_form(count_source, seq_idx_dtype):
    """The CuTe backend's ``seq_idx`` triple selects the same graph and result."""
    seq_lens = [200, 56, 0, 300]
    inputs = _make_inputs(
        seq_lens, 4, 4, seed=61, initial_state=count_source == "initial_states"
    )
    device = inputs["x"].device
    seq_idx = torch.cat(
        [
            torch.full((length,), index, dtype=seq_idx_dtype, device=device)
            for index, length in enumerate(seq_lens)
        ]
    )[None]
    starts = inputs["cu_seqlens"].tolist()
    chunk_indices = torch.tensor(
        [start // 128 for start in starts[:-1]], dtype=torch.int32, device=device
    )
    chunk_offsets = torch.tensor(
        [start % 128 for start in starts[:-1]], dtype=torch.int32, device=device
    )
    reference = _run_cudnn(inputs)
    out, final_state = _run_cudnn(
        inputs,
        cu_seqlens=None,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        num_seqs=len(seq_lens) if count_source == "num_seqs" else None,
    )
    assert torch.equal(out, reference[0])
    assert torch.equal(final_state, reference[1])


def test_cudnn_out_of_range_seq_idx_is_memory_safe():
    """An ID past ``num_seqs`` is clamped: no device fault, the in-range
    sequences before it are untouched (their outputs stay bitwise equal)."""
    seq_lens = [128, 64]
    inputs = _make_inputs(seq_lens + [8], 4, 4, seed=65)
    device = inputs["x"].device
    ids = [0] * 128 + [1] * 64 + [2] * 8
    seq_idx = torch.tensor(ids, dtype=torch.int32, device=device)[None]
    clean = _run_cudnn(inputs)
    out, final_state = _run_cudnn(inputs, cu_seqlens=None, seq_idx=seq_idx, num_seqs=2)
    torch.cuda.synchronize()
    assert out.shape == inputs["x"].shape and final_state.shape[0] == 2
    assert torch.equal(out[:, :128], clean[0][:, :128])
    assert torch.equal(final_state[0], clean[1][0])


def test_cudnn_seq_idx_without_a_count_is_rejected():
    inputs = _make_inputs([128, 128], 4, 4, seed=63)
    seq_idx = torch.repeat_interleave(torch.arange(2, device=inputs["x"].device), 128)[
        None
    ]
    with pytest.raises(ValueError, match="num_seqs"):
        _run_cudnn(inputs, cu_seqlens=None, seq_idx=seq_idx)


@pytest.mark.parametrize("cu_seqlens_dtype", [torch.int32, torch.int64])
def test_cudnn_accepts_both_cu_seqlens_dtypes(cu_seqlens_dtype):
    inputs = _make_inputs([256, 256], 4, 4, seed=67, cu_seqlens_dtype=cu_seqlens_dtype)
    ref_out, _ = _serial(inputs)
    out, _ = _run_cudnn(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)


# ---------------------------------------------------------------------------
# Buffers, layouts and repeatability
# ---------------------------------------------------------------------------


def test_cudnn_honors_output_buffers():
    inputs = _make_inputs([384, 384], 8, 8, seed=71)
    out = torch.empty_like(inputs["x"])
    state = torch.empty(2, 8, HEADDIM, DSTATE, dtype=torch.bfloat16, device="cuda")
    returned_out, returned_state = cudnn_ssd_combined_fwd(
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=True,
        cu_seqlens=inputs["cu_seqlens"],
        out=out,
        final_states=state,
    )
    assert returned_out.data_ptr() == out.data_ptr()
    assert returned_state.data_ptr() == state.data_ptr()
    ref_out, ref_state = _serial(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_writes_no_state_when_not_requested():
    inputs = _make_inputs([256], 4, 4, seed=73, initial_state=True)
    before = inputs["initial_states"].clone()
    out, final_state = _run_cudnn(inputs, return_final_states=False)
    assert final_state is None
    assert torch.equal(inputs["initial_states"], before)
    ref_out, _ = _serial(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)


@pytest.mark.parametrize("tensor", ["x", "B", "C", "dt", "z"])
def test_cudnn_accepts_framework_strided_input_views(tensor):
    """A head-padded view must give the same result as the contiguous tensor."""
    inputs = _make_inputs([256, 128], 4, 4, seed=79, z=True)
    reference = _run_cudnn(inputs)
    strided = dict(inputs)
    source = inputs[tensor]
    wide_shape = list(source.shape)
    wide_shape[2] += 2
    wide = torch.empty(wide_shape, dtype=source.dtype, device=source.device)
    index = (slice(None), slice(None), slice(0, source.shape[2]))
    wide[index] = source
    strided[tensor] = wide[index]
    assert not strided[tensor].is_contiguous()
    out, final_state = _run_cudnn(strided)
    assert torch.equal(out, reference[0])
    assert torch.equal(final_state, reference[1])


def test_cudnn_is_deterministic():
    inputs = _make_inputs([2048, 512], 8, 8, seed=83, initial_state=True)
    first = _run_cudnn(inputs)
    second = _run_cudnn(inputs)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_cudnn_replays_under_cuda_graph_capture():
    """A captured graph replays to the same buffers with the same result."""
    device = torch.device("cuda")
    inputs = _make_inputs([512], 4, 4, seed=89)
    out = torch.empty(1, 512, 4, HEADDIM, dtype=torch.bfloat16, device=device)
    state = torch.zeros(1, 4, HEADDIM, DSTATE, dtype=torch.bfloat16, device=device)

    def run():
        return cudnn_ssd_combined_fwd(
            inputs["x"],
            inputs["dt"],
            inputs["A"],
            inputs["B"],
            inputs["C"],
            dt_bias=inputs["dt_bias"],
            dt_softplus=True,
            cu_seqlens=inputs["cu_seqlens"],
            out=out,
            final_states=state,
        )

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        run()
    capture_stream.synchronize()
    eager_out, eager_state = out.clone(), state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_out, captured_state = run()
    out.fill_(float("nan"))
    state.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert captured_out.data_ptr() == out.data_ptr()
    assert captured_state.data_ptr() == state.data_ptr()
    assert torch.equal(out, eager_out)
    assert torch.equal(state, eager_state)


# ---------------------------------------------------------------------------
# Peer backends
# ---------------------------------------------------------------------------


def _peer_or_skip(backend, **kwargs):
    try:
        return SSDCombined(backend=backend, **kwargs)
    except (ImportError, NotImplementedError, ValueError, RuntimeError) as exc:
        pytest.skip(f"{backend} backend unavailable: {exc}")


@pytest.mark.parametrize("d_layout", ["none", "vector", "matrix"])
@pytest.mark.parametrize("z", [False, True])
def test_cudnn_matches_cute_kernel(d_layout, z):
    """FlashInfer's own CuTe SSD kernel as the oracle on its chunk-aligned domain."""
    batch, seqlen, nheads, ngroups = 2, 512, 8, 2
    inputs = _make_inputs(
        [seqlen] * batch,
        nheads,
        ngroups,
        seed=97,
        d_layout=d_layout,
        z=z,
        initial_state=True,
    )
    dense = _dense_view(inputs, batch, seqlen)
    peer = _peer_or_skip(
        "cute",
        chunk_size=128,
        nheads=nheads,
        headdim=HEADDIM,
        dstate=DSTATE,
        ngroups=ngroups,
        has_d=d_layout != "none",
        d_has_hdim=d_layout == "matrix",
        has_initial_states=True,
        has_z=z,
    )
    peer_out, peer_state = peer.run(
        dense["x"],
        dense["dt"],
        dense["A"],
        dense["B"],
        dense["C"],
        D=dense["D"],
        z=dense["z"],
        dt_bias=dense["dt_bias"],
        dt_softplus=True,
        initial_states=dense["initial_states"],
    )
    out, final_state = _run_cudnn(dense, cu_seqlens=None)
    assert_rel_close("output", out, peer_out, KERNEL_TOLERANCE)
    assert_rel_close("final_state", final_state, peer_state, KERNEL_TOLERANCE)


def test_ssd_combined_class_routes_to_cudnn():
    """``SSDCombined(backend="cudnn").run`` is the functional call."""
    batch, seqlen, nheads = 2, 256, 4
    inputs = _make_inputs(
        [seqlen] * batch, nheads, nheads, seed=101, initial_state=True
    )
    dense = _dense_view(inputs, batch, seqlen)
    runner = SSDCombined(
        chunk_size=128,
        nheads=nheads,
        headdim=HEADDIM,
        dstate=DSTATE,
        ngroups=nheads,
        has_initial_states=True,
        backend="cudnn",
    )
    class_out, class_state = runner.run(
        dense["x"],
        dense["dt"],
        dense["A"],
        dense["B"],
        dense["C"],
        dt_bias=dense["dt_bias"],
        dt_softplus=True,
        initial_states=dense["initial_states"],
    )
    functional_out, functional_state = _run_cudnn(dense, cu_seqlens=None)
    assert torch.equal(class_out, functional_out)
    assert torch.equal(class_state, functional_state)
    with pytest.raises(NotImplementedError, match="prepare"):
        runner.prepare(
            x=dense["x"], dt=dense["dt"], A=dense["A"], B=dense["B"], C=dense["C"]
        )


@pytest.mark.parametrize("prefer", [True, False])
def test_auto_follows_the_cudnn_policy(prefer, monkeypatch):
    """``backend="auto"`` runs cuDNN when ``_ssd_auto_prefers_cudnn`` says so
    and keeps the native defaults otherwise (the CuTe kernel for the class,
    Cake for the function). Every arm matches the serial oracle."""
    from flashinfer.cudnn import linear_attention
    from flashinfer.mamba import ssd_combined

    if not linear_attention._mamba2_auto_available():
        pytest.skip("auto needs cudnn-frontend 1.32+ and CuTe DSL 4.7+")
    monkeypatch.setattr(
        ssd_combined, "_ssd_auto_prefers_cudnn", lambda *args, **kwargs: prefer
    )
    batch, seqlen, nheads = 2, 256, 4
    inputs = _make_inputs(
        [seqlen] * batch, nheads, nheads, seed=139, initial_state=True
    )
    dense = _dense_view(inputs, batch, seqlen)
    executions = []
    run = linear_attention._run_la_graph

    def observe(*args, **kwargs):
        executions.append(args[0])
        return run(*args, **kwargs)

    monkeypatch.setattr(linear_attention, "_run_la_graph", observe)
    ref_out, ref_state = _serial(inputs)
    call = dict(
        dt_bias=dense["dt_bias"],
        dt_softplus=True,
        initial_states=dense["initial_states"],
    )
    arms = []
    try:
        arms.append(
            ssd_combined.SSDCombined(
                chunk_size=128,
                nheads=nheads,
                headdim=HEADDIM,
                dstate=DSTATE,
                ngroups=nheads,
                has_d=False,
                has_initial_states=True,
            ).run(dense["x"], dense["dt"], dense["A"], dense["B"], dense["C"], **call)
        )
        arms.append(
            ssd_combined.ssd_combined_fwd(
                dense["x"], dense["dt"], dense["A"], dense["B"], dense["C"], **call
            )
        )
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        if prefer:
            raise
        pytest.skip(f"native backend unavailable: {exc}")
    assert executions == (["mamba2"] * 2 if prefer else [])
    for out, state in arms:
        assert_rel_close(
            "output",
            out.reshape(1, batch * seqlen, nheads, HEADDIM),
            ref_out,
            SERIAL_TOLERANCE,
        )
        assert_rel_close("final_state", state, ref_state, SERIAL_TOLERANCE)


def test_ssd_combined_class_accepts_unaligned_lengths_on_cudnn():
    """Like Cake, the cuDNN backend is not bound to multiples of ``chunk_size``."""
    inputs = _make_inputs([200], 4, 4, seed=103)
    dense = _dense_view(inputs, 1, 200)
    runner = SSDCombined(
        chunk_size=128,
        nheads=4,
        headdim=HEADDIM,
        dstate=DSTATE,
        ngroups=4,
        backend="cudnn",
    )
    out, final_state = runner.run(
        dense["x"],
        dense["dt"],
        dense["A"],
        dense["B"],
        dense["C"],
        dt_bias=dense["dt_bias"],
        dt_softplus=True,
    )
    ref_out, ref_state = _serial(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_cudnn_class_carries_float16_state():
    """An fp16 state pool, the dtype an fp16 model hands over, crosses as is
    through the class as well as the function."""
    inputs = _make_inputs(
        [256], 4, 4, seed=107, initial_state=True, state_dtype=torch.float16
    )
    dense = _dense_view(inputs, 1, 256)
    runner = SSDCombined(
        chunk_size=128,
        nheads=4,
        headdim=HEADDIM,
        dstate=DSTATE,
        ngroups=4,
        has_initial_states=True,
        state_dtype=torch.float16,
        backend="cudnn",
    )
    out, final_state = runner.run(
        dense["x"],
        dense["dt"],
        dense["A"],
        dense["B"],
        dense["C"],
        dt_bias=dense["dt_bias"],
        dt_softplus=True,
        initial_states=dense["initial_states"],
    )
    assert final_state.dtype == torch.float16
    ref_out, ref_state = _serial(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_rejects_float32_activations():
    inputs = _make_inputs([256], 4, 4, seed=109)
    with pytest.raises(NotImplementedError, match="bfloat16 or float16"):
        cudnn_ssd_combined_fwd(
            inputs["x"].float(),
            inputs["dt"],
            inputs["A"],
            inputs["B"].float(),
            inputs["C"].float(),
            cu_seqlens=inputs["cu_seqlens"],
        )


def test_cudnn_rejects_selective_checkpoints():
    batch, seqlen, nheads = 1, 256, 4
    inputs = _make_inputs([seqlen], nheads, nheads, seed=113)
    dense = _dense_view(inputs, batch, seqlen)
    runner = SSDCombined(
        chunk_size=128,
        nheads=nheads,
        headdim=HEADDIM,
        dstate=DSTATE,
        ngroups=nheads,
        backend="cudnn",
    )
    with pytest.raises(ValueError, match="checkpoint"):
        runner.run(
            dense["x"],
            dense["dt"],
            dense["A"],
            dense["B"],
            dense["C"],
            dt_softplus=True,
            checkpoint_token_indices=torch.tensor(
                [128], dtype=torch.int32, device="cuda"
            ),
            checkpoint_state_slots=torch.tensor([0], dtype=torch.int32, device="cuda"),
            checkpoint_states=torch.empty(
                1, nheads, HEADDIM, DSTATE, dtype=torch.bfloat16, device="cuda"
            ),
        )


def test_backend_argument_is_validated():
    inputs = _make_inputs([128], 4, 4, seed=1)
    with pytest.raises(ValueError, match="backend"):
        _run_cudnn(inputs, backend="nonesuch")
    with pytest.raises(ValueError, match="backend"):
        SSDCombined(
            chunk_size=128,
            nheads=4,
            headdim=HEADDIM,
            dstate=DSTATE,
            ngroups=4,
            backend="nonesuch",
        )


def test_cudnn_rejects_a_packed_call_with_a_batch_axis():
    inputs = _make_inputs([128, 128], 4, 4, seed=2)
    dense = _dense_view(inputs, 2, 128)
    with pytest.raises(ValueError, match="batch == 1"):
        _run_cudnn(dense, cu_seqlens=inputs["cu_seqlens"])
