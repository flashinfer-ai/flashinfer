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

"""``chunk_gated_delta_rule(backend="cudnn")``: cuDNN's fused SM100 GDN engine.

Three oracles, deliberately layered:

* FlashInfer's own SM100 GDN kernel, the tightest bound (both are chunked
  recurrences over the same inputs, so they agree to accumulation order).
* ``serial_delta_rule``, an fp32 token-at-a-time recurrence that shares no code
  with either kernel. It is what catches a bug both chunked kernels could share.
* ``reference_delta_rule.delta_rule``, this directory's own K-major serial
  reference, as a third independent check on the state's orientation.

The graph-cache-key tests follow ``tests/attention/test_cudnn_graph_cache_key.py``:
every host scalar in ``_la_graph_key_fn`` is baked into the built graph as a
compile-time constant, so a same-shape call that differs only in that scalar
must not replay the first call's graph. Each such test runs the same shapes
twice and checks each result against an oracle recomputed for that scalar.

Whether the engine can serve a call at all -- architecture, head dim, dtypes,
head counts -- is cuDNN's decision, so nothing here re-derives it; the module
gate only covers what a test can know without launching.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from flashinfer.cudnn import cudnn_chunk_gated_delta_rule
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from tests.test_helpers.cudnn_linear_attention import (
    HEAD_DIM,
    assert_rel_close,
    assert_state_orientation,
    packed_offsets,
    rel_err,
    requires_cudnn_linear_attention,
    serial_delta_rule,
    widened_view,
)

from .reference_delta_rule import delta_rule

pytestmark = requires_cudnn_linear_attention

KERNEL_TOLERANCE = 5e-3
SERIAL_TOLERANCE = 5e-2

HEAD_CONFIGS = [(8, 8, 8), (16, 4, 4), (4, 4, 8)]


def _make_inputs(
    seq_lens,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    *,
    seed=0,
    initial_state=False,
    normalize=True,
    norm_scale=1.0,
    dtype=torch.bfloat16,
    gate_dtype=torch.float32,
    state_dtype=torch.float32,
    cu_seqlens_dtype=torch.int32,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    total = sum(seq_lens)
    num_sab_heads = max(num_q_heads, num_v_heads)

    def norm(heads):
        x = torch.randn(total, heads, HEAD_DIM, dtype=torch.float32, device=device)
        if normalize:
            x = F.normalize(x, dim=-1)
        return (norm_scale * x).to(dtype).contiguous()

    state = None
    if initial_state:
        state = (
            (
                0.01
                * torch.randn(
                    len(seq_lens),
                    num_sab_heads,
                    HEAD_DIM,
                    HEAD_DIM,
                    dtype=torch.float32,
                    device=device,
                )
            )
            .to(state_dtype)
            .contiguous()
        )
    return {
        "q": norm(num_q_heads),
        "k": norm(num_k_heads),
        "v": torch.randn(
            total, num_v_heads, HEAD_DIM, dtype=dtype, device=device
        ).contiguous(),
        "g": torch.exp(
            -F.softplus(
                torch.randn(total, num_sab_heads, dtype=torch.float32, device=device)
                * 0.5
                - 2.0
            )
        )
        .to(gate_dtype)
        .contiguous(),
        "beta": torch.rand(
            total, num_sab_heads, dtype=torch.float32, device=device
        ).contiguous(),
        "initial_state": state,
        "cu_seqlens": packed_offsets(seq_lens, device, cu_seqlens_dtype),
    }


def _run_cudnn(inputs, **kwargs):
    """``backend="cudnn"`` with this file's positional convention."""
    return chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        kwargs.pop("g", inputs["g"]),
        kwargs.pop("beta", inputs["beta"]),
        kwargs.pop("scale", None),
        cu_seqlens=kwargs.pop("cu_seqlens", inputs["cu_seqlens"]),
        backend="cudnn",
        **kwargs,
    )


def _serial(inputs, *, scale=None, l2norm=False, initial_state=...):
    if initial_state is ...:
        initial_state = inputs["initial_state"]
    return serial_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["cu_seqlens"],
        alpha=inputs["g"],
        beta=inputs["beta"],
        initial_state=initial_state,
        scale=HEAD_DIM**-0.5 if scale is None else scale,
        l2norm=l2norm,
    )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lens", [[512], [256, 768, 129], [64, 1, 2048]])
@pytest.mark.parametrize("num_q_heads,num_k_heads,num_v_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cudnn_backend_matches_default(
    seq_lens, num_q_heads, num_k_heads, num_v_heads, use_initial_state, dtype
):
    """FlashInfer's own SM100 kernel as the oracle."""
    device = torch.device("cuda")
    inputs = _make_inputs(
        seq_lens,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        seed=17,
        initial_state=use_initial_state,
        dtype=dtype,
    )
    num_seqs = len(seq_lens)
    num_sab_heads = max(num_q_heads, num_v_heads)
    state = inputs["initial_state"]

    ref_out, ref_state = chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        None,
        initial_state=None if state is None else state.clone(),
        output_final_state=True,
        cu_seqlens=inputs["cu_seqlens"],
        output_state=torch.empty(
            num_seqs,
            num_sab_heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device=device,
        ),
    )
    cudnn_out, cudnn_state = _run_cudnn(
        inputs,
        initial_state=None if state is None else state.clone(),
        output_final_state=True,
    )

    assert cudnn_out.shape == ref_out.shape
    assert cudnn_out.dtype == ref_out.dtype
    assert cudnn_state.shape == ref_state.shape
    assert_rel_close("output", cudnn_out, ref_out, KERNEL_TOLERANCE)
    assert_rel_close("final_state", cudnn_state, ref_state, KERNEL_TOLERANCE)


@pytest.mark.parametrize("use_initial_state", [False, True])
def test_cudnn_backend_matches_serial_reference(use_initial_state):
    """Token-at-a-time fp32 recurrence as the oracle, independent of any kernel."""
    inputs = _make_inputs([96, 64], 4, 4, 4, seed=23, initial_state=use_initial_state)
    ref_out, ref_state = _serial(inputs)
    state = inputs["initial_state"]
    out, final_state = _run_cudnn(
        inputs,
        initial_state=None if state is None else state.clone(),
        output_final_state=True,
    )
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)
    assert_state_orientation(final_state, ref_state)


@pytest.mark.parametrize("num_q_heads,num_k_heads,num_v_heads", [(8, 8, 8), (16, 4, 4)])
def test_cudnn_backend_matches_directory_reference(
    num_q_heads, num_k_heads, num_v_heads
):
    """This directory's own K-major serial reference, as a third oracle.

    ``reference_delta_rule.delta_rule`` carries the state K-major as
    ``[N, HO, K, V]`` and takes no incoming state, so the comparison
    transposes and runs from zero. Two references written independently
    agreeing on the same kernel is what rules out a shared-oracle bug.
    """
    seq_lens = [128, 96]
    inputs = _make_inputs(seq_lens, num_q_heads, num_k_heads, num_v_heads, seed=41)
    ref_out, ref_state_kv = delta_rule(
        inputs["q"].float(),
        inputs["k"].float(),
        inputs["v"].float(),
        seq_lens,
        alpha=inputs["g"],
        beta=inputs["beta"],
        scale_factor=HEAD_DIM**-0.5,
    )
    out, final_state = _run_cudnn(inputs, output_final_state=True)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close(
        "final_state", final_state, ref_state_kv.transpose(-1, -2), SERIAL_TOLERANCE
    )


def test_cudnn_backend_applies_in_kernel_l2norm():
    """Un-normalized q/k plus the in-kernel norm must match a hand-normalized oracle."""
    inputs = _make_inputs([96, 64], 4, 4, 4, seed=37, normalize=False)
    ref_out, ref_state = _serial(inputs, l2norm=True)
    out, final_state = _run_cudnn(
        inputs, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_backend_honors_scale():
    """A non-default scale reaches the kernel rather than being dropped."""
    inputs = _make_inputs([256], 4, 4, 4, seed=43)
    scale = 3.0 / math.sqrt(HEAD_DIM)
    ref_out, _ = _serial(inputs, scale=scale)
    out = _run_cudnn(inputs, scale=scale)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    default_out = _run_cudnn(inputs)
    assert rel_err(out, default_out) > 0.5, "scale=3/sqrt(d) matched the default"


@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cudnn_backend_accepts_gate_dtypes(gate_dtype):
    """cuDNN reads g at fp32, bf16 or fp16; the log stays at the caller's width."""
    inputs = _make_inputs([256, 128], 4, 4, 4, seed=47, gate_dtype=gate_dtype)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run_cudnn(inputs, output_final_state=True)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16])
def test_cudnn_backend_accepts_beta_dtypes(beta_dtype):
    inputs = _make_inputs([256], 4, 4, 4, seed=53)
    inputs["beta"] = inputs["beta"].to(beta_dtype)
    ref_out, _ = _serial(inputs)
    out = _run_cudnn(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_cudnn_backend_carries_state_dtype(state_dtype):
    """A bf16 state pool crosses untransposed and comes back at its own width."""
    device = torch.device("cuda")
    inputs = _make_inputs(
        [192, 64], 4, 4, 4, seed=59, initial_state=True, state_dtype=state_dtype
    )
    ref_out, ref_state = _serial(inputs)
    output_state = torch.empty(
        2, 4, HEAD_DIM, HEAD_DIM, dtype=state_dtype, device=device
    )
    out, final_state = _run_cudnn(
        inputs,
        initial_state=inputs["initial_state"],
        output_final_state=True,
        output_state=output_state,
    )
    assert final_state.dtype == state_dtype
    assert final_state.data_ptr() == output_state.data_ptr()
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_backend_defaults_gates_to_ones():
    """Omitting g/beta selects the identity gates rather than reading garbage."""
    device = torch.device("cuda")
    inputs = _make_inputs([256], 4, 4, 4, seed=31)
    total = inputs["q"].shape[0]
    implicit = _run_cudnn(inputs, g=None, beta=None)
    explicit = _run_cudnn(
        inputs,
        g=torch.ones(total, 4, dtype=torch.float32, device=device),
        beta=torch.ones(total, 4, dtype=torch.float32, device=device),
    )
    assert rel_err(implicit, explicit) < 1e-6


def test_cudnn_backend_handles_zero_length_sequences():
    """A packed batch with empty rows: the empty states pass straight through."""
    seq_lens = [0, 65, 0, 33]
    inputs = _make_inputs(seq_lens, 4, 4, 4, seed=61, initial_state=True)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run_cudnn(
        inputs,
        initial_state=inputs["initial_state"].clone(),
        output_final_state=True,
    )
    assert out.shape[0] == sum(seq_lens)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)
    for empty in (0, 2):
        assert_rel_close(
            f"state[{empty}]",
            final_state[empty],
            inputs["initial_state"][empty],
            1e-6,
        )


# ---------------------------------------------------------------------------
# Graph cache key
# ---------------------------------------------------------------------------


def test_cudnn_backend_keys_the_graph_cache_on_scale():
    """Same shapes, two scales: the second must not replay the first graph.

    ``scale`` is baked into the built graph as a compile-time constant, so a
    key that omitted it would silently compute the second call with the first
    call's scale. The 3x multiplier puts a stale replay far outside the bound.
    """
    inputs = _make_inputs([256, 128], 4, 4, 4, seed=67)
    for multiplier in (1.0, 3.0):
        scale = multiplier / math.sqrt(HEAD_DIM)
        out = _run_cudnn(inputs, scale=scale)
        ref_out, _ = _serial(inputs, scale=scale)
        assert_rel_close(f"scale={scale}", out, ref_out, SERIAL_TOLERANCE)


def test_cudnn_backend_keys_the_graph_cache_on_qk_l2norm():
    """Same shapes, the in-kernel norm off then on, each against its own oracle.

    q and k are normalized and then halved, so the flag changes the answer
    (the norm is not a no-op) while the un-normalized arm stays a contraction
    rather than diverging over the sequence.
    """
    inputs = _make_inputs([256, 128], 4, 4, 4, seed=71, norm_scale=0.5)
    for l2norm in (False, True):
        out = _run_cudnn(inputs, use_qk_l2norm_in_kernel=l2norm)
        ref_out, _ = _serial(inputs, l2norm=l2norm)
        assert_rel_close(f"use_qk_l2norm={l2norm}", out, ref_out, SERIAL_TOLERANCE)


def test_cudnn_backend_serves_both_batch_invariant_settings():
    """Both split-K settings build and serve the same call correctly.

    Reached through the cuDNN entry point: ``batch_invariant`` is a cuDNN knob
    with no ``chunk_gated_delta_rule`` argument. The two settings are bitwise
    identical on every shape probed, so this cannot pin the knob's membership
    in the graph-cache key; it pins that each setting compiles and runs.
    """
    inputs = _make_inputs([1024], 4, 4, 4, seed=73)
    ref_out, _ = _serial(inputs)
    for batch_invariant in (False, True):
        out = cudnn_chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            cu_seqlens=inputs["cu_seqlens"],
            batch_invariant=batch_invariant,
        )
        assert_rel_close(
            f"batch_invariant={batch_invariant}", out, ref_out, SERIAL_TOLERANCE
        )


def test_cudnn_backend_batch_invariant_is_independent_of_packing():
    """What ``batch_invariant`` buys: a row's result does not depend on its pack.

    The same sequence is run alone and as the head of a two-sequence pack; with
    the split-K partition disabled the two must agree bitwise. The default
    partition happens to agree on this shape too -- it engages on few long
    sequences, not on this one -- so this pins the guarantee, not the contrast.
    """
    device = torch.device("cuda")
    seq_len = 512
    inputs = _make_inputs([seq_len, seq_len], 4, 4, 4, seed=79)
    solo = packed_offsets([seq_len], device)

    def run(cu_seqlens, rows, batch_invariant):
        return cudnn_chunk_gated_delta_rule(
            inputs["q"][:rows],
            inputs["k"][:rows],
            inputs["v"][:rows],
            inputs["g"][:rows],
            inputs["beta"][:rows],
            cu_seqlens=cu_seqlens,
            batch_invariant=batch_invariant,
        )

    packed = run(inputs["cu_seqlens"], 2 * seq_len, True)
    alone = run(solo, seq_len, True)
    assert torch.equal(packed[:seq_len], alone), (
        "batch_invariant=True still let the pack change the reduction order"
    )


# ---------------------------------------------------------------------------
# Buffers, layouts and repeatability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cu_seqlens_dtype", [torch.int32, torch.int64])
def test_cudnn_backend_accepts_both_cu_seqlens_dtypes(cu_seqlens_dtype):
    inputs = _make_inputs(
        [256, 256], 4, 4, 4, seed=29, cu_seqlens_dtype=cu_seqlens_dtype
    )
    ref_out, _ = _serial(inputs)
    out = _run_cudnn(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)


def test_cudnn_backend_honors_output_buffers():
    device = torch.device("cuda")
    num_heads = 8
    inputs = _make_inputs([384, 384], num_heads, num_heads, num_heads, seed=5)
    out = torch.empty_like(inputs["q"])
    state = torch.empty(
        2, num_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
    )
    returned_out, returned_state = _run_cudnn(
        inputs, output_final_state=True, output=out, output_state=state
    )
    assert returned_out.data_ptr() == out.data_ptr()
    assert returned_state.data_ptr() == state.data_ptr()
    ref_out, ref_state = _serial(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_backend_leaves_state_alone_when_not_requested():
    """``output_final_state=False`` returns a bare tensor and writes no state."""
    inputs = _make_inputs([256], 4, 4, 4, seed=83, initial_state=True)
    state = inputs["initial_state"]
    before = state.clone()
    out = _run_cudnn(inputs, initial_state=state)
    assert isinstance(out, torch.Tensor)
    assert torch.equal(state, before)


@pytest.mark.parametrize("tensor", ["q", "k", "v", "g", "beta"])
def test_cudnn_backend_accepts_strided_inputs(tensor):
    """Strides are passed through, so a head-padded view must not be re-read.

    Every buffer is described to cuDNN with the caller's own strides. A wrapper
    that dropped them and assumed compactness would read the padding, so this
    compares a widened non-contiguous view against the contiguous run.
    """
    inputs = _make_inputs([256, 128], 4, 4, 4, seed=89)
    reference = _run_cudnn(inputs, output_final_state=True)
    strided = dict(inputs)
    strided[tensor] = widened_view(inputs[tensor], axis=1)
    assert torch.equal(strided[tensor], inputs[tensor])
    out, final_state = _run_cudnn(strided, output_final_state=True)
    assert torch.equal(out, reference[0])
    assert torch.equal(final_state, reference[1])


def test_cudnn_backend_writes_through_a_strided_output_view():
    device = torch.device("cuda")
    inputs = _make_inputs([256], 4, 4, 4, seed=97)
    wide = torch.zeros(256, 8, HEAD_DIM, dtype=torch.bfloat16, device=device)
    view = wide[:, :4]
    reference = _run_cudnn(inputs)
    out = _run_cudnn(inputs, output=view)
    assert out.data_ptr() == view.data_ptr()
    assert torch.equal(view, reference)
    assert torch.equal(wide[:, 4:], torch.zeros_like(wide[:, 4:])), (
        "the kernel wrote outside the caller's view"
    )


def test_cudnn_backend_is_deterministic():
    """Two identical calls are bitwise equal, split-K reduction included."""
    inputs = _make_inputs([2048, 512], 8, 8, 8, seed=101, initial_state=True)
    first = _run_cudnn(
        inputs, initial_state=inputs["initial_state"].clone(), output_final_state=True
    )
    second = _run_cudnn(
        inputs, initial_state=inputs["initial_state"].clone(), output_final_state=True
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_cudnn_backend_replays_under_cuda_graph_capture():
    """A captured graph replays to the same buffers with the same result."""
    device = torch.device("cuda")
    inputs = _make_inputs([512], 4, 4, 4, seed=103)
    out = torch.empty(512, 4, HEAD_DIM, dtype=torch.bfloat16, device=device)
    state = torch.zeros(1, 4, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
    call = dict(output=out, output_state=state, output_final_state=True)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        _run_cudnn(inputs, **call)
    capture_stream.synchronize()
    eager_out, eager_state = out.clone(), state.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_out, captured_state = _run_cudnn(inputs, **call)
    out.fill_(float("nan"))
    state.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert captured_out.data_ptr() == out.data_ptr()
    assert captured_state.data_ptr() == state.data_ptr()
    assert torch.equal(out, eager_out)
    assert torch.equal(state, eager_state)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_cudnn_backend_rejects_state_indices():
    device = torch.device("cuda")
    num_heads = 8
    inputs = _make_inputs([256], num_heads, num_heads, num_heads, seed=3)
    pool = torch.zeros(
        4, num_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
    )
    with pytest.raises(NotImplementedError, match="state_indices"):
        _run_cudnn(
            inputs,
            initial_state=pool,
            output_final_state=True,
            output_state=pool,
            state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        )


def test_cudnn_backend_rejects_state_checkpoints():
    device = torch.device("cuda")
    num_heads = 8
    inputs = _make_inputs([256], num_heads, num_heads, num_heads, seed=4)
    with pytest.raises(NotImplementedError, match="checkpoint_every_n_tokens"):
        _run_cudnn(
            inputs,
            output_final_state=True,
            state_checkpoints=torch.zeros(
                4, num_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
            ),
            checkpoint_cu_starts=torch.tensor([0, 4], dtype=torch.int64, device=device),
            checkpoint_every_n_tokens=64,
        )


def test_cudnn_backend_rejects_context_parallel():
    inputs = _make_inputs([256], 8, 8, 8, seed=6)
    with pytest.raises(NotImplementedError, match="use_cp"):
        _run_cudnn(inputs, use_cp=True)


def test_cudnn_backend_names_every_unsupported_argument_at_once():
    """The rejection lists all of them, so one round trip fixes the call."""
    device = torch.device("cuda")
    inputs = _make_inputs([256], 8, 8, 8, seed=8)
    with pytest.raises(NotImplementedError) as excinfo:
        _run_cudnn(
            inputs,
            use_cp=True,
            state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        )
    message = str(excinfo.value)
    assert "use_cp" in message and "state_indices" in message


def test_backend_argument_is_validated():
    inputs = _make_inputs([128], 8, 8, 8, seed=1)
    with pytest.raises(ValueError, match="backend"):
        chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            None,
            cu_seqlens=inputs["cu_seqlens"],
            backend="nonesuch",
        )


def test_cudnn_entry_point_requires_cu_seqlens():
    """The cuDNN wrapper's own required-argument check, reached directly."""
    inputs = _make_inputs([128], 8, 8, 8, seed=2)
    with pytest.raises(ValueError, match="cu_seqlens"):
        cudnn_chunk_gated_delta_rule(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        )
