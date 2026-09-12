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

"""``chunk_gated_delta_product``: cuDNN's fused SM100 GDP engine.

FlashInfer carries no GDP kernel of its own, so there is no peer backend to
differentiate against. Two oracles stand in:

* ``serial_delta_product``, an fp32 token-at-a-time GDP recurrence.
* **GDN itself.** ``num_householder == 1`` is exactly the gated delta rule, so
  running ``chunk_gated_delta_rule`` on the same inputs is an independent
  kernel oracle for that corner, and it is the check that would catch a
  sub-token timeline wired off by one.

The graph-cache-key tests follow ``tests/attention/test_cudnn_graph_cache_key.py``.
Everything the engine owns -- architecture, head dims, dtypes, head counts --
is left to cuDNN to accept or decline.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.cudnn import cudnn_chunk_gated_delta_product
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdp_prefill import chunk_gated_delta_product
from tests.test_helpers.cudnn_linear_attention import (
    HEAD_DIM,
    assert_rel_close,
    assert_state_orientation,
    packed_offsets,
    rel_err,
    requires_cudnn_linear_attention,
    serial_delta_product,
    widened_view,
)

pytestmark = requires_cudnn_linear_attention

SERIAL_TOLERANCE = 5e-2
KERNEL_TOLERANCE = 1e-2

HEAD_CONFIGS = [(4, 4, 4), (8, 4, 4), (4, 4, 8)]


def _make_inputs(
    seq_lens,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    num_householder,
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
    expanded = total * num_householder
    num_sab_heads = max(num_q_heads, num_v_heads)

    def norm(rows, heads):
        x = torch.randn(rows, heads, HEAD_DIM, dtype=torch.float32, device=device)
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
        "q": norm(total, num_q_heads),
        "k": norm(expanded, num_k_heads),
        "v": torch.randn(
            expanded, num_v_heads, HEAD_DIM, dtype=dtype, device=device
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
            expanded, num_sab_heads, dtype=torch.float32, device=device
        ).contiguous(),
        "num_householder": num_householder,
        "initial_state": state,
        "cu_seqlens": packed_offsets(seq_lens, device, cu_seqlens_dtype),
    }


def _run(inputs, **kwargs):
    return chunk_gated_delta_product(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        kwargs.pop("g", inputs["g"]),
        kwargs.pop("beta", inputs["beta"]),
        kwargs.pop("num_householder", inputs["num_householder"]),
        kwargs.pop("scale", None),
        cu_seqlens=kwargs.pop("cu_seqlens", inputs["cu_seqlens"]),
        **kwargs,
    )


def _serial(inputs, *, scale=None, l2norm=False, initial_state=...):
    if initial_state is ...:
        initial_state = inputs["initial_state"]
    return serial_delta_product(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["cu_seqlens"],
        num_householder=inputs["num_householder"],
        alpha=inputs["g"],
        beta=inputs["beta"],
        initial_state=initial_state,
        scale=HEAD_DIM**-0.5 if scale is None else scale,
        l2norm=l2norm,
    )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lens", [[96, 64], [512], [64, 1, 1024]])
@pytest.mark.parametrize("num_q_heads,num_k_heads,num_v_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("num_householder", [2, 3])
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gdp_matches_serial_reference(
    seq_lens,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    num_householder,
    use_initial_state,
    dtype,
):
    """Token-at-a-time fp32 recurrence as the oracle, independent of any kernel."""
    inputs = _make_inputs(
        seq_lens,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        num_householder,
        seed=23,
        initial_state=use_initial_state,
        dtype=dtype,
    )
    ref_out, ref_state = _serial(inputs)
    state = inputs["initial_state"]
    out, final_state = _run(
        inputs,
        initial_state=None if state is None else state.clone(),
        output_final_state=True,
    )
    assert out.shape == (sum(seq_lens), max(num_q_heads, num_v_heads), HEAD_DIM)
    assert out.dtype == dtype
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)
    assert_state_orientation(final_state, ref_state)


@pytest.mark.parametrize("use_initial_state", [False, True])
def test_gdp_with_num_householder_one_matches_gdn(use_initial_state):
    """``num_householder == 1`` is exactly the gated delta rule.

    Same inputs through ``chunk_gated_delta_rule`` are an independent kernel
    oracle, and the two calls agree on every shape and every host scalar in
    the graph cache key except ``family``, so this also pins that component.
    """
    seq_lens = [256, 128]
    inputs = _make_inputs(
        [256, 128], 4, 4, 4, 1, seed=7, initial_state=use_initial_state
    )
    state = inputs["initial_state"]
    gdn_out, gdn_state = chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        None,
        initial_state=None if state is None else state.clone(),
        output_final_state=True,
        cu_seqlens=inputs["cu_seqlens"],
        backend="cudnn",
    )
    gdp_out, gdp_state = _run(
        inputs,
        initial_state=None if state is None else state.clone(),
        output_final_state=True,
    )
    assert gdp_out.shape == (sum(seq_lens), 4, HEAD_DIM)
    assert_rel_close("output", gdp_out, gdn_out, KERNEL_TOLERANCE)
    assert_rel_close("final_state", gdp_state, gdn_state, KERNEL_TOLERANCE)
    ref_out, ref_state = _serial(inputs)
    assert_rel_close("output vs serial", gdp_out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("state vs serial", gdp_state, ref_state, SERIAL_TOLERANCE)


def test_gdp_applies_in_kernel_l2norm():
    """Un-normalized q/k plus the in-kernel norm must match a hand-normalized oracle."""
    inputs = _make_inputs([96, 64], 4, 4, 4, 2, seed=37, normalize=False)
    ref_out, ref_state = _serial(inputs, l2norm=True)
    out, final_state = _run(
        inputs, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_gdp_honors_scale():
    inputs = _make_inputs([256], 4, 4, 4, 2, seed=43)
    scale = 3.0 / math.sqrt(HEAD_DIM)
    ref_out, _ = _serial(inputs, scale=scale)
    out = _run(inputs, scale=scale)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert rel_err(out, _run(inputs)) > 0.5, "scale=3/sqrt(d) matched the default"


def test_gdp_defaults_gates_to_ones():
    """Omitting g/beta selects the identity gates."""
    device = torch.device("cuda")
    num_heads = 4
    inputs = _make_inputs([256], num_heads, num_heads, num_heads, 2, seed=31)
    total = inputs["q"].shape[0]
    implicit = _run(inputs, g=None, beta=None)
    explicit = _run(
        inputs,
        g=torch.ones(total, num_heads, dtype=torch.float32, device=device),
        beta=torch.ones(2 * total, num_heads, dtype=torch.float32, device=device),
    )
    assert rel_err(implicit, explicit) < 1e-6


@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_gdp_accepts_forget_gate_dtypes(gate_dtype):
    """``g`` may be fp32, bf16 or fp16, like GDN's."""
    inputs = _make_inputs([256, 128], 4, 4, 4, 2, seed=47, gate_dtype=gate_dtype)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run(inputs, output_final_state=True)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_gdp_carries_state_dtype(state_dtype):
    device = torch.device("cuda")
    inputs = _make_inputs(
        [192, 64], 4, 4, 4, 2, seed=59, initial_state=True, state_dtype=state_dtype
    )
    ref_out, ref_state = _serial(inputs)
    output_state = torch.empty(
        2, 4, HEAD_DIM, HEAD_DIM, dtype=state_dtype, device=device
    )
    out, final_state = _run(
        inputs,
        initial_state=inputs["initial_state"],
        output_final_state=True,
        output_state=output_state,
    )
    assert final_state.dtype == state_dtype
    assert final_state.data_ptr() == output_state.data_ptr()
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_gdp_handles_zero_length_sequences():
    seq_lens = [0, 65, 0, 33]
    inputs = _make_inputs(seq_lens, 4, 4, 4, 2, seed=61, initial_state=True)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run(
        inputs,
        initial_state=inputs["initial_state"].clone(),
        output_final_state=True,
    )
    assert out.shape[0] == sum(seq_lens)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)
    for empty in (0, 2):
        assert_rel_close(
            f"state[{empty}]", final_state[empty], inputs["initial_state"][empty], 1e-6
        )


# ---------------------------------------------------------------------------
# Graph cache key
# ---------------------------------------------------------------------------


def test_gdp_keys_the_graph_cache_on_scale():
    """Same shapes, two scales: the second must not replay the first graph."""
    inputs = _make_inputs([256, 128], 4, 4, 4, 2, seed=67)
    for multiplier in (1.0, 3.0):
        scale = multiplier / math.sqrt(HEAD_DIM)
        out = _run(inputs, scale=scale)
        ref_out, _ = _serial(inputs, scale=scale)
        assert_rel_close(f"scale={scale}", out, ref_out, SERIAL_TOLERANCE)


def test_gdp_keys_the_graph_cache_on_qk_l2norm():
    """Same shapes, the in-kernel norm off then on, each against its own oracle."""
    inputs = _make_inputs([256, 128], 4, 4, 4, 2, seed=71, norm_scale=0.5)
    for l2norm in (False, True):
        out = _run(inputs, use_qk_l2norm_in_kernel=l2norm)
        ref_out, _ = _serial(inputs, l2norm=l2norm)
        assert_rel_close(f"use_qk_l2norm={l2norm}", out, ref_out, SERIAL_TOLERANCE)


def test_gdp_keys_the_graph_cache_on_num_householder():
    """Back-to-back n=2 and n=1 calls, each against its own oracle.

    The row counts differ with ``n``, so the shapes already separate the keys;
    this pins that the wrapper threads ``num_householder`` into the built
    graph rather than baking the first call's value.
    """
    for n in (2, 1):
        inputs = _make_inputs([256], 4, 4, 4, n, seed=79)
        out = _run(inputs)
        ref_out, _ = _serial(inputs)
        assert_rel_close(f"num_householder={n}", out, ref_out, SERIAL_TOLERANCE)


def test_gdp_serves_both_batch_invariant_settings():
    """Both split-K settings build and serve the same call correctly."""
    inputs = _make_inputs([1024], 4, 4, 4, 2, seed=73)
    ref_out, _ = _serial(inputs)
    for batch_invariant in (False, True):
        out = cudnn_chunk_gated_delta_product(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["num_householder"],
            cu_seqlens=inputs["cu_seqlens"],
            batch_invariant=batch_invariant,
        )
        assert_rel_close(
            f"batch_invariant={batch_invariant}", out, ref_out, SERIAL_TOLERANCE
        )


# ---------------------------------------------------------------------------
# Buffers, layouts and repeatability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cu_seqlens_dtype", [torch.int32, torch.int64])
def test_gdp_accepts_both_cu_seqlens_dtypes(cu_seqlens_dtype):
    inputs = _make_inputs(
        [256, 256], 4, 4, 4, 2, seed=29, cu_seqlens_dtype=cu_seqlens_dtype
    )
    ref_out, _ = _serial(inputs)
    assert_rel_close("output", _run(inputs), ref_out, SERIAL_TOLERANCE)


def test_gdp_honors_output_buffers():
    device = torch.device("cuda")
    num_heads = 8
    inputs = _make_inputs([384, 384], num_heads, num_heads, num_heads, 2, seed=5)
    out = torch.empty_like(inputs["q"])
    state = torch.empty(
        2, num_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
    )
    returned_out, returned_state = _run(
        inputs, output_final_state=True, output=out, output_state=state
    )
    assert returned_out.data_ptr() == out.data_ptr()
    assert returned_state.data_ptr() == state.data_ptr()
    ref_out, ref_state = _serial(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", state, ref_state, SERIAL_TOLERANCE)


def test_gdp_leaves_state_alone_when_not_requested():
    inputs = _make_inputs([256], 4, 4, 4, 2, seed=83, initial_state=True)
    state = inputs["initial_state"]
    before = state.clone()
    out = _run(inputs, initial_state=state)
    assert isinstance(out, torch.Tensor)
    assert torch.equal(state, before)


@pytest.mark.parametrize("tensor", ["q", "k", "v", "g", "beta"])
def test_gdp_accepts_strided_inputs(tensor):
    """Strides are passed through, so a head-padded view must not be re-read."""
    inputs = _make_inputs([256, 128], 4, 4, 4, 2, seed=89)
    reference = _run(inputs, output_final_state=True)
    strided = dict(inputs)
    strided[tensor] = widened_view(inputs[tensor], axis=1)
    assert torch.equal(strided[tensor], inputs[tensor])
    out, final_state = _run(strided, output_final_state=True)
    assert torch.equal(out, reference[0])
    assert torch.equal(final_state, reference[1])


def test_gdp_writes_through_a_strided_output_view():
    device = torch.device("cuda")
    inputs = _make_inputs([256], 4, 4, 4, 2, seed=97)
    wide = torch.zeros(256, 8, HEAD_DIM, dtype=torch.bfloat16, device=device)
    view = wide[:, :4]
    reference = _run(inputs)
    out = _run(inputs, output=view)
    assert out.data_ptr() == view.data_ptr()
    assert torch.equal(view, reference)
    assert torch.equal(wide[:, 4:], torch.zeros_like(wide[:, 4:]))


def test_gdp_is_deterministic():
    inputs = _make_inputs([2048, 512], 4, 4, 4, 2, seed=101, initial_state=True)
    first = _run(
        inputs, initial_state=inputs["initial_state"].clone(), output_final_state=True
    )
    second = _run(
        inputs, initial_state=inputs["initial_state"].clone(), output_final_state=True
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_gdp_replays_under_cuda_graph_capture():
    device = torch.device("cuda")
    inputs = _make_inputs([512], 4, 4, 4, 2, seed=103)
    out = torch.empty(512, 4, HEAD_DIM, dtype=torch.bfloat16, device=device)
    state = torch.zeros(1, 4, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)
    call = dict(output=out, output_state=state, output_final_state=True)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        _run(inputs, **call)
    capture_stream.synchronize()
    eager_out, eager_state = out.clone(), state.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_out, captured_state = _run(inputs, **call)
    out.fill_(float("nan"))
    state.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert captured_out.data_ptr() == out.data_ptr()
    assert captured_state.data_ptr() == state.data_ptr()
    assert torch.equal(out, eager_out)
    assert torch.equal(state, eager_state)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_public_api_is_exported():
    """``flashinfer.chunk_gated_delta_product`` is the module function itself."""
    assert flashinfer.chunk_gated_delta_product is chunk_gated_delta_product


@pytest.mark.parametrize("backend", ["auto", "cudnn"])
def test_both_backend_values_reach_cudnn(backend):
    """FlashInfer has no GDP kernel, so ``auto`` and ``cudnn`` are one path."""
    inputs = _make_inputs([256], 4, 4, 4, 2, seed=113)
    ref_out, _ = _serial(inputs)
    assert_rel_close("output", _run(inputs, backend=backend), ref_out, SERIAL_TOLERANCE)


def test_backend_argument_is_validated():
    inputs = _make_inputs([128], 4, 4, 4, 2, seed=1)
    with pytest.raises(ValueError, match="backend"):
        _run(inputs, backend="nonesuch")


def test_cu_seqlens_is_required():
    """The engines take packed THD input, so there is no dense fallback."""
    inputs = _make_inputs([128], 4, 4, 4, 2, seed=1)
    with pytest.raises(ValueError, match="cu_seqlens"):
        chunk_gated_delta_product(inputs["q"], inputs["k"], inputs["v"])


def test_cu_seqlens_dtype_is_validated():
    inputs = _make_inputs([128], 4, 4, 4, 2, seed=1)
    with pytest.raises(ValueError, match="integer dtype"):
        _run(inputs, cu_seqlens=inputs["cu_seqlens"].float())


def test_split_state_dtypes_are_declined_by_the_engine():
    """One state dtype per kernel: cuDNN, not the wrapper, says no."""
    device = torch.device("cuda")
    inputs = _make_inputs([256], 4, 4, 4, 2, seed=127, initial_state=True)
    with pytest.raises(Exception) as excinfo:
        _run(
            inputs,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            output_state=torch.empty(
                1, 4, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=device
            ),
        )
    assert type(excinfo.value).__module__.startswith("cudnn"), (
        f"expected a cuDNN decline, got {type(excinfo.value)!r}"
    )
