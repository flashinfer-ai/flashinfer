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

"""``recurrent_kda(backend="cudnn")``: cuDNN's fused SM100 KDA engine.

Two oracles:

* FlashInfer's own CuTe DSL and Cake prefill backends, the tightest bound.
  Both reject a non-bf16 state and ``use_qk_l2norm_in_kernel=False``, so they
  can only anchor the strict contract.
* ``serial_delta_rule``, an fp32 token-at-a-time recurrence with KDA's
  per-key-channel gate. It is the only oracle for the gate modes FlashInfer's
  own backends do not accept, and the one that shares no code with either
  kernel.

The graph-cache-key tests follow ``tests/attention/test_cudnn_graph_cache_key.py``.
KDA puts the most host scalars in the key of any of the three families --
``scale``, ``use_qk_l2norm``, ``use_beta_sigmoid``, ``safe_gate``,
``gate_lower_bound`` -- and each is baked into the built graph, so each gets a
same-shape pair. (``batch_invariant`` is in the key too, but its two settings
are bitwise identical, so no value-based test can pin it.)

Whether the engine can serve a call is cuDNN's decision; nothing here
re-derives it.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from flashinfer.cudnn import cudnn_recurrent_kda
from flashinfer.kda import recurrent_kda
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from tests.test_helpers.cudnn_linear_attention import (
    HEAD_DIM,
    assert_rel_close,
    assert_state_orientation,
    kda_safe_gate,
    packed_offsets,
    rel_err,
    requires_cudnn_linear_attention,
    serial_delta_rule,
    widened_view,
)

pytestmark = requires_cudnn_linear_attention

LOWER_BOUND = -5.0
KERNEL_TOLERANCE = 2e-2
SERIAL_TOLERANCE = 5e-2


def _make_inputs(
    seq_lens,
    num_heads,
    *,
    num_v_heads=None,
    initial_state=False,
    seed=0,
    gate_bias=0.0,
    dtype=torch.bfloat16,
    gate_dtype=torch.bfloat16,
    state_dtype=torch.bfloat16,
    cu_seqlens_dtype=torch.int64,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    total = sum(seq_lens)
    heads_v = num_heads if num_v_heads is None else num_v_heads
    state = None
    if initial_state:
        state = (
            0.1
            * torch.randn(
                (len(seq_lens), heads_v, HEAD_DIM, HEAD_DIM),
                dtype=torch.float32,
                device=device,
            )
        ).to(state_dtype)
    return {
        "q": torch.randn(1, total, num_heads, HEAD_DIM, dtype=dtype, device=device),
        "k": torch.randn(1, total, num_heads, HEAD_DIM, dtype=dtype, device=device),
        "v": torch.randn(1, total, heads_v, HEAD_DIM, dtype=dtype, device=device),
        "g": (
            0.1
            * torch.randn(
                1, total, heads_v, HEAD_DIM, dtype=torch.float32, device=device
            )
            + gate_bias
        ).to(gate_dtype),
        "beta": torch.randn(1, total, heads_v, dtype=dtype, device=device),
        "A_log": 0.1 * torch.randn(heads_v, dtype=torch.float32, device=device),
        "dt_bias": 0.1
        * torch.randn((heads_v, HEAD_DIM), dtype=torch.float32, device=device),
        "initial_state": state,
        "cu_seqlens": packed_offsets(seq_lens, device, cu_seqlens_dtype),
    }


def _gate_kwargs(inputs, **overrides):
    """The in-kernel gate contract every FlashInfer KDA backend accepts."""
    kwargs = dict(
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=inputs["cu_seqlens"],
        beta_is_logit=True,
    )
    kwargs.update(overrides)
    return kwargs


def _run(inputs, **kwargs):
    return recurrent_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        kwargs.pop("g", inputs["g"]),
        kwargs.pop("beta", inputs["beta"]),
        backend="cudnn",
        **kwargs,
    )


def _run_direct(inputs, **kwargs):
    """The cuDNN entry point, for the knobs ``recurrent_kda`` has no argument
    for: ``output_state`` and ``batch_invariant``."""
    return cudnn_recurrent_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        kwargs.pop("g", inputs["g"]),
        kwargs.pop("beta", inputs["beta"]),
        **kwargs,
    )


def _serial(
    inputs,
    *,
    scale=None,
    safe_gate=True,
    beta_is_logit=True,
    l2norm=True,
    initial_state=...,
    g=None,
    beta=None,
):
    """The fp32 oracle, driven by the same gate modes as the call under test."""
    if initial_state is ...:
        initial_state = inputs["initial_state"]
    gate = inputs["g"] if g is None else g
    raw_beta = inputs["beta"] if beta is None else beta
    alpha = (
        kda_safe_gate(gate[0], inputs["A_log"], inputs["dt_bias"], LOWER_BOUND)
        if safe_gate
        else gate[0].float().exp()
    )
    effective_beta = (
        torch.sigmoid(raw_beta[0].float()) if beta_is_logit else raw_beta[0].float()
    )
    out, state = serial_delta_rule(
        inputs["q"][0],
        inputs["k"][0],
        inputs["v"][0],
        inputs["cu_seqlens"],
        alpha=alpha,
        beta=effective_beta,
        initial_state=initial_state,
        scale=HEAD_DIM**-0.5 if scale is None else scale,
        l2norm=l2norm,
    )
    return out.unsqueeze(0), state


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lens", [[512], [256, 320], [64, 1, 1024]])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("reference_backend", ["cute-dsl", "cake"])
def test_cudnn_backend_matches_default(
    seq_lens, num_heads, use_initial_state, reference_backend
):
    """Each of FlashInfer's own recurrent KDA backends as the oracle."""
    inputs = _make_inputs(seq_lens, num_heads, initial_state=use_initial_state, seed=11)
    state = inputs["initial_state"]
    kwargs = _gate_kwargs(inputs, output_final_state=True)
    args = (inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"])
    try:
        ref_out, ref_state = recurrent_kda(
            *args,
            initial_state=None if state is None else state.clone(),
            backend=reference_backend,
            **kwargs,
        )
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"reference backend {reference_backend} unavailable: {exc}")
    out, final_state = _run(
        inputs, initial_state=None if state is None else state.clone(), **kwargs
    )

    assert out.shape == inputs["q"].shape
    assert out.dtype == inputs["q"].dtype
    assert final_state.shape == (len(seq_lens), num_heads, HEAD_DIM, HEAD_DIM)
    assert_rel_close("output", out, ref_out, KERNEL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, KERNEL_TOLERANCE)
    assert_state_orientation(final_state, ref_state)


@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cudnn_backend_matches_serial_reference(use_initial_state, dtype):
    """Token-at-a-time fp32 recurrence as the oracle, independent of any kernel.

    The fp16 arm runs at mild decay: the kernel carries the inverse of a
    chunk-cumulative per-channel decay, which overflows float16 once alpha
    falls well below ~0.9 per token (e.g. alpha ~ 0.08 over a 16-token chunk
    inverts to ~e^40). The engine's own test generator bounds alpha >= 0.9
    for the same reason; bfloat16's fp32-like exponent is unaffected, so the
    bf16 arm keeps the strong-decay regime.
    """
    inputs = _make_inputs(
        [96, 64],
        4,
        initial_state=use_initial_state,
        seed=23,
        dtype=dtype,
        gate_bias=-4.0 if dtype == torch.float16 else 0.0,
    )
    ref_out, ref_state = _serial(inputs)
    state = inputs["initial_state"]
    out, final_state = _run(
        inputs,
        initial_state=None if state is None else state.clone(),
        **_gate_kwargs(inputs, output_final_state=True),
    )
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)
    assert_state_orientation(final_state, ref_state)


def test_cudnn_backend_matches_serial_reference_with_precomputed_gates():
    """The gate modes off: g arrives log-space and beta already post-sigmoid.

    FlashInfer's own prefill backends reject this contract, so the fp32 serial
    recurrence is the only oracle.
    """
    seq_lens, num_heads = [96, 64], 4
    inputs = _make_inputs(seq_lens, num_heads, seed=19)
    device = torch.device("cuda")
    total = sum(seq_lens)
    log_alpha = -F.softplus(
        torch.randn(1, total, num_heads, HEAD_DIM, device=device) * 0.5 - 2.0
    ).to(torch.bfloat16)
    beta = torch.rand(1, total, num_heads, device=device).to(torch.bfloat16)

    ref_out, ref_state = _serial(
        inputs, safe_gate=False, beta_is_logit=False, g=log_alpha, beta=beta
    )
    out, final_state = _run(
        inputs,
        g=log_alpha,
        beta=beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        cu_seqlens=inputs["cu_seqlens"],
        beta_is_logit=False,
    )
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_backend_can_skip_the_in_kernel_l2norm():
    """``use_qk_l2norm_in_kernel=False``, which FlashInfer's own backends reject."""
    seq_lens, num_heads = [96, 64], 4
    inputs = _make_inputs(seq_lens, num_heads, seed=21)
    device = torch.device("cuda")
    total = sum(seq_lens)
    for name in ("q", "k"):
        inputs[name] = F.normalize(inputs[name].float(), dim=-1).to(torch.bfloat16)
    log_alpha = -F.softplus(
        torch.randn(1, total, num_heads, HEAD_DIM, device=device) * 0.5 - 2.0
    ).to(torch.bfloat16)
    beta = torch.rand(1, total, num_heads, device=device).to(torch.bfloat16)

    ref_out, ref_state = _serial(
        inputs,
        safe_gate=False,
        beta_is_logit=False,
        l2norm=False,
        g=log_alpha,
        beta=beta,
    )
    out, final_state = _run(
        inputs,
        g=log_alpha,
        beta=beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        cu_seqlens=inputs["cu_seqlens"],
        beta_is_logit=False,
    )
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_backend_honors_scale():
    inputs = _make_inputs([256], 4, seed=43)
    scale = 3.0 / math.sqrt(HEAD_DIM)
    ref_out, _ = _serial(inputs, scale=scale)
    out, _ = _run(inputs, scale=scale, **_gate_kwargs(inputs))
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    default_out, _ = _run(inputs, **_gate_kwargs(inputs))
    assert rel_err(out, default_out) > 0.5, "scale=3/sqrt(d) matched the default"


@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cudnn_backend_accepts_gate_dtypes(gate_dtype):
    """cuDNN reads the KDA gate at fp32, bf16 or fp16 and it is forwarded as-is."""
    inputs = _make_inputs([256, 128], 4, seed=47, gate_dtype=gate_dtype)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run(inputs, **_gate_kwargs(inputs, output_final_state=True))
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_cudnn_backend_carries_state_dtype(state_dtype):
    """An fp32 state pool crosses too, which FlashInfer's own backends reject."""
    device = torch.device("cuda")
    inputs = _make_inputs(
        [192, 64], 4, seed=59, initial_state=True, state_dtype=state_dtype
    )
    ref_out, ref_state = _serial(inputs)
    output_state = torch.empty(
        2, 4, HEAD_DIM, HEAD_DIM, dtype=state_dtype, device=device
    )
    out, final_state = _run_direct(
        inputs,
        initial_state=inputs["initial_state"],
        output_state=output_state,
        **_gate_kwargs(inputs, output_final_state=True),
    )
    assert final_state.dtype == state_dtype
    assert final_state.data_ptr() == output_state.data_ptr()
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_backend_supports_more_value_heads_than_query_heads():
    """GVA: cuDNN carries the gate, beta and state at ``HO = max(H, HV)``."""
    inputs = _make_inputs([256], 2, num_v_heads=4, seed=67)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run(inputs, **_gate_kwargs(inputs, output_final_state=True))
    assert out.shape == inputs["v"].shape
    assert final_state.shape == (1, 4, HEAD_DIM, HEAD_DIM)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)


def test_cudnn_backend_rejects_more_query_heads_than_value_heads():
    """FlashInfer puts the KDA state at HV, cuDNN at max(H, HV); they part here."""
    inputs = _make_inputs([256], 4, num_v_heads=2, seed=71)
    with pytest.raises(NotImplementedError, match="max\\(H, HV\\)"):
        _run(inputs, **_gate_kwargs(inputs, output_final_state=True))


def test_cudnn_backend_handles_zero_length_sequences():
    seq_lens = [0, 65, 0, 33]
    inputs = _make_inputs(seq_lens, 4, seed=61, initial_state=True)
    ref_out, ref_state = _serial(inputs)
    out, final_state = _run(
        inputs,
        initial_state=inputs["initial_state"].clone(),
        **_gate_kwargs(inputs, output_final_state=True),
    )
    assert out.shape[1] == sum(seq_lens)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)
    assert_rel_close("final_state", final_state, ref_state, SERIAL_TOLERANCE)
    for empty in (0, 2):
        assert_rel_close(
            f"state[{empty}]", final_state[empty], inputs["initial_state"][empty], 1e-6
        )


# ---------------------------------------------------------------------------
# Graph cache key
# ---------------------------------------------------------------------------


def test_cudnn_backend_keys_the_graph_cache_on_scale():
    inputs = _make_inputs([256, 128], 4, seed=73)
    for multiplier in (1.0, 3.0):
        scale = multiplier / math.sqrt(HEAD_DIM)
        out, _ = _run(inputs, scale=scale, **_gate_kwargs(inputs))
        ref_out, _ = _serial(inputs, scale=scale)
        assert_rel_close(f"scale={scale}", out, ref_out, SERIAL_TOLERANCE)


def test_cudnn_backend_keys_the_graph_cache_on_gate_lower_bound():
    """Two safe-gate calls that differ only in the bound.

    ``gate_lower_bound`` is a compile-time constant of the built graph and the
    only thing separating these two keys.
    """
    inputs = _make_inputs([256, 128], 4, seed=79)
    outs = []
    for lower_bound in (-5.0, -1.0):
        out, _ = _run(inputs, **_gate_kwargs(inputs, lower_bound=lower_bound))
        alpha = kda_safe_gate(
            inputs["g"][0], inputs["A_log"], inputs["dt_bias"], lower_bound
        )
        ref_out, _ = serial_delta_rule(
            inputs["q"][0],
            inputs["k"][0],
            inputs["v"][0],
            inputs["cu_seqlens"],
            alpha=alpha,
            beta=torch.sigmoid(inputs["beta"][0].float()),
            scale=HEAD_DIM**-0.5,
            l2norm=True,
        )
        assert_rel_close(
            f"lower_bound={lower_bound}", out, ref_out.unsqueeze(0), SERIAL_TOLERANCE
        )
        outs.append(out)
    assert rel_err(outs[0], outs[1]) > 1e-3, "the two bounds produced one result"


def test_cudnn_backend_keys_the_graph_cache_on_beta_sigmoid():
    """Same beta buffer layout, read as logits then as post-sigmoid values."""
    inputs = _make_inputs([256], 4, seed=83)
    beta = torch.rand(1, 256, 4, device=torch.device("cuda")).to(torch.bfloat16)
    for beta_is_logit in (True, False):
        out, _ = _run(
            inputs,
            beta=beta,
            **_gate_kwargs(inputs, beta_is_logit=beta_is_logit),
        )
        ref_out, _ = _serial(inputs, beta=beta, beta_is_logit=beta_is_logit)
        assert_rel_close(
            f"beta_is_logit={beta_is_logit}", out, ref_out, SERIAL_TOLERANCE
        )


def test_cudnn_backend_keys_the_graph_cache_on_safe_gate():
    """Same gate buffer layout, read as a raw pre-activation then as log-alpha."""
    device = torch.device("cuda")
    inputs = _make_inputs([256], 4, seed=89)
    log_alpha = -F.softplus(
        torch.randn(1, 256, 4, HEAD_DIM, device=device) * 0.5 - 2.0
    ).to(torch.bfloat16)
    beta = torch.rand(1, 256, 4, device=device).to(torch.bfloat16)

    safe_out, _ = _run(inputs, beta=beta, **_gate_kwargs(inputs, beta_is_logit=False))
    ref_safe, _ = _serial(inputs, beta=beta, beta_is_logit=False)
    assert_rel_close("safe_gate=True", safe_out, ref_safe, SERIAL_TOLERANCE)

    plain_out, _ = _run(
        inputs,
        g=log_alpha,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        cu_seqlens=inputs["cu_seqlens"],
        beta_is_logit=False,
    )
    ref_plain, _ = _serial(
        inputs, safe_gate=False, beta_is_logit=False, g=log_alpha, beta=beta
    )
    assert_rel_close("safe_gate=False", plain_out, ref_plain, SERIAL_TOLERANCE)


def test_cudnn_backend_serves_both_batch_invariant_settings():
    """Both split-K settings build and serve the same call correctly.

    Reached through the cuDNN entry point (no ``recurrent_kda`` argument).
    The two settings are bitwise identical on every shape probed, so this is
    a knob smoke, not a cache-key pin.
    """
    inputs = _make_inputs([1024], 4, seed=97)
    ref_out, _ = _serial(inputs)
    for batch_invariant in (False, True):
        out, _ = _run_direct(
            inputs, batch_invariant=batch_invariant, **_gate_kwargs(inputs)
        )
        assert_rel_close(
            f"batch_invariant={batch_invariant}", out, ref_out, SERIAL_TOLERANCE
        )


# ---------------------------------------------------------------------------
# Buffers, layouts and repeatability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cu_seqlens_dtype", [torch.int32, torch.int64])
def test_cudnn_backend_accepts_both_cu_seqlens_dtypes(cu_seqlens_dtype):
    inputs = _make_inputs([256, 256], 4, seed=29, cu_seqlens_dtype=cu_seqlens_dtype)
    ref_out, _ = _serial(inputs)
    out, _ = _run(inputs, **_gate_kwargs(inputs))
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)


def test_cudnn_backend_updates_initial_state_in_place():
    """Matching the Cake and CuTe DSL backends, an incoming state is advanced."""
    inputs = _make_inputs([384], 4, initial_state=True, seed=7)
    state = inputs["initial_state"]
    before = state.clone()
    _, final_state = _run(
        inputs, initial_state=state, **_gate_kwargs(inputs, output_final_state=True)
    )
    assert final_state.data_ptr() == state.data_ptr()
    assert not torch.equal(state, before)


def test_cudnn_backend_advances_initial_state_without_returning_it():
    """``output_final_state`` gates the return value, not the in-place advance."""
    inputs = _make_inputs([384], 4, initial_state=True, seed=17)
    state = inputs["initial_state"]
    before = state.clone()
    out, final_state = _run(inputs, initial_state=state, **_gate_kwargs(inputs))
    assert final_state is None
    assert not torch.equal(state, before)
    assert out.isfinite().all()


def test_cudnn_backend_writes_a_separate_output_state_without_touching_the_input():
    inputs = _make_inputs([384], 4, initial_state=True, seed=19)
    device = torch.device("cuda")
    state = inputs["initial_state"]
    before = state.clone()
    output_state = torch.empty(
        1, 4, HEAD_DIM, HEAD_DIM, dtype=state.dtype, device=device
    )
    _, final_state = _run_direct(
        inputs,
        initial_state=state,
        output_state=output_state,
        **_gate_kwargs(inputs, output_final_state=True),
    )
    assert final_state.data_ptr() == output_state.data_ptr()
    assert torch.equal(state, before), "initial_state was advanced anyway"


def test_cudnn_backend_honors_output_buffer():
    inputs = _make_inputs([384], 4, seed=13)
    out = torch.empty_like(inputs["v"])
    returned_out, _ = _run(inputs, output=out, **_gate_kwargs(inputs))
    assert returned_out.data_ptr() == out.data_ptr()
    ref_out, _ = _serial(inputs)
    assert_rel_close("output", out, ref_out, SERIAL_TOLERANCE)


@pytest.mark.parametrize("tensor", ["q", "k", "v", "g", "beta"])
def test_cudnn_backend_accepts_strided_inputs(tensor):
    """Strides are passed through, so a head-padded view must not be re-read."""
    inputs = _make_inputs([256, 128], 4, seed=89)
    reference = _run(inputs, **_gate_kwargs(inputs, output_final_state=True))
    strided = dict(inputs)
    strided[tensor] = widened_view(inputs[tensor], axis=2)
    assert torch.equal(strided[tensor], inputs[tensor])
    out, final_state = _run(strided, **_gate_kwargs(strided, output_final_state=True))
    assert torch.equal(out, reference[0])
    assert torch.equal(final_state, reference[1])


def test_cudnn_backend_is_deterministic():
    inputs = _make_inputs([2048, 512], 4, seed=101, initial_state=True)
    first = _run(
        inputs,
        initial_state=inputs["initial_state"].clone(),
        **_gate_kwargs(inputs, output_final_state=True),
    )
    second = _run(
        inputs,
        initial_state=inputs["initial_state"].clone(),
        **_gate_kwargs(inputs, output_final_state=True),
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_cudnn_backend_replays_under_cuda_graph_capture():
    device = torch.device("cuda")
    inputs = _make_inputs([512], 4, seed=103)
    out = torch.empty_like(inputs["v"])
    state = torch.zeros(1, 4, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=device)
    call = _gate_kwargs(inputs, output=out, output_state=state, output_final_state=True)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        _run_direct(inputs, **call)
    capture_stream.synchronize()
    eager_out, eager_state = out.clone(), state.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_out, captured_state = _run_direct(inputs, **call)
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


@pytest.mark.parametrize(
    "argument",
    [
        "num_spec_tokens",
        "num_accepted_tokens",
        "initial_state_source",
        "initial_state_indices",
        "seq_order",
        "prefill_workspace",
        "ssm_state_indices",
        "checkpoint_every_n_tokens",
    ],
)
def test_cudnn_backend_names_the_unsupported_argument(argument):
    """Every argument cuDNN's entry point has no parameter for is named."""
    device = torch.device("cuda")
    inputs = _make_inputs([256], 4, seed=3)
    indices = torch.zeros(1, dtype=torch.int32, device=device)
    values = {
        "num_spec_tokens": 2,
        "num_accepted_tokens": indices,
        "initial_state_source": torch.zeros(
            4, 4, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=device
        ),
        "initial_state_indices": indices,
        "seq_order": indices,
        "prefill_workspace": RecurrentKDAPrefillWorkspace(device),
        "ssm_state_indices": indices,
        "checkpoint_every_n_tokens": 64,
    }
    kwargs = _gate_kwargs(inputs)
    if argument == "checkpoint_every_n_tokens":
        kwargs["state_checkpoints"] = torch.zeros(
            4, 4, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        kwargs["checkpoint_cu_starts"] = torch.tensor(
            [0, 4], dtype=torch.int64, device=device
        )
    with pytest.raises(NotImplementedError, match=argument):
        _run(inputs, **{argument: values[argument]}, **kwargs)


def test_cudnn_backend_names_every_unsupported_argument_at_once():
    device = torch.device("cuda")
    inputs = _make_inputs([256], 4, seed=9)
    indices = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError) as excinfo:
        _run(
            inputs,
            num_spec_tokens=2,
            seq_order=indices,
            **_gate_kwargs(inputs),
        )
    message = str(excinfo.value)
    assert "num_spec_tokens" in message and "seq_order" in message


def test_backend_argument_is_validated():
    inputs = _make_inputs([128], 4, seed=1)
    with pytest.raises(ValueError, match="backend"):
        recurrent_kda(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            cu_seqlens=inputs["cu_seqlens"],
            backend="nonesuch",
        )


def test_cudnn_backend_requires_both_safe_gate_parameters():
    inputs = _make_inputs([128], 4, seed=2)
    with pytest.raises(ValueError, match="A_log and dt_bias"):
        _run(
            inputs,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=LOWER_BOUND,
            cu_seqlens=inputs["cu_seqlens"],
            beta_is_logit=True,
        )


def test_cudnn_entry_point_requires_cu_seqlens():
    inputs = _make_inputs([128], 4, seed=4)
    with pytest.raises(ValueError, match="cu_seqlens"):
        cudnn_recurrent_kda(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        )
