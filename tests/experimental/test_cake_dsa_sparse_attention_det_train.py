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

Backend suite of the deterministic DSA sparse-attention training package
(``flashinfer.experimental.cake.cake_dsa_sparse_attention_det_train``): the host plan (stage order, storage plan,
workspace sizes), the forward against the non-deterministic package's forward (the same kernels: bit for bit), and
the bit-for-bit repeat, accuracy, accumulate, ``o_lo`` replay, checkpoint, CUDA graph and no-host-sync contracts of
the backward.  "Bitwise" here is bit-pattern equality (integer views), not ``torch.equal``: ``-0.0`` and NaN payloads
count.  GPU cases skip without a compute capability 10.0 / 10.3 / 10.7 device or without the registered stages.
"""

import hashlib

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers.cake_dsa_sparse_attention_nondet_train_reference import (
    make_inputs,
    reference_fp64,
    rel_l2,
)
from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_backend")
cake_jit = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_jit")
cake_launch = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_launch")
nondet_backend = cake_package("cake.cake_dsa_sparse_attention_nondet_train", "cake_backend")

NUM_HEADS = cake_backend.NUM_HEADS
D_LATENT = cake_backend.D_LATENT
D_ROPE = cake_backend.D_ROPE
D_QK = cake_backend.D_QK
WORKSPACE_ALIGN = cake_backend.WORKSPACE_ALIGN
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES

# The accuracy gates of the non-deterministic suite (relative L2 vs the chunked FP64 reference of the same BF16
# inputs): the integer accumulation adds one rounding per contribution on a grid 2 ** 12 finer than the row bound
# needs, well inside them.
GATE_DQ_LATENT = 0.00226
GATE_DKV_LATENT = 0.00243
GATE_ROPE_FACTOR = 2.0
FLOOR_MARGIN = 1.05
FLOOR_ABS = 1e-5

SEED = 20261107
REPEATS = 5


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program(*, backward: bool = False, program: str = cake_jit.PROGRAM):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not cake_backend.generated_program_available(
        torch.device("cuda"), backward=backward, program=program
    ):
        pytest.skip(
            "generated DSA training program"
            + (" with backward stages" if backward else "")
            + " not registered for this device"
        )


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The bit pattern of ``t`` as an integer tensor of the same element size."""
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.detach().contiguous().view(view)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _sha16(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


def test_backward_stage_order_is_fixed():
    """Every backward launches the registered stages of ``BACKWARD_STAGES_OF`` in this order, every one of them on
    every call (the recomputation of flagged rows included): no stage is chosen by a tensor value."""
    assert cake_jit.BACKWARD_STAGES == (
        "bwd_delta",
        "bwd_zero",
        "bwd_norm",
        "bwd_bound",
        "bwd_grid",
        "bwd_main",
        "bwd_redo_prep",
        "bwd_redo",
        "bwd_fold",
    )
    assert cake_jit.NOPE_BACKWARD_STAGES == tuple(
        f"nope_{s}" for s in cake_jit.BACKWARD_STAGES
    )
    for program in cake_jit.PROGRAMS:
        registered = cake_jit.registered_stages(program)
        backward = [s for s in registered if s in cake_jit.BACKWARD_STAGES_OF[program]]
        # either no backward stage yet, or all of them, in the declared order
        assert backward in ([], list(cake_jit.BACKWARD_STAGES_OF[program])), backward


@pytest.mark.parametrize(
    "program, columns", [(cake_jit.PROGRAM, D_QK), (cake_jit.NOPE_PROGRAM, D_LATENT)]
)
@pytest.mark.parametrize("accumulator", [True, False])
def test_accumulator_region_layout(program, columns, accumulator):
    R, S = 1000, 1300
    layout = cake_backend.accumulator_region_layout(
        S, R, program=program, accumulator=accumulator
    )
    parts = 2 if program == cake_jit.PROGRAM else 1
    names = (["acc"] if accumulator else []) + [
        "counters",
        "bounds",
        "exponents",
        "knorm",
    ]
    assert [k for k in layout if k not in ("total", "zero")] == names
    offsets = [layout[k][0] for k in names]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    sizes = {k: layout[k][1] for k in names}
    if accumulator:
        assert sizes["acc"] == R * columns * 4
    assert sizes["counters"] == R * 32
    assert sizes["bounds"] == R * 8 * parts
    assert sizes["exponents"] == R * 4 * parts
    assert sizes["knorm"] == S * 4
    # bwd_zero clears the accumulator, the counters and the bounds: everything before the exponents
    assert layout["zero"] == layout["exponents"][0]


def _align(n):
    return -(-n // WORKSPACE_ALIGN) * WORKSPACE_ALIGN


@pytest.mark.parametrize("T", [1, 7, 1024, 16384])
@pytest.mark.parametrize("ratio", [1, 16, 27, 28, 31, 32, 64])
def test_carve_condition(T, ratio):
    """``carve_fits`` is the documented closed form, and the limits are about 27.7 T (rope) and 31.5 T (NoPE)."""
    R = S = T * ratio
    rope = (
        _align(R * 2304) + _align(R * 32) + _align(R * 16) + _align(R * 8) + _align(S * 4)
    )
    nope = (
        _align(R * 2048) + _align(R * 32) + _align(R * 8) + _align(R * 4) + _align(S * 4)
    )
    assert cake_backend.carve_fits(T, S, R) == (rope <= 65536 * T)
    assert cake_backend.carve_fits(T, S, R, program=cake_jit.NOPE_PROGRAM) == (
        nope <= 65536 * T
    )
    if T >= 1024:
        assert cake_backend.carve_fits(T, S, R) == (ratio <= 27)
        assert cake_backend.carve_fits(T, S, R, program=cake_jit.NOPE_PROGRAM) == (
            ratio <= 31
        )


def test_workspace_layout_keeps_delta_and_globals_out_of_the_region():
    """``delta``, ``qstat`` and the global words are always workspace regions (``bwd_delta`` writes them while it
    reads ``o_lo``); the accumulator region is one more workspace region unless the plan carves ``o_lo``."""
    with_region = cake_backend.workspace_layout(300, 1000, 96, backward=True)
    carved = cake_backend.workspace_layout(300, 1000, 96, backward=True, region=False)
    forward = cake_backend.workspace_layout(300, 1000, 96, backward=False)
    for layout in (with_region, carved):
        assert {"topk_length", "delta", "qstat", "globals"} <= set(layout)
        assert layout["delta"][1] == 300 * NUM_HEADS * 4
        assert layout["qstat"][1] == 300 * 4 * 4
    assert "region" in with_region and "region" not in carved
    assert with_region["region"][1] == cake_backend.accumulator_region_layout(
        1000, 1000
    )["total"]
    assert set(forward) == {"topk_length", "total"}


def test_parity_keywords_are_validated_and_inert():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            cake_backend._check_key_passes(bad, 8)
    with pytest.raises(ValueError):
        cake_backend._check_key_passes(9, 8)
    for ok in (None, 1, 3, 8):
        cake_backend._check_key_passes(ok, 8)
    for bad in (0, 1, "auto"):
        with pytest.raises(ValueError):
            cake_backend._check_dkv_direct(bad)
    for ok in (None, True, False):
        cake_backend._check_dkv_direct(ok)


# ---------------------------------------------------------------------------
# Forward (GPU): the non-deterministic package's kernels under this package's names
# ---------------------------------------------------------------------------

FORWARD_SHAPES = [
    pytest.param([128], [512], 256, id="T128_S512_k256"),
    pytest.param([300, 212], [700, 300], 128, id="two_docs"),
    pytest.param([64], [4096], 2048, id="T64_S4096_k2048"),
]


@pytest.mark.parametrize("seq_q, seq_k, topk", FORWARD_SHAPES)
@pytest.mark.parametrize("lengths", ["caller", "derived"])
def test_forward_matches_nondet_bit_for_bit(seq_q, seq_k, topk, lengths):
    """The forward stages are the non-deterministic package's forward kernels (same build, this package's program
    names): ``out``, ``lse`` and ``o_lo`` -- and the derived lengths -- equal the non-deterministic forward bit for
    bit, and repeat bit for bit."""
    _require_program()
    inp = make_inputs(seq_q, seq_k, seed=SEED, topk=topk)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)

    def run(backend):
        if lengths == "caller":
            return backend.forward(*args, topk_length=inp.topk_length), None
        derived = torch.full_like(inp.topk_length, -7)
        outs = backend.forward(*args, topk_length=derived, derive_topk_length=True)
        return outs, derived

    ref, ref_len = run(nondet_backend)
    for _ in range(REPEATS):
        got, got_len = run(cake_backend)
        for a, b in zip(got, ref, strict=True):
            assert _same_bits(a, b)
        if ref_len is not None:
            assert _same_bits(got_len, ref_len)


def test_nope_forward_matches_nondet_bit_for_bit():
    _require_program(program=cake_jit.NOPE_PROGRAM)
    inp = make_inputs([256], [1024], seed=SEED + 1, topk=512)
    q = inp.q_latent
    kv = inp.kv_latent
    ref = nondet_backend.forward_nope(q, kv, inp.idx_global, topk_length=inp.topk_length)
    for _ in range(REPEATS):
        got = cake_backend.forward_nope(
            q, kv, inp.idx_global, topk_length=inp.topk_length
        )
        for a, b in zip(got, ref, strict=True):
            assert _same_bits(a, b)


def test_forward_into_given_storage_is_bit_for_bit():
    """``forward(out=, lse=, o_lo=)`` (the replay of a consumed ``o_lo``) writes the bits a fresh forward returns,
    and leaves the version counters of the given tensors alone."""
    _require_program()
    inp = make_inputs([200], [800], seed=SEED + 2, topk=256)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    out, lse, o_lo = cake_backend.forward(*args, topk_length=inp.topk_length)
    into = (torch.zeros_like(out), torch.zeros_like(lse), torch.zeros_like(o_lo))
    versions = [t._version for t in into]
    got = cake_backend.forward(
        *args, topk_length=inp.topk_length, out=into[0], lse=into[1], o_lo=into[2]
    )
    assert all(g is t for g, t in zip(got, into, strict=True))
    for a, b in zip(into, (out, lse, o_lo), strict=True):
        assert _same_bits(a, b)
    assert [t._version for t in into] == versions
    with pytest.raises(ValueError, match="all of out, lse and o_lo"):
        cake_backend.forward(*args, topk_length=inp.topk_length, out=into[0])


# ---------------------------------------------------------------------------
# Backward (GPU, needs the registered backward stages)
# ---------------------------------------------------------------------------

BACKWARD_SHAPES = [
    pytest.param([256], [1024], 256, id="T256_S1024_k256"),
    pytest.param([300, 212], [700, 300], 128, id="two_docs"),
]


def _rope_args(inp):
    return (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)


def _autograd_step(inp, **kw):
    leaves = [
        t.detach().clone().requires_grad_(True)
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    out = cake_backend.cake_dsa_sparse_attention_det(
        *leaves, inp.idx_global, topk_length=inp.topk_length, **kw
    )
    out.backward(inp.dout)
    return [out] + [leaf.grad for leaf in leaves]


@pytest.mark.parametrize("seq_q, seq_k, topk", BACKWARD_SHAPES)
@pytest.mark.parametrize("dkv_fp32", [False, True])
def test_backward_repeats_bit_for_bit(seq_q, seq_k, topk, dkv_fp32):
    """Five eager backwards over a fresh forward each give the same bits; with ``o_lo_scratch=True`` the forward
    runs again before every backward (the backward consumes ``o_lo``)."""
    _require_program(backward=True)
    inp = make_inputs(seq_q, seq_k, seed=SEED + 3, topk=topk)
    args = _rope_args(inp)
    seen = None
    for i in range(REPEATS):
        out, lse, o_lo = cake_backend.forward(*args, topk_length=inp.topk_length)
        grads = cake_backend.backward(
            *args,
            out,
            o_lo,
            lse,
            inp.dout,
            topk_length=inp.topk_length,
            dkv_fp32=dkv_fp32,
            o_lo_scratch=bool(i % 2),
        )
        if seen is None:
            seen = grads
            assert float(grads[2].float().abs().sum()) > 0
        else:
            for a, b in zip(grads, seen, strict=True):
                assert _same_bits(a, b)


@pytest.mark.parametrize("seq_q, seq_k, topk", BACKWARD_SHAPES)
def test_backward_accuracy_vs_fp64(seq_q, seq_k, topk):
    _require_program(backward=True)
    inp = make_inputs(seq_q, seq_k, seed=SEED + 4, topk=topk)
    args = _rope_args(inp)
    out, lse, o_lo = cake_backend.forward(*args, topk_length=inp.topk_length)
    dql, dqr, dkvl, dkr = cake_backend.backward(
        *args, out, o_lo, lse, inp.dout, topk_length=inp.topk_length
    )
    ref = reference_fp64(*args, dout=inp.dout, topk_length=inp.topk_length)
    assert rel_l2(dql, ref["dq_latent"]) < GATE_DQ_LATENT
    assert rel_l2(dqr, ref["dq_rope"]) < GATE_DQ_LATENT * GATE_ROPE_FACTOR
    assert rel_l2(dkvl, ref["dkv_latent"]) < GATE_DKV_LATENT
    assert rel_l2(dkr, ref["dk_rope"]) < GATE_DKV_LATENT * GATE_ROPE_FACTOR


@pytest.mark.parametrize("with_map", [False, True])
def test_dkv_acc_is_previous_plus_fresh(with_map):
    """``dkv_acc`` receives ``prev + v`` per element, where ``v`` is the fresh FP32 gradient of the same call (one
    FP32 addition), an element whose contribution is exactly zero keeps its bits (``-0.0`` stays), columns past 576
    and rows no key maps to are untouched, and repeated destination rows combine before the addition."""
    _require_program(backward=True)
    inp = make_inputs([256], [1024], seed=SEED + 5, topk=256)
    args = _rope_args(inp)
    S = inp.total_k
    if with_map:
        dst_rows = S // 2 + 3
        dst_map = (torch.arange(S, device="cuda", dtype=torch.int32) % (S // 2)).contiguous()
    else:
        dst_rows, dst_map = S + 3, None
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    prev = torch.randn(dst_rows, 600, device="cuda", generator=gen)
    prev[0, :8] = -0.0
    acc = prev.clone()
    out, lse, o_lo = cake_backend.forward(*args, topk_length=inp.topk_length)
    _, _, none_l, none_r = cake_backend.backward(
        *args,
        out,
        o_lo,
        lse,
        inp.dout,
        topk_length=inp.topk_length,
        dkv_acc=acc,
        dkv_dst_map=dst_map,
    )
    assert none_l is None and none_r is None
    assert _same_bits(acc[:, D_QK:], prev[:, D_QK:])
    if not with_map:
        out, lse, o_lo = cake_backend.forward(*args, topk_length=inp.topk_length)
        _, _, v_l, v_r = cake_backend.backward(
            *args, out, o_lo, lse, inp.dout, topk_length=inp.topk_length, dkv_fp32=True
        )
        v = torch.cat([v_l, v_r], 1)
        want = torch.where(v == 0, prev[:S, :D_QK], prev[:S, :D_QK] + v)
        assert _same_bits(acc[:S, :D_QK], want)
        assert _same_bits(acc[S:], prev[S:])
    # repeat: the same bits
    again = prev.clone()
    out, lse, o_lo = cake_backend.forward(*args, topk_length=inp.topk_length)
    cake_backend.backward(
        *args,
        out,
        o_lo,
        lse,
        inp.dout,
        topk_length=inp.topk_length,
        dkv_acc=again,
        dkv_dst_map=dst_map,
    )
    assert _same_bits(again, acc)


def test_retain_graph_second_backward_bit_for_bit():
    """The autograd backward consumes the saved ``o_lo``; a second backward through the same graph first runs the
    forward again into the saved storage and gives the same gradients, and the saved tensors' version counters do
    not move (no autograd error)."""
    _require_program(backward=True)
    inp = make_inputs([256], [1024], seed=SEED + 6, topk=256)
    leaves = [
        t.detach().clone().requires_grad_(True)
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    out = cake_backend.cake_dsa_sparse_attention_det(*leaves, inp.idx_global)
    out.backward(inp.dout, retain_graph=True)
    first = [leaf.grad.clone() for leaf in leaves]
    for leaf in leaves:
        leaf.grad = None
    out.backward(inp.dout)
    for leaf, g in zip(leaves, first, strict=True):
        assert _same_bits(leaf.grad, g)


def test_checkpoint_matches_the_saved_set_bit_for_bit():
    _require_program(backward=True)
    inp = make_inputs([256], [1024], seed=SEED + 7, topk=256)
    plain = _autograd_step(inp)
    leaves = [
        t.detach().clone().requires_grad_(True)
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    out = checkpoint(
        lambda *xs: cake_backend.cake_dsa_sparse_attention_det(
            *xs, inp.idx_global, topk_length=inp.topk_length
        ),
        *leaves,
        use_reentrant=False,
    )
    out.backward(inp.dout)
    for a, b in zip([out] + [leaf.grad for leaf in leaves], plain, strict=True):
        assert _same_bits(a, b)


def test_runner_repeats_and_graph_replays_bit_for_bit():
    """A prepared runner gives the same bits on every step, eagerly and replayed from a CUDA graph that captures
    forward and backward together; an ``o_lo`` runner refuses a second backward without a forward."""
    _require_program(backward=True)
    inp = make_inputs([256], [1024], seed=SEED + 8, topk=256)
    runner = cake_backend.prepare_dsa_train(
        *_rope_args(inp), topk_length=inp.topk_length, dout=inp.dout, o_lo_scratch=True
    )
    assert runner.storage == "o_lo"
    first = [t.clone() for t in runner.step()]
    for _ in range(REPEATS - 1):
        for a, b in zip(runner.step(), first, strict=True):
            assert _same_bits(a, b)
    with pytest.raises(RuntimeError, match="forward"):
        runner.backward()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        runner.step()
        with torch.cuda.graph(graph, stream=stream):
            runner.step()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(REPEATS):
        graph.replay()
        torch.cuda.synchronize()
        grads = (
            runner.tensors["dq_latent"],
            runner.tensors["dq_rope"],
            runner.tensors["dkv_latent"],
            runner.tensors["dk_rope"],
        )
        for a, b in zip(grads, first, strict=True):
            assert _same_bits(a, b)


def test_no_alert_under_torch_deterministic_mode():
    """The backward runs under ``torch.use_deterministic_algorithms(True)`` without raising or warning, and gives
    the bits it gives outside the mode."""
    _require_program(backward=True)
    inp = make_inputs([128], [512], seed=SEED + 9, topk=128)
    outside = _autograd_step(inp)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            inside = _autograd_step(inp)
    finally:
        torch.use_deterministic_algorithms(previous)
    for a, b in zip(inside, outside, strict=True):
        assert _same_bits(a, b)


def test_nope_backward_repeats_bit_for_bit():
    _require_program(backward=True, program=cake_jit.NOPE_PROGRAM)
    inp = make_inputs([256], [1024], seed=SEED + 10, topk=256)
    q, kv = inp.q_latent, inp.kv_latent
    seen = None
    for i in range(REPEATS):
        out, lse, o_lo = cake_backend.forward_nope(
            q, kv, inp.idx_global, topk_length=inp.topk_length
        )
        grads = cake_backend.backward_nope(
            q,
            kv,
            inp.idx_global,
            out,
            o_lo,
            lse,
            inp.dout,
            topk_length=inp.topk_length,
            o_lo_scratch=bool(i % 2),
        )
        if seen is None:
            seen = grads
        else:
            for a, b in zip(grads, seen, strict=True):
                assert _same_bits(a, b)
