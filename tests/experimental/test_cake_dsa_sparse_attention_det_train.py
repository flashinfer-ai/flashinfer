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
    make_nope_inputs,
    reference_fp64,
    reference_fp64_nope,
    rel_l2,
    rel_l2_rows,
)
from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_backend")
cake_jit = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_jit")
cake_launch = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_launch")
nondet_backend = cake_package(
    "cake.cake_dsa_sparse_attention_nondet_train", "cake_backend"
)

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
        "bwd_norm",
        "bwd_bound",
        "bwd_grid",
        "bwd_main",
        "bwd_redo_prep",
        "bwd_redo",
        "bwd_fold",
    )
    nope_stages = tuple(f"nope_{s}" for s in cake_jit.BACKWARD_STAGES)
    assert nope_stages == cake_jit.NOPE_BACKWARD_STAGES
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
        "grids",
        "row_pairs",
        "redo_mark",
        "redo_rows",
        "knorm",
    ]
    assert [k for k in layout if k != "total"] == names
    offsets = [layout[k][0] for k in names]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    sizes = {k: layout[k][1] for k in names}
    if accumulator:
        assert sizes["acc"] == R * columns * 4
    assert sizes["counters"] == R * 32
    assert sizes["grids"] == R * 4 * parts
    for name in ("row_pairs", "redo_mark", "redo_rows"):
        assert sizes[name] == R * 4
    assert sizes["knorm"] == S * 4
    assert layout["total"] == layout["knorm"][0] + _align(S * 4)


def _align(n):
    return -(-n // WORKSPACE_ALIGN) * WORKSPACE_ALIGN


@pytest.mark.parametrize("T", [1, 7, 1024, 16384])
@pytest.mark.parametrize("ratio", [1, 16, 27, 28, 31, 32, 64])
def test_carve_condition(T, ratio):
    """``carve_fits`` is the documented closed form, and the limits are about 27.7 T (rope) and 31.2 T (NoPE)."""
    R = S = T * ratio
    rope = (
        _align(R * 2304)
        + _align(R * 32)
        + _align(R * 8)
        + 3 * _align(R * 4)
        + _align(S * 4)
    )
    nope = _align(R * 2048) + _align(R * 32) + 4 * _align(R * 4) + _align(S * 4)
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
    """``delta``, ``qstat`` and the global words are never part of the accumulator region (``bwd_delta`` writes them
    while it reads ``o_lo``): workspace regions unless dead output storage holds them; the accumulator region is
    one more workspace region unless the plan carves ``o_lo``."""
    with_region = cake_backend.workspace_layout(300, 1000, 96, backward=True)
    carved = cake_backend.workspace_layout(300, 1000, 96, backward=True, region=False)
    forward = cake_backend.workspace_layout(300, 1000, 96, backward=False)
    for layout in (with_region, carved):
        assert {"topk_length", "delta", "qstat", "globals"} <= set(layout)
        assert layout["delta"][1] == 300 * NUM_HEADS * 4
        assert layout["qstat"][1] == 300 * 4 * 4
    assert "region" in with_region and "region" not in carved
    assert (
        with_region["region"][1]
        == cake_backend.accumulator_region_layout(1000, 1000)["total"]
    )
    assert set(forward) == {"topk_length", "total"}


def test_scratch_layouts_in_dead_output_storage():
    """``qstat`` and ``knorm`` fit dQ (64 KiB per token) up to about ``S = 16380 T``; the redo lists fit any fresh
    dK/dV output; what moves leaves the workspace and the region, and a per-call workspace has no ``topk_length``."""
    T, S = 300, 1000
    dq = cake_backend.dq_scratch_layout(T, S)
    assert dq["qstat"] == (0, T * 16) and dq["knorm"] == (_align(T * 16), S * 4)
    assert dq["total"] <= T * 65536
    assert cake_backend.dq_scratch_layout(1, 16384) is None
    assert cake_backend.redo_lists_fit(S, S)
    assert cake_backend.redo_lists_fit(S, S, dkv_fp32=True)
    region = cake_backend.accumulator_region_layout(S, S, knorm=False, redo=False)
    assert not {"knorm", "redo_mark", "redo_rows"} & set(region)
    full = cake_backend.accumulator_region_layout(S, S)["total"]
    assert region["total"] == full - 3 * _align(S * 4)
    moved = cake_backend.workspace_layout(
        T, S, 96, dq_scratch=True, redo_in_output=True
    )
    assert "qstat" not in moved and moved["region"][1] == region["total"]
    # delta and the global words follow the redo lists into a fresh dK/dV output when they fit it
    front = cake_backend.output_scratch_layout(T, S)
    assert front["delta"] == (2 * _align(S * 4), T * NUM_HEADS * 4)
    assert front["globals"] == (2 * _align(S * 4) + _align(T * NUM_HEADS * 4), 64)
    assert cake_backend.delta_fits_output(T, S, S)
    assert cake_backend.redo_lists_fit(64, 64)
    assert not cake_backend.delta_fits_output(T, 64, 64)
    assert cake_backend.delta_fits_output(T, 64, 64, dkv_fp32=True)
    no_delta = cake_backend.workspace_layout(
        T, S, 96, dq_scratch=True, redo_in_output=True, delta_in_output=True
    )
    assert not {"delta", "globals", "qstat"} & set(no_delta)
    assert no_delta["total"] == moved["total"] - _align(T * NUM_HEADS * 4) - _align(64)
    per_call = cake_backend.workspace_layout(T, S, 96, per_call=True)
    assert "topk_length" not in per_call
    assert per_call["total"] == cake_backend.workspace_layout(T, S, 96)[
        "total"
    ] - _align(T * 4)
    # 8350 rows over 300 tokens fit o_lo only without the key norms and redo lists
    assert cake_backend.carve_fits(T, 8350, 8350, knorm=False, redo=False)
    assert not cake_backend.carve_fits(T, 8350, 8350)


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
    ref = nondet_backend.forward_nope(
        q, kv, inp.idx_global, topk_length=inp.topk_length
    )
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
    """dQ is the non-deterministic package's dQ bit for bit (the same computation; its error against FP64 is that
    package's, reported); dK/dV within the fixed gates."""
    _require_program(backward=True)
    inp = make_inputs(seq_q, seq_k, seed=SEED + 4, topk=topk)
    args = _rope_args(inp)
    tl = inp.topk_length
    out, lse, o_lo = cake_backend.forward(*args, topk_length=tl)
    dql, dqr, dkvl, dkr = cake_backend.backward(
        *args, out, o_lo, lse, inp.dout, topk_length=tl
    )
    n = nondet_backend.forward(*args, topk_length=tl)
    nondet = nondet_backend.backward(*args, n[0], n[2], n[1], inp.dout, topk_length=tl)
    ref = reference_fp64(*args, dout=inp.dout, topk_length=tl)
    _report(
        f"vs_fp64_{len(seq_q)}docs_k{topk}",
        dict(
            dq_latent=rel_l2(dql, ref["dq_latent"]),
            dq_rope=rel_l2(dqr, ref["dq_rope"]),
            dkv_latent=rel_l2(dkvl, ref["dkv_latent"]),
            dkv_latent_nondet=rel_l2(nondet[2], ref["dkv_latent"]),
            dk_rope=rel_l2(dkr, ref["dk_rope"]),
            dk_rope_nondet=rel_l2(nondet[3], ref["dk_rope"]),
        ),
    )
    assert _same_bits(dql, nondet[0])
    assert _same_bits(dqr, nondet[1])
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
        dst_map = (
            torch.arange(S, device="cuda", dtype=torch.int32) % (S // 2)
        ).contiguous()
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


def test_nope_dkv_acc_is_previous_plus_fresh():
    """NoPE: ``dkv_acc`` receives ``prev + v`` per element (``v`` the fresh FP32 ``dkv`` of the same inputs, an
    element whose integer is zero keeps its bits), columns past 512 and rows no key maps to are untouched, and the
    same call repeats bit for bit; through a map with repeated destination rows the result repeats too."""
    _require_program(backward=True, program=cake_jit.NOPE_PROGRAM)
    inp = make_inputs([256], [1024], seed=SEED + 11, topk=256)
    q, kv = inp.q_latent, inp.kv_latent
    S = inp.total_k
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    prev = torch.randn(S + 3, 520, device="cuda", generator=gen)
    prev[0, :8] = -0.0

    def run(**kw):
        out, lse, o_lo = cake_backend.forward_nope(
            q, kv, inp.idx_global, topk_length=inp.topk_length
        )
        return cake_backend.backward_nope(
            q, kv, inp.idx_global, out, o_lo, lse, inp.dout,
            topk_length=inp.topk_length, **kw,
        )  # fmt: skip

    _, v = run()
    acc = prev.clone()
    _, none = run(dkv_acc=acc)
    assert none is None
    want = torch.where(v == 0, prev[:S, :D_LATENT], prev[:S, :D_LATENT] + v)
    assert _same_bits(acc[:S, :D_LATENT], want)
    assert _same_bits(acc[:, D_LATENT:], prev[:, D_LATENT:])
    assert _same_bits(acc[S:], prev[S:])
    dst_map = (
        torch.arange(S, device="cuda", dtype=torch.int32) % (S // 2)
    ).contiguous()
    first = prev.clone()
    run(dkv_acc=first, dkv_dst_map=dst_map)
    assert not _same_bits(first, prev)
    for _ in range(REPEATS - 1):
        again = prev.clone()
        run(dkv_acc=again, dkv_dst_map=dst_map)
        assert _same_bits(again, first)


def _nope_autograd_step(inp, **kw):
    q = inp.q_latent.detach().clone().requires_grad_(True)
    kv = inp.kv_latent.detach().clone().requires_grad_(True)
    out = cake_backend.cake_dsa_sparse_attention_det_nope(
        q, kv, inp.idx_global, topk_length=inp.topk_length, **kw
    )
    out.backward(inp.dout)
    return [out, q.grad, kv.grad]


def test_nope_checkpoint_retain_graph_and_no_alert():
    """NoPE autograd: activation checkpointing, a second backward through a retained graph and
    ``torch.use_deterministic_algorithms(True)`` all give the bits of the plain saved-set step."""
    _require_program(backward=True, program=cake_jit.NOPE_PROGRAM)
    inp = make_inputs([256], [1024], seed=SEED + 12, topk=256)
    plain = _nope_autograd_step(inp)
    q = inp.q_latent.detach().clone().requires_grad_(True)
    kv = inp.kv_latent.detach().clone().requires_grad_(True)
    out = checkpoint(
        lambda a, b: cake_backend.cake_dsa_sparse_attention_det_nope(
            a, b, inp.idx_global, topk_length=inp.topk_length
        ),
        q,
        kv,
        use_reentrant=False,
    )
    out.backward(inp.dout)
    for a, b in zip([out, q.grad, kv.grad], plain, strict=True):
        assert _same_bits(a, b)
    q = inp.q_latent.detach().clone().requires_grad_(True)
    kv = inp.kv_latent.detach().clone().requires_grad_(True)
    out = cake_backend.cake_dsa_sparse_attention_det_nope(
        q, kv, inp.idx_global, topk_length=inp.topk_length
    )
    out.backward(inp.dout, retain_graph=True)
    first = [q.grad.clone(), kv.grad.clone()]
    q.grad = kv.grad = None
    out.backward(inp.dout)
    assert _same_bits(q.grad, first[0]) and _same_bits(kv.grad, first[1])
    assert _same_bits(first[0], plain[1]) and _same_bits(first[1], plain[2])
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            inside = _nope_autograd_step(inp)
    finally:
        torch.use_deterministic_algorithms(previous)
    for a, b in zip(inside, plain, strict=True):
        assert _same_bits(a, b)


@pytest.mark.parametrize("accumulate", [False, True])
def test_nope_runner_graph_replays_bit_for_bit(accumulate):
    """A NoPE runner (fresh ``dkv``, or adding into a zeroed ``dkv_acc`` re-zeroed inside the graph) gives the same
    bits eagerly and replayed from a CUDA graph capturing forward and backward together."""
    _require_program(backward=True, program=cake_jit.NOPE_PROGRAM)
    inp = make_inputs([256], [1024], seed=SEED + 13, topk=256)
    acc = (
        torch.zeros(inp.total_k, D_LATENT, device="cuda", dtype=torch.float32)
        if accumulate
        else None
    )
    runner = cake_backend.prepare_dsa_nope_train(
        inp.q_latent,
        inp.kv_latent,
        inp.idx_global,
        topk_length=inp.topk_length,
        dout=inp.dout,
        dkv_acc=acc,
        o_lo_scratch=True,
    )

    def step():
        if acc is not None:
            acc.zero_()
        runner.forward()
        dq, dkv = runner.backward()
        return dq, (acc if acc is not None else dkv)

    first = [t.clone() for t in step()]
    assert float(first[1].abs().sum()) > 0
    for _ in range(REPEATS - 1):
        for a, b in zip(step(), first, strict=True):
            assert _same_bits(a, b)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        step()
        with torch.cuda.graph(graph, stream=stream):
            got = step()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(REPEATS):
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(got, first, strict=True):
            assert _same_bits(a, b)


# ---------------------------------------------------------------------------
# Accuracy against FP64 at the issue shapes and on adversarial inputs, the recomputation of flagged rows,
# non-finite inputs and host synchronization (GPU, needs the registered backward stages)
# ---------------------------------------------------------------------------

GATE_LSE_ABS = 2e-5  # the non-deterministic suite's lse gate (natural log, absolute)
NOPE_FLOOR_MARGIN = 1.02  # the non-deterministic suite's NoPE dq floor margin


def _globals(runner) -> list[int]:
    """The global words of the runner's last backward (word 8: rows the redo preparation flagged; word 9: rows
    with a non-finite contribution)."""
    return [int(x) for x in runner.tensors["det_globals"].view(torch.int32).tolist()]


def _report(case: str, errors: dict) -> None:
    print(
        "ACCURACY",
        case,
        " ".join(f"{k}={v:.6g}" for k, v in errors.items()),
        flush=True,
    )


def _check_lse(lse, ref):
    lse_ref = ref["lse"]
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), finite)
    if finite.any():
        err = (lse.double()[finite] - lse_ref[finite]).abs().max().item()
        assert err <= GATE_LSE_ABS, err


def _check_rope_accuracy(case, grads, ref, *, canonical, nondet=None, dq_bitwise=False):
    """The non-deterministic suite's rule: every gradient within 1.05x its BF16-P/dS numerics floor on the same
    inputs (+1e-5); the canonical top-k-2048 configurations also within the fixed gates.  ``nondet`` (the
    non-deterministic package's gradients on the same inputs) is reported next to it.  ``dq_bitwise``: dQ (the
    same computation in both packages) is held to the non-deterministic dQ bit for bit instead of the floor,
    on inputs where that package's dQ is above the floor too."""
    names = ("dq_latent", "dq_rope", "dkv_latent", "dk_rope")
    errors, failures = {}, []
    for i, (g, name) in enumerate(zip(grads, names, strict=True)):
        got, floor = rel_l2(g, ref[name]), rel_l2(ref[f"{name}_emu"], ref[name])
        errors[name], errors[f"{name}_floor"] = got, floor
        if nondet is not None:
            errors[f"{name}_nondet"] = rel_l2(nondet[i], ref[name])
        if not torch.isfinite(g.float()).all():
            failures.append(f"{name}: non-finite")
        if dq_bitwise and name.startswith("dq"):
            if not _same_bits(g, nondet[i]):
                failures.append(f"{name}: not the non-deterministic dQ bit for bit")
        elif got > FLOOR_MARGIN * floor + FLOOR_ABS:
            failures.append(f"{name}: rel-L2 {got:.6f} vs floor {floor:.6f}")
    if canonical:
        gates = dict(
            dq_latent=GATE_DQ_LATENT,
            dq_rope=GATE_DQ_LATENT * GATE_ROPE_FACTOR,
            dkv_latent=GATE_DKV_LATENT,
            dk_rope=GATE_DKV_LATENT * GATE_ROPE_FACTOR,
        )
        failures += [
            f"{n}: rel-L2 {errors[n]:.6f} above the gate {g}"
            for n, g in gates.items()
            if errors[n] > g
        ]
    _report(case, errors)
    assert not failures, failures


def _check_nope_accuracy(case, dq, dkv, ref, *, nondet=None, dq_bitwise=False):
    """The non-deterministic suite's NoPE rule: ``dq`` within 1.02x and ``dkv`` within 1.05x their numerics floors
    on the same inputs (+1e-5); ``dq_bitwise`` as in :func:`_check_rope_accuracy`."""
    errors, failures = {}, []
    for i, (got_t, name, margin) in enumerate(
        ((dq, "dq", NOPE_FLOOR_MARGIN), (dkv, "dkv", FLOOR_MARGIN))
    ):
        got, floor = rel_l2(got_t, ref[name]), rel_l2(ref[f"{name}_emu"], ref[name])
        errors[name], errors[f"{name}_floor"] = got, floor
        if nondet is not None:
            errors[f"{name}_nondet"] = rel_l2(nondet[i], ref[name])
        if not torch.isfinite(got_t.float()).all():
            failures.append(f"{name}: non-finite")
        if dq_bitwise and name == "dq":
            if not _same_bits(got_t, nondet[i]):
                failures.append("dq: not the non-deterministic dQ bit for bit")
        elif ref[name].any() and got > margin * floor + FLOOR_ABS:
            failures.append(f"{name}: rel-L2 {got:.6f} vs floor {floor:.6f}")
    _report(case, errors)
    assert not failures, failures


def _p8949_indices(t, s, gen):
    """The flashinfer#8949 repro's index rows: key 0, one key from each of 1023 strata of the older prefix and the
    1024 most recent positions (query ``i`` sits at position ``s - t + i``); distinct and causal."""
    pos = torch.arange(s - t, s, device="cuda")
    recent = pos[:, None] - torch.arange(1024, device="cuda")[None, :]
    older = (pos - 1024).double()
    u = torch.rand((t, 1023), device="cuda", dtype=torch.float64, generator=gen)
    strata = (
        1
        + (
            (torch.arange(1023, device="cuda")[None, :] + u)
            * (older[:, None] - 1)
            / 1023
        ).long()
    )
    sink = torch.zeros((t, 1), device="cuda", dtype=torch.long)
    return torch.cat([sink, strata, recent.flip(1)], 1).int().contiguous()


def _p8949_problem(nope, t=1024, s=8192):
    """The flashinfer#8949 repro's tensors (generator seed 0, N(0, 1) BF16 in the repro's draw order) at the
    accuracy shape ``T = 1024``, ``S = 8192`` (top-k 2048, every slot valid)."""
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(shape, device="cuda", generator=gen).bfloat16()

    p = dict(
        q_latent=randn(t, NUM_HEADS, D_LATENT),
        kv_latent=randn(s, D_LATENT),
        dout=randn(t, NUM_HEADS, D_LATENT),
    )
    if not nope:
        p.update(q_rope=randn(t, NUM_HEADS, D_ROPE), k_rope=randn(s, D_ROPE))
    p["indices"] = _p8949_indices(t, s, gen)
    p["topk_length"] = torch.full((t,), 2048, dtype=torch.int32, device="cuda")
    return p


def test_accuracy_issue_8949_shape():
    """flashinfer#8949's accuracy shape (T 1024, S 8192, its index generator): every gradient within the floors and
    the fixed gates, the BF16 and FP32 dK/dV forms both; adding into a zeroed ``dkv_acc`` gives the FP32 form's
    bits (``+0.0 + v = v``)."""
    _require_program(backward=True)
    p = _p8949_problem(nope=False)
    args = (p["q_latent"], p["q_rope"], p["kv_latent"], p["k_rope"], p["indices"])
    tl = p["topk_length"]
    out, lse, o_lo = cake_backend.forward(*args, topk_length=tl)
    bf16 = cake_backend.backward(*args, out, o_lo, lse, p["dout"], topk_length=tl)
    out, lse, o_lo = cake_backend.forward(*args, topk_length=tl)
    fp32 = cake_backend.backward(
        *args, out, o_lo, lse, p["dout"], topk_length=tl, dkv_fp32=True
    )
    acc = torch.zeros(p["kv_latent"].shape[0], D_QK, device="cuda")
    out, lse, o_lo = cake_backend.forward(*args, topk_length=tl)
    cake_backend.backward(*args, out, o_lo, lse, p["dout"], topk_length=tl, dkv_acc=acc)
    assert _same_bits(acc[:, :D_LATENT], fp32[2]) and _same_bits(
        acc[:, D_LATENT:], fp32[3]
    )
    n_out, n_lse, n_o_lo = nondet_backend.forward(*args, topk_length=tl)
    nondet = nondet_backend.backward(
        *args, n_out, n_o_lo, n_lse, p["dout"], topk_length=tl
    )
    ref = reference_fp64(*args, dout=p["dout"], topk_length=tl)
    _check_lse(lse, ref)
    _check_rope_accuracy("p8949_rope_bf16", bf16, ref, canonical=True, nondet=nondet)
    _check_rope_accuracy("p8949_rope_fp32", fp32, ref, canonical=True)


def test_accuracy_issue_8949_shape_nope():
    _require_program(backward=True, program=cake_jit.NOPE_PROGRAM)
    p = _p8949_problem(nope=True)
    args = (p["q_latent"], p["kv_latent"], p["indices"])
    tl = p["topk_length"]
    out, lse, o_lo = cake_backend.forward_nope(*args, topk_length=tl)
    dq, dkv = cake_backend.backward_nope(
        *args, out, o_lo, lse, p["dout"], topk_length=tl
    )
    acc = torch.zeros(p["kv_latent"].shape[0], D_LATENT, device="cuda")
    out, lse, o_lo = cake_backend.forward_nope(*args, topk_length=tl)
    cake_backend.backward_nope(
        *args, out, o_lo, lse, p["dout"], topk_length=tl, dkv_acc=acc
    )
    assert _same_bits(acc, dkv)
    n_out, n_lse, n_o_lo = nondet_backend.forward_nope(*args, topk_length=tl)
    nondet = nondet_backend.backward_nope(
        *args, n_out, n_o_lo, n_lse, p["dout"], topk_length=tl
    )
    ref = reference_fp64_nope(*args, dout=p["dout"], topk_length=tl)
    _check_lse(lse, ref)
    _check_nope_accuracy("p8949_nope", dq, dkv, ref, nondet=nondet)


def _glm_dst_map(num_kv: int) -> tuple[torch.Tensor, int]:
    """A monotone destination map that merges one pair of neighbouring keys in every 33 (``268757 -> 260611`` rows
    in the recorded GLM-5.2 batch merges about 3 %): ``(map, rows)``."""
    keys = torch.arange(num_kv, device="cuda", dtype=torch.int64)
    dst = (keys * 32) // 33
    return dst.to(torch.int32).contiguous(), int(dst[-1].item()) + 1


def test_accuracy_issue_8946_glm52_packed_dkv_acc_with_map():
    """flashinfer#8946's GLM-5.2 workload form: packed documents (context-parallel tails of longer prefixes), top-k
    2048, the FP32 ``dkv_acc`` addressed through ``dkv_dst_map`` with repeated destination rows.  ``dq`` within the
    floors and gates; ``dkv_acc`` (zeroed) against the FP64 gradient mapped by ``index_add_`` within the gates and
    the mapped BF16 floor; repeats bit for bit."""
    _require_program(backward=True)
    inp = make_inputs([2048, 1024, 1024], [6144, 1024, 3072], seed=SEED + 20, topk=2048)
    args = _rope_args(inp)
    S = inp.total_k
    dst_map, rows = _glm_dst_map(S)

    def run():
        acc = torch.zeros(rows, D_QK, device="cuda")
        out, lse, o_lo = cake_backend.forward(*args, topk_length=inp.topk_length)
        dql, dqr, _, _ = cake_backend.backward(
            *args, out, o_lo, lse, inp.dout, topk_length=inp.topk_length,
            dkv_acc=acc, dkv_dst_map=dst_map,
        )  # fmt: skip
        return dql, dqr, acc, lse

    dql, dqr, acc, lse = run()
    for _ in range(2):
        again = run()
        for a, b in zip(again[:3], (dql, dqr, acc), strict=True):
            assert _same_bits(a, b)
    ref = reference_fp64(*args, dout=inp.dout, topk_length=inp.topk_length)
    _check_lse(lse, ref)
    index = dst_map.long()

    def mapped(t):
        return torch.zeros(
            rows, t.shape[1], dtype=torch.float64, device="cuda"
        ).index_add_(0, index, t.double())

    grads = (dql, dqr, acc[:, :D_LATENT], acc[:, D_LATENT:])
    ref_mapped = dict(ref)
    for name in ("dkv_latent", "dk_rope"):
        ref_mapped[name] = mapped(ref[name])
        ref_mapped[f"{name}_emu"] = mapped(ref[f"{name}_emu"])
    _check_rope_accuracy("i8946_glm52_packed_map", grads, ref_mapped, canonical=True)


def test_accuracy_issue_8946_glm53_flash_nope():
    """flashinfer#8946's GLM-5.3-Flash NoPE form: the pooled index structure (four-token pools, 2176 slots), scale
    1/16, fresh ``dkv`` and a zeroed ``dkv_acc`` (the same bits)."""
    _require_program(backward=True, program=cake_jit.NOPE_PROGRAM)
    inp = make_nope_inputs([2048, 2048], [8192, 4096], seed=SEED + 21)
    args = (inp.q, inp.kv, inp.idx_global)
    tl = inp.topk_length
    out, lse, o_lo = cake_backend.forward_nope(*args, topk_length=tl)
    dq, dkv = cake_backend.backward_nope(
        *args, out, o_lo, lse, inp.dout, topk_length=tl
    )
    acc = torch.zeros(inp.total_k, D_LATENT, device="cuda")
    out, lse, o_lo = cake_backend.forward_nope(*args, topk_length=tl)
    cake_backend.backward_nope(
        *args, out, o_lo, lse, inp.dout, topk_length=tl, dkv_acc=acc
    )
    assert _same_bits(acc, dkv)
    n_out, n_lse, n_o_lo = nondet_backend.forward_nope(*args, topk_length=tl)
    nondet = nondet_backend.backward_nope(
        *args, n_out, n_o_lo, n_lse, inp.dout, topk_length=tl
    )
    ref = reference_fp64_nope(*args, dout=inp.dout, topk_length=tl)
    _check_lse(lse, ref)
    _check_nope_accuracy("i8946_glm53_nope", dq, dkv, ref, nondet=nondet)


def _sink_beta(q_parts, k_parts, scale, target=0.95):
    """``beta`` such that adding ``beta * (key 0)`` to every query puts a mean softmax mass of about ``target`` on
    key 0 (causal rows over all earlier keys; bisection on an FP32 evaluation)."""
    T, S = q_parts[0].shape[0], k_parts[0].shape[0]
    causal = torch.arange(S, device="cuda")[None, :] <= (
        torch.arange(T, device="cuda")[:, None] + (S - T)
    )
    lo, hi = 0.0, 4.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        logits = (
            sum(
                torch.einsum("thd,sd->ths", (q.float() + mid * k[0].float()), k.float())
                for q, k in zip(q_parts, k_parts, strict=True)
            )
            * scale
        )
        logits = logits.masked_fill(~causal[:, None, :], float("-inf"))
        w = torch.softmax(logits, -1)[..., 0].mean().item()
        if w < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sink_problem(nope, seed, T=256):
    """A dominant-key ("attention sink") problem: every query selects every earlier key (top-k >= T), key 0 carries
    about 95 % of the softmax mass of every row, and ``dout`` is aligned across tokens and heads (all ones plus 1/8
    of noise), so key 0's dK/dV row adds ``T x 64`` nearly equal contributions: its S2 bound is loose by only about
    two orders of magnitude, and the row overflows the grid 12 bits finer than the bound needs."""
    inp = make_inputs([T], [T], seed=seed, topk=T)
    q_parts = [inp.q_latent] if nope else [inp.q_latent, inp.q_rope]
    k_parts = [inp.kv_latent] if nope else [inp.kv_latent, inp.k_rope]
    scale = cake_backend.NOPE_SOFTMAX_SCALE if nope else D_QK**-0.5
    beta = _sink_beta(q_parts, k_parts, scale)
    for q, k in zip(q_parts, k_parts, strict=True):
        q.add_(k[0][None, None, :], alpha=beta)
    dout = (torch.ones_like(inp.dout.float()) + 0.125 * inp.dout.float()).bfloat16()
    return inp, dout


def test_backward_dominant_key_exercises_the_redo():
    """The dominant-key problem flags key 0's row for recomputation on the default grid (the redo runs: coverage
    guard), no row is left non-finite, the gradients are within the floors, and the runner and the eager entry
    give the same bits on every repeat."""
    _require_program(backward=True)
    inp, dout = _sink_problem(nope=False, seed=SEED + 22)
    args = _rope_args(inp)
    runner = cake_backend.prepare_dsa_train(
        *args, topk_length=inp.topk_length, dout=dout, dkv_fp32=True
    )
    first = [t.clone() for t in runner.step()]
    words = _globals(runner)
    assert words[cake_backend.REDO_WORD] > 0, (
        "coverage: no row was flagged for the redo"
    )
    assert words[cake_backend.NONFINITE_WORD] == 0
    for _ in range(REPEATS - 1):
        for a, b in zip(runner.step(), first, strict=True):
            assert _same_bits(a, b)
    out, lse, o_lo = cake_backend.forward(*args, topk_length=inp.topk_length)
    eager = cake_backend.backward(
        *args, out, o_lo, lse, dout, topk_length=inp.topk_length, dkv_fp32=True
    )
    for a, b in zip(eager, first, strict=True):
        assert _same_bits(a, b)
    n_out, n_lse, n_o_lo = nondet_backend.forward(*args, topk_length=inp.topk_length)
    nondet = nondet_backend.backward(
        *args, n_out, n_o_lo, n_lse, dout, topk_length=inp.topk_length, dkv_fp32=True
    )
    ref = reference_fp64(
        *args,
        dout=dout,
        topk_length=inp.topk_length,
        own_key=torch.zeros(inp.total_q, dtype=torch.long, device="cuda"),
    )
    print(
        "SINK rope flagged", words[cake_backend.REDO_WORD], "weight", ref["self_weight"]
    )
    assert 0.8 < ref["self_weight"] < 0.999
    _check_rope_accuracy("sink_rope", first, ref, canonical=False, nondet=nondet)


def test_backward_dominant_key_exercises_the_redo_nope():
    _require_program(backward=True, program=cake_jit.NOPE_PROGRAM)
    inp, dout = _sink_problem(nope=True, seed=SEED + 23)
    args = (inp.q_latent, inp.kv_latent, inp.idx_global)
    runner = cake_backend.prepare_dsa_nope_train(
        *args, topk_length=inp.topk_length, dout=dout
    )
    runner.forward()
    first = [t.clone() for t in runner.backward()]
    words = _globals(runner)
    assert words[cake_backend.REDO_WORD] > 0, (
        "coverage: no row was flagged for the redo"
    )
    assert words[cake_backend.NONFINITE_WORD] == 0
    for _ in range(REPEATS - 1):
        runner.forward()
        for a, b in zip(runner.backward(), first, strict=True):
            assert _same_bits(a, b)
    out, lse, o_lo = cake_backend.forward_nope(*args, topk_length=inp.topk_length)
    eager = cake_backend.backward_nope(
        *args, out, o_lo, lse, dout, topk_length=inp.topk_length
    )
    for a, b in zip(eager, first, strict=True):
        assert _same_bits(a, b)
    n_out, n_lse, n_o_lo = nondet_backend.forward_nope(
        *args, topk_length=inp.topk_length
    )
    nondet = nondet_backend.backward_nope(
        *args, n_out, n_o_lo, n_lse, dout, topk_length=inp.topk_length
    )
    ref = reference_fp64_nope(
        *args,
        dout=dout,
        topk_length=inp.topk_length,
        own_key=torch.zeros(inp.total_q, dtype=torch.long, device="cuda"),
    )
    print(
        "SINK nope flagged", words[cake_backend.REDO_WORD], "weight", ref["self_weight"]
    )
    assert 0.8 < ref["self_weight"] < 0.999
    _check_nope_accuracy("sink_nope", first[0], first[1], ref, nondet=nondet)


@pytest.mark.parametrize("nope", [False, True], ids=["rope", "nope"])
def test_forced_redo_of_most_rows_repeats_bit_for_bit(monkeypatch, nope):
    """With the grid 24 bits finer than the bound needs (``GRID_SHIFT = 24``, ``MIN_BUMP = 25``) most rows overflow
    and are recomputed by the redo: no row is left flagged or non-finite, ``dq`` keeps its bits (it does not depend
    on the grid), and five steps give the same bits.  The error against FP64 is reported (this grid is not a
    shipped configuration)."""
    program = cake_jit.NOPE_PROGRAM if nope else cake_jit.PROGRAM
    _require_program(backward=True, program=program)
    inp = make_inputs([256], [1024], seed=SEED + 24, topk=256)

    def prepare():
        if nope:
            return cake_backend.prepare_dsa_nope_train(
                inp.q_latent, inp.kv_latent, inp.idx_global,
                topk_length=inp.topk_length, dout=inp.dout,
            )  # fmt: skip
        return cake_backend.prepare_dsa_train(
            *_rope_args(inp), topk_length=inp.topk_length, dout=inp.dout, dkv_fp32=True
        )

    def step(runner):
        runner.forward()
        return [t.clone() for t in runner.backward()]

    shipped = step(prepare())
    monkeypatch.setattr(cake_backend, "GRID_SHIFT", 24)
    monkeypatch.setattr(cake_backend, "MIN_BUMP", 25)
    runner = prepare()
    first = step(runner)
    words = _globals(runner)
    print(
        "FORCED_REDO",
        "nope" if nope else "rope",
        "flagged",
        words[cake_backend.REDO_WORD],
    )
    assert words[cake_backend.REDO_WORD] > inp.total_k // 4
    assert words[cake_backend.NONFINITE_WORD] == 0
    assert all(torch.isfinite(t.float()).all() for t in first)
    assert _same_bits(first[0], shipped[0])
    if not nope:
        assert _same_bits(first[1], shipped[1])
    for _ in range(REPEATS - 1):
        for a, b in zip(step(runner), first, strict=True):
            assert _same_bits(a, b)
    dkv = first[-1] if nope else torch.cat(first[2:], 1)
    ref_dkv = shipped[-1] if nope else torch.cat(shipped[2:], 1)
    print("FORCED_REDO rel-L2 vs the shipped grid", rel_l2(dkv, ref_dkv))


@pytest.mark.parametrize("nope", [False, True], ids=["rope", "nope"])
def test_redo_persistent_grid_gives_the_bits_of_a_cta_per_token(monkeypatch, nope):
    """The redo launches one persistent CTA per SM (``redo_launch_grid``) whatever the key count; with most rows
    recomputed by the redo (``GRID_SHIFT = 24``) it gives the bits of a redo launched with a CTA per token."""
    _require_program(
        backward=True, program=cake_jit.NOPE_PROGRAM if nope else cake_jit.PROGRAM
    )
    inp = make_inputs([256], [1024], seed=SEED + 32, topk=256)
    T = inp.total_q
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    assert cake_backend.redo_launch_grid(T, sm_count) == min(T, sm_count)
    assert cake_backend.redo_launch_grid(T, 1 << 20) == T
    assert cake_backend.redo_launch_grid(0, sm_count) == 1
    monkeypatch.setattr(cake_backend, "GRID_SHIFT", 24)
    monkeypatch.setattr(cake_backend, "MIN_BUMP", 25)
    stage = "nope_bwd_redo" if nope else "bwd_redo"
    table = "NOPE_PERSISTENT_GRIDS" if nope else "PERSISTENT_GRIDS"

    def run():
        if nope:
            runner = cake_backend.prepare_dsa_nope_train(
                inp.q_latent, inp.kv_latent, inp.idx_global,
                topk_length=inp.topk_length, dout=inp.dout,
            )  # fmt: skip
        else:
            runner = cake_backend.prepare_dsa_train(
                *_rope_args(inp), topk_length=inp.topk_length, dout=inp.dout,
            )  # fmt: skip
        runner.forward()
        out = [t.clone() for t in runner.backward() if t is not None]
        return runner, out

    persistent, got = run()
    assert tuple(persistent.launches[stage].grid) == (min(T, sm_count), 1, 1)
    assert _globals(persistent)[cake_backend.REDO_WORD] > inp.total_k // 4
    grids = dict(getattr(cake_backend, table))
    grids[stage] = lambda num_queries, num_kv, sms: num_queries
    monkeypatch.setattr(cake_backend, table, grids)
    per_token, want = run()
    assert tuple(per_token.launches[stage].grid) == (T, 1, 1)
    for a, b in zip(got, want, strict=True):
        assert _same_bits(a, b)


def _inside(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Whether the bytes of ``x`` lie inside the bytes of ``y``."""
    lo, hi = y.data_ptr(), y.data_ptr() + y.numel() * y.element_size()
    return lo <= x.data_ptr() and x.data_ptr() + x.numel() * x.element_size() <= hi


@pytest.mark.parametrize("mode", ["bf16", "fp32", "acc", "nope", "nope_acc"])
def test_scratch_in_dead_output_storage_gives_the_same_bits(monkeypatch, mode):
    """A runner's default plan keeps ``qstat`` and ``knorm`` in dQ and (fresh RoPE outputs) the redo lists,
    ``delta`` and the global words in the dK/dV output; with most rows recomputed by the redo (``GRID_SHIFT = 24``)
    it gives the bits of the plan that keeps only the redo lists there and of the plan that keeps all of them in the
    workspace."""
    nope = mode.startswith("nope")
    program = cake_jit.NOPE_PROGRAM if nope else cake_jit.PROGRAM
    _require_program(backward=True, program=program)
    inp = make_inputs([256], [1024], seed=SEED + 31, topk=256)
    S = inp.total_k
    monkeypatch.setattr(cake_backend, "GRID_SHIFT", 24)
    monkeypatch.setattr(cake_backend, "MIN_BUMP", 25)

    def run():
        acc = None
        if mode.endswith("acc"):
            acc = torch.zeros(S, 512 if nope else 576, device="cuda")
        if nope:
            runner = cake_backend.prepare_dsa_nope_train(
                inp.q_latent, inp.kv_latent, inp.idx_global,
                topk_length=inp.topk_length, dout=inp.dout, dkv_acc=acc,
            )  # fmt: skip
        else:
            runner = cake_backend.prepare_dsa_train(
                *_rope_args(inp), topk_length=inp.topk_length, dout=inp.dout,
                dkv_fp32=mode == "fp32", dkv_acc=acc,
            )  # fmt: skip
        runner.forward()
        grads = [g.clone() for g in runner.backward() if g is not None]
        assert _globals(runner)[cake_backend.REDO_WORD] > S // 4
        return runner, grads + ([acc.clone()] if acc is not None else [])

    moved, got = run()
    t = moved.tensors
    assert _inside(t["qstat"], t["dq_latent"]) and _inside(t["knorm"], t["dq_latent"])
    fresh = mode in ("bf16", "fp32")
    out_key = "dkv_latent_fp32" if mode == "fp32" else "dkv_latent"
    if fresh:
        out = t[out_key]
        for name in ("redo_mark", "redo_rows", "delta", "det_globals"):
            assert _inside(t[name], out), name
        # the redo lists alone in the output, delta and the global words in the workspace
        monkeypatch.setattr(cake_backend, "delta_fits_output", lambda *a, **k: False)
        half, same = run()
        h = half.tensors
        assert _inside(h["redo_mark"], h[out_key])
        assert not _inside(h["delta"], h[out_key])
        assert not _inside(h["det_globals"], h[out_key])
        for a, b in zip(got, same, strict=True):
            assert _same_bits(a, b)
    monkeypatch.setattr(cake_backend, "dq_scratch_layout", lambda *a: None)
    monkeypatch.setattr(cake_backend, "redo_lists_fit", lambda *a, **k: False)
    kept, want = run()
    k = kept.tensors
    assert not _inside(k["qstat"], k["dq_latent"])
    assert not _inside(k["knorm"], k["dq_latent"])
    if fresh:
        assert not _inside(k["redo_mark"], k[out_key])
        assert not _inside(k["delta"], k[out_key])
    for a, b in zip(got, want, strict=True):
        assert _same_bits(a, b)


@pytest.mark.parametrize("nope", [False, True], ids=["rope", "nope"])
def test_backward_skewed_magnitudes(nope):
    """Key rows scaled by powers of two over 2 ** +-3 and ``dout`` tokens over 2 ** +-8, independently (per-row
    grids far apart; the key scale stays small enough to keep the softmax unsaturated): dQ is the
    non-deterministic dQ bit for bit (both are above the dQ floor on these inputs), dK/dV within the floors, and
    the 99th percentile of the per-row relative error of the FP32 dK/dV within that of the BF16
    floor."""
    program = cake_jit.NOPE_PROGRAM if nope else cake_jit.PROGRAM
    _require_program(backward=True, program=program)
    inp = make_inputs([512], [1024], seed=SEED + 25, topk=256)
    gen = torch.Generator(device="cuda").manual_seed(SEED + 25)
    S, T = inp.total_k, inp.total_q
    k_scale = torch.exp2(
        torch.randint(-3, 4, (S, 1), device="cuda", generator=gen).float()
    )
    t_scale = torch.exp2(
        torch.randint(-8, 9, (T, 1, 1), device="cuda", generator=gen).float()
    )
    kv = (inp.kv_latent.float() * k_scale).bfloat16()
    dout = (inp.dout.float() * t_scale).bfloat16()
    tl = inp.topk_length
    if nope:
        args = (inp.q_latent, kv, inp.idx_global)
        out, lse, o_lo = cake_backend.forward_nope(*args, topk_length=tl)
        dq, dkv = cake_backend.backward_nope(
            *args, out, o_lo, lse, dout, topk_length=tl
        )
        n = nondet_backend.forward_nope(*args, topk_length=tl)
        nondet = nondet_backend.backward_nope(
            *args, n[0], n[2], n[1], dout, topk_length=tl
        )
        ref = reference_fp64_nope(*args, dout=dout, topk_length=tl)
        _check_nope_accuracy(
            "skewed_nope", dq, dkv, ref, nondet=nondet, dq_bitwise=True
        )
        got_rows, floor_rows = (
            rel_l2_rows(dkv, ref["dkv"]),
            rel_l2_rows(ref["dkv_emu"], ref["dkv"]),
        )
    else:
        k_rope = (inp.k_rope.float() * k_scale).bfloat16()
        args = (inp.q_latent, inp.q_rope, kv, k_rope, inp.idx_global)
        out, lse, o_lo = cake_backend.forward(*args, topk_length=tl)
        grads = cake_backend.backward(
            *args, out, o_lo, lse, dout, topk_length=tl, dkv_fp32=True
        )
        n = nondet_backend.forward(*args, topk_length=tl)
        nondet = nondet_backend.backward(
            *args, n[0], n[2], n[1], dout, topk_length=tl, dkv_fp32=True
        )
        ref = reference_fp64(*args, dout=dout, topk_length=tl)
        _check_rope_accuracy(
            "skewed_rope", grads, ref, canonical=False, nondet=nondet, dq_bitwise=True
        )
        got_rows = rel_l2_rows(grads[2], ref["dkv_latent"])
        floor_rows = rel_l2_rows(ref["dkv_latent_emu"], ref["dkv_latent"])
    touched = (ref["dkv"] if nope else ref["dkv_latent"]).abs().sum(1) > 0
    p99 = got_rows[touched].quantile(0.99).item()
    p99_floor = floor_rows[touched].quantile(0.99).item()
    print("SKEWED", "nope" if nope else "rope", "row p99", p99, "floor p99", p99_floor)
    assert p99 <= FLOOR_MARGIN * p99_floor + FLOOR_ABS


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("mode", ["fresh", "acc"])
@pytest.mark.parametrize("nope", [False, True], ids=["rope", "nope"])
def test_nonfinite_dout_writes_nan_rows(nope, mode, value):
    """One non-finite ``dout`` element of token ``t0``, head ``h0``: every element of every dK/dV row ``t0``
    selects is NaN (fresh output or ``dkv_acc`` alike), ``dq[t0, h0]`` is non-finite, and every other row and
    ``dq`` element keeps the bits of the same call with a finite element there (``t0``'s ``dout`` is scaled by
    ``2 ** -20`` in both calls so its bound terms do not set the maxima); the non-finite call repeats bit for
    bit."""
    program = cake_jit.NOPE_PROGRAM if nope else cake_jit.PROGRAM
    _require_program(backward=True, program=program)
    inp = make_inputs([256], [1024], seed=SEED + 26, topk=256)
    S = inp.total_k
    t0, h0 = 200, 5
    dout = inp.dout.clone()
    dout[t0] = (dout[t0].float() * 2**-20).bfloat16()
    bad = dout.clone()
    bad[t0, h0, 11] = value
    tl = inp.topk_length
    cols = D_LATENT if nope else D_QK

    def run(d):
        kw = {}
        acc = None
        if mode == "acc":
            gen = torch.Generator(device="cuda").manual_seed(SEED)
            acc = torch.randn(S, cols, device="cuda", generator=gen)
            kw["dkv_acc"] = acc
        if nope:
            args = (inp.q_latent, inp.kv_latent, inp.idx_global)
            out, lse, o_lo = cake_backend.forward_nope(*args, topk_length=tl)
            dq, dkv = cake_backend.backward_nope(
                *args, out, o_lo, lse, d, topk_length=tl, **kw
            )
            return [dq], (acc if acc is not None else dkv)
        args = _rope_args(inp)
        out, lse, o_lo = cake_backend.forward(*args, topk_length=tl)
        if acc is None:
            kw["dkv_fp32"] = True
        g = cake_backend.backward(*args, out, o_lo, lse, d, topk_length=tl, **kw)
        return [g[0], g[1]], (acc if acc is not None else torch.cat([g[2], g[3]], 1))

    clean_dq, clean_dkv = run(dout)
    got_dq, got_dkv = run(bad)
    again_dq, again_dkv = run(bad)
    assert _same_bits(got_dkv, again_dkv)
    for a, b in zip(got_dq, again_dq, strict=True):
        assert _same_bits(a, b)
    sel = inp.idx_global[t0, : int(tl[t0])]
    rows = torch.zeros(S, dtype=torch.bool, device="cuda")
    rows[sel[(sel >= 0) & (sel < S)].long()] = True
    assert rows.any() and (~rows).any()
    assert torch.isnan(got_dkv[rows]).all()
    assert _same_bits(got_dkv[~rows], clean_dkv[~rows])
    keep = torch.ones(inp.total_q, NUM_HEADS, dtype=torch.bool, device="cuda")
    keep[t0, h0] = False
    for got, clean in zip(got_dq, clean_dq, strict=True):
        assert not torch.isfinite(got[t0, h0].float()).any()
        assert _same_bits(got[keep], clean[keep])


@pytest.mark.parametrize("form", ["rope", "rope_acc_map", "nope", "nope_acc"])
def test_autograd_step_does_not_synchronize(form):
    """An autograd step (forward + backward) of the entries makes no pageable host-to-device copy, no synchronizing
    call, and returns while device work queued before it still runs (``tests/test_helpers/cake_host_sync.py``)."""
    from tests.test_helpers.cake_host_sync import census, nonblocking, summary

    nope = form.startswith("nope")
    _require_program(
        backward=True, program=cake_jit.NOPE_PROGRAM if nope else cake_jit.PROGRAM
    )
    inp = make_inputs([256], [1024], seed=SEED + 27, topk=256)
    S = inp.total_k
    kw = {}
    if form == "rope_acc_map":
        kw["dkv_dst_map"] = (
            torch.arange(S, device="cuda", dtype=torch.int32) % (S // 2)
        ).contiguous()
        kw["dkv_acc"] = torch.zeros(S // 2, D_QK, device="cuda")
    elif form == "nope_acc":
        kw["dkv_acc"] = torch.zeros(S, D_LATENT, device="cuda")
    if nope:
        leaves = [inp.q_latent.detach().clone().requires_grad_(True)]
        kv = (
            inp.kv_latent
            if "dkv_acc" in kw
            else inp.kv_latent.detach().clone().requires_grad_(True)
        )

        def step():
            out = cake_backend.cake_dsa_sparse_attention_det_nope(
                leaves[0], kv, inp.idx_global, topk_length=inp.topk_length, **kw
            )
            out.backward(inp.dout)
    else:
        leaves = [
            t.detach().clone().requires_grad_(True) for t in (inp.q_latent, inp.q_rope)
        ]
        kv_parts = (inp.kv_latent, inp.k_rope)
        if "dkv_acc" not in kw:
            kv_parts = tuple(t.detach().clone().requires_grad_(True) for t in kv_parts)

        def step():
            out = cake_backend.cake_dsa_sparse_attention_det(
                *leaves, *kv_parts, inp.idx_global, topk_length=inp.topk_length, **kw
            )
            out.backward(inp.dout)

    step()
    torch.cuda.synchronize()
    c = census(step, torch.device("cuda"))
    assert nonblocking(c), summary(c)


def test_runner_step_does_not_synchronize_or_allocate():
    from tests.test_helpers.cake_host_sync import census, nonblocking, summary

    _require_program(backward=True)
    inp = make_inputs([256], [1024], seed=SEED + 28, topk=256)
    runner = cake_backend.prepare_dsa_train(
        *_rope_args(inp), topk_length=inp.topk_length, dout=inp.dout, o_lo_scratch=True
    )
    runner.step()
    torch.cuda.synchronize()
    c = census(runner.step, torch.device("cuda"))
    assert nonblocking(c), summary(c)
    assert c["device_alloc"] == 0, summary(c)
