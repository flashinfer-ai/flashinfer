"""Tests for the MoE routing prologue and weighted-sum finalize.

The tests are grouped by what they protect:

* allowlist well-formedness -- the measured bound is asserted so re-widening
  has to be deliberate;
* the dispatch guards -- what must NOT reach the specialized kernels, for each
  half separately;
* numerics -- specialized vs the composable path for each half and for the two
  chained together, including a forced skewed expert load that ordinary random
  routing never produces;
* the token-count ceiling -- the descriptor pass keeps every assignment in one
  CTA, so there is a hard ceiling, and it must be REPORTED rather than silently
  producing nothing;
* what the entry points refuse to trust -- an expert id the caller supplied is
  not a shared-memory index until it has been checked, and an operand on
  another device is not this launch's memory.  Both go to the module directly,
  since the dispatch guards are what keeps them away from it in normal use;
* the build latch -- the JIT is this op's only build path, so a build that
  fails is attempted once per process rather than by every later dispatch;
* CUDA graphs -- capture after precompile records the specialized kernels, cold
  capture falls back cleanly;
* launch independence -- the split has no persistent device state and no
  inter-CTA rendezvous, so a launch's result cannot depend on any earlier
  launch.  That is asserted directly, by interleaving problem sizes and
  comparing against the same calls made in isolation;
* packaging -- the code lives under ``flashinfer/fused_moe/experimental/`` but
  is called as ``flashinfer.<name>``, importing it costs nothing, the kernel
  source ships next to the op, and there is no AOT entry.  These run without a
  GPU.
"""

import json
import pathlib
from importlib import resources

import pytest
import torch

from flashinfer.fused_moe.experimental import moe_routing as mr

HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 8
BLOCK_M = mr.BLOCK_SIZE_M
MAX_TOKENS = mr._MAX_TOKENS

# The measured win surface actually shipped.  Widening this is a measurement
# decision, so the test states the bound instead of deriving it.
SHIPPED_MAX_M = 4


def _sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


requires_sm120 = pytest.mark.skipif(
    not _sm120(), reason="the specialized MoE routing kernels are SM120 only"
)


def _inputs(m, seed=0, device="cuda", hidden=HIDDEN, experts=NUM_EXPERTS, top_k=TOP_K):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def randn(*shape, std=1.0):
        return (
            torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * std
        ).to(torch.bfloat16)

    return dict(
        hidden_states=randn(m, hidden),
        gate_weight=randn(experts, hidden, std=0.04),
        shared_gate_weight=randn(1, hidden, std=0.04),
        expert_out=randn(m, top_k, hidden),
        shared_out=randn(m, hidden, std=0.25),
    )


def _prologue_outputs(m, device="cuda", experts=NUM_EXPERTS, top_k=TOP_K):
    return dict(
        topk_weights=torch.zeros(m, top_k, dtype=torch.float32, device=device),
        topk_ids=torch.zeros(m, top_k, dtype=torch.int32, device=device),
        sorted_token_ids=torch.zeros(64 * m, dtype=torch.int32, device=device),
        expert_ids=torch.zeros(BLOCK_M * m, dtype=torch.int32, device=device),
        num_tokens_post_pad=torch.zeros(1, dtype=torch.int32, device=device),
        shared_gate=torch.zeros(m, dtype=torch.bfloat16, device=device),
        router_logits=torch.zeros(m, experts, dtype=torch.bfloat16, device=device),
    )


def _call_prologue(ins, outs):
    return mr.moe_routing_prologue(
        ins["hidden_states"],
        ins["gate_weight"],
        ins["shared_gate_weight"],
        **outs,
    )


def _composable_prologue(ins, m, experts=NUM_EXPERTS, top_k=TOP_K):
    outs = _prologue_outputs(m, experts=experts, top_k=top_k)
    mr._reference_prologue(
        ins["hidden_states"],
        ins["gate_weight"],
        ins["shared_gate_weight"],
        outs["router_logits"],
        outs["shared_gate"],
        outs["topk_weights"],
        outs["topk_ids"],
        outs["sorted_token_ids"],
        outs["expert_ids"],
        outs["num_tokens_post_pad"],
        BLOCK_M,
    )
    return outs


def _composable_finalize(ins, pro, m, hidden=HIDDEN):
    out = torch.zeros(m, hidden, dtype=torch.bfloat16, device="cuda")
    mr._reference_finalize(
        ins["expert_out"],
        ins["shared_out"],
        pro["topk_weights"],
        pro["shared_gate"],
        out,
    )
    return out


def _per_expert_assignments(outs, m):
    """Expert -> multiset of assignment indices, ignoring the block split.

    The consumer processes block ``b`` for ``expert_ids[b]`` over the entries in
    ``sorted_token_ids[8b:8b+8]``, so any partition of one expert's assignments
    across that expert's own blocks is the same work.  Intra-expert order is
    resolved by an atomicAdd and is deliberately not constrained.
    """
    total = int(outs["num_tokens_post_pad"][0].item())
    numel = m * TOP_K
    sti = outs["sorted_token_ids"].tolist()
    eid = outs["expert_ids"].tolist()
    got = {}
    for b, e in enumerate(eid):
        if b * BLOCK_M >= total or e < 0:
            continue
        got.setdefault(e, []).extend(
            v for v in sti[b * BLOCK_M : (b + 1) * BLOCK_M] if v != numel
        )
    return {e: sorted(v) for e, v in got.items()}


def _assert_prologue_matches(actual, expected, m):
    torch.testing.assert_close(
        actual["topk_weights"], expected["topk_weights"], rtol=1e-4, atol=1e-6
    )
    assert torch.equal(actual["topk_ids"], expected["topk_ids"])
    assert torch.equal(actual["expert_ids"], expected["expert_ids"])
    assert torch.equal(actual["num_tokens_post_pad"], expected["num_tokens_post_pad"])
    assert _per_expert_assignments(actual, m) == _per_expert_assignments(expected, m)
    torch.testing.assert_close(
        actual["shared_gate"].to(torch.float32),
        expected["shared_gate"].to(torch.float32),
        rtol=8e-3,
        atol=1e-3,
    )


def _assert_close_bf16(a, b, tol=4e-3):
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    assert torch.isfinite(a).all()
    scale = float(b.abs().max().item()) or 1.0
    assert float((a - b).abs().max().item()) / scale <= tol
    cosine = torch.nn.functional.cosine_similarity(
        a.reshape(1, -1), b.reshape(1, -1), dim=1
    ).item()
    assert cosine >= 0.9999


# --------------------------------------------------------------- allowlist
def test_allowlist_is_well_formed():
    payload = json.loads(
        resources.files("flashinfer.fused_moe.experimental")
        .joinpath("moe_routing_sm120_workloads.json")
        .read_text()
    )
    assert tuple(payload["fields"]) == ("m", "hidden_size", "num_experts", "top_k")
    rows = [tuple(int(v) for v in row) for row in payload["workloads"]]
    assert rows, "the allowlist must not be empty"
    assert len(set(rows)) == len(rows), "duplicate allowlist rows"
    for m, hidden, experts, top_k in rows:
        # the prologue keeps every assignment in one CTA
        assert 1 <= m <= MAX_TOKENS
        # the kernels are written for exactly one MoE geometry
        assert (hidden, experts, top_k) == (HIDDEN, NUM_EXPERTS, TOP_K)
        # the measured bound: widening past it is a measurement decision
        assert m <= SHIPPED_MAX_M
    assert frozenset(rows) == mr.load_moe_routing_sm120_workloads()


# ------------------------------------------------------------------- guard
@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
def test_supported_matches_the_allowlist(m):
    assert mr.moe_routing_supported(m, HIDDEN, NUM_EXPERTS, TOP_K)


@requires_sm120
@pytest.mark.parametrize(
    "args",
    [
        (3, HIDDEN, NUM_EXPERTS, TOP_K),  # inside the ceiling, not measured
        (24, HIDDEN, NUM_EXPERTS, TOP_K),  # a vLLM capture size, not shipped
        (8, HIDDEN, NUM_EXPERTS, TOP_K),  # servable but not measured
        (64, HIDDEN, NUM_EXPERTS, TOP_K),  # past the prologue ceiling
        (1, 4096, NUM_EXPERTS, TOP_K),  # other geometry
        (1, HIDDEN, 128, TOP_K),
        (1, HIDDEN, NUM_EXPERTS, 4),
    ],
)
def test_supported_declines_off_surface(args):
    assert not mr.moe_routing_supported(*args)


@requires_sm120
def test_supported_declines_non_bf16():
    assert not mr.moe_routing_supported(
        1, HIDDEN, NUM_EXPERTS, TOP_K, dtype=torch.float16
    )


@requires_sm120
def test_supported_declines_other_block_size_m():
    assert not mr.moe_routing_supported(1, HIDDEN, NUM_EXPERTS, TOP_K, block_size_m=16)


@requires_sm120
def test_kill_switch_takes_the_composable_path(monkeypatch):
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    assert not mr.moe_routing_supported(1, HIDDEN, NUM_EXPERTS, TOP_K)
    ins = _inputs(1, seed=7)
    outs = _prologue_outputs(1)
    before = mr.moe_routing_stats()
    _call_prologue(ins, outs)
    out = mr.moe_routing_finalize(
        ins["expert_out"], ins["shared_out"], outs["topk_weights"], outs["shared_gate"]
    )
    after = mr.moe_routing_stats()
    assert after["prologue_launch_count"] == before["prologue_launch_count"]
    assert after["finalize_launch_count"] == before["finalize_launch_count"]
    expected = _composable_prologue(ins, 1)
    _assert_prologue_matches(outs, expected, 1)
    _assert_close_bf16(out, _composable_finalize(ins, expected, 1))


@requires_sm120
def test_non_contiguous_input_takes_the_composable_path():
    ins = _inputs(4, seed=11)
    padded = torch.zeros(4, HIDDEN + 8, dtype=torch.bfloat16, device="cuda")
    padded[:, :HIDDEN] = ins["hidden_states"]
    ins_nc = dict(ins, hidden_states=padded[:, :HIDDEN])
    assert not ins_nc["hidden_states"].is_contiguous()
    outs = _prologue_outputs(4)
    before = mr.moe_routing_stats()["prologue_launch_count"]
    _call_prologue(ins_nc, outs)
    assert mr.moe_routing_stats()["prologue_launch_count"] == before


@requires_sm120
def test_finalize_declines_a_non_contiguous_expert_out():
    ins = _inputs(2, seed=13)
    outs = _prologue_outputs(2)
    _call_prologue(ins, outs)
    padded = torch.zeros(2, TOP_K, HIDDEN + 8, dtype=torch.bfloat16, device="cuda")
    padded[:, :, :HIDDEN] = ins["expert_out"]
    eo = padded[:, :, :HIDDEN]
    assert not eo.is_contiguous()
    before = mr.moe_routing_stats()["finalize_launch_count"]
    out = mr.moe_routing_finalize(
        eo, ins["shared_out"], outs["topk_weights"], outs["shared_gate"]
    )
    assert mr.moe_routing_stats()["finalize_launch_count"] == before
    _assert_close_bf16(out, _composable_finalize(ins, outs, 2))


# ---------------------------------------------------------------- numerics
@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prologue_matches_composable(m, seed):
    ins = _inputs(m, seed=seed)
    outs = _prologue_outputs(m)
    before = mr.moe_routing_stats()["prologue_launch_count"]
    _call_prologue(ins, outs)
    assert mr.moe_routing_stats()["prologue_launch_count"] == before + 1, (
        "the specialized prologue did not dispatch for an allowlisted size"
    )
    torch.cuda.synchronize()
    _assert_prologue_matches(outs, _composable_prologue(ins, m), m)


@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_finalize_matches_composable(m, seed):
    ins = _inputs(m, seed=seed)
    ref = _composable_prologue(ins, m)
    before = mr.moe_routing_stats()["finalize_launch_count"]
    out = mr.moe_routing_finalize(
        ins["expert_out"], ins["shared_out"], ref["topk_weights"], ref["shared_gate"]
    )
    assert mr.moe_routing_stats()["finalize_launch_count"] == before + 1, (
        "the specialized finalize did not dispatch for an allowlisted size"
    )
    torch.cuda.synchronize()
    _assert_close_bf16(out, _composable_finalize(ins, ref, m))


@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
def test_chained_halves_match_the_composable_chain(m):
    """The two entry points in their serving order, against the pure fallback.

    ``expert_out`` is materialised independently here, which is exactly what the
    expert GEMMs do in a serving block: they read the prologue's descriptors and
    write this tensor, and the finalize reads it back.
    """
    ins = _inputs(m, seed=31 + m)
    outs = _prologue_outputs(m)
    _call_prologue(ins, outs)
    out = mr.moe_routing_finalize(
        ins["expert_out"], ins["shared_out"], outs["topk_weights"], outs["shared_gate"]
    )
    torch.cuda.synchronize()
    ref = _composable_prologue(ins, m)
    _assert_prologue_matches(outs, ref, m)
    _assert_close_bf16(out, _composable_finalize(ins, ref, m))


@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
def test_finalize_without_the_shared_expert(m):
    """The shape a consumer that combines the shared expert elsewhere calls.

    vLLM adds the shared expert outside the routed-expert kernel, so its route
    passes neither shared operand; the result must be exactly the routed
    weighted sum, and the specialized kernel must still dispatch.
    """
    ins = _inputs(m, seed=51 + m)
    ref = _composable_prologue(ins, m)
    before = mr.moe_routing_stats()["finalize_launch_count"]
    out = mr.moe_routing_finalize(ins["expert_out"], None, ref["topk_weights"], None)
    assert mr.moe_routing_stats()["finalize_launch_count"] == before + 1
    torch.cuda.synchronize()
    want = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    mr._reference_finalize(ins["expert_out"], None, ref["topk_weights"], None, want)
    _assert_close_bf16(out, want)
    # and it really is the shared-expert term that is missing
    withshared = mr.moe_routing_finalize(
        ins["expert_out"], ins["shared_out"], ref["topk_weights"], ref["shared_gate"]
    )
    assert not torch.equal(out, withshared)


@requires_sm120
def test_finalize_rejects_half_a_shared_expert():
    """Half the pair would silently drop the gate or read an unset buffer."""
    ins = _inputs(2, seed=53)
    ref = _composable_prologue(ins, 2)
    with pytest.raises(ValueError):
        mr.moe_routing_finalize(
            ins["expert_out"], ins["shared_out"], ref["topk_weights"], None
        )
    with pytest.raises(ValueError):
        mr.moe_routing_finalize(
            ins["expert_out"], None, ref["topk_weights"], ref["shared_gate"]
        )


@requires_sm120
def test_finalize_owns_the_routing_weights():
    """The G0 hazard, asserted rather than only documented.

    The finalize applies ``topk_weights``; a caller whose expert GEMM also
    applies them gets them twice.  If this op ever stopped applying them the
    double-application would become invisible, so the dependence is a test.
    """
    m = 2
    ins = _inputs(m, seed=41)
    ref = _composable_prologue(ins, m)
    base = mr.moe_routing_finalize(
        ins["expert_out"], ins["shared_out"], ref["topk_weights"], ref["shared_gate"]
    ).clone()
    doubled = mr.moe_routing_finalize(
        ins["expert_out"],
        ins["shared_out"],
        ref["topk_weights"] * 2.0,
        ref["shared_gate"],
    )
    torch.cuda.synchronize()
    # Only the routed part scales; the gated shared expert does not, so the
    # result is neither equal nor a clean 2x -- it just has to move.
    assert not torch.equal(base, doubled)
    routed = (base.to(torch.float32) - doubled.to(torch.float32)).abs().max().item()
    assert routed > 0.0


@requires_sm120
@pytest.mark.parametrize("m", [3, 8, 24, 40])
def test_off_allowlist_is_correct_and_never_dispatches(m):
    """An uncovered size must take the composable path and still be right."""
    ins = _inputs(m, seed=5)
    outs = _prologue_outputs(m)
    for t in outs.values():
        t.fill_(-1)
    before = mr.moe_routing_stats()
    _call_prologue(ins, outs)
    out = mr.moe_routing_finalize(
        ins["expert_out"], ins["shared_out"], outs["topk_weights"], outs["shared_gate"]
    )
    torch.cuda.synchronize()
    after = mr.moe_routing_stats()
    assert after["prologue_launch_count"] == before["prologue_launch_count"]
    assert after["finalize_launch_count"] == before["finalize_launch_count"]
    ref = _composable_prologue(ins, m)
    _assert_prologue_matches(outs, ref, m)
    _assert_close_bf16(out, _composable_finalize(ins, ref, m))


@requires_sm120
def test_prologue_reports_a_token_count_it_cannot_serve():
    """Past the ceiling the C++ entry point RAISES; it never no-ops.

    The dispatch guard already keeps such a call away from the kernel, so this
    goes to the module directly: a launcher that quietly did nothing here would
    leave the caller's buffers untouched, which is a wrong answer with no error.
    """
    assert mr.moe_routing_precompile()
    m = MAX_TOKENS + 1
    ins = _inputs(m, seed=17)
    outs = _prologue_outputs(m)
    # B017: blind on purpose. The ceiling check lives in the C++ entry point
    # and surfaces through tvm_ffi, whose error class is not a stable public
    # type to import here -- what this test pins is that it RAISES rather than
    # quietly no-ops, not which exception it picks.
    with pytest.raises(Exception):  # noqa: B017
        mr._MODULE.moe_routing_prologue_sm120(
            ins["hidden_states"],
            ins["gate_weight"],
            ins["shared_gate_weight"],
            outs["router_logits"],
            outs["shared_gate"],
            outs["topk_weights"],
            outs["topk_ids"],
            outs["sorted_token_ids"],
            outs["expert_ids"],
            outs["num_tokens_post_pad"],
        )


@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
def test_skewed_expert_load(m):
    """Every token forced onto the same experts: per-expert load = m.

    Random routing on this geometry produces a maximum per-expert load of a
    handful, so the descriptor's counting/scan/scatter path is otherwise never
    exercised at its bound.
    """
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(99 + m)
    hidden = torch.full((m, HIDDEN), 0.5, device=device, dtype=torch.float32)
    hidden += (
        torch.randn(m, HIDDEN, generator=g, device=device, dtype=torch.float32) * 1e-3
    )
    gate = torch.zeros(NUM_EXPERTS, HIDDEN, device=device, dtype=torch.float32)
    for e in range(TOP_K):
        gate[e] = 0.01 * (TOP_K - e)
    ins = dict(
        hidden_states=hidden.to(torch.bfloat16),
        gate_weight=gate.to(torch.bfloat16),
        shared_gate_weight=(
            torch.randn(1, HIDDEN, generator=g, device=device, dtype=torch.float32)
            * 0.04
        ).to(torch.bfloat16),
        expert_out=torch.randn(
            m, TOP_K, HIDDEN, generator=g, device=device, dtype=torch.float32
        ).to(torch.bfloat16),
        shared_out=(
            torch.randn(m, HIDDEN, generator=g, device=device, dtype=torch.float32)
            * 0.25
        ).to(torch.bfloat16),
    )
    outs = _prologue_outputs(m)
    _call_prologue(ins, outs)
    out = mr.moe_routing_finalize(
        ins["expert_out"], ins["shared_out"], outs["topk_weights"], outs["shared_gate"]
    )
    torch.cuda.synchronize()
    loads = torch.bincount(
        outs["topk_ids"].reshape(-1).to(torch.int64), minlength=NUM_EXPERTS
    )
    assert int(loads.max().item()) == m
    assert int((loads > 0).sum().item()) == TOP_K
    ref = _composable_prologue(ins, m)
    _assert_prologue_matches(outs, ref, m)
    _assert_close_bf16(out, _composable_finalize(ins, ref, m))


# ------------------------------------------------------------------- align
def _align_outputs(m, device="cuda"):
    return dict(
        sorted_token_ids=torch.zeros(64 * m, dtype=torch.int32, device=device),
        expert_ids=torch.zeros(BLOCK_M * m, dtype=torch.int32, device=device),
        num_tokens_post_pad=torch.zeros(1, dtype=torch.int32, device=device),
    )


def _composable_align(topk_ids, m):
    outs = _align_outputs(m)
    mr._reference_descriptors(
        topk_ids,
        NUM_EXPERTS,
        BLOCK_M,
        outs["sorted_token_ids"],
        outs["expert_ids"],
        outs["num_tokens_post_pad"],
    )
    return outs


def _assert_descriptors_match(actual, expected, m):
    """expert_ids and the padded total must be exact; sorted_token_ids only as
    per-expert assignment sets -- intra-expert rank is an atomicAdd above
    BLOCK_M tokens per expert, and the consumer reads a block's entries as a
    set."""
    assert torch.equal(actual["expert_ids"], expected["expert_ids"])
    assert torch.equal(actual["num_tokens_post_pad"], expected["num_tokens_post_pad"])
    assert _per_expert_assignments(actual, m) == _per_expert_assignments(expected, m)


@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_align_matches_composable(m, seed):
    ins = _inputs(m, seed=seed)
    ref = _composable_prologue(ins, m)
    outs = _align_outputs(m)
    before = mr.moe_routing_stats()["align_launch_count"]
    mr.moe_routing_align(ref["topk_ids"], NUM_EXPERTS, **outs)
    assert mr.moe_routing_stats()["align_launch_count"] == before + 1, (
        "the specialized align did not dispatch for an allowlisted size"
    )
    torch.cuda.synchronize()
    _assert_descriptors_match(outs, _composable_align(ref["topk_ids"], m), m)


@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
def test_align_reproduces_the_prologues_own_descriptors(m):
    """The two entry points share one kernel, so they must agree exactly.

    This is the property that makes moe_routing_align safe to drop in beside a
    caller that uses the prologue elsewhere: a descriptor that disagreed with
    the one the expert GEMM was built against would be a silent wrong answer.
    At the shipped sizes the descriptor path has no atomics, so equality here is
    byte-for-byte.
    """
    ins = _inputs(m, seed=61 + m)
    pro = _prologue_outputs(m)
    _call_prologue(ins, pro)
    outs = _align_outputs(m)
    mr.moe_routing_align(pro["topk_ids"], NUM_EXPERTS, **outs)
    torch.cuda.synchronize()
    assert torch.equal(outs["expert_ids"], pro["expert_ids"])
    assert torch.equal(outs["num_tokens_post_pad"], pro["num_tokens_post_pad"])
    assert torch.equal(outs["sorted_token_ids"], pro["sorted_token_ids"])


@requires_sm120
@pytest.mark.parametrize("m", [3, 8, 24])
def test_align_off_allowlist_is_correct_and_never_dispatches(m):
    ins = _inputs(m, seed=71)
    ref = _composable_prologue(ins, m)
    outs = _align_outputs(m)
    before = mr.moe_routing_stats()["align_launch_count"]
    mr.moe_routing_align(ref["topk_ids"], NUM_EXPERTS, **outs)
    torch.cuda.synchronize()
    assert mr.moe_routing_stats()["align_launch_count"] == before
    _assert_descriptors_match(outs, _composable_align(ref["topk_ids"], m), m)


@requires_sm120
def test_align_declines_another_block_size_m():
    ins = _inputs(2, seed=73)
    ref = _composable_prologue(ins, 2)
    before = mr.moe_routing_stats()["align_launch_count"]
    mr.moe_routing_align(ref["topk_ids"], NUM_EXPERTS, block_size_m=16)
    assert mr.moe_routing_stats()["align_launch_count"] == before


@requires_sm120
def test_align_kill_switch_takes_the_composable_path(monkeypatch):
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    ins = _inputs(1, seed=75)
    ref = _composable_prologue(ins, 1)
    outs = _align_outputs(1)
    before = mr.moe_routing_stats()["align_launch_count"]
    mr.moe_routing_align(ref["topk_ids"], NUM_EXPERTS, **outs)
    assert mr.moe_routing_stats()["align_launch_count"] == before
    _assert_descriptors_match(outs, _composable_align(ref["topk_ids"], 1), 1)


@requires_sm120
def test_align_reports_a_token_count_it_cannot_serve():
    assert mr.moe_routing_precompile()
    m = MAX_TOKENS + 1
    ins = _inputs(m, seed=77)
    ref = _composable_prologue(ins, m)
    outs = _align_outputs(m)
    # B017: blind on purpose. The ceiling check lives in the C++ entry point
    # and surfaces through tvm_ffi, whose error class is not a stable public
    # type to import here -- what this test pins is that it RAISES rather than
    # quietly no-ops, not which exception it picks.
    with pytest.raises(Exception):  # noqa: B017
        mr._MODULE.moe_routing_align_sm120(
            ref["topk_ids"],
            outs["sorted_token_ids"],
            outs["expert_ids"],
            outs["num_tokens_post_pad"],
            NUM_EXPERTS,
            BLOCK_M,
        )


@requires_sm120
def test_descriptor_drops_out_of_range_expert_ids():
    """A caller-supplied expert id is not trusted as a shared-memory index.

    ``moe_routing_align`` takes ``topk_ids`` from the caller's own router, and
    above 4 tokens the descriptor pass indexes per-expert shared arrays
    (``s_cnt`` / ``s_cur`` / ``s_start``, ``NUM_EXPERTS`` wide) with them.  An
    id outside ``[0, NUM_EXPERTS)`` is dropped -- its slot keeps the padding
    sentinel -- rather than scribbling on shared memory.

    The shipped allowlist is m in {1, 2, 4}, so the guards never send a size
    this large to the kernel and this goes to the module directly.  That is
    also why the check costs the shipped path nothing: at those sizes the
    warp-local branch runs instead, and it indexes nothing by expert id.
    """
    assert mr.moe_routing_precompile()
    m = 8
    assert m > SHIPPED_MAX_M
    ins = _inputs(m, seed=101)
    topk_ids = _composable_prologue(ins, m)["topk_ids"].clone()
    for (row, col), value in {
        (0, 0): -1,
        (1, 3): NUM_EXPERTS,
        (4, 7): NUM_EXPERTS + 11,
        (6, 5): -(1 << 20),
        (7, 2): 1 << 20,
    }.items():
        topk_ids[row, col] = value

    outs = _align_outputs(m)
    mr._MODULE.moe_routing_align_sm120(
        topk_ids,
        outs["sorted_token_ids"],
        outs["expert_ids"],
        outs["num_tokens_post_pad"],
        NUM_EXPERTS,
        BLOCK_M,
    )
    torch.cuda.synchronize()

    flat = topk_ids.reshape(-1).tolist()
    kept = {}
    for index, expert in enumerate(flat):
        if 0 <= expert < NUM_EXPERTS:
            kept.setdefault(expert, []).append(index)
    kept = {expert: sorted(v) for expert, v in kept.items()}

    # The valid assignments are described exactly as if the invalid ones had
    # never been passed...
    assert _per_expert_assignments(outs, m) == kept
    blocks = sum((len(v) + BLOCK_M - 1) // BLOCK_M for v in kept.values())
    assert int(outs["num_tokens_post_pad"][0].item()) == blocks * BLOCK_M
    # ... and the dropped ones appear nowhere in the descriptor.
    dropped = {i for i, e in enumerate(flat) if not (0 <= e < NUM_EXPERTS)}
    assert len(dropped) == 5
    assert dropped.isdisjoint(set(outs["sorted_token_ids"].tolist()))


@requires_sm120
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="an operand on another device needs two"
)
def test_prologue_rejects_an_operand_on_another_device():
    """Outputs are checked against the launch's device too, not just inputs.

    ``CHECK_INPUT_AND_TYPE`` only says "some CUDA tensor".  An output allocated
    on a different device would be written through a pointer that does not
    belong to the context this launch guards to.
    """
    assert mr.moe_routing_precompile()
    m = 1
    ins = _inputs(m, seed=103)
    outs = _prologue_outputs(m)
    outs["topk_ids"] = outs["topk_ids"].to("cuda:1")
    # B017: blind on purpose, as above -- tvm_ffi's error class is not a stable
    # public type to import here.
    with pytest.raises(Exception):  # noqa: B017
        mr._MODULE.moe_routing_prologue_sm120(
            ins["hidden_states"],
            ins["gate_weight"],
            ins["shared_gate_weight"],
            outs["router_logits"],
            outs["shared_gate"],
            outs["topk_weights"],
            outs["topk_ids"],
            outs["sorted_token_ids"],
            outs["expert_ids"],
            outs["num_tokens_post_pad"],
        )


def test_a_failed_build_is_attempted_once_per_process(monkeypatch):
    """A build that fails must not be retried by every later dispatch.

    The JIT is this op's only build path, so a failure here is the difference
    between a slow first call and a file lock plus a ``ninja`` invocation per
    MoE layer per decode step.  The reason a build fails does not resolve
    itself mid-process, so the answer is latched and later dispatches take the
    composable path directly.
    """
    attempts = []

    class _FailingSpec:
        def build_and_load(self):
            attempts.append(1)
            raise RuntimeError("nvcc will not be there on the next call either")

    monkeypatch.setattr(mr, "_sm120_module_generator", lambda: _FailingSpec)
    monkeypatch.setattr(mr, "_MODULE", None)
    monkeypatch.setattr(mr, "_MODULE_BUILD_FAILED", False)

    for _ in range(5):
        assert mr.moe_routing_precompile() is False
    assert len(attempts) == 1
    assert mr.moe_routing_ready_for_graph_capture() is False

    if torch.cuda.is_available():
        # ... and the dispatch path, which is what a serving engine actually
        # calls, goes through the same latch.
        for _ in range(5):
            assert mr._dispatch_ready() is False
        assert len(attempts) == 1


# ------------------------------------------------------------- CUDA graphs
@requires_sm120
@pytest.mark.parametrize("m", [1, 2, 4])
def test_capture_after_precompile_records_the_specialized_kernels(m):
    assert mr.moe_routing_precompile()
    assert mr.moe_routing_ready_for_graph_capture()
    ins = _inputs(m, seed=3)
    outs = _prologue_outputs(m)
    out = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    _call_prologue(ins, outs)
    mr.moe_routing_finalize(
        ins["expert_out"],
        ins["shared_out"],
        outs["topk_weights"],
        outs["shared_gate"],
        output=out,
    )
    torch.cuda.synchronize()

    before = mr.moe_routing_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_prologue(ins, outs)
        mr.moe_routing_finalize(
            ins["expert_out"],
            ins["shared_out"],
            outs["topk_weights"],
            outs["shared_gate"],
            output=out,
        )
    after = mr.moe_routing_stats()
    assert after["prologue_launch_count"] == before["prologue_launch_count"] + 1, (
        "capture recorded the composable prologue, not the kernel"
    )
    assert after["finalize_launch_count"] == before["finalize_launch_count"] + 1, (
        "capture recorded the composable finalize, not the kernel"
    )

    fresh = _inputs(m, seed=4)
    for key, value in fresh.items():
        ins[key].copy_(value)
    graph.replay()
    torch.cuda.synchronize()
    ref = _composable_prologue(ins, m)
    _assert_prologue_matches(outs, ref, m)
    _assert_close_bf16(out, _composable_finalize(ins, ref, m))


@requires_sm120
def test_cold_capture_falls_back_cleanly(monkeypatch):
    """Capture with no compiled module must not compile; it must fall back."""
    monkeypatch.setattr(mr, "_MODULE", None)
    monkeypatch.setattr(mr, "moe_routing_precompile", lambda: False)
    ins = _inputs(1, seed=8)
    outs = _prologue_outputs(1)
    out = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="cuda")
    before = mr.moe_routing_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_prologue(ins, outs)
        mr.moe_routing_finalize(
            ins["expert_out"],
            ins["shared_out"],
            outs["topk_weights"],
            outs["shared_gate"],
            output=out,
        )
    after = mr.moe_routing_stats()
    assert after["prologue_launch_count"] == before["prologue_launch_count"]
    assert after["finalize_launch_count"] == before["finalize_launch_count"]
    graph.replay()
    torch.cuda.synchronize()
    ref = _composable_prologue(ins, 1)
    _assert_prologue_matches(outs, ref, 1)
    _assert_close_bf16(out, _composable_finalize(ins, ref, 1))


# ------------------------------------------------------- launch independence
@requires_sm120
def test_interleaved_sizes_are_bit_identical_to_isolated_calls():
    """No cross-launch state: replaces the fused op's publication-tag digest.

    The fused kernel kept per-problem-size generation tags in persistent
    ``__device__`` memory, and their producer/consumer lockstep had to be
    asserted.  The split has no such state at all, so the stronger and simpler
    statement is testable directly: the same call, made after an adversarial
    interleaving of other sizes and after many repetitions, is bit-identical to
    the call made in isolation.
    """
    assert mr.moe_routing_precompile()
    sizes = [1, 2, 4]
    state = {m: (_inputs(m, seed=20 + m), _prologue_outputs(m)) for m in sizes}
    isolated = {}
    for m in sizes:
        ins, outs = state[m]
        _call_prologue(ins, outs)
        out = mr.moe_routing_finalize(
            ins["expert_out"],
            ins["shared_out"],
            outs["topk_weights"],
            outs["shared_gate"],
        )
        torch.cuda.synchronize()
        isolated[m] = (
            {k: v.clone() for k, v in outs.items()},
            out.clone(),
        )

    # 300 launches at m=1 (past any 8-bit wrap a tagged design would have) with
    # the other sizes touched rarely, then all three round-robin.
    schedule = [1] * 300 + [2] + [1] * 300 + [4] + [1, 2, 4] * 30
    for m in schedule:
        ins, outs = state[m]
        _call_prologue(ins, outs)
        mr.moe_routing_finalize(
            ins["expert_out"],
            ins["shared_out"],
            outs["topk_weights"],
            outs["shared_gate"],
        )
    torch.cuda.synchronize()

    for m in sizes:
        ins, outs = state[m]
        _call_prologue(ins, outs)
        out = mr.moe_routing_finalize(
            ins["expert_out"],
            ins["shared_out"],
            outs["topk_weights"],
            outs["shared_gate"],
        )
        torch.cuda.synchronize()
        ref_outs, ref_out = isolated[m]
        for key, value in ref_outs.items():
            # every shipped size takes the warp-local descriptor path, which
            # has no atomics, so even sorted_token_ids is bit-reproducible
            assert torch.equal(outs[key], value), f"{key} drifted at m={m}"
        assert torch.equal(out, ref_out), f"output drifted at m={m}"


# ------------------------------------------------------------------ stats
@requires_sm120
def test_stats_report_a_single_compiled_variant_and_no_persistent_state():
    assert mr.moe_routing_precompile()
    stats = mr.moe_routing_stats()
    assert stats["available"]
    assert stats["entry_points"] == 3
    assert stats["distinct_kernels_for_allowlist"] == 3
    assert stats["compiled_variants"] == 1
    assert stats["persistent_device_state_bytes"] == 0
    assert stats["ready_for_graph_capture"]
    assert stats["allowlist_rows"] == len(mr.load_moe_routing_sm120_workloads())
    assert stats["launch_count"] == (
        stats["prologue_launch_count"]
        + stats["align_launch_count"]
        + stats["finalize_launch_count"]
    )
    assert stats["has_align_entry_point"]
    assert stats["finalize_optional_shared_expert"]


def test_import_does_not_require_a_gpu():
    """The dispatch module must import and answer `supported` off-device."""
    assert isinstance(mr.moe_routing_stats(), dict)
    if not torch.cuda.is_available():
        assert not mr.moe_routing_supported(1, HIDDEN, NUM_EXPERTS, TOP_K)


# ------------------------------------------------------------------ packaging
# The op's code lives under flashinfer/fused_moe/experimental/, but that is a
# file location, not an import path.  These pin the consumer-facing half of
# that arrangement, off-device, because it is exactly the kind of property a
# later "tidy-up" breaks without failing anything: a consumer that resolves
# nothing keeps every gate green and silently runs the stock path.
def test_public_names_are_top_level():
    """One spelling: ``flashinfer.<name>``.

    A consumer probes for this op by reading an ATTRIBUTE off ``flashinfer``
    (``getattr(flashinfer, "moe_routing_finalize", None)``), so the top-level
    names must be the very functions this module defines -- and the older
    ``flashinfer.fused_moe.<name>`` alias must be gone rather than lingering as
    a second spelling of one API.
    """
    import flashinfer
    import flashinfer.fused_moe

    for name in (
        "moe_routing_prologue",
        "moe_routing_align",
        "moe_routing_finalize",
        "moe_routing_supported",
        "moe_routing_precompile",
        "moe_routing_ready_for_graph_capture",
        "moe_routing_stats",
    ):
        assert getattr(flashinfer, name) is getattr(mr, name), name
        assert not hasattr(flashinfer.fused_moe, name), name


def test_importing_flashinfer_does_not_import_a_kernel():
    """The capability check a consumer runs must cost nothing.

    ``getattr(flashinfer, "moe_routing_finalize", None)`` has to answer without
    importing the JIT toolchain or touching the kernel -- otherwise a
    consumer's fail-closed probe pays for machinery it may never use, on every
    install.  Pinned here because the lazy imports that make it true are easy
    to "simplify" away.
    """
    import os
    import subprocess
    import sys

    # No trailing dot on the kernel-package prefix: the package itself is as
    # much an eager import as any module under it -- importing it runs the
    # package __init__ -- and a trailing dot would only match the submodules.
    probe = (
        "import sys, flashinfer;"
        "assert callable(flashinfer.moe_routing_prologue);"
        "assert callable(flashinfer.moe_routing_align);"
        "assert callable(flashinfer.moe_routing_finalize);"
        "assert callable(flashinfer.moe_routing_supported);"
        "eager = [m for m in sys.modules"
        " if 'fused_moe.experimental.kernel' in m"
        " or 'jit.moe_routing' in m];"
        "print(eager)"
    )
    # Hand the child this interpreter's search path so it imports the same
    # flashinfer this test did (source checkout or installed wheel alike).
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_kernel_source_is_package_data_next_to_the_op():
    """The JIT spec reads the .cu from inside the package, and it is there.

    The source moved out of csrc/ into the op's own ``kernel/`` directory, so
    the path the spec builds is the thing that would break -- and it would
    break at first *dispatch*, as a caught exception and a silent fallback to
    the composable path, not at import.
    """
    from flashinfer.jit import moe_routing as jit_moe_routing

    source = jit_moe_routing._MOE_ROUTING_KERNEL_DIR / "moe_routing_sm120.cu"
    assert source.is_file(), source
    # ... and it is inside the installed package, not the repo's csrc/.
    package_root = resources.files("flashinfer.fused_moe.experimental.kernel")
    assert source.parent.resolve() == pathlib.Path(str(package_root)).resolve()


def test_the_op_is_not_in_the_aot_build():
    """No AOT entry: the module JIT-compiles on first non-capturing dispatch.

    Asserted against aot.py's source text rather than by importing it, so this
    runs anywhere.  If the trade is ever revisited, this test is the place that
    says so out loud.
    """
    import importlib.util

    origin = importlib.util.find_spec("flashinfer.aot").origin
    assert origin is not None
    assert "gen_moe_routing_sm120_module" not in pathlib.Path(origin).read_text()
