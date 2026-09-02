"""Targeted trace coverage for the fused AlphaMoE gating router template.

Covers definition-name encoding, committed fi_trace_out artifacts that must
exec and run standalone, the rendered init source, and end-to-end fi_trace
emission through the public ``fi_trace`` attribute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flashinfer.trace.templates.moe import (
    alphamoe_fused_router_trace as ALPHAMOE,
)

FI_TRACE_OUT = Path(__file__).parent / "fi_trace_out"
PLAIN_JSON = FI_TRACE_OUT / "alphamoe_fused_router_e256_k8_b16.json"
SHARED_JSON = FI_TRACE_OUT / "alphamoe_fused_router_e257_k9_b8.json"


def _exec_in_fresh_namespace(source: str) -> dict:
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 — exercising the emitted source is the point
    return namespace


def _axes(**overrides):
    base = dict(num_experts=256, top_k=8, block_m=16)
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Definition-name encoding
# ---------------------------------------------------------------------------


def test_alphamoe_name_encodes_const_axes():
    """The definition name is the prefix plus the const-axis abbreviation vector."""
    assert (
        ALPHAMOE.definition_name(_axes()) == "alphamoe_fused_router_e256_k8_b16"
    )
    assert (
        ALPHAMOE.definition_name(_axes(num_experts=257, top_k=9, block_m=8))
        == "alphamoe_fused_router_e257_k9_b8"
    )


def test_alphamoe_routed_and_shared_geometries_have_distinct_names():
    """Shared-expert geometry (E+1 experts, top-k+1) never collides with plain."""
    plain = ALPHAMOE.definition_name(_axes())
    shared = ALPHAMOE.definition_name(_axes(num_experts=257, top_k=9, block_m=8))
    assert plain != shared


def test_alphamoe_var_axes_do_not_enter_the_name():
    """num_tokens is a var axis: a batch resize must not rename the definition."""
    name = ALPHAMOE.definition_name(_axes())
    named_with_batch = ALPHAMOE.definition_name({**_axes(), "num_tokens": 128})
    assert named_with_batch == name


# ---------------------------------------------------------------------------
# 2. Committed artifacts are runnable standalone
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PLAIN_JSON.exists(), reason="alphamoe trace not generated")
def test_committed_alphamoe_init_runs_standalone():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = json.loads(PLAIN_JSON.read_text())["init"]
    namespace = _exec_in_fresh_namespace(source)
    init = namespace["_alphamoe_fused_router_init"]

    out = init(num_tokens=4, num_experts=32, top_k=4, block_m=8, device="cuda")
    assert out["router_logits"].shape == (4, 32)
    assert out["router_logits"].dtype == torch.float32
    assert out["router_logits"].is_cuda
    assert out["top_k"] == 4 and out["block_m"] == 8
    assert out["has_shared_expert"] is False


@pytest.mark.skipif(not SHARED_JSON.exists(), reason="alphamoe trace not generated")
@pytest.mark.parametrize("has_shared_expert", [False, True])
def test_committed_alphamoe_reference_runs_standalone(has_shared_expert):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = json.loads(SHARED_JSON.read_text())["reference"]
    namespace = _exec_in_fresh_namespace(source)
    reference = namespace["_alphamoe_fused_router_reference"]

    T, E, K, B = 6, 33, 5, 8
    logits = torch.randn(T, E, dtype=torch.float32, device="cuda")
    (
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        expert_counts,
        expert_offsets,
        expert_scatter_offsets,
    ) = reference(logits, K, B, has_shared_expert)

    # softmax over the selected logits: rows sum to one.
    assert topk_weights.shape == (T, K)
    assert torch.allclose(
        topk_weights.sum(dim=-1), torch.ones(T, device="cuda"), atol=1e-6
    )
    # stable descending selection sanity: ids are in range and unique per token.
    assert topk_ids.shape == (T, K)
    assert int(topk_ids.min()) >= 0 and int(topk_ids.max()) < E
    for row in topk_ids.tolist():
        assert len(set(row)) == K
    # shared expert column is the last selected id for every token.
    if has_shared_expert:
        assert (topk_ids[:, -1] == E - 1).all()
    # histogram and padded-inclusive offsets are consistent.
    assert expert_counts.shape == (E,)
    assert int(expert_counts.sum()) == T * K
    assert expert_offsets.shape == (E + 1,)
    assert int(expert_offsets[0]) == 0
    assert int(expert_offsets[-1]) == int(num_tokens_post_padded[0])
    padded_block = ((expert_counts + B - 1) // B) * B
    assert int(padded_block.sum()) == int(expert_offsets[-1])
    # scatter offsets mirror the upstream counts.clone() routing plan output.
    assert torch.equal(expert_scatter_offsets, expert_counts)
    # per-block expert ids: every nonempty expert's blocks carry its id.
    expert = 5
    start, end = int(expert_offsets[expert]), int(expert_offsets[expert + 1])
    if start < end:
        assert (expert_ids[start // B : end // B] == expert).all()
        # sentinel tail inside the segment.
        count = int(expert_counts[expert])
        if start + count < end:
            assert (sorted_token_ids[start + count : end] == T * K).all()
        # valid routes reference (token, slot) pairs of the same expert.
        routes = sorted_token_ids[start : start + count].to(torch.int64)
        assert (topk_ids.flatten()[routes] == expert).all()


# ---------------------------------------------------------------------------
# 3. Rendered source is standalone (name-resolution trap)
# ---------------------------------------------------------------------------


def test_alphamoe_init_renders_standalone():
    """Dump-time globals must not leak into the committed init source."""
    from flashinfer.trace.template import _render_init_source
    from flashinfer.trace.templates.moe import _alphamoe_fused_router_init

    namespace = _exec_in_fresh_namespace(
        _render_init_source(_alphamoe_fused_router_init)
    )
    inputs = namespace["_alphamoe_fused_router_init"](
        num_tokens=4, num_experts=8, top_k=2, block_m=4, device="cpu"
    )
    assert inputs["router_logits"].shape == (4, 8)
    assert inputs["router_logits"].dtype == torch.float32


def test_alphamoe_reference_renders_standalone():
    from flashinfer.trace.template import _render_reference_source
    from flashinfer.trace.templates.moe import (
        _alphamoe_fused_router_init,
        _alphamoe_fused_router_reference,
    )

    namespace = _exec_in_fresh_namespace(
        _render_reference_source(_alphamoe_fused_router_reference)
    )
    inputs = _alphamoe_fused_router_init(
        num_tokens=3,
        num_experts=8,
        top_k=3,
        block_m=8,
        has_shared_expert=True,
        device="cpu",
    )
    out = namespace["_alphamoe_fused_router_reference"](
        inputs["router_logits"],
        inputs["top_k"],
        inputs["block_m"],
        inputs["has_shared_expert"],
    )
    assert len(out) == 8
    assert out[0].shape == (3, 3) and out[1].shape == (3, 3)
    assert (out[1][:, -1] == 7).all()


# ---------------------------------------------------------------------------
# 4. End-to-end: public fi_trace emits a complete definition
# ---------------------------------------------------------------------------


def test_fi_trace_emits_alphamoe_definition():
    """Exercise the symbolic trace through the decorated public API."""
    import flashinfer

    defn = flashinfer.fused_moe.alphamoe_fused_router.fi_trace(
        router_logits=torch.zeros(8, 32, dtype=torch.float32),
        top_k=4,
        block_m=8,
        has_shared_expert=True,
    )
    assert defn["op_type"] == "moe_routing"
    assert defn["name"] == "alphamoe_fused_router_e32_k4_b8"
    assert defn["axes"]["num_experts"]["value"] == 32
    assert defn["axes"]["top_k"]["value"] == 4
    assert defn["axes"]["block_m"]["value"] == 8
    assert "unknown" not in str(defn["inputs"])
    assert "unknown" not in str(defn["outputs"])
