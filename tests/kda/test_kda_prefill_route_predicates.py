"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""The prepared KDA prefill runtime shares the planner predicates of
``flashinfer.kda_prefill`` (CPU-only: no GPU, no module build).

The runtime used to carry its own copies of six planner predicates.  Over the
domain below (SM100a / SM103a; 148, 152 and 160 SMs; 1 to 128 sequences; 1 to
96 heads; 64 to 8192 tokens; fixed, packed-uniform and packed-mixed layouts)
these tests pin that the shared predicates reproduce the policy those copies
encoded, and that the one rule the runtime still keeps (the recurrence-piece
rule without the 148 / 152-SM gate, which the TF32 auto route requests)
differs from the shared rule only at other SM counts.
"""

import itertools

import pytest
import torch

import flashinfer.cake_kda_tf32_runtime as runtime
import flashinfer.kda_prefill as planner

CAPABILITIES = ((10, 0), (10, 3))
GPU_ARCH = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
SM_COUNTS = (148, 152, 160)
BATCHES = (1, 2, 4, 8, 16, 128)
HEADS = (1, 4, 6, 8, 12, 16, 24, 32, 64, 96)
TOKENS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
DIRECT = runtime.BF16_ROUTE_DIRECT_M128
PIECE = runtime.BF16_ROUTE_PIECE_M128
EXPORTED_BF16_ROUTES = {
    DIRECT,
    runtime.BF16_ROUTE_DIRECT_M128_N16,
    runtime.BF16_ROUTE_M64,
}


def _domain():
    """Yield (capability, sm_count, B, H, fixed_layout, lengths) over the domain."""
    for capability, sm_count, B, H, T in itertools.product(
        CAPABILITIES, SM_COUNTS, BATCHES, HEADS, TOKENS
    ):
        yield capability, sm_count, B, H, True, (T,) * B  # fixed [B, T] layout
        yield capability, sm_count, B, H, False, (T,) * B  # packed, uniform
        if B > 1:  # packed, mixed lengths
            yield capability, sm_count, B, H, False, (T,) + (max(32, T // 2),) * (B - 1)


def _lpt_loads(capability, sm_count, H, lengths):
    if (
        planner._uses_measured_sm100_persistent_policy(
            compute_capability=capability, sm_count=sm_count
        )
        and len(set(lengths)) > 1
        and len(lengths) * H > sm_count
    ):
        _ids, _offsets, loads = planner._make_lpt_task_bins(
            tuple(sorted(lengths, reverse=True)), num_heads=H, sm_count=sm_count
        )
        return loads
    return ()


def _kept_piece_rule(
    *,
    compute_capability,
    sm_count,
    num_sequences,
    num_heads,
    uniform_sequences,
    max_sequence_length,
    use_initial_state=True,
    store_final_state=True,
):
    """The runtime's piece rule (no SM-count gate) in the planner's vocabulary."""
    return runtime._should_use_uniform_piece_persistent(
        gpu_arch=GPU_ARCH[compute_capability],
        sm_count=sm_count,
        num_seqs=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_seq_len=max_sequence_length,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    )


def _bf16_auto_route(route, direct_route):
    """The prepared BF16 launch maps unexported families to the direct family."""
    return route if route in EXPORTED_BF16_ROUTES else direct_route


def test_runtime_imports_the_shared_predicates():
    assert runtime._select_bf16_route is planner._select_bf16_route
    assert runtime._pair_packed_beta_tma_source is planner._pair_packed_beta_tma_source
    for name in ("_requires_exact_n16_recurrence", "select_bf16_schedule_route"):
        assert not hasattr(runtime, name)


def test_exact_n16_holdout_matches_the_former_runtime_rule():
    """148 SMs, packed layout, 128 uniform sequences of 96 heads, no checkpoints."""
    for capability, sm_count, B, H, fixed, lengths in _domain():
        uniform = len(set(lengths)) == 1
        expected = sm_count == 148 and not fixed and B == 128 and H == 96 and uniform
        assert (
            planner._requires_exact_n16_recurrence(
                compute_capability=capability,
                sm_count=sm_count,
                fixed_layout=fixed,
                num_sequences=B,
                num_heads=H,
                uniform_sequences=uniform,
                checkpoint_every_n_tokens=0,
            )
            is expected
        )


def test_small_bh_owner_helper_matches_the_runtime_bf16_arm():
    """The runtime keeps a TF32 superset of the rule; its BF16 arm is the shared one."""
    for capability, sm_count, B, H, _fixed, lengths in _domain():
        shared = planner._should_use_small_bh_owner_helper(
            compute_capability=capability,
            sm_count=sm_count,
            num_sequences=B,
            num_heads=H,
            sequence_length=max(lengths),
        )
        kept = runtime._should_use_small_bh_owner_helper(
            gpu_arch=GPU_ARCH[capability],
            sm_count=sm_count,
            num_seqs=B,
            num_heads=H,
            max_seq_len=max(lengths),
        )
        assert shared is kept


def test_recurrence_piece_rule_differs_only_at_other_sm_counts():
    """The shared rule adds the 148 / 152-SM gate the runtime's rule lacks."""
    other_sm_points = 0
    for capability, sm_count, B, H, _fixed, lengths in _domain():
        kwargs = dict(
            compute_capability=capability,
            sm_count=sm_count,
            num_sequences=B,
            num_heads=H,
            uniform_sequences=len(set(lengths)) == 1,
            max_sequence_length=max(lengths),
        )
        shared = planner._should_use_uniform_piece_persistent(**kwargs)
        kept = _kept_piece_rule(**kwargs)
        if sm_count in (148, 152):
            assert shared is kept
        else:
            assert shared is False
            other_sm_points += kept
    # the 148 points of this domain where the ungated rule fires at 160 SMs
    assert other_sm_points == 148


def test_select_bf16_route_matches_the_former_runtime_chain(monkeypatch):
    """The runtime's former copy was this chain with its SM-count-free piece rule.

    Where the two differ (recurrence pieces at 160 SMs) the prepared BF16 launch
    maps both results to the direct family and the TF32 launch restores the
    piece request, so neither launch selects a different module.
    """
    differing = 0
    for capability, sm_count, B, H, fixed, lengths in _domain():
        piece_kwargs = dict(
            compute_capability=capability,
            sm_count=sm_count,
            num_sequences=B,
            num_heads=H,
            uniform_sequences=len(set(lengths)) == 1,
            max_sequence_length=max(lengths),
        )
        kwargs = dict(
            fixed_layout=fixed,
            lpt_loads=_lpt_loads(capability, sm_count, H, lengths),
            **piece_kwargs,
        )
        shared = planner._select_bf16_route(**kwargs)
        with monkeypatch.context() as patched:
            patched.setattr(
                planner, "_should_use_uniform_piece_persistent", _kept_piece_rule
            )
            former = planner._select_bf16_route(**kwargs)
        direct_route = planner._direct_m128_route(
            num_heads=H, max_sequence_length=max(lengths)
        )
        if shared != former:
            differing += 1
            assert sm_count not in (148, 152)
            assert (former, shared, direct_route) == (PIECE, DIRECT, DIRECT)
        assert _bf16_auto_route(shared, direct_route) == _bf16_auto_route(
            former, direct_route
        )
        tf32_route = (
            PIECE
            if shared == direct_route == DIRECT and _kept_piece_rule(**piece_kwargs)
            else shared
        )
        assert tf32_route == former
    assert differing == 143


@pytest.mark.parametrize(
    ("shape", "paired"),
    [
        ((1, 512, 12), (256, 24)),  # dense H12 rows, even token count
        ((2, 256, 12), (256, 24)),  # dense H12 rows of two sequences
        ((1, 511, 12), None),  # odd token count
        ((1, 512, 16), None),  # not twelve heads
    ],
)
def test_pair_packed_beta_source_takes_the_flat_view_as_one_batch(shape, paired):
    """The runtime hands the shared rule its flattened ``[total_tokens, H]`` view."""
    beta = torch.empty(shape, dtype=torch.bfloat16)
    flat = beta.reshape(shape[0] * shape[1], shape[2])
    source = runtime._pair_packed_beta_tma_source(flat.unsqueeze(0))
    if paired is None:
        assert source is None
    else:
        assert tuple(source.shape) == paired
        assert source.data_ptr() == flat.data_ptr()


def test_pair_packed_beta_source_declines_a_strided_head_slice():
    """A beta slice of a wider row (serving allocators) keeps the padded carrier."""
    wide = torch.empty((1, 512, 36), dtype=torch.bfloat16)
    flat = wide[0, :, 8:20]  # [512, 12] view with a 36-element token pitch
    assert runtime._pair_packed_beta_tma_source(flat.unsqueeze(0)) is None
