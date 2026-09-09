# Chunk-parallel workspace layouts and composition contract

Three conventions and one contract. Every one of them has already been got
wrong once in this work, and none of them is caught by a shape check, because
`head_size` is 128 on both axes of the state and the two orientations are the
same shape.

This is the document, not the comments. A comment sits beside one call site; a
stage ported next month reads this.

## 1. The state is (V, K) at the op and (K, V) in the references

    flashinfer.gdn_prefill.chunk_gated_delta_rule
        initial_state, output_state, state_checkpoints   [N, H, V, K]

    tests/gdn/reference_delta_rule.py
        blockwise_cp_delta_rule_pre_transposed  -> transfer, state   [H, K, V]
        cp_delta_rule_fixup                     -> per-chunk states  [H, K, V]

Feeding a reference state to the op without transposing does not raise. It
produces a state wrong by its own magnitude: measured at 1.10 relative against
the kernel's own checkpoints, against 2.9e-03 once transposed.

**Name the axis in the identifier.** A workspace tensor is `state_KV` or
`state_VK`, never `state`. A stage that takes one and returns the other says so
in both names. `cp_compose_proof.py` in the measurement harness transposes at exactly one place
-- the boundary between the reference stages and the op -- and says why there.

## 2. `cp_delta_rule_fixup` returns (final, per-chunk), in that order

    reference_delta_rule.py:703
        return torch.stack(final_states), fixed_states_by_seq

`[0]` is one state per *sequence*. `[1]` is a list, one entry per sequence, each
`[n_chunks, H, K, V]`.

Taking `[0]` where per-chunk states were wanted broadcasts `(1, H, K, V)` across
`(n_chunks, H, K, V)` without error, and every chunk after the first then starts
from the sequence's *final* state. That was a 1.10 relative error that looked
like a decay artefact for two rounds.

`tests/gdn/test_prefill_cp_delta_rule.py:501` is the consuming pattern to copy:

    _, ref_by_seq = reference.cp_delta_rule_fixup_transposed(
        ref_transfers_by_seq, ref_states_by_seq,
        ref_initial_states_by_seq if use_initial_state else None,
    )

It discards `[0]`, uses the `_transposed` variant, and passes the initial state
as a parameter rather than folding it by hand.

## 3. The transfer multiplies from the left

    reference_delta_rule.py:691
        state = bmm(local_transfers[chunk], state) + local_states[chunk]

So a sequence's own initial state enters its first chunk as
`bmm(transfer[0], initial)`, not `initial @ transfer[0]`. Both are valid matrix
products of the right shape.

Prefer passing `initial_states_by_seq` to `cp_delta_rule_fixup_transposed` over
folding it in by hand; the parameter exists and the tests use it.

## 4. An empty sequence's output state is left untouched

`_FullyFusedDeltaRuleSm80` guards its whole body on `seq_len != 0`
(`delta_rule_sm80.py`, `if work_desc.seq_len != cutlass.Int32(0)`). A sequence
of zero tokens therefore never writes its row of `output_state`, and that row
keeps whatever the caller passed in.

This is the kernel's property, not the CP path's, so the assertion belongs at
the public entry point in `gdn_prefill.py` and not in a CP adapter -- a non-CP
caller can walk into it just as easily.

For a chunked call this means: a sequence that owns no chunk is not written by
anything, so an adapter must not synthesise a value for it. Writing its initial
state instead is wrong by the size of the state -- 1.977e-01 on the case that
found it.

## The composition, as the sm120 kernel states it

    delta_rule_cp_sm90.py:3954   fixed_state_idx = cp_chunk_idx - 1
        chunk j starts from the state *after* chunk j-1; chunk 0 starts from
        the sequence's own initial state

    delta_rule_cp_sm90.py:3958   store_state = chunk_idx_in_seq == n - 1
        only a sequence's last chunk writes that sequence's final state

so chunk states go to a workspace with one row per chunk, and the sequence's
final state is gathered from its last chunk. Several chunks writing the
sequence's row would be blocks racing for it.

Proven bit-exact against the fused kernel over fourteen cases, including ragged
packs and a zero-length sequence: `logs/cp_compose/RESULT.md` in the
measurement harness.

## Probes

`PROBE_RULES.md` in the measurement harness. Read it before writing one,
not after. Every
convention on this page was found by a probe, and five probes in the session
that found them returned answers about themselves instead.

## Checking a stage against the references

Consume them the way `tests/gdn/test_prefill_cp_delta_rule.py` does -- `[1]`,
the `_transposed` variant, the initial state as a parameter. Every one of the
three defects above came from a hand-rolled driver that did it differently.
