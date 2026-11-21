.. _apimamba:

flashinfer.mamba
================

Mamba / Mamba-2 state-space-model kernels. These wrap the selective scan and
state-update primitives used in SSM blocks.

.. currentmodule:: flashinfer.mamba

.. autosummary::
    :toctree: ../generated

    selective_state_update
    checkpointing_ssu
    ssd_combined_fwd
    replayssm_materialize
    causal_conv1d_fn
    causal_conv1d_update

ReplaySSM checkpointing update
-----------------------------

``checkpointing_ssu`` supports speculative Mamba-2 state updates with a replay
ring and a persistent checkpoint. Only accepted history contributes to a flush;
unaccepted ring padding is ignored by both monolithic and two-kernel paths.

The default ``sr_rng="philox"`` preserves Philox stochastic rounding.
Under this policy, omitting ``rand_seed`` selects round-to-nearest in
``checkpointing_ssu``; a supplied seed requires positive ``philox_rounds``
(default 10). For FP16, FP8 E4M3 and INT8 persistent state, the explicit
``sr_rng="shared_seed"`` option uses the invocation seed's uniform low 32 bits
without running Philox. The repeated random words are correlated. Supply a
fresh uniform seed for each invocation and layer through ``rand_seed``; in this
mode the ``philox_rounds`` count is unused. Zero is normalized to 10 for JIT
compatibility; existing positive counts are retained. Shared-seed mode rejects
negative rounds, a missing seed, BF16 state and FP32 state. Neither
policy changes the current round-to-nearest working-state readout.

The optional ``main_pipeline_stages`` and ``main_ctas_per_sm`` controls expose
the existing two-kernel MAIN launch settings. Their defaults, both zero, retain
the normal heuristic/autotuner behavior. Positive controls select the
two-kernel algorithm, require its scratch tensors, and cannot be combined with
``algorithm="monolith"``. Pipeline stages may be 1 or 2; the CTA cap is a
nonnegative int32 value and is capped at the total work. The autotuner's search
currently covers CTA caps 1 through 32.

ReplaySSM prefix materialization accepts the same explicit ``sr_rng`` policy.
The default Philox path keeps its scalar base-seed contract and derives a stream
per layer. With ``sr_rng="shared_seed"``, provide a contiguous CUDA int64 tensor
of shape ``(L,)`` with one fresh uniform-low-32-bit seed for each layer. No
round-count argument is required for shared-seed FP16, FP8 E4M3 or INT8 state;
omitted or zero rounds normalize to 10. The default Philox policy keeps
round-to-nearest with the materializer's default zero rounds. Zero-prefix copies and
inactive requests retain their existing behavior. Launch tuning never changes
the caller's rounding policy or round count.

Inference causal convolution
---------------------------

``causal_conv1d_fn`` and ``causal_conv1d_update`` provide width-four,
per-channel convolution with fused SiLU on SM100/SM103. Packed inputs and
outputs are BF16 ``[tokens, channels]``. Weights are FP32 or BF16
``[channels, 4]``; tap 3 multiplies the current token. The optional bias
initializes an FP32 accumulator, followed by four FMAs in tap order and FP32 SiLU, then one
BF16 output conversion.

The caller owns a BF16 ``[slots, channels, history_capacity]`` state pool.
History stores raw convolution inputs. Slots may have padding between them;
either the channels or the history axis must be contiguous. Slot zero is
valid, and index ``-1`` denotes padding. Nonempty active sequences must have
distinct, in-range slots. Padding produces zero output without changing
state, and empty sequences leave state untouched.

Prefill reads ``cu_seqlens`` and ``has_initial_state`` on the device, consumes
three old history values (or zero history) and writes the final three values
to the same slot. For more than 32 sequences and nonempty input, its optional
``workspace`` needs ``1 + 2 * min(rows, ceil(rows / B) + num_sequences)`` int32
elements. B is 8 if ``rows < 64 * num_sequences`` and 16 otherwise. A GPU
kernel builds the chunk-to-sequence map shared by all channel tiles.
Smaller batches use no workspace:
one sequence maps directly to token tiles; 2-32 sequences build the prefix
inside the convolution kernel. The scratch map is
rebuilt on every call, including graph replay. Concurrent calls
need separate workspaces and nonoverlapping state slots. The ordinary update
uses no workspace and commits all its input tokens in one launch.

For speculative verification, ``num_accepted_tokens`` selects the history
window from the **preceding** call; it includes the accepted bonus token.
For a positive count ``a``, the input history is ``state[..., a-1:a+2]``.
Nonpositive counts denote an inactive row and produce zero output without updating
state. The current T-token call writes ``[old_h1, old_h2, x0, ..., x(T-1)]``
to positions ``0:T+2``. Provision at least ``T_max + 2`` history positions
and require ``1 <= a <= history_capacity - 2`` for active rows. Select only
windows produced by the preceding call. After prefill the
first speculative call uses ``a=1``. This interface covers linear accepted
prefixes, not tree speculation.

For CUDA graphs, warm the kernels and supply ``out`` and the prefill
``workspace`` before capture. Shapes and addresses remain fixed while input
values, offsets, masks, slots and accepted counts may change in place.
Packed offsets must be nondecreasing, start at zero and end at or before the
input's row count. Rows beyond the final offset are not written, including
for a single sequence. Device-value constraints are caller-owned and introduce no
device-to-host validation synchronization. Output, state and workspace must
not alias their input operands.

The unified benchmark compares complete calls, including state updates,
against the repository's framework Triton reference. ``--input_row_stride``
sets the input row stride in elements and defaults to ``--dim``. Reference checks use
separate initial state snapshots, and state resets and JIT compilation are
outside timing. Reported bandwidth is a logical-byte estimate, not measured
DRAM traffic. Eager host-submission time is reported separately from device
time. ``--profile_launches`` counts CUDA kernels and copies in one warmed
eager call per backend in a separate process; this profiling is outside the
timing samples. ``--seq_lens`` supplies explicit prefill lengths; ``--varlen``
generates unequal lengths instead::

    python benchmarks/flashinfer_benchmark.py --routine causal_conv1d_fn \
        --batch_size 8 --dim 4096 --seq_len 1024 --varlen \
        --has_initial_state --backends flashinfer triton --refcheck \
        --dry_run_iters 10 --num_iters 20 --profile_launches

    python benchmarks/flashinfer_benchmark.py --routine causal_conv1d_update \
        --batch_size 32 --dim 4096 --seq_len 5 --speculative \
        --backends flashinfer triton --refcheck --dry_run_iters 10 --num_iters 20

The prefill, decode and speculative comparison shapes can be run with::

    python benchmarks/flashinfer_benchmark.py \
        --testlist benchmarks/samples/causal_conv1d.txt \
        --output_path causal_conv1d.csv

The review prefill matrix, including the 32/33-sequence boundary and batches
of 64-256 sequences, is reproducible with::

    python benchmarks/flashinfer_benchmark.py \
        --testlist benchmarks/samples/causal_conv1d_prefill.txt \
        --output_path causal_conv1d_prefill.csv
