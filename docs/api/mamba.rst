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

Training causal conv1d (Cake)
-----------------------------

Packed width-4 depthwise causal convolution + SiLU forward / backward for
training on SM100, SM103 and SM107: documents delimited by ``cu_seqlens`` and
convolved on their own, FP32 or BF16 taps and an optional FP32 or BF16
per-channel bias (added after the taps) with FP32 arithmetic and one BF16
rounding, a fused ``dx`` + deterministic ``dweight`` (+ ``dbias``) backward in
two ordinary launches (a main program and a programmatically dependent
fixed-order reduce; no cooperative grid, so kernels resident on other streams
never stall it) whose SiLU' is evaluated at the BF16-rounded
preactivation (default) or, with ``silu_grad_preactivation="fp32"``, at the
unrounded FP32 one for a weight gradient within about 2e-7 rel-L2 of exact, and
an autograd wrapper that saves only the forward's inputs.  The forward's
co-resident CTAs take each channel tile's tokens in chunks from int32 counters
in a forward workspace (``cake_causal_conv1d_det_train_forward_workspace`` /
``cake_causal_conv1d_det_train_forward_workspace_bytes``; zero before its first
launch and left zero by every launch, so one buffer serves every later call of
one stream); when the caller passes none the package keeps one per device and
stream.

.. currentmodule:: flashinfer.cake_causal_conv1d_det_train

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d_det
    cake_causal_conv1d_det_train_forward
    cake_causal_conv1d_det_train_backward
    cake_causal_conv1d_det_train_backward_workspace_bytes
    cake_causal_conv1d_det_train_backward_workspace
    CakeCausalConv1dDetFunction

Context-parallel training causal conv1d (Cake)
----------------------------------------------

The context-parallel (KCP) route of the same convolution: every rank convolves
its fragments of the packed batch, and the first tokens of a fragment that
continues a document read the history rows of the preceding fragment through
the halo collective of the KDA step's context.  ``flashinfer.cake_causal_conv1d_det_train``
keeps the previous names ``cake_causal_conv1d_kcp``, ``cake_causal_conv1d_kcp_multi``
and ``KcpRecomputeHandle`` as aliases of the same objects.

.. currentmodule:: flashinfer.cake_causal_conv1d_det_train_kcp

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d_det_kcp
    cake_causal_conv1d_det_multi_kcp
    RecomputeHandleDetKcp
