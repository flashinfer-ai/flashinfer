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

.. rubric:: Numerical mode

The module selects the numerical mode: the deterministic package
(``flashinfer.cake_causal_conv1d_det_train[_kcp]``) or the non-deterministic
package (``flashinfer.cake_causal_conv1d_nondet_train[_kcp]``).  No entry point
takes a ``deterministic`` keyword; passing one raises ``TypeError``.

.. list-table:: Training causal conv1d modules by numerical mode
   :header-rows: 1
   :widths: 16 34 25 25

   * - Mode and route
     - Module and entry points
     - Results
     - Under ``torch.use_deterministic_algorithms(True)``
   * - deterministic, single-rank
     - ``flashinfer.cake_causal_conv1d_det_train``: ``cake_causal_conv1d_det``,
       ``cake_causal_conv1d_det_train_forward``,
       ``cake_causal_conv1d_det_train_backward``
     - ``y``, ``dx``, ``dweight`` and ``dbias`` bitwise reproducible
     - no effect
   * - deterministic, context-parallel (KCP)
     - ``flashinfer.cake_causal_conv1d_det_train_kcp``:
       ``cake_causal_conv1d_det_kcp``, ``cake_causal_conv1d_det_kcp_multi``
     - ``y``, ``dx`` and the rank-local FP32 ``dweight`` partial bitwise
       reproducible
     - no effect
   * - non-deterministic, single-rank
     - ``flashinfer.cake_causal_conv1d_nondet_train``:
       ``cake_causal_conv1d_nondet``,
       ``cake_causal_conv1d_nondet_train_forward``,
       ``cake_causal_conv1d_nondet_train_backward``
     - ``y`` and ``dx`` bitwise the deterministic package's; ``dweight`` and
       ``dbias`` within its error budget, not bitwise reproducible
     - the forward runs; the backward raises ``RuntimeError`` (with
       ``warn_only=True``: ``UserWarning``, then it runs); nothing falls back
       to the deterministic package
   * - non-deterministic, context-parallel (KCP)
     - ``flashinfer.cake_causal_conv1d_nondet_train_kcp``:
       ``cake_causal_conv1d_nondet_kcp``,
       ``cake_causal_conv1d_nondet_kcp_multi``
     - ``y`` and ``dx`` bitwise the deterministic package's; the rank-local
       FP32 ``dweight`` partial within its error budget, not bitwise
       reproducible
     - the forward runs; the backward raises ``RuntimeError`` (with
       ``warn_only=True``: ``UserWarning``, then it runs); nothing falls back
       to the deterministic package

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
the halo collective of the KDA step's context.

.. currentmodule:: flashinfer.cake_causal_conv1d_det_train_kcp

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d_det_kcp
    cake_causal_conv1d_det_kcp_multi
    CakeCausalConv1dDetKcpFunction
    CakeCausalConv1dDetKcpMultiFunction
    CakeCausalConv1dDetKcpRecomputeHandle

Training causal conv1d, non-deterministic (Cake)
-------------------------------------------------

The same convolution, single-rank and context-parallel, with a backward whose
``dweight`` / ``dbias`` reductions are unordered: the non-deterministic package
``flashinfer.cake_causal_conv1d_nondet_train[_kcp]``.  These entry points are
stable APIs (``@flashinfer_api``); their backend package lives under
``flashinfer.experimental.cake``.  The module selects the mode, and nothing
here runs the deterministic package.

* ``flashinfer.cake_causal_conv1d_nondet_train``:
  ``cake_causal_conv1d_nondet`` (its autograd function
  ``CakeCausalConv1dNondetFunction``),
  ``cake_causal_conv1d_nondet_train_forward``,
  ``cake_causal_conv1d_nondet_train_backward``, the forward / backward
  workspace helpers
  ``cake_causal_conv1d_nondet_train_{forward,backward}_workspace{,_bytes}``
  and ``is_cake_causal_conv1d_nondet_train_supported``.  The parameters are the
  deterministic entry points'.
* ``flashinfer.cake_causal_conv1d_nondet_train_kcp``:
  ``cake_causal_conv1d_nondet_kcp``, ``cake_causal_conv1d_nondet_kcp_multi``
  and ``CakeCausalConv1dNondetKcpRecomputeHandle``, with the deterministic KCP
  entry points' parameters and the keyword-only ``bwd_reserve_sms`` (SMs the
  main backward leaves free; ``bwd_reserve_sms >= 1`` raises ``ValueError`` on
  SM107 and, elsewhere, runs the one-program backward after the halo exchange
  instead of the overlapped backward).  ``cooperative=True`` raises
  ``ValueError``: it applies to the deterministic KCP package only, and the
  non-deterministic backward launches no cooperative grid.

.. currentmodule:: flashinfer.cake_causal_conv1d_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d_nondet
    cake_causal_conv1d_nondet_train_forward
    cake_causal_conv1d_nondet_train_backward
    cake_causal_conv1d_nondet_train_backward_workspace_bytes
    cake_causal_conv1d_nondet_train_backward_workspace
    CakeCausalConv1dNondetFunction

.. currentmodule:: flashinfer.cake_causal_conv1d_nondet_train_kcp

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d_nondet_kcp
    cake_causal_conv1d_nondet_kcp_multi
    CakeCausalConv1dNondetKcpRecomputeHandle

The package behind both,
``flashinfer.experimental.cake.cake_causal_conv1d_nondet_train``, exports the
single-rank names; its ``kcp`` module exports the KCP names, the explicit
entry points ``cake_causal_conv1d_nondet_train_kcp_{forward,backward}`` and
``cake_causal_conv1d_nondet_train_kcp_multi_{forward,backward}`` and the
autograd functions ``CakeCausalConv1dNondetKcpFunction`` and
``CakeCausalConv1dNondetKcpMultiFunction``.  The package is not marked
experimental and follows the same rules.

Each name of the two modules maps to the deterministic name with ``det`` in
place of ``nondet`` (``cake_causal_conv1d_nondet`` ->
``cake_causal_conv1d_det``, ``CakeCausalConv1dNondetFunction`` ->
``CakeCausalConv1dDetFunction``,
``cake_causal_conv1d_nondet_kcp`` -> ``cake_causal_conv1d_det_kcp``,
``cake_causal_conv1d_nondet_kcp_multi`` -> ``cake_causal_conv1d_det_kcp_multi``,
``CakeCausalConv1dNondetKcpRecomputeHandle`` ->
``CakeCausalConv1dDetKcpRecomputeHandle``).
Example::

    from flashinfer.cake_causal_conv1d_nondet_train import (
        cake_causal_conv1d_nondet,
    )
    from flashinfer.cake_causal_conv1d_nondet_train_kcp import (
        cake_causal_conv1d_nondet_kcp,
    )

    y = cake_causal_conv1d_nondet(x, weight, cu_seqlens, bias)
    y_local = cake_causal_conv1d_nondet_kcp(x_local, weight, ctx)

Numerical contract: the forward runs copies of the deterministic forward
programs, so ``y`` is bitwise the same.  The backward is one ordinary launch
per call (no reduce program, no cooperative grid): ``dx`` is bitwise the
deterministic backward's, and the CTAs add their ``dweight`` / ``dbias`` rows
into zeroed accumulators with unordered global reductions (FP32, or FP64 under
``silu_grad_preactivation="fp32"``), rounded once into the requested dtype.
``dweight`` / ``dbias`` stay within the deterministic error budget but are not
bitwise reproducible from run to run.  The KCP backward overlaps the halo
exchange: its main program runs before the exchange is waited, as if no
fragment had tail rows, with CTAs that claim their tokens at run time (Cluster
Launch Control), so a kernel resident on another stream slows it about in
proportion to the SMs it holds; a tail-row program then writes ``dx`` of the
last three rows of each fragment that receives tail rows.  ``dx`` is bitwise
the deterministic KCP backward's; the rank-local FP32 ``dweight`` partial is
summed into four FP32 accumulator copies in the order the CTAs end their runs,
so it is not bitwise reproducible from run to run.

Under ``torch.use_deterministic_algorithms(True)`` the forward and the
checkpoint recompute run (they are fixed-order).  The entry points that launch
the unordered reduction (the single-rank backward, the KCP backward, the KCP
multi backward, and so the autograd backward) raise PyTorch's alert for an
operation without a deterministic implementation before they launch anything:
a ``RuntimeError``, or a ``UserWarning`` with ``warn_only=True``, after which
the backward runs.  The mode is read when the backward runs.  Use the
deterministic package (``flashinfer.cake_causal_conv1d_det_train[_kcp]``) for
deterministic results.

The non-deterministic backward workspace holds the accumulators and one
ticket counter per channel tile.  Allocate it with
``cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)``
(a zeroed ``uint8`` buffer); every launch leaves it zero, and its size depends
on ``C`` and the route, not on ``S``.  Do not share one workspace between two
backward launches that can run at the same time, for example on two streams.
Without a caller workspace the package keeps one zeroed buffer per device and
stream; inside a CUDA Graph capture it keeps one zeroed buffer per capture and
stream (when CUDA's capture id is not available, every captured call allocates
its own).

.. rubric:: Renamed non-deterministic modules and names

The non-deterministic modules and names below were renamed.  The old names
were removed without aliases: importing an old module raises
``ModuleNotFoundError``.  The ``deterministic`` keyword was removed as well:
the module selects the mode.  The old names ran the deterministic package by
default: a call that relied on that default moves to the deterministic
package's names (``flashinfer.cake_causal_conv1d_det_train[_kcp]``), not to
the names below.

.. list-table:: Old and new non-deterministic names
   :header-rows: 1
   :widths: 50 50

   * - Old name (removed, no alias)
     - New name
   * - ``flashinfer.experimental.cake.cake_causal_conv1d_train``
     - ``flashinfer.cake_causal_conv1d_nondet_train``
   * - ``flashinfer.cake_causal_conv1d_train_kcp``,
       ``flashinfer.experimental.cake.cake_causal_conv1d_train_kcp``
     - ``flashinfer.cake_causal_conv1d_nondet_train_kcp``
   * - ``flashinfer.experimental.cake.cake_causal_conv1d_train_backend``
     - ``flashinfer.experimental.cake.cake_causal_conv1d_nondet_train``
   * - ``flashinfer.experimental.cake.cake_causal_conv1d_train_backend.kcp``
     - ``flashinfer.experimental.cake.cake_causal_conv1d_nondet_train.kcp``
   * - ``cake_causal_conv1d``
     - ``cake_causal_conv1d_nondet``
   * - ``CakeCausalConv1dFunction``
     - ``CakeCausalConv1dNondetFunction``
   * - ``cake_causal_conv1d_train_forward``
     - ``cake_causal_conv1d_nondet_train_forward``
   * - ``cake_causal_conv1d_train_backward``
     - ``cake_causal_conv1d_nondet_train_backward``
   * - ``cake_causal_conv1d_train_forward_workspace``
     - ``cake_causal_conv1d_nondet_train_forward_workspace``
   * - ``cake_causal_conv1d_train_backward_workspace``
     - ``cake_causal_conv1d_nondet_train_backward_workspace``
   * - ``cake_causal_conv1d_train_forward_workspace_bytes``
     - ``cake_causal_conv1d_nondet_train_forward_workspace_bytes``
   * - ``cake_causal_conv1d_train_backward_workspace_bytes``
     - ``cake_causal_conv1d_nondet_train_backward_workspace_bytes``
   * - ``is_cake_causal_conv1d_train_supported``
     - ``is_cake_causal_conv1d_nondet_train_supported``
   * - ``cake_causal_conv1d_kcp``
     - ``cake_causal_conv1d_nondet_kcp``
   * - ``cake_causal_conv1d_multi_kcp``
     - ``cake_causal_conv1d_nondet_kcp_multi``
   * - ``RecomputeHandleKcp``
     - ``CakeCausalConv1dNondetKcpRecomputeHandle``
   * - ``CakeCausalConv1dFunctionKcp``
     - ``CakeCausalConv1dNondetKcpFunction``
   * - ``CakeCausalConv1dMultiFunctionKcp``
     - ``CakeCausalConv1dNondetKcpMultiFunction``
   * - ``cake_causal_conv1d_forward_kcp``
     - ``cake_causal_conv1d_nondet_train_kcp_forward``
   * - ``cake_causal_conv1d_backward_kcp``
     - ``cake_causal_conv1d_nondet_train_kcp_backward``
   * - ``cake_causal_conv1d_multi_forward_kcp``
     - ``cake_causal_conv1d_nondet_train_kcp_multi_forward``
   * - ``cake_causal_conv1d_multi_backward_kcp``
     - ``cake_causal_conv1d_nondet_train_kcp_multi_backward``
   * - ``graph_cache_clear`` (of the backend's ``kcp`` module)
     - ``cake_causal_conv1d_nondet_train_kcp_graph_cache_clear``
   * - ``host_cache_clear`` (of the backend's ``kcp`` module)
     - ``cake_causal_conv1d_nondet_train_kcp_host_cache_clear``
   * - ``plan_cache_clear`` (of the backend)
     - ``cake_causal_conv1d_nondet_train_plan_cache_clear``

The deterministic package's names before its ``det`` rename (the modules
``flashinfer.cake_causal_conv1d_train`` and
``flashinfer.cake_causal_conv1d_train_backend``, the top-level
``flashinfer.cake_causal_conv1d`` / ``CakeCausalConv1dFunction`` /
``cake_causal_conv1d_train_{forward,backward}`` and the KCP names
``cake_causal_conv1d_kcp`` / ``cake_causal_conv1d_kcp_multi`` /
``KcpRecomputeHandle`` of ``flashinfer.cake_causal_conv1d_det_train``) are
removed too; use the ``det`` names.
