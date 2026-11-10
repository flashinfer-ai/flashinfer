.. _apinorm:

flashinfer.norm
===============

Kernels for normalization layers.

.. currentmodule:: flashinfer.norm

.. autosummary::
    :toctree: ../generated

    rmsnorm
    rmsnorm_quant
    fused_add_rmsnorm
    fused_add_rmsnorm_quant
    fused_add_rmsnorm_fp8_block_quant
    gemma_rmsnorm
    gemma_fused_add_rmsnorm
    layernorm
    layernorm_quant
    fused_rmsnorm_silu
    fused_qk_rmsnorm_rope
    fused_dit_residual_layernorm_scale_shift
    fused_dit_gate_residual_layernorm_scale_shift
    fused_dit_gate_residual_layernorm_gamma_beta

Fixed-Width Fused Norms
-----------------------

Fused gated-residual and per-head gated RMSNorm kernels for SM100, SM103 and
SM107. They support only the widths in their docstrings (hidden size 8192, head
dim 64); check the matching ``supports_*`` helper and fall back otherwise.

.. autosummary::
    :toctree: ../generated

    gated_residual_rmsnorm
    supports_gated_residual_rmsnorm
    gated_residual_next_rmsnorm
    supports_gated_residual_next_rmsnorm
    rmsnorm_sigmoid_gate
    supports_rmsnorm_sigmoid_gate

Training RMSNorm (Cake)
-----------------------

Fused BF16 RMSNorm forward / backward for training on SM100, SM103 and SM107:
FP32 statistics and accumulation, one FP32 reciprocal RMS saved per row,
fused ``dx`` + deterministic FP32 ``dw`` backward, optional residual-add
fusion and an autograd wrapper.

.. note::

   **Mode and dispatch rules.**

   * The mode is part of the name, and only of the name. Every public name of
     ``flashinfer.cake_rmsnorm_det_train`` (this section, the deterministic
     package) carries ``det``; every public name of
     ``flashinfer.cake_rmsnorm_nondet_train`` (next section, the
     non-deterministic package) carries ``nondet``. The constants ``ARCHES``
     and ``SUPPORTED_COMPUTE_CAPABILITIES`` keep their names. Each package
     runs its own kernels only; neither runs the other.
   * No entry point of either package takes a ``deterministic`` keyword:
     passing one raises Python's ``TypeError``. The deterministic package's
     weight gradient ``dw`` is bitwise reproducible; the non-deterministic
     package's is not guaranteed to be.
   * Under ``torch.use_deterministic_algorithms(True)`` the non-deterministic
     package alerts only before work whose result can vary from run to run:
     every backward call (and the backward of the autograd
     ``cake_rmsnorm_nondet`` / ``CakeRMSNormNondetFunction``, when it runs)
     raises ``RuntimeError`` naming the ``det`` entry point, or, with
     ``warn_only=True``, warns once and runs the non-deterministic backward
     (never the deterministic one). The forward (no cross-row reduction,
     bitwise reproducible), the workspace helpers,
     ``cake_rmsnorm_nondet_train_route_key`` and
     ``is_cake_rmsnorm_nondet_train_supported`` never alert. The deterministic
     package never alerts.
   * The ``flashinfer`` top level binds the ``det`` entry points
     (``cake_rmsnorm_det``, ``CakeRMSNormDetFunction``,
     ``cake_rmsnorm_det_train_forward``, ``cake_rmsnorm_det_train_backward``,
     ``cake_rmsnorm_det_train_backward_workspace_bytes``) and the modules
     ``flashinfer.cake_rmsnorm_det_train`` and
     ``flashinfer.cake_rmsnorm_nondet_train``. It binds none of the
     non-deterministic package's names: that package is reached through its
     module only.
   * The names from before the rename are removed, with no aliases; code that
     uses them fails at import.  In v0.7.3 every former name ran the
     deterministic programs (``deterministic=False`` raised an error).  In
     v0.7.4rc1 a former entry ran the deterministic package by default
     (``deterministic=True``) and the non-deterministic package with
     ``deterministic=False``.  A v0.7.4rc1 ``deterministic=False`` call under
     ``torch.use_deterministic_algorithms(True)`` raised ``RuntimeError``
     (with ``warn_only=True`` it warned and computed deterministically), so
     it moves to the deterministic name.  The deterministic package now
     carries every deterministic name; the non-deterministic package never
     falls back to it:

     .. list-table::
         :header-rows: 1

         * - Former name
           - v0.7.3, and the v0.7.4rc1 default (``deterministic=True``)
           - v0.7.4rc1 ``deterministic=False``
         * - module ``flashinfer.cake_rmsnorm_train``
           - module ``flashinfer.cake_rmsnorm_det_train``
           - module ``flashinfer.cake_rmsnorm_nondet_train``
         * - ``cake_rmsnorm``
           - ``cake_rmsnorm_det``
           - ``cake_rmsnorm_nondet``
         * - ``CakeRMSNormFunction``
           - ``CakeRMSNormDetFunction``
           - ``CakeRMSNormNondetFunction``
         * - ``cake_rmsnorm_train_forward``
           - ``cake_rmsnorm_det_train_forward``
           - ``cake_rmsnorm_nondet_train_forward``
         * - ``cake_rmsnorm_train_backward``
           - ``cake_rmsnorm_det_train_backward``
           - ``cake_rmsnorm_nondet_train_backward``
         * - ``cake_rmsnorm_train_backward_workspace``
           - ``cake_rmsnorm_det_train_backward_workspace``
           - ``cake_rmsnorm_nondet_train_backward_workspace``
         * - ``cake_rmsnorm_train_backward_workspace_bytes``
           - ``cake_rmsnorm_det_train_backward_workspace_bytes``
           - ``cake_rmsnorm_nondet_train_backward_workspace_bytes``
         * - ``is_cake_rmsnorm_train_supported``
           - ``is_cake_rmsnorm_det_train_supported``
           - ``is_cake_rmsnorm_nondet_train_supported``
         * - package ``flashinfer.cake_rmsnorm_train_backend``
           - ``flashinfer.experimental.cake.cake_rmsnorm_det_train``
           - ``flashinfer.experimental.cake.cake_rmsnorm_nondet_train``

     Names that did not depend on ``deterministic``:

     .. list-table::
         :header-rows: 1

         * - Former name
           - Name
         * - ``route_key`` of ``flashinfer.cake_rmsnorm_train`` (v0.7.3)
           - ``cake_rmsnorm_det_train_route_key``
         * - ``route_key`` of ``flashinfer.cake_rmsnorm_train`` (v0.7.4rc1)
           - ``cake_rmsnorm_nondet_train_route_key``
         * - ``route_key``, ``program_windows`` and ``weight_dtypes`` of
             ``flashinfer.cake_rmsnorm_det_train``
           - ``cake_rmsnorm_det_train_route_key``,
             ``cake_rmsnorm_det_train_program_windows`` and
             ``cake_rmsnorm_det_train_weight_dtypes``
         * - the names without ``det`` of ``flashinfer.cake_rmsnorm_det_train``
             (v0.7.4rc1: ``cake_rmsnorm``, ``CakeRMSNormFunction``,
             ``cake_rmsnorm_train_*``, ``is_cake_rmsnorm_train_supported``)
           - the same name with ``det`` in ``flashinfer.cake_rmsnorm_det_train``
         * - the ``det`` names that ``flashinfer.cake_rmsnorm_train`` re-exported
             (v0.7.4rc1)
           - the same name in ``flashinfer.cake_rmsnorm_det_train``
         * - package ``flashinfer.cake_rmsnorm_det_train_backend``
           - ``flashinfer.experimental.cake.cake_rmsnorm_det_train``

.. currentmodule:: flashinfer.cake_rmsnorm_det_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_det
    cake_rmsnorm_det_train_forward
    cake_rmsnorm_det_train_backward
    cake_rmsnorm_det_train_backward_workspace_bytes
    cake_rmsnorm_det_train_backward_workspace
    is_cake_rmsnorm_det_train_supported
    cake_rmsnorm_det_train_program_windows
    cake_rmsnorm_det_train_weight_dtypes
    CakeRMSNormDetFunction

Training RMSNorm, non-deterministic weight gradient (Cake)
----------------------------------------------------------

The deterministic package's forward programs, rendered into this package, and a
backward with the non-deterministic weight gradient (the note above gives the
mode and dispatch rules):

* The backward runs the form its registry lists for each token-count window
  (the notes of the backend's backward,
  ``flashinfer.experimental.cake.cake_rmsnorm_nondet_train.cake_rmsnorm_nondet_train_backward``,
  list the windows). In the windows that ship the
  single-kernel form, one kernel per call computes ``dx`` row by row and adds
  the ``dw`` sums of its CTAs into FP64 sums in the workspace, in the order the
  CTAs finish: ``dw`` is not guaranteed bitwise reproducible; ``dx`` is, and it
  is within one BF16 ulp of ``max(|dx|, rms(dx))`` of the deterministic
  backward's ``dx`` (not always bitwise equal to it). In the windows those
  notes name, a twosum single kernel keeps each thread's ``dw``
  column sums exact (error-free FP32 products, TwoSum compensation) before the
  FP64 additions, so its ``dw`` error does not grow with the token count. The
  other windows run the deterministic package's backward pair (main kernel and
  fixed-order reduce), rendered into this package.
* The backward workspace (``cake_rmsnorm_nondet_train_backward_workspace_bytes``) has
  one layout with fixed offsets: the pair's counters, the single kernel's
  counters and FP64 column sums and, for the pair windows, the partial rows.
  Zeroed once, it serves any sequence of backward calls it is large enough for
  with no host write: the kernels leave the counters at rest and the sums at
  zero.
* It also accepts ``[..., H]`` inputs whose leading dimensions collapse into
  rows and an FP32 weight on the routes that ship the FP32-weight programs.

.. currentmodule:: flashinfer.cake_rmsnorm_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_nondet
    cake_rmsnorm_nondet_train_forward
    cake_rmsnorm_nondet_train_backward
    cake_rmsnorm_nondet_train_backward_workspace_bytes
    cake_rmsnorm_nondet_train_backward_workspace
    CakeRMSNormNondetFunction
    is_cake_rmsnorm_nondet_train_supported

Training SiLU-gated RMSNorm (Cake)
----------------------------------

``RMSNorm(x) * weight * SiLU(z)`` with a separate gate input ``z``, forward /
backward for training on SM100, SM103 and SM107: rows of 128 (``[T, H, 128]``
or ``[T * H, 128]``) normalized per head with FP32 statistics and gate, BF16
inputs, outputs and gradients, a weight shared by every token and head whose
gradient is summed in FP32 runs of at most 16 rows and added exactly in fixed
point (repeated calls with the same row count on a device with the same SM
count return the same bits), and an autograd wrapper that saves only its
inputs.  The forward computes ``rstd`` and the gate with the approximate FP32
reciprocal square root, base-2 exponential and reciprocal, so ``y`` can differ
from the correctly rounded FP32 formula by one BF16 unit in the last place near
a BF16 rounding boundary.  The backward computes them with the same approximate
instructions but keeps the subnormal sigmoid down to about ``-88.72``: ``dx``
and ``dz`` lie within one BF16 unit in the last place of the correctly rounded
FP32 formulas plus 1/64 of a unit of the row's largest term, the same bits at
every call.  The forward's CTAs each walk at most two row blocks, and the
backward's CTAs claim row blocks past one wave.  Beside a kernel on another
stream that holds ``k`` of the ``N`` SMs, the forward takes at most
``N / (N - k)`` times its time alone, plus one CTA's share of the row blocks,
plus 0.02.  The backward's workspace (18952 bytes, zero-filled) is left zero
by every call, so one workspace serves every later call on its stream.

.. rubric:: Mode and dispatch rules

1. The mode is part of every name.  Every public name of the deterministic
   module ``flashinfer.cake_rmsnorm_silu_gate_det_train`` (backend package
   ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train``) carries
   ``det``; every public name of the non-deterministic module
   ``flashinfer.cake_rmsnorm_silu_gate_nondet_train`` (backend package
   ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``)
   carries ``nondet``.  The module constants ``ARCHES`` and
   ``SUPPORTED_COMPUTE_CAPABILITIES`` are exempt: they keep their names,
   without a mode word, in both modules and both backend packages.  The
   ``flashinfer`` top level exports only names with a mode word.
   Each module imports its backend package on first use (``import
   flashinfer`` loads neither backend package); its autograd class and
   its architecture constants are the backend package's objects.
2. No entry point takes a ``deterministic`` keyword: passing one is a
   ``TypeError``.  A call of one package never runs the other package.
3. Under ``torch.use_deterministic_algorithms(True)`` no call of the
   deterministic package alerts: every deterministic launch gives the same
   bits for the same inputs on the same device (a sum across CTAs is exact).
   A non-deterministic backward call that computes a BF16 weight
   gradient in its fused band (see its section below) raises PyTorch's
   alert immediately before that launch, whose weight-gradient sum adds in an
   order that depends on scheduling (``RuntimeError``; with ``warn_only=True``
   a ``UserWarning``, then the non-deterministic kernel runs).  No other call of
   the non-deterministic package alerts: its other launches repeat their bits
   (their weight-gradient sum is exact, or in a fixed order, so the order of
   the additions does not change the bits).  The check runs on the host: a
   CUDA Graph checks it when it is captured, not when it is replayed.
4. Autograd: ``CakeRMSNormSiLUGateDetFunction`` and
   ``CakeRMSNormSiLUGateNondetFunction`` save only ``x``, ``z`` and ``weight``
   and run the backward of their own package.  A backward workspace serves only
   its own package (FP32 for the deterministic package, a ``torch.uint8``
   buffer for the non-deterministic one): each backward refuses the other
   package's workspace (``ValueError``), also in a call with no rows.
5. The former names are gone, with no alias or shim: the former modules fail
   at import, and the former names without a mode word are attributes of
   neither ``flashinfer`` nor either module.

The former names and their names in each mode.  In v0.7.3 the former
module had no ``deterministic`` keyword and every former name ran the
deterministic programs, so v0.7.3 callers use the deterministic column.
From v0.7.4rc1 a former entry ran the deterministic package by default
(``deterministic=True``) and the non-deterministic package with
``deterministic=False``:

.. list-table::
    :header-rows: 1

    * - Former name
      - Name (former default, ``deterministic=True``)
      - Name (former ``deterministic=False``)
    * - ``flashinfer.cake_rmsnorm_silu_gate_train`` (module)
      - ``flashinfer.cake_rmsnorm_silu_gate_det_train``
      - ``flashinfer.cake_rmsnorm_silu_gate_nondet_train``
    * - ``flashinfer.cake_rmsnorm_silu_gate_det_train_backend`` (package;
        from v0.7.4rc1)
      - ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train``
      - (none)
    * - ``flashinfer.cake_rmsnorm_silu_gate_train_backend`` (package) in
        v0.7.3, where it held the deterministic programs
      - ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train``
      - (none)
    * - ``flashinfer.cake_rmsnorm_silu_gate_train_backend`` (package) from
        v0.7.4rc1, where it took ``deterministic=False`` only
      - (none)
      - ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``
    * - ``cake_rmsnorm_silu_gate``
      - ``cake_rmsnorm_silu_gate_det``
      - ``cake_rmsnorm_silu_gate_nondet``
    * - ``CakeRMSNormSiLUGateFunction``
      - ``CakeRMSNormSiLUGateDetFunction``
      - ``CakeRMSNormSiLUGateNondetFunction``
    * - ``cake_rmsnorm_silu_gate_train_forward``
      - ``cake_rmsnorm_silu_gate_det_train_forward``
      - ``cake_rmsnorm_silu_gate_nondet_train_forward``
    * - ``cake_rmsnorm_silu_gate_train_backward``
      - ``cake_rmsnorm_silu_gate_det_train_backward``
      - ``cake_rmsnorm_silu_gate_nondet_train_backward``
    * - ``cake_rmsnorm_silu_gate_train_backward_workspace_bytes``
      - ``cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes``
      - ``cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes``
    * - ``cake_rmsnorm_silu_gate_train_backward_workspace``
      - ``cake_rmsnorm_silu_gate_det_train_backward_workspace``
      - ``cake_rmsnorm_silu_gate_nondet_train_backward_workspace``
    * - ``cake_rmsnorm_silu_gate_train_reset_workspace``
      - ``cake_rmsnorm_silu_gate_det_train_reset_workspace``
      - ``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``
    * - ``is_cake_rmsnorm_silu_gate_train_supported``
      - ``is_cake_rmsnorm_silu_gate_det_train_supported``
      - ``is_cake_rmsnorm_silu_gate_nondet_train_supported``

.. currentmodule:: flashinfer.cake_rmsnorm_silu_gate_det_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_silu_gate_det
    cake_rmsnorm_silu_gate_det_train_forward
    cake_rmsnorm_silu_gate_det_train_backward
    cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
    cake_rmsnorm_silu_gate_det_train_backward_workspace
    cake_rmsnorm_silu_gate_det_train_reset_workspace
    CakeRMSNormSiLUGateDetFunction
    is_cake_rmsnorm_silu_gate_det_train_supported

Training SiLU-gated RMSNorm, non-deterministic package (Cake)
-------------------------------------------------------------

The non-deterministic package (``nondet`` names; see the mode and dispatch
rules above). Its entry points run only the standalone non-deterministic
kernels (``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``).
``y``, ``dx``, ``dz`` and an FP32 weight gradient repeat bit for bit from run
to run everywhere.  A BF16 weight gradient repeats bit for bit too, except in
the fused band (below: past 512 rows on SM100, past 256 rows on SM103, up to
1000 rows per SM; on SM107 past 64 up to 256 rows per SM), where its last
bits may vary from run to run.  A band runs a backward whose weight gradient
may vary only where that makes the checkpointed training step (2 x forward +
backward) at least 3 % faster than the repeatable backward, which runs past
the fused band (on SM107 also before it, past 512 rows up to 64 rows per
SM).  ``y`` is the
deterministic package's bits where a pipelined forward runs (below) and not
elsewhere; ``dx`` and ``dz`` are the same bits in every band but the fused and
the repeatable ones (below: a BF16 weight gradient), within one BF16 unit in
the last place of the FP64 value plus 1/64 of a unit of the tensor's largest
finite magnitude in every band (the deterministic package's backward need not
give these bits, but in the fused and the repeatable bands they are its bits
in this release), and outside the short backward's band (a BF16 weight
gradient up to 256 rows), the fused band and the repeatable band ``dw`` is the
bits of the deterministic package at FlashInfer commit 2e989556, except where
the exact sum lies within that package's FP64 sum's error of a rounding
boundary.  In the repeatable band ``dx``, ``dz`` and ``dw`` are the bits of
the deterministic package in this release, the same for the rows and the SM
count.  In the short
backward's band ``dw`` is that kernel's own fixed-order sum, the same bits on
every call, which need not be the deterministic package's.  The FP64 error of
each output (relative L2 and the largest error relative to the largest
reference value) is at most 1.1 times the deterministic package's on the same
inputs, and within the deterministic package's absolute bounds; for an FP32
weight gradient the 1.1 times bound holds against the deterministic package at
FlashInfer 2e989556, and against the deterministic package in this release
(whose FP32 weight gradient need not be these bits) the bound is
``|dw_det| * 2^-7 + sum(|terms|) * 2^-20``.

* The forward uses approximate FP32 intrinsics (``rsqrt.approx`` for
  ``rstd``; ``ex2.approx`` and ``rcp.approx`` for the sigmoid, the
  deterministic forward's split gate, nonzero down to ``z`` about -88.72).
  It runs one
  of several modules chosen by bands of rows per SM: where ``x`` and ``z``
  hold packed rows (row ``n`` at element ``128 n``, as contiguous inputs do;
  fewer than 2^24 rows) a tiny forward up to 38 rows per SM on SM100, 36 on
  SM103 and 65 on SM107, one 128-thread CTA per 8-row block; for other
  layouts a short forward up to 12 rows per SM on SM100 and SM103 (59 on
  SM107), one CTA per 16-row block.  Past them the bands run the mid forward
  (whole waves of 6 CTAs per SM, each CTA at most 2 blocks of 16 rows), the
  long forward (whole waves of 3 CTAs per SM, each CTA at most 6 blocks of
  32 rows), the extra-long forward (the long forward's grid, also prefetching
  its next block), the pipelined forward, this package's copy of the
  deterministic package's forward (128 threads, one CTA per 16-row block
  below 8192 rows per SM, from there whole waves of 6 CTAs per SM, each CTA
  at most 2 blocks; it loads ``x`` and ``z`` with the L2 ``evict_first``
  policy), or for other layouts the plain forward (one 64-thread CTA per
  8-row block).  The cuts follow the checkpointed training step (two
  forwards and the backward), since where the rows fit in L2 the pipelined
  forward is faster alone but makes the step slower.  Packed rows on SM100:
  pipelined up to 85, mid up to 160, long up to 400, extra-long up to 520,
  pipelined up to 6500, mid past it; on SM103 mid up to 240, extra-long up
  to 600, pipelined up to 6000, mid up to 8192, pipelined past it; on SM107
  mid up to 175, extra-long up to 280, pipelined past it (where alone it is
  about 10 % faster and the step at most about 3.5 % slower).  Other layouts
  on SM100: mid up to 140, long up to 400, extra-long up to 700, the plain
  forward past it; on SM103 mid up to 250, extra-long up to 700, the plain
  forward past it; on SM107 mid up to 480, the plain forward past it.  Each
  module's arithmetic is fixed, but the modules associate a
  row's sum of squares differently (the tiny and the short forward: 8-term
  FMA chains of 16 lanes folded by XOR shuffles 1, 2, 4, 8; the others:
  16-term chains of 8 lanes folded by 1, 2, 4, on SM107 the pipelined
  forward as four interleaved chains), so ``y``'s bits can differ between
  bands, and ``y`` repeats bit for bit for the same inputs on the same
  device.  Beside a kernel on another stream that holds ``k`` of the ``N``
  SMs, the pipelined, mid, long and extra-long forwards take at most ``N / (N
  - k)`` times their time alone, plus one CTA's share of the row blocks, plus
  0.02, and the tiny and the short forward end no later than the
  deterministic package's forward (within 2 % or 1.5 microseconds).  On
  SM100 and SM103 with fewer than 64 SMs and on SM107 with fewer than 205, as
  on a small MIG instance, the short forward's grid at the end of its band has
  more CTAs than the SMs beside 16 held ones take in one wave, and this bound
  of the short forward is not shown there.
* The backward is one kernel.  ``dx`` and ``dz`` use one row arithmetic in
  every band but the fused and the repeatable ones.  A call of few rows runs
  one thread-block cluster of 16 CTAs that sums the weight gradient on chip in
  a fixed order and takes no workspace.  The short backward (256-thread CTAs
  with the row-claims backward's row arithmetic; CTA ``r`` adds every CTA's
  FP32 partials of the columns ``8r`` to ``8r + 7`` in rank order in FP64)
  runs a BF16 weight gradient (or ``compute_grad_weight=False``) up to 256
  rows.  The exact tiny backward (4 teams of the 128-thread CTA of the
  deterministic backward at FlashInfer 2e989556 a CTA, adding that backward's
  chunk partials in its reduction order in FP64) runs an FP32 weight gradient
  up to 512 rows on every architecture, and a BF16 one (or
  ``compute_grad_weight=False``) from 257 up to 512 rows on SM100 and SM107;
  SM103 has no BF16 exact tiny backward.  Past those rows the rows (``T * H``)
  and the SM count pick the kernel by bands of rows per SM.  The chunked
  backwards launch at most 8 CTAs per SM; they sum the chunk partials of the
  deterministic backward at FlashInfer 2e989556 exactly in 64-bit integer
  windows and round that sum to FP32 from its FP64 value,
  so the weight gradient does not depend on the grid or on the order in which
  CTAs finish; they run an FP32 weight gradient.  A static backward (CTA
  ``b`` runs the chunks ``b``, ``b + grid``, ...) runs up to 256 rows per SM
  on SM100 and SM103 and up to 64 on SM107; on SM107 it keeps 4 copies of the
  weight-gradient windows plus occupancy words.  Past the static band a
  claimed backward runs, whose CTAs claim runs of chunks at run time, so a CTA slowed
  by a kernel on another stream takes fewer; on SM100 and SM103 it warms each
  CTA's next block in L2 with one bulk prefetch per tensor where ``dy``,
  ``x`` and ``z`` hold packed rows, on SM107 the block two ahead.
* Past the one-cluster backwards a BF16 weight gradient (or
  ``compute_grad_weight=False``) runs the fused backward on every
  architecture: past 512 rows on SM100, past 256 rows on SM103, up to 1000
  rows per SM, and on SM107 past 64 up to 256 rows per SM, where its varying
  weight gradient is to make the checkpointed step at least 3 % faster than
  the repeatable backward.  An
  FP32 weight gradient runs no fused backward.  It runs the deterministic
  package's fused backward's schedule and row arithmetic: 256-thread CTAs (16
  lanes a row, 16-row blocks, ``min(ceil(rows / 16), 4 * SMs)`` of them),
  which claim the blocks in chunks when there are more blocks than CTAs, with
  that backward's approximate gate and ``rstd``, so ``dx`` and ``dz`` are the
  deterministic package's bits in this release at any grid.  It adds each
  thread's FP32 runs of at most 16 rows, folded into FP32 two-sum pairs, to
  FP64 accumulators with atomics, so the order of the sum, and the last bits
  of the BF16 weight gradient, depend on scheduling, within the family's
  error budget.  The BF16 bands
  up to 512 rows follow measurements of a checkpointed step (FlashInfer's
  benchmark timer, cold L2): on GB300 (SM103) the step with the short
  backward is 7.8 % to 14.8 % faster than with the faster of the exact tiny
  and the row-claims backward at 48 to 256 rows (13.1 % to 18.7 % faster
  than with the exact tiny backward), and the row-claims backward saves
  6.9 % to 13.1 % of the step against the exact tiny backward at 336 to 512
  rows, so SM103 runs no BF16 exact tiny backward; on B200 (SM100) the steps
  with the short and with the exact tiny backward are within 3 % of each
  other at up to 256 rows, and under FlashInfer's benchmark timer the short
  backward alone is the faster of the two at every row count up to 256.  On
  R200 (SM107) the step with the exact tiny backward takes 2.9 % to 6.4 %
  longer than with the short backward at 96 to 256 rows (the mean of 4
  passes; longer in every pass) and 1.1 % less at 48 rows, within that row
  count's pass-to-pass spread; at 336 to 512 rows the step with the
  row-claims backward takes 18.7 % to 24.5 % longer than with the exact tiny
  backward, so SM107 runs the exact tiny backward from 257 to 512 rows.
  Under
  ``torch.use_deterministic_algorithms(True)`` a call that computes the
  weight gradient there raises PyTorch's alert (rule 3); no other call
  alerts.
* Past 1000 rows per SM a BF16 weight gradient (or
  ``compute_grad_weight=False``) runs the repeatable backward on SM100 and
  SM103 (on SM107 past 512 rows up to 64 rows per SM and past 256 rows per
  SM), this package's copy of the deterministic package's BF16
  backward: the fused backward's grid and row arithmetic, its CTAs claiming
  the blocks in chunks whose rows are fixed by the row count and the grid,
  each chunk's weight-gradient terms added exactly into fixed-point digits in
  the workspace and the sum rounded once by the CTA that arrives last.  Its
  ``dx``, ``dz`` and weight gradient are the deterministic package's bits in
  this release, the same for the rows and the SM count, within the same
  bounds; no call in this band alerts.  Packed rows (``grad_out``, ``x`` and
  ``z`` each with head stride 128 and token stride ``H * 128``) run its
  copy of the deterministic package's look-ahead program, the same build,
  from 2560 rows per SM on SM100 and SM103 and from 1536 on SM107, with the
  same bits.
* The workspace (``cake_rmsnorm_silu_gate_nondet_train_backward_workspace``)
  is a zero-filled ``torch.uint8`` tensor of 29192 bytes on SM100 and SM103
  and 36360 bytes on SM107, whatever the token count (the largest layout of
  the architecture's backward kernels; the fused backward lays its FP64
  accumulators and two counter words in the first 4104 bytes, the repeatable
  backward its fixed-point digits, a flag word a column and two counter words
  in the first 18952 bytes), that serves
  every backward kernel of its architecture and is zero after every call: zero it
  once and reuse it, one per
  concurrently running stream.  A buffer that anything else wrote into is
  zeroed with ``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``.  Calls
  without a workspace use one per stream and backward module that takes one,
  kept for the process, and such eager calls also keep a reserve of
  zero-filled workspaces per device for CUDA Graph captures (about 1.8 MiB per
  64 workspaces on SM100 and SM103, 2.2 MiB on SM107, at most 1024).  The autograd wrapper saves ``x``, ``z`` and
  ``weight`` by reference and ``eps``, no intermediate; no workspace is saved
  state.

.. currentmodule:: flashinfer.cake_rmsnorm_silu_gate_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_silu_gate_nondet
    cake_rmsnorm_silu_gate_nondet_train_forward
    cake_rmsnorm_silu_gate_nondet_train_backward
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace
    CakeRMSNormSiLUGateNondetFunction
    is_cake_rmsnorm_silu_gate_nondet_train_supported
