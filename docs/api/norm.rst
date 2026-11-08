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
gradient is summed so that repeated calls return the same bits, and an
autograd wrapper that saves only its inputs.  The deterministic forward runs
one of several fixed-order modules chosen from the row count and the SM count
(a short forward up to 12 rows per SM, 59 on SM107; the plain forward up to
975 rows per SM, 799 on SM107; from 976 rows per SM, 800 on SM107,
persistent forwards whose CTAs each walk a few row blocks); every module
returns the same bits.  Beside a kernel on another stream that holds ``k`` of
the ``N`` SMs, a persistent forward takes at most ``N / (N - k)`` times its
time alone, plus one CTA's share of the row blocks, plus 0.02, and the short
forward ends no later than the plain forward (within 2 % or 1.5
microseconds).  On SM107 with fewer than 205 SMs, as on a small MIG instance,
the short forward's grid at 59 rows per SM is more than one wave, and this
bound of the short forward is not shown there.  The non-deterministic
package's forward bands are in its section below.

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
   deterministic package alerts: every deterministic launch sums in a fixed
   order.  A non-deterministic backward call that computes a BF16 weight
   gradient in its row-claims bands (see its section below) raises PyTorch's
   alert immediately before that launch, whose weight-gradient sum adds in an
   order that depends on scheduling (``RuntimeError``; with ``warn_only=True``
   a ``UserWarning``, then the non-deterministic kernel runs).  No other call of
   the non-deterministic package alerts: its other launches repeat their bits
   (their weight-gradient sum is exact, or in a fixed order, so the order of
   the additions does not change the bits).  The check runs on the host: a
   CUDA Graph checks it when it is captured, not when it is replayed.
4. Autograd: ``CakeRMSNormSiLUGateDetFunction`` and
   ``CakeRMSNormSiLUGateNondetFunction`` save only ``x``, ``z`` and ``weight``
   and run the backward of their own package.  A backward workspace holds the
   raw partial sums of one package and serves only that package (FP32 for the
   deterministic package, a ``torch.uint8`` buffer for the non-deterministic
   one).  The non-deterministic backward refuses the deterministic package's
   workspace (``ValueError``) in every call, also with no rows; the
   deterministic backward refuses the non-deterministic package's workspace in
   a call with rows.
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
      - (none: the deterministic workspace needs no reset)
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
    CakeRMSNormSiLUGateDetFunction
    is_cake_rmsnorm_silu_gate_det_train_supported

Training SiLU-gated RMSNorm, non-deterministic package (Cake)
-------------------------------------------------------------

The non-deterministic package (``nondet`` names; see the mode and dispatch
rules above). Its entry points run only the standalone non-deterministic
kernels (``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``).
``y``, ``dx``, ``dz`` and an FP32 weight gradient repeat bit for bit from run
to run everywhere.  A BF16 weight gradient repeats bit for bit too, except in
the row-claims bands (below: SM100 513 rows to 1329 rows per SM, SM103 513
rows to 1294 rows per SM, SM107 above 24 to 68 rows per SM), where its last
bits may vary from run to run.  ``y`` is not the deterministic package's bits;
``dx`` and ``dz`` are in every band, and outside the row-claims bands ``dw`` is
too, except where the exact sum lies within the deterministic FP64 sum's
error of a rounding boundary.  The FP64 error of
each output (relative L2 and the largest error relative to the largest
reference value) is at most 1.1 times the deterministic package's on the same
inputs, and within the deterministic package's absolute bounds.

* The forward uses approximate FP32 intrinsics (``rsqrt.approx`` for
  ``rstd``; ``ex2.approx`` and ``rcp.approx`` for the sigmoid).  It runs one
  of several modules chosen by bands of rows per SM: where ``x`` and ``z``
  hold packed rows (row ``n`` at element ``128 n``, as contiguous inputs do)
  a tiny forward up to 140 rows per SM on SM100, 120 on SM103 and 65 on
  SM107, one 128-thread CTA per 8-row block; for other layouts a short
  forward up to 12 rows per SM on SM100 and SM103 (59 on SM107), one CTA per
  16-row block; past that the mid forward (whole waves of 6 CTAs per SM, each CTA at most 2
  blocks of 16 rows), the long forward (whole waves of 3 CTAs per SM, each
  CTA at most 6 blocks of 32 rows) or the extra-long forward (the long
  forward's grid, also prefetching its next block); the plain forward (one
  64-thread CTA per 8-row block) is in no band.  The bands past the tiny or
  the short forward: on SM100 long up to 400 rows per SM (other layouts: mid
  up to 140, then long), extra-long up to 1000, mid past it; on SM103 mid up
  to 250, extra-long up to 550, mid past it; on SM107 mid up to 200,
  extra-long up to 520, mid past it.  Each module's arithmetic is fixed, but
  the modules associate a row's sum of squares differently (the tiny and the
  short forward: 8-term FMA chains of 16 lanes folded by XOR shuffles 1, 2, 4,
  8; the others: 16-term chains of 8 lanes folded by 1, 2, 4), so ``y``'s bits
  can differ between bands, and ``y`` repeats bit for bit for the same inputs
  on the same device.  Beside a kernel on another stream that holds ``k`` of
  the ``N`` SMs, the mid, long and extra-long forwards take at most ``N / (N
  - k)`` times their time alone, plus one CTA's share of the row blocks, plus
  0.02, and the tiny and the short forward end no later than the
  deterministic package's plain forward (within 2 % or 1.5 microseconds).  On
  SM100 and SM103 with fewer than 64 SMs and on SM107 with fewer than 205, as
  on a small MIG instance, the short forward's grid at the end of its band has
  more CTAs than the SMs beside 16 held ones take in one wave, and this bound
  of the short forward is not shown there.
* The backward is one kernel.  ``dx`` and ``dz`` use the deterministic
  backward's row arithmetic in every band.  Up to 512 rows, with a BF16 or an
  FP32 weight gradient or with ``compute_grad_weight=False``, the backward is
  one thread-block cluster of 16 CTAs that sums the weight gradient on chip
  in a fixed order and takes no workspace.  Past that the rows (``T * H``)
  and the SM count pick the kernel by bands of rows per SM.  The chunked
  backwards launch at most 8 CTAs per SM; they sum the deterministic
  backward's chunk partials exactly in 64-bit integer windows and round that
  sum to FP32 from its FP64 value (then to BF16 for a BF16 weight gradient),
  so the weight gradient does not depend on the grid or on the order in which
  CTAs finish.  A static backward (CTA ``b`` runs the chunks ``b``, ``b +
  grid``, ...) runs an FP32 weight gradient up to 256 rows per SM on SM100 and
  SM103 and up to 64 on SM107, and a BF16 weight gradient up to 24 rows per SM
  on SM107; on SM107 it keeps 4 copies of the weight-gradient windows plus
  occupancy words.  Past the static and the row-claims bands a claimed
  backward runs, whose CTAs claim runs of chunks at run time, so a CTA slowed
  by a kernel on another stream takes fewer; on SM100 and SM103 it warms each
  CTA's next block in L2 with one bulk prefetch per tensor where ``dy``,
  ``x`` and ``z`` hold packed rows, on SM107 the block two ahead.
* A BF16 weight gradient runs a row-claims backward where it saves at least
  3 % of a checkpointed training step: on SM100 from 513 rows up to 1329 rows
  per SM (the row-claims backward up to 800 rows per SM, the row-claims mid
  backward past it), on SM103 from 513 rows up to 1294 rows per SM (the
  row-claims backward up to 700 rows per SM, the row-claims mid backward past
  it), on SM107 above 24 up to 68 rows per SM (the row-claims backward).
  Its 256-thread CTAs take 16-row blocks (4 CTAs per SM, 5 for the
  row-claims mid backward; past one wave they claim blocks from the
  workspace's claim counter) and add each thread's FP32 run of rows to
  FP64 accumulators with atomics (the row-claims mid backward: FP32 two-sum
  pairs, added once as FP64), so the order of the sum, and the last bits of
  the BF16 weight gradient, depend on scheduling, within the family's error
  budget.  As currently measured (CUPTI, cold L2) these bands save about
  3.2 % to 16.3 % of a checkpointed step (2 x forward + backward with a BF16
  weight gradient) on SM100, about 4.0 % (an estimate) to 18.4 %
  (provisional) on SM103 and about 3.3 % to 5.5 % on SM107.  Under
  ``torch.use_deterministic_algorithms(True)`` a call that computes the
  weight gradient there raises PyTorch's alert (rule 3); no other call
  alerts.
* The workspace (``cake_rmsnorm_silu_gate_nondet_train_backward_workspace``)
  is a zero-filled ``torch.uint8`` tensor of 29192 bytes on SM100 and SM103
  and 36360 bytes on SM107, whatever the token count (the largest layout of
  the architecture's backward kernels; a row-claims backward lays its FP64
  accumulators and two counter words in the first 4104 bytes), that serves
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
