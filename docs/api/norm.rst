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
gradient is summed in a fixed order (repeated calls return the same bits), and
an autograd wrapper that saves only its inputs.  The forward runs one of several
fixed-order modules chosen from the row count and the SM count (a short forward
up to 12 rows per SM, 59 on SM107; from 976 rows per SM, 800 on SM107,
persistent forwards whose CTAs each walk a few row blocks); every module returns
the same bits.  Beside a kernel on another stream that holds ``k`` of the ``N``
SMs, a persistent forward takes at most ``N / (N - k)`` times its time alone,
plus one CTA's share of the row blocks, plus 0.02, and the short forward ends
no later than the plain forward (within 2 % or 1.5 microseconds).
On SM107 with fewer than 205 SMs, as on a small MIG instance, the short
forward's grid at 59 rows per SM is more than one wave, and this bound of the
short forward is not shown there.

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
3. Under ``torch.use_deterministic_algorithms(True)`` no call of either
   package alerts: every deterministic launch sums in a fixed order, and every
   output of the non-deterministic package repeats bit for bit (its
   weight-gradient sum is exact, so the order of its additions does not change
   the bits).
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

Training SiLU-gated RMSNorm, non-deterministic weight gradient (Cake)
---------------------------------------------------------------------

The non-deterministic package (``nondet`` names; see the mode and dispatch
rules above). Its entry points run only the standalone non-deterministic
kernels (``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``).
Every output (``y``, ``dx``, ``dz`` and ``dw``) repeats bit for bit from run to
run.  ``y`` is not the deterministic package's bits; ``dx`` and ``dz`` are, and
``dw`` is except where the exact sum lies within the deterministic FP64 sum's
error of a rounding boundary.  The FP64 error of
each output (relative L2 and the largest error relative to the largest
reference value) is at most 1.1 times the deterministic package's on the same
inputs, and within the deterministic package's absolute bounds.

* The forward uses approximate FP32 intrinsics (``rsqrt.approx`` for
  ``rstd``; ``ex2.approx`` and ``rcp.approx`` for the sigmoid).  It runs one
  of several modules chosen from the row count and the SM count: a short
  forward up to 12 rows per SM on SM100 and SM103 (59 on SM107), the plain
  forward, from 976 rows per SM (800 on SM107) a persistent forward whose CTAs
  each walk at most 2 row blocks, and from 2400 rows per SM on SM100 and SM103
  a long forward (on SM103 from 3000 and on SM107 from 1740, an extra-long
  forward that also prefetches its next block).  Each module's arithmetic is
  fixed, but the modules may associate a row's sum of squares differently, so
  ``y`` repeats bit for bit for the same inputs on the same device.
* The backward is one kernel.  ``dx`` and ``dz`` use the deterministic
  backward's row arithmetic.  The weight gradient sums the deterministic
  backward's chunk partials exactly in 64-bit integer windows and rounds once,
  so it does not depend on the grid or on the order in which CTAs finish.  Up
  to 64 rows per SM every chunk has its own CTA; past that the CTAs claim runs
  of chunks at run time, so a CTA slowed by a kernel on another stream takes
  fewer.  Up to 256 rows with a BF16 weight gradient, or with
  ``compute_grad_weight=False``, the backward is one thread-block cluster of
  16 CTAs that sums on chip and takes no workspace.
* The workspace (``cake_rmsnorm_silu_gate_nondet_train_backward_workspace``)
  is a zero-filled ``torch.uint8`` tensor of 29192 bytes whatever the token
  count, zero after every call: zero it once and reuse it, one per
  concurrently running stream.  A buffer that anything else wrote into is
  zeroed with ``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``.  Calls
  without a workspace use one per stream and backward module that takes one,
  kept for the
  process, and each device keeps a reserve of zero-filled workspaces for CUDA
  Graph captures (about 1.8 MiB per 64 workspaces, at most 1024).

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
