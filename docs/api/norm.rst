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

   * ``flashinfer.cake_rmsnorm_det_train`` (this section) is the deterministic
     package: its weight gradient ``dw`` is bitwise reproducible and every
     forward, backward, workspace, support-query and autograd entry point name
     carries ``det``. Its entry points take no ``deterministic`` keyword
     (passing one raises ``TypeError``), and
     ``torch.use_deterministic_algorithms`` changes nothing for them.
   * ``flashinfer.cake_rmsnorm_train`` (next section) is the non-deterministic
     package (names without ``det``). Every entry point requires
     ``deterministic=False``: any other value, the default ``True`` included,
     raises ``ValueError`` naming the matching ``det`` entry point. A ``False``
     default would silently switch callers of the former deterministic default
     to the non-deterministic kernels; the explicit ``False`` keeps the mode
     visible at every call, for example
     ``cake_rmsnorm(x, w, eps, deterministic=False)``.
   * Under ``torch.use_deterministic_algorithms(True)`` the non-deterministic
     package alerts only before work whose result can vary from run to run:
     every backward call (and the backward of the autograd ``cake_rmsnorm`` /
     ``CakeRMSNormFunction``, when it runs) raises ``RuntimeError`` naming the
     ``det`` entry point, or, with ``warn_only=True``, warns once and runs the
     non-deterministic backward. The forward (no cross-row reduction, bitwise
     reproducible), the workspace helpers and
     ``is_cake_rmsnorm_train_supported`` never alert.
   * The ``flashinfer`` top level binds only the ``det`` names
     (``cake_rmsnorm_det``, ``CakeRMSNormDetFunction``,
     ``cake_rmsnorm_det_train_forward``, ``cake_rmsnorm_det_train_backward``,
     ``cake_rmsnorm_det_train_backward_workspace_bytes``). The non-deterministic
     package is reached only as the module ``flashinfer.cake_rmsnorm_train``:
     the top-level ``cake_rmsnorm``, ``CakeRMSNormFunction`` and
     ``cake_rmsnorm_train_*`` names are removed, so code that imports them fails
     at import.
   * There are no aliases: a name without ``det`` never runs the deterministic
     package, and neither package runs the other.

.. currentmodule:: flashinfer.cake_rmsnorm_det_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_det
    cake_rmsnorm_det_train_forward
    cake_rmsnorm_det_train_backward
    cake_rmsnorm_det_train_backward_workspace_bytes
    cake_rmsnorm_det_train_backward_workspace
    is_cake_rmsnorm_det_train_supported
    program_windows
    weight_dtypes
    CakeRMSNormDetFunction

Training RMSNorm, non-deterministic weight gradient (Cake)
----------------------------------------------------------

The deterministic package's forward programs, rendered into this package, and a
backward with the non-deterministic weight gradient (the note above gives the
mode and dispatch rules: every call passes ``deterministic=False``):

* The backward runs the form its registry lists for each token-count window
  (the backward's docstring lists the windows). In the windows that ship the
  single-kernel form, one kernel per call computes ``dx`` row by row and adds
  the ``dw`` sums of its CTAs into FP64 sums in the workspace, in the order the
  CTAs finish: ``dw`` is not guaranteed bitwise reproducible; ``dx`` is, and it
  is within one BF16 ulp of ``max(|dx|, rms(dx))`` of the deterministic
  backward's ``dx`` (not always bitwise equal to it). In the windows the
  backward's notes name, a twosum single kernel keeps each thread's ``dw``
  column sums exact (error-free FP32 products, TwoSum compensation) before the
  FP64 additions, so its ``dw`` error does not grow with the token count. The
  other windows run the deterministic package's backward pair (main kernel and
  fixed-order reduce), rendered into this package.
* The backward workspace (``cake_rmsnorm_train_backward_workspace_bytes``) has
  one layout with fixed offsets: the pair's counters, the single kernel's
  counters and FP64 column sums and, for the pair windows, the partial rows.
  Zeroed once, it serves any sequence of backward calls it is large enough for
  with no host write: the kernels leave the counters at rest and the sums at
  zero.
* It also accepts ``[..., H]`` inputs whose leading dimensions collapse into
  rows and an FP32 weight on the routes that ship the FP32-weight programs.

.. currentmodule:: flashinfer.cake_rmsnorm_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm
    cake_rmsnorm_train_forward
    cake_rmsnorm_train_backward
    cake_rmsnorm_train_backward_workspace_bytes
    cake_rmsnorm_train_backward_workspace
    CakeRMSNormFunction
    is_cake_rmsnorm_train_supported

Training SiLU-gated RMSNorm (Cake)
----------------------------------

``RMSNorm(x) * weight * SiLU(z)`` with a separate gate input ``z``, forward /
backward for training on SM100, SM103 and SM107: rows of 128 (``[T, H, 128]``
or ``[T * H, 128]``) normalized per head with FP32 statistics and gate, BF16
inputs, outputs and gradients, a weight shared by every token and head whose
gradient is summed in a fixed order (repeated calls return the same bits), and
an autograd wrapper that saves only its inputs.

.. currentmodule:: flashinfer.cake_rmsnorm_silu_gate_det_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_silu_gate_det
    cake_rmsnorm_silu_gate_det_train_forward
    cake_rmsnorm_silu_gate_det_train_backward
    cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
    cake_rmsnorm_silu_gate_det_train_backward_workspace
    CakeRMSNormSiLUGateDetFunction

Training SiLU-gated RMSNorm, opt-in non-deterministic weight gradient (Cake)
---------------------------------------------------------------------------

The names the training SiLU-gated RMSNorm had before the deterministic package
took its ``det`` names. Every entry point takes the keyword ``deterministic``
(default ``True``; a value that is not a ``bool`` raises ``TypeError``), and
``CakeRMSNormSiLUGateFunction`` takes it as its last forward input:

* ``deterministic=True`` runs the deterministic package above and returns its
  bits.
* ``deterministic=False`` runs the standalone non-deterministic package
  (``flashinfer.cake_rmsnorm_silu_gate_train_backend``, whose compute and
  workspace entry points take ``deterministic=False`` only): the same forward
  kernel (up to 12 rows per SM on SM100 and SM103 and up to 59 on SM107, a
  forward with 16 threads to a row, and from 976 rows per SM on SM100 and
  SM103 and from 800 on SM107,
  a forward whose CTAs each walk several row blocks: the same bits) and a
  one-kernel backward: ``dx`` and ``dz`` are the deterministic backward's
  bits, and the CTAs take row blocks (each CTA its own block when
  the grid has a CTA for every block; past one wave of CTAs they claim the
  blocks in chunks from the workspace's claim counter) and add their FP64
  weight-gradient column sums into FP64 accumulators of the workspace in the
  order they finish. The claimed blocks, and so the FP32 partial sums, and that
  order depend on scheduling: the weight gradient can differ between calls,
  always within the deterministic backward's error bound (for FP32 normal
  gates, terms and sums; with subnormal values both backwards can leave it).
  Up to 256 rows
  the backward is one thread-block cluster of 16 CTAs that adds
  the CTAs' partial sums in a fixed order on chip: the same bits on every
  call, within the same bound, and no workspace. The workspace
  (``cake_rmsnorm_silu_gate_train_backward_workspace(..., deterministic=False)``,
  a few KiB whatever the token count) is zero after every call: zero it once
  and reuse it, one per concurrently running stream. A buffer that anything
  else wrote into is zeroed with
  ``cake_rmsnorm_silu_gate_train_reset_workspace(..., deterministic=False)``.
* Under ``torch.use_deterministic_algorithms(True)``, ``deterministic=False``
  raises ``RuntimeError`` from the compute, workspace and autograd entry
  points; with ``warn_only=True`` they warn and run the deterministic kernels.
  The backward of a ``deterministic=False`` autograd node that runs after the
  mode was enabled raises PyTorch's alert (``warn_only=True``: a warning, then
  the non-deterministic backward). The support query and the workspace reset
  do not depend on that setting.

.. currentmodule:: flashinfer.cake_rmsnorm_silu_gate_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_silu_gate
    cake_rmsnorm_silu_gate_train_forward
    cake_rmsnorm_silu_gate_train_backward
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes
    cake_rmsnorm_silu_gate_train_backward_workspace
    cake_rmsnorm_silu_gate_train_reset_workspace
    CakeRMSNormSiLUGateFunction
    is_cake_rmsnorm_silu_gate_train_supported
