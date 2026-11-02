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

.. currentmodule:: flashinfer.cake_rmsnorm_det_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_det
    cake_rmsnorm_det_train_forward
    cake_rmsnorm_det_train_backward
    cake_rmsnorm_det_train_backward_workspace_bytes
    cake_rmsnorm_det_train_backward_workspace
    program_windows
    weight_dtypes
    CakeRMSNormDetFunction

Training RMSNorm, opt-in non-deterministic weight gradient (Cake)
-----------------------------------------------------------------

The names the training RMSNorm had before the deterministic package took its
``det`` names. Every entry point takes ``deterministic`` (default ``True``):

* ``deterministic=True`` runs the deterministic package above and returns its
  bits.
* ``deterministic=False`` runs this package's backward in the form its
  registry lists for each token-count window (the backward's docstring lists
  the windows). In the windows that ship the single-kernel form, one kernel per
  call computes ``dx`` row by row and adds the ``dw`` sums of its CTAs into
  FP64 sums in the workspace, in the order the CTAs finish: ``dw`` is not
  guaranteed bitwise reproducible; ``dx`` is, and it is within one BF16 ulp of
  max(|dx|, rms(dx)) of the deterministic backward's ``dx`` (not always bitwise
  equal to it). In the windows the backward's notes name, a twosum single kernel
  keeps each thread's ``dw`` column sums exact (error-free FP32 products, TwoSum
  compensation) before the FP64 additions, so its ``dw`` error does not grow
  with the token count. The
  other windows run the deterministic package's backward pair (main kernel and
  fixed-order reduce), rendered into this package. The backward workspace
  (``cake_rmsnorm_train_backward_workspace_bytes(..., deterministic=False)``)
  has one layout with fixed offsets: the pair's counters, the single kernel's
  counters and FP64 column sums and, for the pair windows, the partial rows.
  Zeroed once, it serves any sequence of ``deterministic=False`` calls it is
  large enough for with no host write: the kernels leave the counters at rest
  and the sums at zero. It serves calls of one mode.
* Under ``torch.use_deterministic_algorithms(True)``, ``deterministic=False``
  raises ``RuntimeError``. With ``warn_only=True`` it warns; the forward runs
  this package's (deterministic) forward programs and the backward runs the
  deterministic package with its own workspace (a caller's ``workspace`` is
  not used), or raises one ``RuntimeError`` when that package cannot serve the
  call (``[..., H]`` inputs, an FP32 weight, a width or residual route it does
  not export). The workspace helpers with ``deterministic=False`` warn and
  return the deterministic package's workspace in that mode.

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

.. list-table:: Names before the ``det`` rename and their deterministic names
    :header-rows: 1

    * - Name (``deterministic=True`` by default)
      - Deterministic name (always deterministic)
    * - ``flashinfer.cake_rmsnorm_train``
      - ``flashinfer.cake_rmsnorm_det_train``
    * - ``cake_rmsnorm``
      - ``cake_rmsnorm_det``
    * - ``cake_rmsnorm_train_forward``
      - ``cake_rmsnorm_det_train_forward``
    * - ``cake_rmsnorm_train_backward``
      - ``cake_rmsnorm_det_train_backward``
    * - ``cake_rmsnorm_train_backward_workspace_bytes``
      - ``cake_rmsnorm_det_train_backward_workspace_bytes``
    * - ``cake_rmsnorm_train_backward_workspace``
      - ``cake_rmsnorm_det_train_backward_workspace``
    * - ``CakeRMSNormFunction``
      - ``CakeRMSNormDetFunction``
    * - ``is_cake_rmsnorm_train_supported``
      - ``is_cake_rmsnorm_det_train_supported``

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
  kernel (on SM100 and SM103 up to 10 rows per SM, a forward with 16 threads
  to a row, and from 976 rows per SM, a forward whose CTAs each walk several
  row blocks: the same bits) and a one-kernel backward: ``dx``
  and ``dz`` are the deterministic backward's bits, and the CTAs take row
  blocks (each CTA its own block when
  the grid has a CTA for every block; past one wave of CTAs they claim the
  blocks in chunks from the workspace's claim counter) and add their FP64
  weight-gradient column sums into FP64 accumulators of the workspace in the
  order they finish. The claimed blocks, and so the FP32 partial sums, and that
  order depend on scheduling: the weight gradient can differ between calls,
  always within the deterministic backward's error bound. The workspace
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
