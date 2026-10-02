.. _apicake_moe_grouped_gemm:

flashinfer.experimental.cake_moe_grouped_gemm
==============================================

.. currentmodule:: flashinfer.experimental.cake_moe_grouped_gemm

``cake_moe_grouped_gemm`` is an experimental Cake backend: generated ragged BF16
grouped GEMM programs for the expert projections of Mixture-of-Experts layers on
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200).  Both the API and the
backend are experimental (see :ref:`experimental`); calling one of the functions
below, or naming the backend in the stable API, is the explicit opt-in and emits an
``ExperimentalWarning`` once.

Groups are consecutive row ranges of the packed activations, described by an int32
**device** tensor: either ``offs[E]`` of cumulative end offsets or ``m_indptr[E + 1]``
with a leading zero (the ``torch._grouped_mm`` and :func:`flashinfer.grouped_mm.grouped_mm_bf16`
conventions).  The kernels read the offsets on the device; the host never synchronises
on them, no padding is required and group sizes may be imbalanced, empty, one row or
any size that is not a tile multiple.  All operands are bfloat16 with fp32 tensor-core
accumulation; the reductions are bitwise deterministic (no atomics) and empty groups
produce exactly zero weight gradients.

============ ================================================================== ============================
operation    computation                                                         output
============ ================================================================== ============================
``fwd``      ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``             ``[sum_m, N]`` bfloat16
``dgrad``    ``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` (W in place)  ``[sum_m, K]`` bfloat16
``wgrad``    ``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]``             ``[E, N, K]`` bfloat16 or float32
============ ================================================================== ============================

Stable API opt-in
-----------------

The forward projection is reachable from the stable grouped GEMM API by naming the
backend::

    out = flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr, backend="cake")

It accepts the ``[cum_m, K]`` / ``[E, N, K]`` / ``[E + 1]`` arguments of the cuDNN
backend, produces bfloat16 outputs only, has no tactic index and requires
``N % 256 == 0`` and ``K % 64 == 0``.

One-shot functions
------------------

.. autosummary::
    :toctree: ../generated

    grouped_gemm_fwd
    grouped_gemm_dgrad
    grouped_gemm_wgrad

Autograd
--------

.. autosummary::
    :toctree: ../generated

    cake_grouped_mm

``cake_grouped_mm(x, w, offs, deterministic=True)`` is a ``torch.autograd.Function``
wrapper (:class:`CakeGroupedMm`): its ``backward`` produces ``x.grad`` with the
``dgrad`` program and ``w.grad`` with the ``wgrad`` program in the dtype of ``w``.  The
Cake programs always reduce in a fixed order, so ``deterministic=False`` currently runs
the same programs.  Use :func:`grouped_gemm_wgrad` with ``out_dtype=torch.float32``
for an fp32 weight gradient.

Prepared launches
-----------------

``prepare_grouped_gemm_fwd``, ``prepare_grouped_gemm_dgrad`` and
``prepare_grouped_gemm_wgrad`` validate the arguments, bind one generated program and
return a ``GroupedGemmLaunch`` whose ``launch()`` performs no allocation and no host
synchronisation and is CUDA-graph capturable; the same prepared launch stays valid
when new offsets or values are written into the bound tensors.
