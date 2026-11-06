"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Cake ragged BF16 MoE grouped GEMM (SM100 / SM103 / SM107), the
non-deterministic package's experimental-API namespace.

The functions of this module carry the experimental-API marking and hand off
to the package ``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train`` with the
same arguments: the weight gradient with the 512-wide k tile adds its split
tail (the tiles of its last partial wave, cut into k-step units over the
clusters) in place with unordered fp32 reductions, in one launch.  The
deterministic counterpart is a separate package with ``det`` in every name,
``flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train`` (also
``flashinfer.grouped_mm.grouped_det_gemm_*`` / ``cake_moe_grouped_gemm_det``), to
which nothing here routes.

Mode and dispatch rules (one row per call)::

    call                                                  result
    deterministic=False                                   the non-deterministic package
    deterministic omitted or True, or any value that      ValueError naming the deterministic
      is not the bool False (None, 0, 1, "False")           entry; nothing runs
    deterministic=False under                             PyTorch's non-determinism alert
      torch.use_deterministic_algorithms(True)              immediately before an order-varying
                                                            launch (RuntimeError, nothing launches;
                                                            warn_only=True: UserWarning, then it
                                                            runs); never a deterministic fallback

    entry                 deterministic entry (no deterministic argument)
    grouped_gemm_fwd      flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train.cake_moe_grouped_gemm_det_train_forward
    grouped_gemm_dgrad    ...cake_moe_grouped_gemm_det_train_dgrad
    grouped_gemm_wgrad    ...cake_moe_grouped_gemm_det_train_wgrad
    cake_grouped_mm       ...cake_moe_grouped_gemm_det

An order-varying launch is the weight gradient whose 512-wide k tile splits
its tail (``plan["unordered_tail"]`` and ``plan["tail_splits"] > 1``), in
overwrite or accumulate mode: :func:`grouped_gemm_wgrad` and the backward of
:func:`cake_grouped_mm` when it computes ``dW``.  The forward pass, the data
gradient, the 256-wide k tile's weight gradient and an unsplit tail run in a
fixed order and never alert.

Numerical contract of ``deterministic=False``: only the run-to-run bitwise
reproducibility of the weight gradient ``dW`` is relaxed.  ``Y`` and ``dX`` are
bitwise the deterministic package's (copies of the same programs; on sm_100a the
data gradient's L2 evict-first twin is the package's own build of the
deterministic wide program, whose cache policy changes no arithmetic), and so is
``dW`` wherever the 256-wide k tile runs or the 512-wide k tile does not split
its tail.  Every whole tile of ``dW`` is reduced by one CTA pair over the
group's rows in a fixed order; only the units of a split tail tile are added
in an order the launch does not fix.  The operands (bf16), the accumulation
(fp32), the one round-to-nearest rounding to bf16 (or the fp32 store), the
layouts, the device-resident ``offs`` contract (no host synchronisation, no
padding), the exact zeros of empty groups, the error budgets and the fixes the
deterministic package carries are unchanged.  The split tail's fp32 sums and
tickets are workspaces kept per (device, CUDA stream) at rest at zero (every
launch leaves them zero); a CUDA-graph capture takes zeroed buffers of its own.

Weight-gradient accumulation (``grouped_gemm_wgrad(..., out=out, accumulate=True)``,
``out[e] += G[e].T @ X[e]``, and ``cake_grouped_mm(..., fuse_wgrad_accumulation=True)``,
which adds into ``w.main_grad`` in the backward; Megatron's
``gradient_accumulation_fusion``) is the package's accumulate mode (fused into
the 512-wide k tile's epilogue where served, else the overwrite into a scratch
and ``out.add_``), with the same mode rules.

This is a local operator: on a context-parallel rank it runs on that rank's
local rows; it adds no collective, and the reduction of a parameter gradient
across ranks stays with the caller.

Usage::

    from flashinfer.cake_moe_grouped_gemm_nondet_train import (
        cake_grouped_mm,
        grouped_gemm_wgrad,
    )

    y = cake_grouped_mm(x, w, offs, deterministic=False)   # unordered dW split tail
    dw = grouped_gemm_wgrad(g, x, offs, out_dtype=torch.float32, deterministic=False)

    # The deterministic entries take no deterministic argument:
    from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import cake_moe_grouped_gemm_det

    y = cake_moe_grouped_gemm_det(x, w, offs)

The kernels, the host planning, the autograd wrapper and the JIT registration
live in the package, which every function imports inside its body (a
deferred import), so importing this module loads none of it.
"""

from __future__ import annotations

from typing import Optional

import torch

from .api_logging import flashinfer_experimental_api

__all__ = [
    "cake_grouped_mm",
    "grouped_gemm_dgrad",
    "grouped_gemm_fwd",
    "grouped_gemm_wgrad",
]


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_fwd")
def grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T`` for every group ``e``.

    Parameters
    ----------
    x : torch.Tensor
        ``[sum_m, K]`` bfloat16 rows of all groups, group after group.
    w : torch.Tensor
        ``[E, N, K]`` bfloat16 weights, one ``[N, K]`` matrix per group.
    offs : torch.Tensor
        ``[E]`` int32 cumulative end offsets on the device (``offs[-1] == sum_m``),
        or ``m_indptr`` ``[E + 1]`` with a leading 0; read by the kernel, never
        on the host.
    out : Optional[torch.Tensor]
        ``[sum_m, N]`` bfloat16 output (allocated when omitted); may be row-padded.
    deterministic : bool
        Must be ``False``: the non-deterministic package's copy of the
        deterministic forward programs (bitwise the same ``Y``).  Every other
        value, the default ``True`` included, raises ``ValueError`` naming
        ``cake_moe_grouped_gemm_det_train_forward``.  Never alerts under torch's deterministic
        mode (fixed order).

    Returns
    -------
    torch.Tensor
        ``out``.  ``N % 256 == 0`` and ``K % 64 == 0`` are required.
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        grouped_gemm_fwd as _impl,
    )

    return _impl(x, w, offs, out=out, deterministic=deterministic)


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_dgrad")
def grouped_gemm_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
) -> torch.Tensor:
    """``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` for every group ``e``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``w`` is ``[E, N, K]`` bfloat16 and is
    read in place (no transpose copy), ``offs`` is ``[E]`` int32 device end
    offsets (or ``m_indptr`` ``[E + 1]``) and ``out`` is ``[sum_m, K]`` bfloat16.
    ``K % 256 == 0`` and ``N % 64 == 0`` are required.  ``deterministic`` must
    be ``False`` (every other value, the default ``True`` included, raises
    ``ValueError`` naming ``cake_moe_grouped_gemm_det_train_dgrad``): the non-deterministic
    package's copy of the deterministic programs (on sm_100a, where the
    evict-first rule holds, its own build of the deterministic wide program
    with L2 evict-first loads and stores; bitwise the same ``dX``).  Never
    alerts under torch's deterministic mode (fixed order).
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        grouped_gemm_dgrad as _impl,
    )

    return _impl(g, w, offs, out=out, deterministic=deterministic)


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.grouped_gemm_wgrad")
def grouped_gemm_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
    *,
    deterministic: bool = True,
    accumulate: bool = False,
) -> torch.Tensor:
    """``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`` for every group ``e``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``x`` is ``[sum_m, K]`` bfloat16,
    ``offs`` is ``[E]`` int32 device end offsets (``E = offs.numel()``; pass
    ``num_groups=E`` to hand in ``m_indptr`` ``[E + 1]`` instead) and ``out``
    is ``[E, N, K]`` in ``out_dtype`` (bfloat16 or float32; defaults to the
    dtype of ``out``, else bfloat16).  fp32 accumulation, one rounding at the
    store, exact zeros for empty groups.  ``N % 256 == 0`` and ``K`` a multiple
    of the selected k tile (256, or 512 when chosen) are required.

    ``deterministic`` must be ``False`` (every other value, the default
    ``True`` included, raises ``ValueError`` naming
    ``cake_moe_grouped_gemm_det_train_wgrad``): with the 512-wide k tile a split tail is
    added in place with unordered fp32 reductions (one launch instead of two),
    within the same error budgets but not bitwise reproducible run to run;
    with the 256-wide k tile the deterministic programs run.  Under torch's
    deterministic mode a launch with such a split tail raises PyTorch's
    non-determinism alert immediately before it runs (``RuntimeError``;
    ``warn_only=True``: a ``UserWarning``, then it runs); a fixed-order launch
    does not alert.

    ``accumulate=True`` adds into ``out`` (required, bfloat16 or float32;
    ``out_dtype`` None or its dtype) and returns it: ``out[e] += G[e].T @ X[e]``,
    the package's accumulate mode (fused into the 512-wide k tile's epilogue
    where served, else the overwrite into a scratch and ``out.add_``), after
    the checks of ``out`` (no overlap with itself, ``g``, ``x`` or ``offs``).
    Every call adds again.
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        grouped_gemm_wgrad as _impl,
    )

    return _impl(
        g,
        x,
        offs,
        out=out,
        out_dtype=out_dtype,
        num_groups=num_groups,
        deterministic=deterministic,
        accumulate=accumulate,
    )


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.cake_grouped_mm")
def cake_grouped_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    *,
    deterministic: bool = True,
    fuse_wgrad_accumulation: bool = False,
) -> torch.Tensor:
    """Differentiable ragged grouped GEMM ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``.

    Parameters
    ----------
    x : torch.Tensor
        ``[sum_m, K]`` bfloat16 rows of all groups, group after group.
    w : torch.Tensor
        ``[E, N, K]`` bfloat16 weights (``[N, K]`` per group).
    offs : torch.Tensor
        int32 device tensor: ``[E]`` cumulative end offsets or ``m_indptr``
        ``[E + 1]``.  Read by the kernels, never on the host.
    deterministic : bool
        Keyword-only; must be ``False`` (every other value, the default ``True``
        included, raises ``ValueError`` naming ``cake_moe_grouped_gemm_det``):
        ``CakeGroupedMm`` of the non-deterministic package, whose backward runs
        the deterministic data-gradient programs and the weight gradient with
        the unordered split tail.  Under torch's deterministic mode the
        backward raises PyTorch's non-determinism alert immediately before a
        weight-gradient launch with a split tail (``RuntimeError``;
        ``warn_only=True``: a ``UserWarning``, then it runs); the forward,
        ``dX`` and a fixed-order ``dW`` do not alert.
    fuse_wgrad_accumulation : bool
        ``False`` (default): ``dW`` is returned as ``w``'s gradient.  ``True``
        (Megatron's ``gradient_accumulation_fusion``): the backward adds ``dW``
        into ``w.main_grad`` (fp32 or bf16) and returns a placeholder or no
        gradient for ``w``, as
        ``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_grouped_mm``
        documents.

    Returns
    -------
    torch.Tensor
        ``[sum_m, N]`` bfloat16 (the deterministic forward's bits).
        ``backward`` produces ``x.grad`` ``[sum_m, K]`` bfloat16 and ``w.grad``
        ``[E, N, K]`` in ``w``'s dtype; use :func:`grouped_gemm_wgrad` for an
        fp32 weight gradient.
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_autograd import (
        cake_grouped_mm as _impl,
    )

    return _impl(
        x,
        w,
        offs,
        deterministic=deterministic,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
    )
