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

Packed GDN (Gated DeltaNet) chunked training forward and backward on one rank
(SM100 / SM103 / SM107), the Cake backend: the entry points of the
non-deterministic package ``flashinfer.experimental.cake.cake_gdn_nondet_train``.
Head dimension 128, chunk 64, ``HV`` a multiple of ``H``, FLA's
``chunk_gated_delta_rule`` signature, no context parallelism.

Mode rules
----------

* The mode is in the name: this module runs the non-deterministic package
  only.  No entry takes a ``deterministic`` keyword; passing one raises
  ``TypeError``.  Nothing is routed to the deterministic package and nothing
  falls back to it.
* Every launch is fixed-order today, so two runs of one call on one device give
  the same bits.  The contract does not promise run-to-run bits.  The results
  match the deterministic package within its accuracy budgets, not bitwise.
* ``torch.use_deterministic_algorithms(True)``, with or without
  ``warn_only=True``, runs the call unchanged: no alert, no warning, no
  fallback.
* The deterministic counterpart is ``flashinfer.cake_gdn_det_train``.

A stable FlashInfer API: thin ``@flashinfer_api`` entry points that import the
package on first use.  The host plan, the autograd Function and the generated
kernels live in the package; this module forwards every argument.  Every plan
pin is a keyword argument; no environment variable is read.

Usage::

    from flashinfer.cake_gdn_nondet_train import cake_gdn_nondet

    o, final_state = cake_gdn_nondet(
        q, k, v, g, beta,                 # [B, T, H, 128] / [B, T, HV, 128] bf16; g, beta [B, T, HV]
        initial_state=None,               # [N, HV, 128, 128] fp32 or None
        output_final_state=False,
        cu_seqlens=cu_seqlens,            # packed batch (B == 1): int32 / int64 offsets on the device
        cu_seqlens_cpu=cu_seqlens_cpu,    # the same offsets on the host: host-table route; None: device route
        use_qk_l2norm_in_kernel=True,
    )
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api


# The package is imported on first use and cached: a call does not repeat the import.
@functools.cache
def _pkg():
    from .experimental.cake import cake_gdn_nondet_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_gdn_nondet_train


@flashinfer_api
def cake_gdn_nondet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    scan_config: Optional[tuple[int, int]] = None,
    segment_chunks: Optional[int] = None,
    finalize: Optional[str] = None,
    warmup_log2: Optional[float] = None,
    save_state: bool = True,
    host_cache: bool = True,
    dhu_scan: Optional[str] = None,
    prep: Optional[str] = None,
    lpt: bool = True,
    streams: int = 2,
    tail: Optional[str] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Non-deterministic chunked gated delta rule with FLA's signature (autograd).

    Parameters
    ----------
    q, k : torch.Tensor
        BF16 ``[B, T, H, 128]``.
    v : torch.Tensor
        BF16 ``[B, T, HV, 128]``, ``HV`` a positive multiple of ``H``.
    g : torch.Tensor
        ``[B, T, HV]`` natural-log decay gate (computed in FP32).
    beta : torch.Tensor
        ``[B, T, HV]`` FP32 or BF16; its gradient comes back in its dtype.
    scale : float, optional
        Output scale; ``1 / sqrt(128)`` when ``None``.
    initial_state : torch.Tensor, optional
        FP32 ``[N, HV, 128, 128]``.
    output_final_state : bool
        Return the final state.
    cu_seqlens : torch.Tensor, optional
        Packed variable-length batch (``B = 1``): int32 / int64 offsets on the device.
    cu_seqlens_cpu : torch.Tensor, optional
        The same offsets on the host: selects the host-table route.  Without it the device builds the tables.
    use_qk_l2norm_in_kernel : bool
        Normalize ``q`` / ``k`` per token and head inside the kernels.
    scan_config, segment_chunks, finalize, warmup_log2, save_state, host_cache, dhu_scan, prep, lpt, streams, tail
        Keyword-only plan pins (the package's ``ChainOptions``; see the package README).

    Returns
    -------
    ``(o, final_state)``: ``o`` like ``v``; ``final_state`` is ``None`` unless ``output_final_state``.

    The inputs are checked before the first launch (``ValueError``).  There is no ``deterministic`` keyword.  Every
    launch is fixed-order today, so ``torch.use_deterministic_algorithms(True)`` (strict or ``warn_only``) changes
    nothing; the contract does not promise run-to-run bits.
    """
    return _pkg().cake_gdn_nondet(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scan_config=scan_config,
        segment_chunks=segment_chunks,
        finalize=finalize,
        warmup_log2=warmup_log2,
        save_state=save_state,
        host_cache=host_cache,
        dhu_scan=dhu_scan,
        prep=prep,
        lpt=lpt,
        streams=streams,
        tail=tail,
    )


@flashinfer_api
def is_cake_gdn_nondet_train_supported(
    device: torch.device | str | int | None = None,
) -> bool:
    r"""``True`` when ``device`` (default: the current CUDA device) is an exported target of the package
    (``sm_100a``, ``sm_103a`` or ``sm_107a``); ``False`` without CUDA or for a device that is not CUDA.

    The package's loader is imported first: a package that does not import raises, it is not reported as
    "not supported".

    Parameters
    ----------
    device : torch.device, str, int, optional
        Device to query; ``None`` selects the current CUDA device.
    """
    from .experimental.cake.cake_gdn_nondet_train import (  # noqa: PLC0415 - imported on first use
        cake_backend,
    )

    if not torch.cuda.is_available():
        return False
    if device is None:
        index = torch.cuda.current_device()
    else:
        dev = torch.device(device)
        if dev.type != "cuda":
            return False
        index = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        capability = torch.cuda.get_device_capability(index)
    except (AssertionError, RuntimeError, ValueError):  # no such device index
        return False
    return cake_backend.arch_for_capability(capability) is not None


# The autograd Function of the package, resolved on first access so that importing this module does not import the
# package.
_LAZY_NAMES = ("CakeGDNNondetFunction",)


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        return getattr(_pkg(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_NAMES))


__all__ = [
    "CakeGDNNondetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_gdn_nondet",
    "is_cake_gdn_nondet_train_supported",
]
