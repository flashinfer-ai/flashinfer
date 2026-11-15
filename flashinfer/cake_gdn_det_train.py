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

Deterministic packed GDN (Gated DeltaNet) chunked training forward and
backward on one rank, no context parallelism (SM100 / SM103 / SM107), the
Cake backend.  With the same inputs, hardware, layout and plan, the outputs
and gradients repeat bit for bit from run to run.

Mode and dispatch rules: every name here carries the mode word ``det``; no
entry takes a ``deterministic`` keyword (passing one raises ``TypeError``);
``torch.use_deterministic_algorithms(True)`` changes nothing (every launch is
fixed-order, so no alert is raised).  The non-deterministic route is the
separate module ``flashinfer.cake_gdn_nondet_train``; nothing here routes to
it.

Thin entry points (a stable API: no ``ExperimentalWarning``).  The contract, the chunk and segment tables,
the scan planner, the launch order, the autograd Function, the low-level
chain (``cake_gdn_det_train_forward`` / ``cake_gdn_det_train_backward`` over
the package's chunk table and tape) and the generated kernels live in the
backend package ``flashinfer.experimental.cake.cake_gdn_det_train``; this
module validates nothing itself and forwards every argument.  Every switch is
a keyword argument; no environment variable is read.

Operands (FLA's signature, head dimension 128): ``q, k`` BF16 ``[B, T, H,
128]``, ``v`` BF16 ``[B, T, HV, 128]`` with ``HV`` a multiple of ``H``,
``g`` ``[B, T, HV]`` (the natural-log decay, computed in FP32), ``beta``
``[B, T, HV]`` FP32 or BF16 (its gradient comes back in its dtype).  A packed
batch passes ``B = 1`` with ``cu_seqlens`` on the device; ``cu_seqlens_cpu``
(the same offsets on the host) selects the host-table route.  ``scale`` is
``1 / sqrt(128)`` when ``None``.

Usage::

    from flashinfer.cake_gdn_det_train import cake_gdn_det, is_cake_gdn_det_train_supported

    o, final_state = cake_gdn_det(
        q, k, v, g, beta,
        initial_state=None,                  # [N, HV, 128, 128] FP32 or None
        output_final_state=False,
        cu_seqlens=cu_seqlens,               # packed batch (B == 1): int32 / int64 offsets on the device
        cu_seqlens_cpu=cu_seqlens_cpu,       # the same offsets on the host; None: the device-table route
        use_qk_l2norm_in_kernel=True,
    )
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .api_logging import flashinfer_api


def _pkg():
    from .experimental.cake import cake_gdn_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_gdn_det_train


@flashinfer_api
def cake_gdn_det(
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
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Deterministic packed GDN chunked training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``(o, final_state)``: ``o`` like ``v``, ``final_state`` FP32 ``[N, HV, 128, 128]`` or ``None`` unless
    ``output_final_state``; gradients to ``q, k, v, g, beta`` (and ``initial_state``).  The keyword-only
    arguments are the plan pins of the package (``scan_config``, ``segment_chunks``, ``finalize``,
    ``warmup_log2``, ``dhu_scan``, ``prep``, ``lpt``, ``streams``) and its ``save_state`` / ``host_cache``
    switches; see ``flashinfer.experimental.cake.cake_gdn_det_train.cake_gdn_det`` for the contract and the input
    checks (``ValueError`` before the first launch).

    Parameters
    ----------
    q, k : torch.Tensor
        BF16 ``[B, T, H, 128]`` query and key tensors.
    v : torch.Tensor
        BF16 ``[B, T, HV, 128]`` values, with ``HV`` a multiple of ``H``.
    g : torch.Tensor
        ``[B, T, HV]`` natural-log decay gate, computed in FP32.
    beta : torch.Tensor
        FP32 or BF16 ``[B, T, HV]`` delta-rule gate after the sigmoid.
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    initial_state : torch.Tensor, optional
        FP32 ``[N, HV, 128, 128]`` initial state; zero when omitted.
    output_final_state : bool, default=False
        Return the final state in addition to the output.
    cu_seqlens : torch.Tensor, optional
        Device document offsets of a packed batch (``B == 1``).
    cu_seqlens_cpu : torch.Tensor, optional
        The same offsets on the host; selects the host-table route when given.
    use_qk_l2norm_in_kernel : bool, default=False
        Normalize raw query and key rows inside the kernels.
    scan_config : tuple of int, optional
        Pin the backend scan configuration.
    segment_chunks : int, optional
        Pin the number of chunks per segment.
    finalize : str, optional
        Pin the finalize stage of the unfused backward tail.
    warmup_log2 : float, optional
        Override the pre-scans' log2 warm-up threshold.
    save_state : bool, default=True
        Keep the forward state tape for the backward.
    host_cache : bool, default=True
        Cache the host-side plan and tables.
    dhu_scan : str, optional
        Pin the backward adjoint scan implementation.
    prep : str, optional
        Pin the backend preparation implementation.
    lpt : bool, default=True
        Enable longest-processing-time scheduling in the backend plan.
    streams : int, default=2
        Number of streams used by the backend plan: ``1`` or ``2``.
    """
    return _pkg().cake_gdn_det(
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
    )


@flashinfer_api
def is_cake_gdn_det_train_supported(
    device: torch.device | str | int | None = None,
) -> bool:
    r"""``True`` when ``device`` (default: the current CUDA device) is an exported target of the package
    (compute capability 10.0, 10.3 or 10.7); ``False`` without CUDA or for a non-CUDA device.

    Parameters
    ----------
    device : torch.device, str, int, optional
        Device to query; ``None`` selects the current CUDA device.
    """
    if not torch.cuda.is_available():
        return False
    try:
        if device is None:
            index = torch.cuda.current_device()
        else:
            dev = torch.device(device)
            if dev.type != "cuda":
                return False
            index = dev.index if dev.index is not None else torch.cuda.current_device()
        from .experimental.cake.cake_gdn_det_train import cake_backend  # noqa: PLC0415

        capability = tuple(torch.cuda.get_device_capability(index))
        return cake_backend.arch_for_capability(capability) is not None
    except Exception:  # noqa: BLE001 - an unexported device or a package without its registry is "not supported"
        return False


# The autograd Function of the package, resolved on first access so that importing the facade imports the package
# lazily.
_LAZY_TYPES = ("CakeGDNDetFunction",)


def __getattr__(name: str) -> Any:
    if name in _LAZY_TYPES:
        return getattr(_pkg(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))


__all__ = [
    "CakeGDNDetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_gdn_det",
    "is_cake_gdn_det_train_supported",
]
