# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight admission for the Qwen 3.5 GDN block: the geometry, the typed declines and the support probe.

Imports torch only. The backend registry (``api``) is imported on demand and imports no kernel either, so a serving
stack's init-time dispatch question costs nothing when the answer is no. One gate, :func:`check_decline`, serves both
the probe (:func:`qwen_3_5_gdn_block_support`, never raises) and plan construction (:func:`require_served`, typed
errors), so a shape the probe advertises is one ``prepare`` accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

FEATURE = "Qwen 3.5 GDN block"
FORMS: Tuple[str, ...] = ("decode", "verify")
#: what a decline is about: a malformed argument, an unregistered backend, an unknown form, the device, the CuTe DSL
#: install, the model geometry, or the step shape (requests / tokens per request)
DECLINE_KINDS: Tuple[str, ...] = (
    "argument",
    "backend",
    "form",
    "device",
    "dsl",
    "geometry",
    "shape",
)


@dataclass(frozen=True)
class Qwen35GdnGeometry:
    """The GDN sub-layer's shape at this tensor-parallel rank.

    Qwen3.8-27B at TP 1: ``Qwen35GdnGeometry(d_model=5120, num_v_heads=48, num_k_heads=16, head_k_dim=128,
    head_v_dim=128, conv_kernel=4)``. ``norm_eps`` is the gated RMSNorm epsilon (compiled into a plan). The constructor
    checks the form of the fields; which geometries a backend serves is that backend's admission
    (:func:`qwen_3_5_gdn_block_support`).
    """

    d_model: int
    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    norm_eps: float = 1e-6

    def __post_init__(self):
        for name in (
            "d_model",
            "num_v_heads",
            "num_k_heads",
            "head_k_dim",
            "head_v_dim",
            "conv_kernel",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(
                    f"Qwen35GdnGeometry.{name} must be a positive int, got {v!r}"
                )
        if self.num_v_heads % self.num_k_heads:
            raise ValueError(
                f"Qwen35GdnGeometry: num_v_heads={self.num_v_heads} must be a multiple of "
                f"num_k_heads={self.num_k_heads} (each K head serves a whole group of V heads)"
            )
        if self.conv_kernel < 2:
            raise ValueError(
                f"Qwen35GdnGeometry: conv_kernel must be at least 2, got {self.conv_kernel}"
            )
        if isinstance(self.norm_eps, bool) or not float(self.norm_eps) > 0.0:
            raise ValueError(
                f"Qwen35GdnGeometry: norm_eps must be > 0, got {self.norm_eps!r}"
            )

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        """Channels of the causal conv: the ``q | k | v`` columns of ``qkv``."""
        return 2 * self.key_dim + self.value_dim

    def conv_state_len(self, tokens_per_request: int = 1) -> int:
        """History columns a conv-state slot holds: ``conv_kernel - 1`` for decode, ``T + conv_kernel - 2`` for a verify
        of ``T`` tokens per request (the speculative layout keeps the newest ``T - 1`` inputs past the window)."""
        return int(tokens_per_request) + self.conv_kernel - 2


@dataclass(frozen=True)
class Qwen35GdnDecline:
    """Why a backend does not serve a request: ``kind`` is one of :data:`DECLINE_KINDS`."""

    kind: str
    reason: str


@dataclass(frozen=True)
class Qwen35GdnSupport:
    """The static answer to "can this process run this form of the block on this device with this backend"."""

    served: bool
    backend: str
    form: str
    kind: Optional[str] = None
    reason: Optional[str] = None
    device_cc: Optional[Tuple[int, int]] = None


def device_cc(device) -> Optional[Tuple[int, int]]:
    """``(major, minor)`` of a CUDA device, ``None`` for anything else (never raises)."""
    try:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 - a probe never raises
        return None


def _count_decline(name: str, value) -> Optional[Qwen35GdnDecline]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return Qwen35GdnDecline(
            "argument", f"{name} must be a positive int or None, got {value!r}"
        )
    return None


def check_decline(
    device,
    geometry,
    form: str,
    backend: str = "frost",
    *,
    requests: Optional[int] = None,
    tokens_per_request: Optional[int] = None,
) -> Optional[Qwen35GdnDecline]:
    """The one gate: ``None`` when ``backend`` serves ``form`` for this geometry on this device (and, when given, this
    step shape), else the typed decline. Never raises; a backend support function that raises is reported as a
    ``"backend"`` decline."""
    if not isinstance(geometry, Qwen35GdnGeometry):
        return Qwen35GdnDecline(
            "argument",
            f"geometry must be a Qwen35GdnGeometry, got {type(geometry).__name__}",
        )
    if form not in FORMS:
        return Qwen35GdnDecline("form", f"form must be one of {FORMS}, got {form!r}")
    for name, value in (
        ("requests", requests),
        ("tokens_per_request", tokens_per_request),
    ):
        decline = _count_decline(name, value)
        if decline is not None:
            return decline
    if form == "decode" and tokens_per_request not in (None, 1):
        return Qwen35GdnDecline(
            "shape",
            f"the decode form takes one token per request, got tokens_per_request={tokens_per_request}",
        )
    from . import api  # the registry; imports no kernel

    entry = api.get_backend(backend) if isinstance(backend, str) else None
    if entry is None:
        return Qwen35GdnDecline(
            "backend",
            f"no Qwen 3.5 GDN block backend named {backend!r}; registered: {api.registered_backends()}",
        )
    if form == "decode":
        tokens_per_request = 1
    try:
        decline = entry.support_fn(device, geometry, form, requests, tokens_per_request)
    except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
        return Qwen35GdnDecline(
            "backend",
            f"the {backend!r} support check raised {type(exc).__name__}: {exc}",
        )
    if decline is not None and decline.kind not in DECLINE_KINDS:
        return Qwen35GdnDecline(
            "backend",
            f"the {backend!r} support check returned an unknown decline kind {decline.kind!r}: {decline.reason}",
        )
    return decline


def unsupported_reason(
    device,
    geometry,
    form: str,
    backend: str = "frost",
    *,
    requests: Optional[int] = None,
    tokens_per_request: Optional[int] = None,
) -> Optional[str]:
    """Why ``backend`` does not serve this request (``None`` = served); the reason of :func:`check_decline`."""
    decline = check_decline(
        device,
        geometry,
        form,
        backend,
        requests=requests,
        tokens_per_request=tokens_per_request,
    )
    return None if decline is None else decline.reason


def qwen_3_5_gdn_block_support(
    device,
    geometry,
    form: str,
    backend: str = "frost",
    *,
    requests: Optional[int] = None,
    tokens_per_request: Optional[int] = None,
) -> Qwen35GdnSupport:
    """See :func:`flashinfer.qwen_3_5_gdn_block.qwen_3_5_gdn_block_support` (the public, decorated entry)."""
    decline = check_decline(
        device,
        geometry,
        form,
        backend,
        requests=requests,
        tokens_per_request=tokens_per_request,
    )
    return Qwen35GdnSupport(
        served=decline is None,
        backend=str(backend),
        form=str(form),
        kind=None if decline is None else decline.kind,
        reason=None if decline is None else decline.reason,
        device_cc=device_cc(device),
    )


def require_served(
    device,
    geometry,
    form: str,
    backend: str,
    *,
    requests: int,
    tokens_per_request: int,
) -> None:
    """The plan-time gate: ``ValueError`` for a malformed argument, ``NotImplementedError`` naming what is not served."""
    decline = check_decline(
        device,
        geometry,
        form,
        backend,
        requests=requests,
        tokens_per_request=tokens_per_request,
    )
    if decline is None:
        return
    message = f"qwen_3_5_gdn_block ({backend}, {form}): {decline.reason}"
    if decline.kind == "argument":
        raise ValueError(message)
    raise NotImplementedError(message)
