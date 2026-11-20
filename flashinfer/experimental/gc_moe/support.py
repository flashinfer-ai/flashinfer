"""Support checks for the experimental GC MoE API.

Kept deliberately lightweight: importing this module pulls in torch only, so
callers can validate before the vendored kernels, ``nvidia-cutlass-dsl`` or a
CUDA context are touched.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re

import torch

BACKENDS = ("sm90", "sm100", "sm120")

# Compute capability -> architecture path. SM100 covers both B200 (10, 0) and
# B300/GB300 (10, 3); the upstream SM100 path accepts either.
_CC_TO_BACKEND = {
    (9, 0): "sm90",
    (10, 0): "sm100",
    (10, 3): "sm100",
    (12, 0): "sm120",
}

_MIN_CUTLASS_DSL = (4, 8, 1)


def _device_index(device=None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    index = torch.device(device).index
    return torch.cuda.current_device() if index is None else index


def compute_capability(device=None) -> tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(_device_index(device))
    return int(major), int(minor)


def _format_cc(cc: tuple[int, int]) -> str:
    return f"SM{cc[0]}{cc[1]}"


def backend_for_compute_capability(cc: tuple[int, int]) -> str:
    backend = _CC_TO_BACKEND.get(cc)
    if backend is None:
        supported = ", ".join(_format_cc(pair) for pair in sorted(_CC_TO_BACKEND))
        raise ValueError(
            f"GC MoE supports {supported} GPUs; this device is {_format_cc(cc)}"
        )
    return backend


def resolve_backend(backend: str | None, device=None) -> str:
    """Return the architecture path for ``backend`` and the current device."""
    if not torch.cuda.is_available():
        raise RuntimeError("GC MoE requires a CUDA device")
    cc = compute_capability(device)
    detected = backend_for_compute_capability(cc)
    if backend is None:
        return detected
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS} or None, got {backend!r}")
    if backend != detected:
        raise ValueError(
            f"backend={backend!r} does not match this device, which selects "
            f"{detected!r} ({_format_cc(cc)})"
        )
    return backend


def is_supported(device=None) -> bool:
    if not torch.cuda.is_available():
        return False
    return compute_capability(device) in _CC_TO_BACKEND


def check_cutlass_dsl() -> None:
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "GC MoE is a CUTLASS Python DSL kernel and needs the DSL runtime; "
            "install it with: pip install 'nvidia-cutlass-dsl==4.8.1'"
        ) from exc
    release = tuple(int(part) for part in re.findall(r"\d+", version)[:3])
    if release < _MIN_CUTLASS_DSL:
        raise RuntimeError(
            f"GC MoE needs nvidia-cutlass-dsl >= 4.8.1, found {version}; "
            "install it with: pip install 'nvidia-cutlass-dsl==4.8.1'"
        )


def check_symmetric_memory() -> None:
    """Reject builds without the module every vendored tree imports at load."""
    try:
        importlib.import_module("torch.distributed._symmetric_memory")
    except ImportError as exc:
        raise RuntimeError(
            "GC MoE needs torch.distributed._symmetric_memory; the vendored trees "
            "import it even for a single rank, and this PyTorch build "
            f"({torch.__version__}) does not provide it"
        ) from exc


def validate_group(group) -> int:
    size = getattr(group, "size", None)
    if not callable(size) or not callable(getattr(group, "barrier", None)):
        raise TypeError("group must be a torch.distributed ProcessGroup")
    ranks = int(size())
    if ranks < 1:
        raise ValueError("group must contain at least one rank")
    return ranks


def validate_activation(activation) -> None:
    if activation != "swiglu":
        raise ValueError(
            f"GC MoE supports activation='swiglu' only, got {activation!r}"
        )


def validate_launch(activation, activation_alpha, num_tokens_bound, backend) -> None:
    validate_activation(activation)
    if activation_alpha is not None:
        raise ValueError(
            "activation_alpha parameterizes the swiglu_oai activation, which "
            "GC MoE does not support; leave it unset"
        )
    if num_tokens_bound is not None and backend == "sm100":
        raise ValueError(
            "num_tokens_bound is a capture hint for the SM90 and SM120 backends; "
            "the SM100 backend sizes its graph from the staged tokens"
        )
    if num_tokens_bound is not None:
        if isinstance(num_tokens_bound, bool) or not isinstance(num_tokens_bound, int):
            raise ValueError("num_tokens_bound must be a positive integer")
        if num_tokens_bound < 1:
            raise ValueError("num_tokens_bound must be a positive integer")
