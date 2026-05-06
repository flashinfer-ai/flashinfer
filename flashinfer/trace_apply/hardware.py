from __future__ import annotations

import torch

# SKU name (as it appears in TraceRecord.environment.hardware and
# Solution.spec.target_hardware) -> SM arch string ("sm{major}{minor}").
_SKU_TO_SM: dict[str, str] = {
    "NVIDIA A100": "sm80",
    "NVIDIA A100 80GB PCIe": "sm80",
    "NVIDIA A100-SXM4-80GB": "sm80",
    "NVIDIA L4": "sm89",
    "NVIDIA L40": "sm89",
    "NVIDIA L40S": "sm89",
    "NVIDIA H20": "sm90",
    "NVIDIA H100": "sm90",
    "NVIDIA H100 PCIe": "sm90",
    "NVIDIA H100 80GB HBM3": "sm90",
    "NVIDIA H200": "sm90",
    "NVIDIA B200": "sm100",
    "NVIDIA GB200": "sm100",
    "NVIDIA B100": "sm100",
}


def sm_for_sku(sku: str) -> str | None:
    """Return the SM arch string for a SKU name, or None if unknown."""
    return _SKU_TO_SM.get(sku)


def sms_for_skus(skus: list[str]) -> set[str]:
    """Return the set of known SMs covered by a list of SKU names."""
    out: set[str] = set()
    for sku in skus:
        sm = _SKU_TO_SM.get(sku)
        if sm is not None:
            out.add(sm)
    return out


def current_sm(device: torch.device | int | str | None = None) -> str:
    """Return the SM arch of the current (or specified) CUDA device."""
    if device is None:
        device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"
