"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Shared NVFP4 quantization helpers.
"""

from dataclasses import dataclass
from enum import IntEnum
import os

import torch


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


class NVFP44Over6ErrMode(IntEnum):
    """Error metric for selecting between NVFP4 4over6 scale candidates."""

    MAE = 0
    MSE = 1


@dataclass(frozen=True)
class NVFP44Over6Config:
    """NVFP4 4over6 configuration shared by Python drivers and kernels."""

    e4m3_max: int = 448
    err_mode: NVFP44Over6ErrMode | str = NVFP44Over6ErrMode.MAE
    err_use_fast_math: bool = False

    def __post_init__(self) -> None:
        if self.e4m3_max not in (256, 448):
            raise ValueError("NVFP4 4over6 E4M3 max must be either 256 or 448.")
        try:
            if isinstance(self.err_mode, str):
                err_mode = NVFP44Over6ErrMode[self.err_mode.upper()]
            else:
                err_mode = NVFP44Over6ErrMode(self.err_mode)
        except (KeyError, ValueError):
            raise ValueError("NVFP4 4over6 error mode must be MAE or MSE.") from None
        object.__setattr__(self, "err_mode", err_mode)

    @property
    def err_mode_name(self) -> str:
        if isinstance(self.err_mode, str):
            return self.err_mode.upper()
        return self.err_mode.name


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def current_nvfp4_4over6_config() -> NVFP44Over6Config | None:
    if not env_flag_enabled("FLASHINFER_NVFP4_4OVER6"):
        return None

    return NVFP44Over6Config(
        e4m3_max=256
        if env_flag_enabled("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256")
        else 448,
        err_mode=os.environ.get("FLASHINFER_NVFP4_4OVER6_ERR_MODE", "MAE"),
        err_use_fast_math=env_flag_enabled("FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH"),
    )


def nvfp4_e4m3_max(nvfp4_4over6_config: NVFP44Over6Config | None) -> float:
    if nvfp4_4over6_config is not None:
        return float(nvfp4_4over6_config.e4m3_max)
    return FLOAT8_E4M3_MAX


def make_nvfp4_global_scale(
    input_tensor: torch.Tensor,
    per_token_activation: bool,
    global_scale: float | None = None,
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
) -> torch.Tensor:
    e4m3_max = nvfp4_e4m3_max(nvfp4_4over6_config)
    if per_token_activation:
        scale = 1.0 / (e4m3_max * FLOAT4_E2M1_MAX)
    elif global_scale is not None:
        scale = global_scale
    else:
        amax = input_tensor.abs().max().to(torch.float32)
        if amax == 0:
            return torch.full(
                (1,),
                torch.finfo(torch.float32).max,
                dtype=torch.float32,
                device=input_tensor.device,
            )
        return (e4m3_max * FLOAT4_E2M1_MAX / amax).reshape(1).to(input_tensor.device)

    return torch.tensor(
        [scale],
        dtype=torch.float32,
        device=input_tensor.device,
    )


def nvfp4_4over6_mode_label(
    per_token_activation: bool, nvfp4_4over6_config: NVFP44Over6Config | None
) -> str:
    parts = []
    if per_token_activation:
        parts.append("per-token")
    if nvfp4_4over6_config is not None:
        parts.append(f"4over6-{nvfp4_4over6_config.err_mode_name.lower()}")
        parts.append(f"e4m3-{nvfp4_4over6_config.e4m3_max}")
        parts.append(
            "err-fastmath" if nvfp4_4over6_config.err_use_fast_math else "err-exactmath"
        )
    return ", ".join(parts) if parts else "standard"


__all__ = [
    "FLOAT4_E2M1_MAX",
    "FLOAT8_E4M3_MAX",
    "NVFP44Over6Config",
    "NVFP44Over6ErrMode",
    "current_nvfp4_4over6_config",
    "env_flag_enabled",
    "make_nvfp4_global_scale",
    "nvfp4_4over6_mode_label",
    "nvfp4_e4m3_max",
]
