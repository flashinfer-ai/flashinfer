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


Global compilation context management for FlashInfer.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def _get_arch_suffix(major: int, minor: int) -> str:
    """Determine the architecture suffix ('a' or 'f') based on GPU capability and CUDA version.

    For SM >= 9, architectures use the 'a' (architecture-specific) suffix by default.
    For SM120 (Blackwell GeForce/workstation), the 'f' (feature-set / family) suffix is
    preferred when the CUDA toolkit supports it (>= 12.9), as it enables additional
    instructions such as native FP4 conversion (cvt.rn.satfinite.e2m1x2.f32).

    Note: 'a' and 'f' are different feature sets, not a superset relationship.
    We only auto-select 'f' for SM120 where it's been verified to improve FP4 performance.
    Other architectures (SM100, SM103, SM110) keep 'a' by default; specific modules can
    opt into 'f' explicitly via sm100f_nvcc_flags etc.
    """
    from packaging import version as pkg_version

    cuda_version = torch.version.cuda
    suffix = "a"
    # Use 'f' suffix for SM120 when CUDA >= 12.9, enabling native FP4 conversion
    if major == 12 and minor == 0 and cuda_version is not None:
        try:
            if pkg_version.parse(cuda_version) >= pkg_version.parse("12.9"):
                suffix = "f"
        except Exception:
            pass
    return str(minor) + suffix


class CompilationContext:
    COMMON_NVCC_FLAGS = [
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
    ]

    def __init__(self):
        self.TARGET_CUDA_ARCHS = set()
        if "FLASHINFER_CUDA_ARCH_LIST" in os.environ:
            for arch in os.environ["FLASHINFER_CUDA_ARCH_LIST"].split(" "):
                major, minor = arch.split(".")
                major = int(major)
                self.TARGET_CUDA_ARCHS.add((int(major), str(minor)))
        else:
            try:
                for device in range(torch.cuda.device_count()):
                    major, minor = torch.cuda.get_device_capability(device)
                    if major >= 9:
                        minor = _get_arch_suffix(major, minor)
                    self.TARGET_CUDA_ARCHS.add((int(major), str(minor)))
            except Exception as e:
                logger.warning(f"Failed to get device capability: {e}.")

    def get_nvcc_flags_list(
        self, supported_major_versions: list[int] = None
    ) -> list[str]:
        if supported_major_versions:
            supported_cuda_archs = [
                major_minor_tuple
                for major_minor_tuple in self.TARGET_CUDA_ARCHS
                if major_minor_tuple[0] in supported_major_versions
            ]
        else:
            supported_cuda_archs = self.TARGET_CUDA_ARCHS
        if len(supported_cuda_archs) == 0:
            raise RuntimeError(
                f"No supported CUDA architectures found for major versions {supported_major_versions}."
            )
        return [
            f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
            for major, minor in supported_cuda_archs
        ] + self.COMMON_NVCC_FLAGS
