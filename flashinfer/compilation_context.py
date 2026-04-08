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


class CompilationContext:
    COMMON_NVCC_FLAGS = [
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
    ]

    @staticmethod
    def _normalize_cuda_arch(major: int, minor: int) -> tuple[int, str]:
        """Normalize a (major, minor) capability pair into a (major, minor_str)
        tuple with the correct architecture suffix for nvcc.

        SM 9.x  -> 'a' suffix (e.g. compute_90a)
        SM 12.x -> 'f' suffix with minor version preserved (e.g. compute_120f for SM120, compute_121a for SM121).
        Each SM 12.x variant gets its own cubin to avoid running SM120 code on SM121 (DGX Spark) which
        can cause cudaErrorIllegalInstruction. Requires CUDA >= 12.9.
        SM 10+  -> 'a' suffix (e.g. compute_100a)
        SM < 9  -> no suffix
        """
        if major == 9:
            return (major, str(minor) + "a")
        elif major == 12:
            from flashinfer.jit.cpp_ext import is_cuda_version_at_least

            if is_cuda_version_at_least("12.9"):
                if minor == 0:
                    return (major, "0f")
                else:
                    return (major, str(minor) + "a")
            else:
                raise RuntimeError("SM 12.x requires CUDA >= 12.9")
        elif major >= 10:
            return (major, str(minor) + "a")
        return (major, str(minor))

    def __init__(self):
        self.TARGET_CUDA_ARCHS = set()
        if "FLASHINFER_CUDA_ARCH_LIST" in os.environ:
            for arch in os.environ["FLASHINFER_CUDA_ARCH_LIST"].split(" "):
                major, minor = arch.split(".")
                major = int(major)
                # If the user already provided a suffix (e.g. "12.0f"),
                # respect it as-is; otherwise normalise.
                if minor[-1].isalpha():
                    self.TARGET_CUDA_ARCHS.add((major, minor))
                else:
                    self.TARGET_CUDA_ARCHS.add(
                        self._normalize_cuda_arch(major, int(minor))
                    )
        else:
            try:
                for device in range(torch.cuda.device_count()):
                    major, minor = torch.cuda.get_device_capability(device)
                    self.TARGET_CUDA_ARCHS.add(self._normalize_cuda_arch(major, minor))
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
            for major, minor in sorted(supported_cuda_archs)
        ] + self.COMMON_NVCC_FLAGS
