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
import subprocess
import re

logger = logging.getLogger(__name__)


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
                        minor = str(minor) + "a"
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

    # I've included the two version here:
    #  - torch.version.cuda is the version which torch was compiled with
    #  - nvcc --version is the version of CUDA toolkit installed on the system
    def get_cuda_version(self, use_nvcc: bool = False):
        if use_nvcc:
            return self.get_cuda_version_from_nvcc()
        else:
            return torch.version.cuda.split(".")

    def get_cuda_version_from_nvcc(self):
        """Return (major, minor) CUDA version detected from `nvcc --version`.

        We assume `nvcc` is installed and available on PATH.
        """
        try:
            proc = subprocess.run(
                ["nvcc", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out = proc.stdout
        except Exception as e:
            raise RuntimeError(f"Failed to execute nvcc --version: {e}") from e

        # Try common patterns in nvcc output
        # Example lines:
        #   Cuda compilation tools, release 12.4, V12.4.131
        #   Build cuda_12.3.r12.3/compiler....
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if not m:
            m = re.search(r"V(\d+)\.(\d+)", out)
        if not m:
            m = re.search(r"cuda_(\d+)\.(\d+)", out)
        if not m:
            raise RuntimeError(f"Unable to parse CUDA version from nvcc output:\n{out}")

        major, minor = m.group(1), m.group(2)
        return major, minor
