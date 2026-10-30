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
"""

# Shared infrastructure of the experimental FP8 / UE8M0 paged MQA-logits
# backends (flashinfer/experimental/cake_fp8_ue8m0_paged_mqa_logits_decode and
# _prefill): their architectures, the loader of their generated programs and
# the launch-device guard. Each backend keeps its own programs, launch
# parameters, input checks and launch.

from .programs import (
    ARCHES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    check_launcher,
    import_program,
    on_device,
    program_arch,
    program_files,
)

__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "check_launcher",
    "import_program",
    "on_device",
    "program_arch",
    "program_files",
]
