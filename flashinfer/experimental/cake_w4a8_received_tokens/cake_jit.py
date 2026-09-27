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

"""Target-owned JIT registration for received-token W4A8 MoE."""

from functools import cache
import hashlib
import importlib
import os
from pathlib import Path

from ...compilation_context import CompilationContext
from ...jit.cute_dsl_core import JitSpecCuteDsl

# Filled mechanically from the resolved source export.
MODULES = {
    "fc1_n16": "cake_w4a8_received_tokens_60a8a50c07eb606b97ce",
    "fc1_n32": "cake_w4a8_received_tokens_a9a05f3f9668cd5ffeb9",
    "fc1_n64": "cake_w4a8_received_tokens_d3ed383766e4ccbb1a79",
    "fc2_n16": "cake_w4a8_received_tokens_19aff305db55d3968abc",
    "fc2_n32": "cake_w4a8_received_tokens_60e032d5480bdee93e8a",
    "fc2_n64": "cake_w4a8_received_tokens_0bdc8d560be95f90da81",
    "finalize": "cake_w4a8_received_tokens_6155b894995b7b084893",
    "routing": "cake_w4a8_received_tokens_9c4fa8da35e8dbc19fde",
}


@cache
def spec(stage):
    archs = CompilationContext().TARGET_CUDA_ARCHS
    if not any(
        int(major) == 10 and str(minor).rstrip("af") == "3" for major, minor in archs
    ):
        raise RuntimeError("W4A8 MoE requires SM103a in FLASHINFER_CUDA_ARCH_LIST")
    if os.environ.get("CUTE_DSL_ARCH", "sm_103a").replace("_", "") != "sm103a":
        raise RuntimeError("W4A8 MoE requires CUTE_DSL_ARCH=sm_103a")
    name = MODULES[stage]
    package = f"{__package__}.kernels.sm_103a"
    device = importlib.import_module(f"{package}.{name}_kernel")
    binding = importlib.import_module(f"{package}.{name}_binding")
    digest = hashlib.sha256(
        Path(device.__file__).read_bytes() + Path(binding.__file__).read_bytes()
    ).hexdigest()
    return JitSpecCuteDsl(name, "run", device.compile_program, digest)


@cache
def load(stage):
    name = MODULES[stage]
    binding = importlib.import_module(f"{__package__}.kernels.sm_103a.{name}_binding")
    # Normalization caches are per prepared call; only the compiled FFI entry is shared.
    return binding.Kernel, spec(stage).build_and_load()
