# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone JIT for the experimental W4A8 forward."""

import functools
import hashlib
from pathlib import Path

from ...jit import env
from ...jit.core import gen_jit_spec, sm103a_nvcc_flags


@functools.cache
def load_module():
    source_dir = Path(__file__).resolve().parent / "csrc"
    sources = [
        source_dir / name
        for name in ("cake_w4a8_megamoe_ep16.cu", "cake_w4a8_megamoe_ep16_binding.cu")
    ]
    identity = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    headers = env.FLASHINFER_CSRC_DIR
    include = env.FLASHINFER_INCLUDE_DIR
    if not (headers / "tvm_ffi_utils.h").is_file():
        headers = Path(__file__).resolve().parents[3] / "csrc"
        include = Path(__file__).resolve().parents[3] / "include"
    return gen_jit_spec(
        name=f"cake_w4a8_megamoe_ep16_{identity}",
        sources=sources,
        extra_cuda_cflags=[*sm103a_nvcc_flags],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[headers, include],
        use_fast_math=False,
    ).build_and_load()
