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
"""

from .core import JitSpec, gen_jit_spec
from .env import FLASHINFER_CSRC_DIR


def gen_comm_alltoall_module() -> JitSpec:
    return gen_jit_spec(
        "comm",
        [
            FLASHINFER_CSRC_DIR / "trtllm_alltoall.cu",
            FLASHINFER_CSRC_DIR / "trtllm_alltoall_prepare.cu",
        ],
    )


def gen_trtllm_mnnvl_comm_module() -> JitSpec:
    return gen_jit_spec(
        "trtllm_mnnvl_comm",
        [
            FLASHINFER_CSRC_DIR / "trtllm_mnnvl_allreduce.cu",
        ],
    )
