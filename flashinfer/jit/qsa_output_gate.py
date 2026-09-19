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

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_qsa_output_gate_module() -> JitSpec:
    return gen_jit_spec(
        "qsa_output_gate",
        [
            jit_env.FLASHINFER_CSRC_DIR / "qsa_output_gate.cu",
            jit_env.FLASHINFER_CSRC_DIR / "qsa_output_gate_jit_binding.cu",
        ],
        # The gate reproduces an elementwise expression a caller could have
        # written itself, so it is judged against one. Fast math would replace
        # the logistic with its approximate intrinsic and flush a subnormal
        # result to zero, and the second of those is not a rounding difference:
        # a gate below about -87 would take the output to zero rather than to a
        # small number. The kernel is memory-bound, so there is nothing to gain
        # in exchange.
        use_fast_math=False,
    )
