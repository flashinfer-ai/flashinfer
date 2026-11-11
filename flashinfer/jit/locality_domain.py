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

import functools

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, write_if_different


@functools.cache
def gen_locality_domain_module() -> JitSpec:
    """Return the JIT spec for the CUDA locality-domain support library."""

    uri = "locality_domain"
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)
    source = gen_directory / "locality_domain.cu"
    write_if_different(
        source,
        (jit_env.FLASHINFER_CSRC_DIR / "locality_domain.cu").read_text(),
    )
    return gen_jit_spec(
        uri,
        [source],
        extra_ldflags=["-lcuda"],
        use_fast_math=False,
    )


__all__ = ["gen_locality_domain_module"]
