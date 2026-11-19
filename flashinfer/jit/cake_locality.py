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

import contextlib
import functools
import os
import pathlib
import secrets

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm107a_nvcc_flags


def _publish_if_different(path: pathlib.Path, content: str) -> None:
    """Write ``content`` to ``path`` unless the file already holds it.

    Source generation runs before the JIT build lock, so another process can be
    compiling ``path`` while this one writes it. A changed file is therefore written
    to a temporary file in the same directory and moved into place with
    ``os.replace``: a reader sees either the old or the new contents, never a
    truncated file. An unchanged file is not rewritten, so warm builds keep their
    source timestamps."""
    try:
        if path.read_text() == content:
            return
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f"{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    # Mode 0o666 less the umask, as a plain open() creates it.
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


@functools.cache
def gen_cake_locality_module() -> JitSpec:
    """Return the JIT spec of the Cake locality runtime: the SM-table probe kernel
    (``flashinfer.cake_locality.PROBE_SOURCE``, sm_107a, CUDA 13.5 or newer) and the
    split locality-domain allocator (``csrc/cake_locality.cu``)."""
    from ..cake_locality import PROBE_SOURCE

    uri = "cake_locality"
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)
    allocator = gen_directory / "cake_locality.cu"
    _publish_if_different(
        allocator,
        (jit_env.FLASHINFER_CSRC_DIR / "cake_locality.cu").read_text(),
    )
    probe = gen_directory / "cake_locality_probe.cu"
    _publish_if_different(probe, PROBE_SOURCE)
    return gen_jit_spec(
        uri,
        [allocator, probe],
        extra_cuda_cflags=sm107a_nvcc_flags,
        extra_ldflags=["-lcuda"],
        use_fast_math=False,
    )


__all__ = ["gen_cake_locality_module"]
