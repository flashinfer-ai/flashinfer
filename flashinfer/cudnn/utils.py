"""
Copyright (c) 2024 by FlashInfer team.

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
import threading
import weakref
from typing import Optional, Tuple

import torch

from ..jit.cubin_loader import setup_cubin_loader
from ..jit import gen_cudnn_fmha_module


_attention_handles = threading.local()


class _AttentionHandle:
    def __init__(self, backend):
        self.value = backend.create_handle()
        weakref.finalize(self, backend.destroy_handle, self.value)


def get_cudnn_attention_handle(backend, stream):
    """A handle belongs to its CUDA device and must not race across threads."""
    handles = getattr(_attention_handles, "handles", None)
    if handles is None:
        handles = _attention_handles.handles = {}
    owner = handles.get(stream.device)
    if owner is None:
        with torch.cuda.device(stream.device):
            owner = handles[stream.device] = _AttentionHandle(backend)
    backend.set_stream(owner.value, stream.cuda_stream)
    return owner.value


@functools.cache
def get_cudnn_fmha_gen_module():
    mod = gen_cudnn_fmha_module()
    op = mod.build_and_load()
    for library_path in mod.get_library_paths():
        setup_cubin_loader(library_path)
    return op


# cudnn-frontend 1.30.0 is the first release whose DEFAULT SM100 f16/bf16 SDPA
# forward engine (the CuTe-DSL "FROST" row) serves paged decode: multi-token rows
# ride its decode tile, and an attention sink at q_len_per_req == 1, which the
# classic backend engine declines, falls through to it at plan time. Older
# frontends serve FlashInfer's decode graphs with the backend engine only.
CUDNN_FRONTEND_FROST_DECODE_MIN_VERSION: Tuple[int, int, int] = (1, 30, 0)


def cudnn_frontend_version() -> Optional[Tuple[int, ...]]:
    """The installed cudnn-frontend python package's version, or ``None``.

    ``"1.31.0.dev123"`` parses to ``(1, 31, 0)``; a missing package or a
    version string without a numeric prefix gives ``None``.
    """
    try:
        import cudnn  # noqa: PLC0415 -- optional dependency, imported lazily
    except Exception:  # noqa: BLE001 -- any import failure means "not available"
        return None
    raw = getattr(cudnn, "__version__", None)
    if not raw:
        return None
    parts = []
    for piece in str(raw).split("+")[0].split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def cudnn_frontend_serves_frost_decode(compute_capability: Tuple[int, int]) -> bool:
    """Whether cuDNN's FROST SDPA decode paths serve this device by default.

    True on SM100 / SM103 (the Blackwell parts with the decode tile; Rubin's
    row has none yet) with cudnn-frontend at least
    :data:`CUDNN_FRONTEND_FROST_DECODE_MIN_VERSION`. No environment variable is
    involved: the row is a default engine of the frontend.
    """
    version = cudnn_frontend_version()
    if version is None:
        return False
    # "1.30" parses to (1, 30): pad before comparing so it is not short of 1.30.0.
    if (tuple(version) + (0, 0, 0))[:3] < CUDNN_FRONTEND_FROST_DECODE_MIN_VERSION:
        return False
    return tuple(compute_capability) in ((10, 0), (10, 3))


# cudnn-frontend 1.30 ranks its FROST decode tile ahead of the backend engine for
# multi-token rows only from a declared 2048-token cache; below that the backend's
# prefill-class engine goes first (~8x fa2's time). 1.31 ranks the tile first at
# every cache length (NVIDIA/cudnn-frontend#1420).
CUDNN_FRONTEND_SHORT_CACHE_LEAD_MIN_VERSION: Tuple[int, int, int] = (1, 31, 0)


def cudnn_frontend_leads_short_caches() -> bool:
    """Whether the installed cudnn-frontend ranks its decode tile first for
    multi-token rows at every declared cache length (1.31+)."""
    version = cudnn_frontend_version()
    if version is None:
        return False
    return (tuple(version) + (0, 0, 0))[
        :3
    ] >= CUDNN_FRONTEND_SHORT_CACHE_LEAD_MIN_VERSION
