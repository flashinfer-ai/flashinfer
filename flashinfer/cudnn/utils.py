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
