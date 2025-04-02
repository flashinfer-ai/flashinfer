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

import logging

import pytest
import torch
import torch.multiprocessing as mp

from flashinfer.jit.core import FLASHINFER_JIT_DIR, cleanup_compiled_ops
from flashinfer.sampling import get_sampling_module


def _compile_sampling_kernel(rank: int):
    get_sampling_module()


def test_multiprocess_jit_compile_same_kernel():
    # print pid
    cleanup_compiled_ops("sampling")
    # create 4 processes, each process should compile the same kernel
    mp.spawn(fn=_compile_sampling_kernel, args=(), nprocs=4, join=True)
