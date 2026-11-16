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

# Backend of ``flashinfer.cake_gqa_det_train``: packed-varlen causal GQA
# attention training with head dimension 256 on SM100 / SM103 / SM107, the
# deterministic (``det``) package of the family: every sum has one fixed
# order, so the outputs and gradients are bitwise reproducible run to run.  Q ``[T_q, H_q, 256]``, K / V
# ``[T_k, H_kv, 256]`` (BF16, ``H_q / H_kv`` even), segments packed by
# ``cu_seqlens_q`` / ``cu_seqlens_k`` (int32 on the device), causal with the
# bottom-right alignment per segment.  The entry point with autograd is
# ``cake_gqa_det``; ``cake_gqa_det_train_forward`` /
# ``cake_gqa_det_train_backward`` are the two halves.  The registry and the
# host plan live in ``cake_jit``, the loader, launches and autograd in
# ``cake_backend``.  Every switch is an explicit keyword argument; the package
# reads no environment variable.

from .cake_backend import (
    ARCHES,
    HEAD_DIM,
    CakeGQADetDkvAccSource,
    CakeGQADetFunction,
    CakeGQADetWorkspace,
    cake_gqa_det,
    cake_gqa_det_train_backward,
    cake_gqa_det_train_forward,
    cake_gqa_det_train_workspace_bytes,
)

__all__ = [
    "ARCHES",
    "CakeGQADetDkvAccSource",
    "CakeGQADetFunction",
    "CakeGQADetWorkspace",
    "HEAD_DIM",
    "cake_gqa_det",
    "cake_gqa_det_train_backward",
    "cake_gqa_det_train_forward",
    "cake_gqa_det_train_workspace_bytes",
]
