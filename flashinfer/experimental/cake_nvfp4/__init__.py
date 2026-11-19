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

Experimental ``backend="cake-nvfp4"`` of
:func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache` and
:func:`flashinfer.prefill.trtllm_batch_context_with_kv_cache`: attention
kernels for Nemotron 3.5 Super with an NVFP4 KV cache on ``sm_100a``. See
README.md in this directory.

Importing this package does not load or build the kernels.
"""

from .cake_backend import prefill_workspace_bytes

__all__ = ["prefill_workspace_bytes"]
