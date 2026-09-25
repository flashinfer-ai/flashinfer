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

"""Explicitly selected Hopper VSA backend; kernel imports are deferred to run."""

import torch

from .metadata import prepare_metadata


class VsaSm90Plan:
    """Wrapper-owned GPU descriptors, with no process-global metadata cache."""

    def __init__(self, metadata, device):
        self.device = torch.device(device)
        self.num_heads = metadata.num_heads
        self.qo_len = metadata.qo_len
        self.kv_len = metadata.kv_len
        self.sm_scale = metadata.sm_scale
        self.schedule = metadata.schedule
        self.captured = False
        with torch.cuda.device(self.device):
            self.indices = metadata.indices.to(self.device)
            self.counts = metadata.counts.to(self.device)
            self.order = metadata.order.to(self.device)
            if self.schedule == "pipelined" and self.order.numel() == 0:
                # Preserve the original pipelined launcher ABI for its unused
                # order argument. A real permutation always has >264 entries.
                self.order = torch.zeros((1,), dtype=torch.int64, device=self.device)
            # Capture must retain an external wait node for the descriptor
            # upload, which was recorded outside the captured stream.
            self.ready = torch.cuda.Event(external=True)
            self.ready.record(torch.cuda.current_stream(self.device))

    def run(self, q, k, v, out=None, lse=None, return_lse=False, enable_pdl=None):
        if return_lse or lse is not None:
            raise ValueError("vsa_sm90_blk64 does not support log-sum-exp outputs")
        if enable_pdl:
            raise ValueError("vsa_sm90_blk64 does not support PDL")
        for name, tensor, length in (
            ("q", q, self.qo_len),
            ("k", k, self.kv_len),
            ("v", v, self.kv_len),
        ):
            self._check_tensor(name, tensor, (self.num_heads, length, 128))
        # Preserve VariableBlockSparseAttentionWrapper's existing DPS ABI:
        # the provided output is [H*M, 1, D], while the return value is HND.
        if out is None:
            result = torch.empty_like(q)
        else:
            self._check_tensor("out", out, (self.num_heads * self.qo_len, 1, 128))
            result = out.view(self.num_heads, self.qo_len, 128)
            begin = out.data_ptr()
            end = begin + out.numel() * out.element_size()
            for tensor in (q, k, v):
                tensor_begin = tensor.data_ptr()
                tensor_end = tensor_begin + tensor.numel() * tensor.element_size()
                if begin < tensor_end and tensor_begin < end:
                    raise ValueError("out must not overlap Q/K/V storage")

        from .aligned_copy import aligned_empty_like, copy_bf16, ensure_aligned

        if self.schedule == "single":
            from .attention_single import run_prepared
        elif self.schedule == "dsplit":
            from .attention96_dsplit import run_prepared
        elif self.schedule == "pipelined":
            from .attention_fast96 import run_prepared
        else:
            from .attention96 import run_prepared

        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if torch.cuda.is_current_stream_capturing():
                if out is None:
                    raise ValueError(
                        "CUDA graph capture requires a preallocated out buffer"
                    )
                self.captured = True
            # A plan may have been built on another stream. Event waits enqueue
            # a dependency without synchronizing the CPU or copying metadata.
            stream.wait_event(self.ready)
            for tensor in (self.indices, self.counts, self.order):
                tensor.record_stream(stream)
            aligned_q = ensure_aligned(q, stream)
            aligned_k = ensure_aligned(k, stream)
            aligned_v = ensure_aligned(v, stream)
            target = (
                result if result.data_ptr() % 16 == 0 else aligned_empty_like(result)
            )
            for tensor in (aligned_q, aligned_k, aligned_v, target, result):
                tensor.record_stream(stream)
            run_prepared(
                aligned_q,
                aligned_k,
                aligned_v,
                self.indices,
                self.counts,
                self.order,
                self.sm_scale,
                target,
            )
            if target is not result:
                copy_bf16(target, result, stream)
        return result

    def check_replan(self):
        if self.captured:
            raise RuntimeError(
                "A captured VSA plan cannot be replaced; create a new wrapper and "
                "keep the original wrapper alive while its graph may replay"
            )

    def _check_tensor(self, name, tensor, shape):
        if tensor.shape != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must have shape {shape} and dtype bfloat16")
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous on the planned device {self.device}"
            )


def create_plan(device, *args, **kwargs):
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("vsa_sm90_blk64 requires Hopper SM90")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Call plan() outside CUDA graph capture")
        metadata = prepare_metadata(*args, **kwargs)
        return VsaSm90Plan(metadata, device)
