# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepared one-launch MXFP4/MXFP8 MegaMoE session."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import tvm_ffi

from ...comm.torch_symmetric_memory import _enable_symm_mem_for_group
from ...moe_ep.cake_w4a8_megamoe_ep16 import CakeW4A8MegaMoeEp16Weights
from ...moe_ep.weights import PrequantizedMoEWeights
from .jit import load_module
from .workspace import WorkspaceLayout


def _tensor(t, name, shape, dtype, device=None):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if not t.is_cuda or not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA storage")
    if device is not None and t.device != device:
        raise ValueError(f"{name} must be on {device}")


def preprocess_weights(weights):
    from ...moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.weights import (
        preprocess_mega_weights,
    )

    quantized = isinstance(weights, PrequantizedMoEWeights)
    divisor = 2 if quantized else 1
    device = weights.w13.device
    for name, shape in (
        ("w13", (32, 10240, 3072 // divisor)),
        ("w2", (32, 3072, 5120 // divisor)),
    ):
        t = getattr(weights, name)
        if t.dtype not in (
            (torch.int8, torch.uint8) if quantized else (torch.bfloat16,)
        ):
            raise TypeError(f"{name} has an unsupported weight dtype")
        _tensor(t, name, shape, t.dtype, device)
    if quantized:
        for name, shape in (
            ("w13_scale", (32, 10240, 96)),
            ("w2_scale", (32, 3072, 160)),
        ):
            scale = getattr(weights, name)
            if scale.dtype not in (torch.uint8, torch.float32):
                raise TypeError(
                    f"{name} must contain UE8M0 bytes or FP32 powers of two"
                )
            _tensor(scale, name, shape, scale.dtype, device)
            if scale.dtype == torch.float32:
                exponent = torch.log2(scale)
                if not bool(
                    (
                        torch.isfinite(exponent)
                        & (exponent == exponent.round())
                        & (exponent >= -127)
                        & (exponent <= 127)
                    )
                    .all()
                    .item()
                ):
                    raise ValueError(
                        f"{name} must contain representable UE8M0 powers of two"
                    )
    prepared = preprocess_mega_weights(
        weights, hidden_size=3072, intermediate_size=5120
    )
    return CakeW4A8MegaMoeEp16Weights(
        prepared[0][0].reshape(32 * 10240, 1536).view(torch.uint8),
        prepared[1][0].reshape(32 * 3072, 2560).view(torch.uint8),
        prepared[0][1]
        .permute(0, 2, 1)
        .contiguous()
        .view(torch.uint32)
        .reshape(32 * 24, 10240),
        prepared[1][1]
        .permute(0, 2, 1)
        .contiguous()
        .view(torch.uint32)
        .reshape(32 * 40, 3072),
    )


class Session:
    """Collective setup; serial, same-stream forwards with immutable weights."""

    def __init__(self, weights, topk_ids, *, process_group=None):
        if not dist.is_initialized():
            raise RuntimeError(
                "initialize an NCCL process group before creating a session"
            )
        self.group = dist.group.WORLD if process_group is None else process_group
        if (
            dist.get_world_size(self.group) != 16
            or dist.get_backend(self.group) != "nccl"
        ):
            raise ValueError("the session requires an NCCL group of exactly 16 ranks")
        error = None
        self.tokens = -1
        valid_ids = False
        # Every rank must reach this collective, even if a local input is invalid.
        try:
            if not isinstance(weights, CakeW4A8MegaMoeEp16Weights):
                raise TypeError("weights must be prepared CakeW4A8MegaMoeEp16Weights")
            if not isinstance(weights.w13, torch.Tensor):
                raise TypeError("w13 must be a tensor")
            self.device = weights.w13.device
            for name, shape, dtype in (
                ("w13", (327680, 1536), torch.uint8),
                ("w2", (98304, 2560), torch.uint8),
                ("w13_scale", (768, 10240), torch.uint32),
                ("w2_scale", (1280, 3072), torch.uint32),
            ):
                _tensor(getattr(weights, name), name, shape, dtype, self.device)
            prop = torch.cuda.get_device_properties(self.device)
            if (prop.major, prop.minor, prop.multi_processor_count) != (10, 3, 152):
                raise ValueError("the schedule requires SM103a with 152 physical SMs")
            if self.device.index != torch.cuda.current_device():
                raise ValueError("set the current CUDA device to the weights' device")
            if not isinstance(topk_ids, torch.Tensor):
                raise TypeError("topk_ids must be a tensor")
            if topk_ids.ndim != 2 or not 0 <= topk_ids.shape[0] <= 384:
                raise ValueError("topk_ids must have 0 through 384 token rows")
            self.tokens = int(topk_ids.shape[0])
            _tensor(topk_ids, "topk_ids", (self.tokens, 8), torch.int64, self.device)
            valid_ids = bool(((topk_ids >= 0) & (topk_ids < 512)).all().item())
        except (TypeError, ValueError) as exc:
            error = str(exc)
        gathered = [None] * 16
        dist.all_gather_object(
            gathered, (self.tokens, valid_ids, error), group=self.group
        )
        errors = [(rank, e) for rank, (_, _, e) in enumerate(gathered) if e is not None]
        if errors:
            raise ValueError(f"session setup rejected on ranks {errors}")
        if any(n != self.tokens or not valid for n, valid, _ in gathered):
            raise ValueError(
                "all ranks need equal token counts and expert IDs in [0, 512)"
            )
        self.ids = topk_ids.clone()
        self.weights = weights
        self.rank = dist.get_rank(self.group)
        self._stream = torch.cuda.current_stream(self.device).cuda_stream
        self._module = load_module()
        symm_mem.set_backend("NVSHMEM")
        if symm_mem.get_backend(self.device) != "NVSHMEM":
            raise RuntimeError("the NVSHMEM symmetric-memory backend is required")
        _enable_symm_mem_for_group(self.group.group_name)
        self._layout = layout = WorkspaceLayout()
        # Match the current-device allocation form used by native MegaMoE.
        # NVSHMEM TeamManager records that device argument across allocations.
        self._workspace = symm_mem.empty(
            layout.nbytes, dtype=torch.uint8, device="cuda"
        )
        self._handle = symm_mem.rendezvous(self._workspace, group=self.group.group_name)
        pointers = [int(p) for p in self._handle.buffer_ptrs]
        if len(pointers) != 16 or any(p == 0 for p in pointers):
            raise RuntimeError("all 16 symmetric peer mappings must be available")
        bases = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        self._peers = {
            name: (bases + layout.regions[name].offset).view(torch.uint64)
            for name in (
                "indices",
                "recv",
                "totals",
                "signals",
                "input",
                "input_sf",
                "route_weights",
                "combine",
            )
        }
        self._workspace.zero_()
        self._views = v = {
            name: layout.view(self._workspace, name) for name in layout.regions
        }
        p = self._peers
        self._args = [
            None,
            None,
            v["input"].view(torch.uint32),
            v["input_sf"].view(torch.uint8),
            v["ids"],
            v["route_weights"],
            weights.w13,
            weights.w2,
            v["l1"],
            v["l2"],
            weights.w13_scale,
            weights.w2_scale,
            v["l1_sf"],
            v["l2_sf"],
            v["l1_weights"],
            v["l2"],
            v["l2"],
            v["l2_sf"].view(torch.uint8),
            p["combine"],
            v["combine"].view(torch.uint8),
            None,
            v["grid"][1:2],
            v["l1_full"],
            v["l1_empty"],
            v["l2_full"],
            v["l2_empty"],
            self.ids,
            v["send"],
            v["recv"],
            v["indices"],
            p["indices"],
            p["recv"],
            p["totals"],
            v["grid"][:1],
            v["signals"],
            v["status"],
            p["signals"],
            self.rank,
            self.tokens,
            p["input"],
            p["input_sf"],
            p["route_weights"],
            v["l1"],
            v["l1_sf"],
            v["metadata"],
            v["totals"],
            v["claims"],
            layout.sf_ring_tokens // 128,
            32,
            32,
            1,
            40,
            12,
            24,
            40,
            152,
            1,
            1,
        ]
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.group)

    def forward(self, x, router_weights, *, out):
        """Submit one complete forward on the setup stream, without allocation.

        All ranks must call in the same order. Input/output storage must be
        disjoint, and remain alive until the setup stream finishes. Concurrent
        session use and CUDA graph capture are not supported by this API.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graph capture is not supported by this experimental session"
            )
        if torch.cuda.current_stream(self.device).cuda_stream != self._stream:
            raise ValueError(
                "forward must use the stream selected during session setup"
            )
        _tensor(x, "x", (self.tokens, 3072), torch.bfloat16, self.device)
        _tensor(
            router_weights,
            "router_weights",
            (self.tokens, 8),
            torch.float32,
            self.device,
        )
        _tensor(out, "out", (self.tokens, 3072), torch.bfloat16, self.device)
        if self.tokens:
            lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
            for tensor in (
                x,
                router_weights,
                self.ids,
                self._workspace,
                self.weights.w13,
                self.weights.w2,
                self.weights.w13_scale,
                self.weights.w2_scale,
            ):
                begin = tensor.data_ptr()
                end = begin + tensor.numel() * tensor.element_size()
                if lo < end and begin < hi:
                    raise ValueError("out must not overlap input, weights or workspace")
        args = self._args.copy()
        args[0], args[1], args[20] = x, router_weights, out.view(torch.uint8)
        with tvm_ffi.use_torch_stream():
            self._module.run(*args)
        return out
