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

from ...comm.torch_symmetric_memory import _enable_symm_mem_for_group
from ...moe_ep.cake_w4a8_megamoe_ep16_cutedsl import CakeW4A8MegaMoeEp16CuteDslWeights
from ...moe_ep.weights import PrequantizedMoEWeights
from .jit import load_module
from .workspace import WorkspaceLayout
from .weights import pack_pair_major_weights


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
    return CakeW4A8MegaMoeEp16CuteDslWeights(
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
        if not isinstance(weights, CakeW4A8MegaMoeEp16CuteDslWeights):
            raise TypeError(
                "weights must be prepared CakeW4A8MegaMoeEp16CuteDslWeights"
            )
        self.device = weights.w13.device
        prop = torch.cuda.get_device_properties(self.device)
        if (prop.major, prop.minor, prop.multi_processor_count) != (10, 3, 152):
            raise ValueError("the schedule requires SM103a with 152 physical SMs")
        if self.device.index != torch.cuda.current_device():
            raise ValueError("set the current CUDA device to the weights' device")
        for name, shape, dtype in (
            ("w13", (327680, 1536), torch.uint8),
            ("w2", (98304, 2560), torch.uint8),
            ("w13_scale", (768, 10240), torch.uint32),
            ("w2_scale", (1280, 3072), torch.uint32),
        ):
            _tensor(getattr(weights, name), name, shape, dtype, self.device)
        if topk_ids.ndim != 2 or not 0 <= topk_ids.shape[0] <= 384:
            raise ValueError("topk_ids must have 0 through 384 token rows")
        self.tokens = int(topk_ids.shape[0])
        _tensor(topk_ids, "topk_ids", (self.tokens, 8), torch.int64, self.device)
        info = (self.tokens, bool(((topk_ids >= 0) & (topk_ids < 512)).all().item()))
        gathered = [None] * 16
        dist.all_gather_object(gathered, info, group=self.group)
        if any(n != self.tokens or not valid for n, valid in gathered):
            raise ValueError(
                "all ranks need equal token counts and expert IDs in [0, 512)"
            )
        self.ids = topk_ids.clone()
        self.weights = weights
        self._pair_weights = (
            pack_pair_major_weights(weights.w13),
            pack_pair_major_weights(weights.w2),
        )
        self._cached_forward = None
        self._cached_bindings = None
        self.rank = dist.get_rank(self.group)
        self._stream = torch.cuda.current_stream(self.device).cuda_stream
        self._module = load_module(self.group)
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
        multicast = int(self._handle.multicast_ptr)
        if not multicast or multicast in pointers:
            raise RuntimeError("the schedule requires a real NVSHMEM multicast mapping")
        signal_offset = layout.regions["signals"].offset
        self._peers["signals"] = torch.tensor(
            [pointer + signal_offset for pointer in pointers]
            + [multicast + signal_offset],
            dtype=torch.uint64,
            device=self.device,
        )
        self._workspace.zero_()
        self._views = {
            name: layout.view(self._workspace, name) for name in layout.regions
        }
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.group)

    def _validate_forward(self, x, router_weights, out):
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
                *self._pair_weights,
            ):
                begin = tensor.data_ptr()
                end = begin + tensor.numel() * tensor.element_size()
                if lo < end and begin < hi:
                    raise ValueError("out must not overlap input, weights or workspace")

    def prepare(self, x, router_weights, *, out):
        """Bind tensors and the setup stream without reading input contents.

        Call the returned forward repeatedly with live tensor contents. A new
        binding requires prepare again. It retains all storage and always
        submits eagerly on the setup stream; serialize use of the session.
        """
        self._validate_forward(x, router_weights, out)
        arguments = dict(
            W1=self._pair_weights[0],
            W2=self._pair_weights[1],
            X1=self._views["l1"],
            X2=self._views["l2"],
            SW1=self.weights.w13_scale,
            SW2=self.weights.w2_scale,
            SX1=self._views["l1_sf"],
            SX2=self._views["l2_sf"],
            RW=self._views["l1_weights"],
            Q=self._views["l2"],
            QData=self._views["l2"],
            SF=self._views["l2_sf"].view(torch.uint8),
            output_peers=self._peers["combine"],
            slots=self._views["combine"].view(torch.uint8),
            output=out.view(torch.uint8),
            epilogue_grid=self._views["grid"][1:2],
            l1_full=self._views["l1_full"],
            l1_empty=self._views["l1_empty"],
            l2_full=self._views["l2_full"],
            l2_empty=self._views["l2_empty"],
            ids=self.ids,
            send=self._views["send"],
            rank_counts=self._views["recv"],
            indices=self._views["indices"],
            src_peers=self._peers["indices"],
            recv_peers=self._peers["recv"],
            sum_peers=self._peers["totals"],
            grid_counter=self._views["grid"][:1],
            signals=self._views["signals"],
            status=self._views["status"],
            signal_peers=self._peers["signals"],
            rank=self.rank,
            live_tokens=self.tokens,
            token_peers=self._peers["input"],
            sf_peers=self._peers["input_sf"],
            weight_peers=self._peers["route_weights"],
            XData=self._views["l1"],
            XSFData=self._views["l1_sf"],
            metadata=self._views["metadata"],
            recv=self._views["totals"],
            claims=self._views["claims"],
            pool_blocks=self._layout.sf_ring_tokens // 128,
            active_n=32,
            valid_m=32,
            epoch=1,
            fc1_tiles=40,
            fc2_tiles=12,
            fc1_k=24,
            fc2_k=40,
            caller_x=x,
            caller_rw=router_weights,
            staged_x=self._views["input"].view(torch.uint32),
            staged_sf=self._views["input_sf"].view(torch.uint8),
            staged_ids=self._views["ids"],
            staged_rw=self._views["route_weights"],
        )
        arguments.update(
            W1_pair=arguments["W1"],
            W2_pair=arguments["W2"],
            Q8=arguments["Q"],
            Q32=arguments["Q"],
            X1_32=arguments["X1"],
            X2_32=arguments["X2"],
            X1_8=arguments["X1"],
            X2_8=arguments["X2"],
            SW1_1=arguments["SW1"],
            SW2_1=arguments["SW2"],
            SX1_1=arguments["SX1"],
            SX2_1=arguments["SX2"],
            active_n=64,
            valid_m=64,
        )
        owners = (self._handle, self._workspace, self._module)
        launch = self._module.prepare(
            arguments, torch.cuda.current_stream(self.device), owners
        )
        return PreparedForward(launch, out)

    def forward(self, x, router_weights, *, out):
        """Run the full forward, retaining one cached set of fixed bindings."""
        self._validate_forward(x, router_weights, out)
        bindings = (
            id(x),
            id(router_weights),
            id(out),
            x.data_ptr(),
            router_weights.data_ptr(),
            out.data_ptr(),
        )
        if bindings != self._cached_bindings:
            self._cached_forward = self.prepare(x, router_weights, out=out)
            self._cached_bindings = bindings
        return self._cached_forward()


class PreparedForward:
    def __init__(self, launch, output):
        self.launch = launch
        self.output = output

    def __call__(self):
        self.launch()
        return self.output

    def close(self):
        self.launch.close()
